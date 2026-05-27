#!/usr/bin/env python3
"""
Exercise 04 — Built-in Callbacks + WandbLogger (+ a tiny custom callback).

Everything from ex03 (DataModule / Dataset / Module) is unchanged in *role*.
The diff is entirely in how we **observe and persist** training:

  ex03                                      ex04
  ----                                      ----
  CSVLogger only                            WandbLogger (optional) + CSV fallback
  Trainer default checkpoint (last.ckpt)    ModelCheckpoint(monitor="val/loss")
  manual self.log("lr", ...)                LearningRateMonitor callback
  no images                                 LogBEVHeatmapCallback -> wandb panels

Research code in ``nusc_det/`` is still untouched.

When reading messy team code, search for:
  * ``callbacks=[...]`` on ``Trainer``
  * ``WandbLogger`` / ``TensorBoardLogger``
  * classes inheriting ``pl.Callback`` with ``on_*`` methods
Those three searches find 90% of "where did logging / checkpointing happen?"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nusc_det.dataset import NuScenesLidarDetDataset
from nusc_det.losses import detection_loss
from nusc_det.model import BEVDetector
from nusc_det.targets import build_center_targets
from nusc_det.voxelize import BEVConfig, points_to_bev


@dataclass
class Config:
    data_root: str = (
        "/mnt/data_archive/test/j6gen2/e0305816-afe6-4c89-9b5d-1b8aaab1f8b1"
    )
    categories: tuple[str, ...] = ("car",)

    bev: BEVConfig = field(
        default_factory=lambda: BEVConfig(
            x_min=-50.0,
            x_max=50.0,
            y_min=-50.0,
            y_max=50.0,
            z_min=-3.0,
            z_max=3.0,
            voxel_size=0.4,
        )
    )

    base_channels: int = 32
    num_classes: int = 1

    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hm_weight: float = 1.0
    reg_weight: float = 0.1

    log_every: int = 10
    max_points: int = 60_000
    max_frames: int | None = None
    val_frames: int = 2
    num_workers: int = 0
    output_dir: str = "runs/04_callbacks_wandb"

    # wandb
    wandb_project: str = "wandb_training"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    heatmap_log_every_n_epochs: int = 10


# ---------------------------------------------------------------------------
# Data stack — identical to ex03 (Dataset / collate / DataModule).
# ---------------------------------------------------------------------------
class NuScenesBEVDataset(Dataset):
    def __init__(
        self,
        base: NuScenesLidarDetDataset,
        bev_cfg: BEVConfig,
        num_classes: int = 1,
        max_points: int = 60_000,
        random_subsample: bool = True,
    ):
        self.base = base
        self.bev_cfg = bev_cfg
        self.num_classes = num_classes
        self.max_points = max_points
        self.random_subsample = random_subsample

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        points = sample["points"]
        if points.shape[0] > self.max_points:
            n = points.shape[0]
            if self.random_subsample:
                sel = torch.randperm(n)[: self.max_points]
            else:
                sel = torch.arange(min(n, self.max_points))
            points = points[sel]

        bev = points_to_bev(points, self.bev_cfg)
        targets = build_center_targets(
            sample["boxes"], self.bev_cfg, num_classes=self.num_classes
        )
        return {"bev": bev, "targets": targets, "meta": sample["meta"]}


def collate_bev_batch(batch: Sequence[dict]) -> dict:
    return {
        "bev": torch.stack([b["bev"] for b in batch], dim=0),
        "targets": {
            k: torch.stack([b["targets"][k] for b in batch], dim=0)
            for k in batch[0]["targets"]
        },
        "meta": [b["meta"] for b in batch],
    }


class LitBEVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        bev_cfg: BEVConfig,
        categories: Sequence[str] = ("car",),
        num_classes: int = 1,
        max_points: int = 60_000,
        max_frames: int | None = None,
        val_frames: int = 2,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["bev_cfg"])
        self.bev_cfg = bev_cfg
        self._train_ds: Subset | None = None
        self._val_ds: Subset | None = None

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: str | None = None) -> None:
        point_range = (
            self.bev_cfg.x_min,
            self.bev_cfg.y_min,
            self.bev_cfg.z_min,
            self.bev_cfg.x_max,
            self.bev_cfg.y_max,
            self.bev_cfg.z_max,
        )
        base = NuScenesLidarDetDataset(
            self.hparams.data_root,
            categories=tuple(self.hparams.categories),
            point_range=point_range,
        )

        n_total = len(base)
        if self.hparams.max_frames is not None:
            n_total = min(n_total, self.hparams.max_frames)

        n_val = min(self.hparams.val_frames, max(n_total - 1, 0))
        n_train = n_total - n_val
        train_indices = list(range(0, n_train))
        val_indices = list(range(n_train, n_train + n_val))

        train_full = NuScenesBEVDataset(
            base,
            self.bev_cfg,
            num_classes=self.hparams.num_classes,
            max_points=self.hparams.max_points,
            random_subsample=True,
        )
        self._train_ds = Subset(train_full, train_indices)

        if val_indices:
            val_full = NuScenesBEVDataset(
                base,
                self.bev_cfg,
                num_classes=self.hparams.num_classes,
                max_points=self.hparams.max_points,
                random_subsample=False,
            )
            self._val_ds = Subset(val_full, val_indices)
        else:
            self._val_ds = None

        print(
            f"[data] train_frames={len(train_indices)} "
            f"val_frames={len(val_indices)}"
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_bev_batch,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._val_ds is None:
            return None
        return DataLoader(
            self._val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_bev_batch,
        )


class LitBEVDetector(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 1,
        base_channels: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        hm_weight: float = 1.0,
        reg_weight: float = 0.1,
        epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.detector = BEVDetector(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    def forward(self, bev: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.detector(bev)

    def _shared_step(self, batch: dict) -> dict[str, torch.Tensor]:
        outputs = self(batch["bev"])
        return detection_loss(
            outputs,
            batch["targets"],
            hm_weight=self.hparams.hm_weight,
            reg_weight=self.hparams.reg_weight,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses = self._shared_step(batch)
        self.log("train/loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_hm", losses["loss_hm"], on_step=False, on_epoch=True)
        self.log("train/loss_reg", losses["loss_reg"], on_step=False, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        losses = self._shared_step(batch)
        self.log("val/loss", losses["loss"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/loss_hm", losses["loss_hm"], on_epoch=True, sync_dist=True)
        self.log("val/loss_reg", losses["loss_reg"], on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Custom callback — preview of ex06, but minimal: log one val heatmap to wandb.
#
# Callback hook order (simplified, one val epoch):
#   on_validation_epoch_start
#   for each batch:
#       on_validation_batch_start
#       validation_step          (Module)
#       on_validation_batch_end
#   on_validation_epoch_end    <- we log images here
#
# Built-in callbacks (ModelCheckpoint, LR monitor) hook into the same graph.
# Order among callbacks = list order on Trainer, unless a callback sets
# ``priority`` (advanced; rarely needed).
# ---------------------------------------------------------------------------
class LogBEVHeatmapCallback(pl.Callback):
    """Log pred vs GT BEV heatmaps to wandb once every N epochs."""

    def __init__(self, log_every_n_epochs: int = 10):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def _wandb_logger(self, trainer: pl.Trainer):
        # Trainer may hold a single logger or a list (LoggerCollection).
        loggers = trainer.loggers if trainer.loggers else []
        for lg in loggers:
            if lg.__class__.__name__ == "WandbLogger":
                return lg
        return None

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LitBEVDetector) -> None:
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        if trainer.sanity_checking:
            return

        wb = self._wandb_logger(trainer)
        if wb is None:
            return

        val_loader = trainer.datamodule.val_dataloader() if trainer.datamodule else None
        if val_loader is None:
            return

        batch = next(iter(val_loader))
        device = pl_module.device
        bev = batch["bev"].to(device)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}

        pl_module.eval()
        outputs = pl_module(bev)

        pred_hm = outputs["heatmap"].sigmoid()[0, 0].detach().float().cpu().numpy()
        gt_hm = targets["heatmap"][0, 0].detach().float().cpu().numpy()
        panel = np.concatenate([pred_hm, gt_hm], axis=1)

        pcd_name = Path(batch["meta"][0].get("pcd_path", "frame0")).name
        import wandb

        wb.experiment.log(
            {
                "val/bev_heatmap": wandb.Image(
                    panel,
                    caption=f"epoch={trainer.current_epoch}  left=pred  right=gt  {pcd_name}",
                ),
                "epoch": trainer.current_epoch,
            }
        )


def build_logger(cfg: Config, out_dir: Path, use_wandb: bool) -> Logger | list[Logger]:
    """Wandb when available; always keep CSV as a local fallback."""
    loggers: list[Logger] = [CSVLogger(save_dir=str(out_dir), name="csv")]

    if use_wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger

            loggers.append(
                WandbLogger(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    entity=cfg.wandb_entity,
                    save_dir=str(out_dir),
                    log_model=False,
                )
            )
            print(f"[wandb] project={cfg.wandb_project}  run={cfg.wandb_run_name}")
        except ImportError:
            print("[warn] wandb not installed; use --no-wandb or pip install wandb")

    return loggers[0] if len(loggers) == 1 else loggers


def build_callbacks(cfg: Config, ckpt_dir: Path, monitor: str) -> list[pl.Callback]:
    return [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:03d}-{" + monitor.replace("/", "_") + ":.4f}",
            monitor=monitor,
            mode="min",
            save_top_k=1,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        LogBEVHeatmapCallback(log_every_n_epochs=cfg.heatmap_log_every_n_epochs),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--max-points", type=int, default=Config.max_points)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--val-frames", type=int, default=Config.val_frames)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--log-every", type=int, default=Config.log_every)
    parser.add_argument("--wandb-project", type=str, default=Config.wandb_project)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--heatmap-every",
        type=int,
        default=Config.heatmap_log_every_n_epochs,
        help="Log BEV heatmap panel to wandb every N epochs.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip WandbLogger (CSV + local checkpoints still work).",
    )
    args = parser.parse_args()

    max_frames = None if args.max_frames < 0 else args.max_frames
    cfg = Config(
        data_root=args.data_root,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        max_points=args.max_points,
        max_frames=max_frames,
        val_frames=args.val_frames,
        num_workers=args.num_workers,
        log_every=args.log_every,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        heatmap_log_every_n_epochs=args.heatmap_every,
    )

    out_dir = REPO_ROOT / cfg.output_dir
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    monitor = "val/loss" if cfg.val_frames > 0 else "train/loss"
    if cfg.val_frames == 0:
        print("[warn] val_frames=0 -> ModelCheckpoint monitors train/loss instead")

    datamodule = LitBEVDataModule(
        data_root=cfg.data_root,
        bev_cfg=cfg.bev,
        categories=cfg.categories,
        num_classes=cfg.num_classes,
        max_points=cfg.max_points,
        max_frames=cfg.max_frames,
        val_frames=cfg.val_frames,
        batch_size=1,
        num_workers=cfg.num_workers,
    )

    model = LitBEVDetector(
        in_channels=5,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        hm_weight=cfg.hm_weight,
        reg_weight=cfg.reg_weight,
        epochs=cfg.epochs,
    )
    print(f"[model] LitBEVDetector  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"[bev] grid: {cfg.bev.grid_w} x {cfg.bev.grid_h} (voxel={cfg.bev.voxel_size}m)")
    print(f"[checkpoint] monitor={monitor}  dir={ckpt_dir}")

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=10.0,
        log_every_n_steps=cfg.log_every,
        default_root_dir=str(out_dir),
        logger=build_logger(cfg, out_dir, use_wandb=use_wandb),
        callbacks=build_callbacks(cfg, ckpt_dir, monitor=monitor),
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)

    print(f"[done] artifacts under: {out_dir}")
    print(f"       best checkpoint: {ckpt_dir}/ (look for best-*.ckpt)")


if __name__ == "__main__":
    main()
