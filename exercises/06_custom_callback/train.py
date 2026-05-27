#!/usr/bin/env python3
"""
Exercise 06 — A proper custom Callback: dump BEV heatmaps to disk.

ex04 logged **one** heatmap to wandb inside a tiny callback.
ex06 replaces that with a **production-style** visualization callback:

  * runs at ``on_validation_epoch_end`` (after all val metrics are logged)
  * optionally iterates the **entire** val DataLoader (not just batch 0)
  * writes ``pred / gt / panel`` PNGs under ``runs/06_.../visualizations/epoch_XXX/``
  * respects DDP: only ``trainer.is_global_zero`` writes files

Module / DataModule / Trainer flags (ex05) are unchanged in *role*.
The lesson is how to **extend** Lightning without touching ``training_step``.

When you inherit messy code, custom callbacks are often 200–800 lines in a
``callbacks/`` package — search ``class .*Callback`` and read ``on_*`` methods.
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
    batch_size: int = 1
    output_dir: str = "runs/06_custom_callback"

    wandb_project: str = "wandb_training"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None

    precision: str = "auto"
    accumulate_grad_batches: int = 1
    devices: int = 1

    # ex06 — disk dump
    dump_heatmap_every_n_epochs: int = 5
    dump_all_val_frames: bool = True


# ---------------------------------------------------------------------------
# Data stack (same as ex03–ex05)
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
# ex06 core — custom Callback with multiple hooks & helpers.
#
# Why NOT put this in validation_step?
#   * validation_step should stay "compute metrics" — fast, simple.
#   * disk I/O + matplotlib + looping extra batches are side effects.
#   * you can disable visualization by removing one callback from the list.
#
# Why on_validation_epoch_end (not on_train_epoch_end)?
#   * we want eval-mode predictions (BatchNorm uses running stats).
#   * val set is the canonical "generalization snapshot" for debugging.
# ---------------------------------------------------------------------------
def _save_heatmap_png(path: Path, heatmap: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> None:
    """Write a single-channel heatmap as PNG (no GUI; safe on headless servers)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, heatmap, cmap="inferno", vmin=vmin, vmax=vmax)
    plt.close("all")


class DumpBEVHeatmapCallback(pl.Callback):
    """Dump predicted / GT car heatmaps to disk every N validation epochs.

    Directory layout::

        {save_root}/
          epoch_000/
            000_00000_pred.png
            000_00000_gt.png
            000_00000_panel.png   # pred | gt side-by-side
          epoch_005/
            ...

    Parameters
    ----------
    save_root:
        Base folder (typically ``runs/06_custom_callback/visualizations``).
    every_n_epochs:
        Run when ``(epoch + 1)`` is divisible by this number.
    dump_all_val_frames:
        If True, iterate the whole val DataLoader. If False, only the first batch.
    max_batches:
        Safety cap when ``dump_all_val_frames`` is True (``None`` = no cap).
    """

    def __init__(
        self,
        save_root: str | Path,
        every_n_epochs: int = 5,
        dump_all_val_frames: bool = True,
        max_batches: int | None = None,
    ):
        super().__init__()
        self.save_root = Path(save_root)
        self.every_n_epochs = every_n_epochs
        self.dump_all_val_frames = dump_all_val_frames
        self.max_batches = max_batches

    def _should_run(self, trainer: pl.Trainer) -> bool:
        if trainer.sanity_checking:
            return False
        if not trainer.is_global_zero:
            # DDP: eight processes would otherwise write the same files eight times.
            return False
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return False
        return True

    def _epoch_dir(self, trainer: pl.Trainer) -> Path:
        return self.save_root / f"epoch_{trainer.current_epoch:03d}"

    @torch.no_grad()
    def _dump_batch(
        self,
        trainer: pl.Trainer,
        pl_module: LitBEVDetector,
        batch: dict,
        batch_idx: int,
        epoch_dir: Path,
    ) -> int:
        device = pl_module.device
        bev = batch["bev"].to(device)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}

        was_training = pl_module.training
        pl_module.eval()
        outputs = pl_module(bev)
        if was_training:
            pl_module.train()

        pred = outputs["heatmap"].sigmoid().detach().cpu()
        gt = targets["heatmap"].detach().cpu()
        bsz = pred.shape[0]

        for i in range(bsz):
            stem = Path(batch["meta"][i].get("pcd_path", f"sample{i}")).stem
            tag = f"{batch_idx:03d}_{stem}"

            pred_hm = pred[i, 0].numpy()
            gt_hm = gt[i, 0].numpy()
            panel = np.concatenate([pred_hm, gt_hm], axis=1)

            _save_heatmap_png(epoch_dir / f"{tag}_pred.png", pred_hm)
            _save_heatmap_png(epoch_dir / f"{tag}_gt.png", gt_hm)
            _save_heatmap_png(epoch_dir / f"{tag}_panel.png", panel)

        return bsz

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LitBEVDetector) -> None:
        if not self._should_run(trainer):
            return

        dm = trainer.datamodule
        val_loader = dm.val_dataloader() if dm is not None else None
        if val_loader is None:
            print("[DumpBEVHeatmapCallback] no val_dataloader — skip")
            return

        epoch_dir = self._epoch_dir(trainer)
        epoch_dir.mkdir(parents=True, exist_ok=True)

        n_saved = 0
        if self.dump_all_val_frames:
            for b_idx, batch in enumerate(val_loader):
                if self.max_batches is not None and b_idx >= self.max_batches:
                    break
                n_saved += self._dump_batch(trainer, pl_module, batch, b_idx, epoch_dir)
        else:
            batch = next(iter(val_loader))
            n_saved += self._dump_batch(trainer, pl_module, batch, 0, epoch_dir)

        print(
            f"[DumpBEVHeatmapCallback] epoch={trainer.current_epoch} "
            f"saved {n_saved} frame(s) -> {epoch_dir}"
        )


def build_logger(cfg: Config, out_dir: Path, use_wandb: bool) -> Logger | list[Logger]:
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
        except ImportError:
            print("[warn] wandb not installed; use --no-wandb")
    return loggers[0] if len(loggers) == 1 else loggers


def build_callbacks(
    cfg: Config, ckpt_dir: Path, vis_dir: Path, monitor: str
) -> list[pl.Callback]:
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
        DumpBEVHeatmapCallback(
            save_root=vis_dir,
            every_n_epochs=cfg.dump_heatmap_every_n_epochs,
            dump_all_val_frames=cfg.dump_all_val_frames,
        ),
    ]


def resolve_precision(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return "bf16-mixed"
            return "16-mixed"
        return "32-true"
    if requested in ("32", "fp32"):
        return "32-true"
    if requested in ("16-mixed", "bf16-mixed"):
        if not torch.cuda.is_available():
            print(f"[warn] precision={requested} needs CUDA; using 32-true")
            return "32-true"
        if requested == "bf16-mixed" and not torch.cuda.is_bf16_supported():
            print("[warn] no bf16; falling back to 16-mixed")
            return "16-mixed"
        return requested
    raise ValueError(f"Unknown precision={requested!r}")


def resolve_strategy(devices: int) -> str:
    return "ddp" if devices > 1 else "auto"


def build_trainer(
    cfg: Config,
    out_dir: Path,
    ckpt_dir: Path,
    vis_dir: Path,
    monitor: str,
    use_wandb: bool,
) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=cfg.devices,
        strategy=resolve_strategy(cfg.devices),
        precision=resolve_precision(cfg.precision),
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gradient_clip_val=10.0,
        log_every_n_steps=cfg.log_every,
        default_root_dir=str(out_dir),
        logger=build_logger(cfg, out_dir, use_wandb=use_wandb),
        callbacks=build_callbacks(cfg, ckpt_dir, vis_dir, monitor),
        enable_progress_bar=True,
    )


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
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--log-every", type=int, default=Config.log_every)
    parser.add_argument("--wandb-project", type=str, default=Config.wandb_project)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--precision", type=str, default=Config.precision,
                        choices=["auto", "32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--accumulate-grad-batches", type=int, default=Config.accumulate_grad_batches)
    parser.add_argument("--devices", type=int, default=Config.devices)
    # ex06
    parser.add_argument(
        "--dump-every",
        type=int,
        default=Config.dump_heatmap_every_n_epochs,
        help="Dump val heatmap PNGs every N epochs.",
    )
    parser.add_argument(
        "--dump-first-val-batch-only",
        action="store_true",
        help="If set, only dump batch 0 (like ex04 wandb callback).",
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
        batch_size=args.batch_size,
        log_every=args.log_every,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        devices=args.devices,
        dump_heatmap_every_n_epochs=args.dump_every,
        dump_all_val_frames=not args.dump_first_val_batch_only,
    )

    out_dir = REPO_ROOT / cfg.output_dir
    ckpt_dir = out_dir / "checkpoints"
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    monitor = "val/loss" if cfg.val_frames > 0 else "train/loss"
    if cfg.val_frames == 0:
        print("[warn] val_frames=0 — DumpBEVHeatmapCallback needs val data")

    datamodule = LitBEVDataModule(
        data_root=cfg.data_root,
        bev_cfg=cfg.bev,
        categories=cfg.categories,
        num_classes=cfg.num_classes,
        max_points=cfg.max_points,
        max_frames=cfg.max_frames,
        val_frames=cfg.val_frames,
        batch_size=cfg.batch_size,
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
    print(f"[vis]   heatmaps -> {vis_dir}  every {cfg.dump_heatmap_every_n_epochs} epoch(s)")

    trainer = build_trainer(cfg, out_dir, ckpt_dir, vis_dir, monitor, use_wandb)
    trainer.fit(model, datamodule=datamodule)

    print(f"[done] PNGs under: {vis_dir}")


if __name__ == "__main__":
    main()
