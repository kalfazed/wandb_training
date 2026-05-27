#!/usr/bin/env python3
"""
Exercise 03 — Extract a LightningDataModule + a wrapper Dataset.

Compared to ex02, the *responsibilities* are realigned (the research code in
``nusc_det/`` is still untouched):

  ex02                                       ex03
  ----                                       ----
  train.py builds Dataset + DataLoader   ->  LitBEVDataModule owns them
  LitBEVDetector.on_before_batch_transfer    NuScenesBEVDataset.__getitem__
    voxelize + build_center_targets          voxelize + build_center_targets
  (no val)                                   train / val split in DataModule
                                             + validation_step in the Module

The point of this exercise is the SHAPE of the code, not new tricks:
  * `LightningModule` becomes a pure "research" object: forward + losses + log.
  * `LightningDataModule` is the single place that knows about paths, splits,
    DataLoader kwargs, and the data pipeline.
  * Anyone reading this file can answer "where does data X come from?" by
    looking in exactly one place.

This mirrors how the cleaner Lightning projects you'll inherit are laid out.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nusc_det.dataset import NuScenesLidarDetDataset
from nusc_det.losses import detection_loss
from nusc_det.model import BEVDetector
from nusc_det.targets import build_center_targets
from nusc_det.voxelize import BEVConfig, points_to_bev


# ---------------------------------------------------------------------------
# Config — same fields as ex02, with two new knobs for the val split.
# ---------------------------------------------------------------------------
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
    val_frames: int = 2  # NEW: last N frames are held out as the val split
    num_workers: int = 0
    output_dir: str = "runs/03_lightning_datamodule"


# ---------------------------------------------------------------------------
# Wrapper Dataset — pushes voxelize + target rendering DOWN into __getitem__,
# so the LightningModule doesn't need on_before_batch_transfer anymore.
#
# Trade-offs to be aware of (these are the kind of decisions you'll need to
# explain at code review in the new team):
#   + Module becomes thin and "research-only".
#   + Multi-worker DataLoaders parallelize voxelize/target across CPUs.
#   + The same wrapper is reusable for train / val / test / predict.
#   - The pipeline runs once per __getitem__, so randomness here cannot
#     depend on training step or self.hparams. (In ex02 it could, because
#     the hook ran inside the Module.) For training-state-dependent
#     transforms, prefer keeping them in the Module hook.
# ---------------------------------------------------------------------------
class NuScenesBEVDataset(Dataset):
    """Adapter: raw NuScenes frame -> (BEV tensor, target tensors)."""

    def __init__(
        self,
        base: NuScenesLidarDetDataset,
        bev_cfg: BEVConfig,
        num_classes: int = 1,
        max_points: int = 60_000,
    ):
        self.base = base
        self.bev_cfg = bev_cfg
        self.num_classes = num_classes
        self.max_points = max_points

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        points = sample["points"]
        if points.shape[0] > self.max_points:
            sel = torch.randperm(points.shape[0])[: self.max_points]
            points = points[sel]

        bev = points_to_bev(points, self.bev_cfg)  # (5, H, W) — NO batch dim
        targets = build_center_targets(
            sample["boxes"], self.bev_cfg, num_classes=self.num_classes
        )
        return {"bev": bev, "targets": targets, "meta": sample["meta"]}


def collate_bev_batch(batch: Sequence[dict]) -> dict:
    """Stack tensor fields along a new batch dim; carry ``meta`` as a list.

    Why a custom collate? ``meta`` contains strings (file paths, tokens) that
    default_collate turns into awkward per-key lists; doing it ourselves keeps
    a clean ``batch["meta"] -> list[dict]`` shape, which is what callbacks and
    visualization code (later exercises) will want.
    """
    return {
        "bev": torch.stack([b["bev"] for b in batch], dim=0),
        "targets": {
            k: torch.stack([b["targets"][k] for b in batch], dim=0)
            for k in batch[0]["targets"]
        },
        "meta": [b["meta"] for b in batch],
    }


# ---------------------------------------------------------------------------
# LightningDataModule — the new contract for "data".
#
# Five conventional hooks (you can override any subset):
#
#   __init__       : remember paths / kwargs, no I/O
#   prepare_data() : one-time, single-process work (downloads etc).
#                    NOT called per-rank in DDP, so don't mutate self here.
#   setup(stage)   : build Datasets for the given stage
#                    ("fit", "validate", "test", "predict").
#                    Called on every rank.
#   train_dataloader / val_dataloader / test_dataloader / predict_dataloader
#                  : return DataLoaders Trainer can iterate.
#   teardown(stage): cleanup, mirror of setup.
#
# In bigger projects this is the file where 80% of "where does the data
# actually come from?" questions are answered. Make it boring and explicit.
# ---------------------------------------------------------------------------
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
        # save_hyperparameters() also exists on DataModule; same semantics as
        # in the Module — args get stashed on self.hparams and serialized.
        self.save_hyperparameters(ignore=["bev_cfg"])
        self.bev_cfg = bev_cfg

        self._train_ds: Subset | None = None
        self._val_ds: Subset | None = None

    def prepare_data(self) -> None:
        # No download / no global file mutation. Defined here purely so you
        # see the hook exist — empty implementations are normal.
        return None

    def setup(self, stage: str | None = None) -> None:
        """Build the train/val Datasets. Called by Trainer on every rank."""
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

        # Apply --max-frames first, then carve off the last --val-frames as
        # a deterministic held-out split (no shuffling: the dataset is tiny
        # and we want runs to be reproducible across exercises).
        n_total = len(base)
        if self.hparams.max_frames is not None:
            n_total = min(n_total, self.hparams.max_frames)

        n_val = min(self.hparams.val_frames, max(n_total - 1, 0))
        n_train = n_total - n_val
        train_indices = list(range(0, n_train))
        val_indices = list(range(n_train, n_train + n_val))

        # ``Subset`` is the idiomatic way to take a view of a Dataset by
        # index. ``NuScenesBEVDataset`` only uses ``len()`` and ``[i]`` on its
        # base, so wrapping a Subset works transparently.
        full = NuScenesBEVDataset(
            base,
            self.bev_cfg,
            num_classes=self.hparams.num_classes,
            max_points=self.hparams.max_points,
        )
        self._train_ds = Subset(full, train_indices)
        self._val_ds = Subset(full, val_indices) if val_indices else None

        print(
            f"[data] train_frames={len(train_indices)} "
            f"val_frames={len(val_indices)}"
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "call setup() first"
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


# ---------------------------------------------------------------------------
# LightningModule — slimmed down. No more data-pipeline hook here.
#
# Diff vs ex02:
#   - DELETED: on_before_batch_transfer  (lives in NuScenesBEVDataset now)
#   - DELETED: max_points hyperparameter (it's a DataModule concern)
#   - ADDED  : validation_step           (mirrors training_step, no backward)
#   - ADDED  : _shared_step              (so train/val don't drift apart)
#
# This is the form a Module "should" have once the data side is sorted: it
# basically reads as `outputs = self(x); loss = criterion(outputs, y); log`.
# ---------------------------------------------------------------------------
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
        # on_step + on_epoch lets the progress bar show live loss AND the
        # logger get one epoch-averaged scalar — common Lightning idiom.
        self.log("train/loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_hm", losses["loss_hm"], on_step=False, on_epoch=True)
        self.log("train/loss_reg", losses["loss_reg"], on_step=False, on_epoch=True)
        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
        )
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        # No gradients here — Lightning wraps validation_step in torch.no_grad
        # and switches BatchNorm / Dropout to eval mode automatically. You
        # don't need to write `with torch.no_grad():` or `model.eval()`.
        losses = self._shared_step(batch)
        # sync_dist=True is the right default for val on multi-GPU; harmless
        # on single-GPU. Including it here so the habit transfers.
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
# main — even thinner than ex02. Notice that train.py now knows about NEITHER
# Dataset internals NOR data-pipeline transforms. It only assembles three
# things: DataModule, Module, Trainer.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--max-points", type=int, default=Config.max_points)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Use only the first N frames (default: all). Try 8 for a quick run.",
    )
    parser.add_argument(
        "--val-frames",
        type=int,
        default=Config.val_frames,
        help="Hold out the last N frames as a val split.",
    )
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--log-every", type=int, default=Config.log_every)
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
    )

    out_dir = REPO_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] LitBEVDetector  params={n_params:.2f}M")
    print(
        f"[bev] grid: {cfg.bev.grid_w} x {cfg.bev.grid_h} "
        f"(voxel_size={cfg.bev.voxel_size}m)"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=10.0,
        log_every_n_steps=cfg.log_every,
        default_root_dir=str(out_dir),
        logger=CSVLogger(save_dir=str(out_dir), name="csv"),
        enable_progress_bar=True,
        enable_checkpointing=True,
        # Validate every epoch by default; you'll change this in ex05 via
        # `check_val_every_n_epoch` / `val_check_interval`.
    )

    # Notice: we pass `datamodule=` instead of `train_dataloaders=`. This is
    # the canonical Lightning entry point once you have a DataModule — Trainer
    # picks up train_dataloader / val_dataloader / test_dataloader from it.
    trainer.fit(model, datamodule=datamodule)

    print(f"[done] checkpoints + metrics under: {out_dir}")


if __name__ == "__main__":
    main()
