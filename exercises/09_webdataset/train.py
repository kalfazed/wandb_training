#!/usr/bin/env python3
"""
Exercise 09 (part 2/2) — Train from pre-packed WebDataset shards.

The only thing that changes vs ex07/ex08 is the DataModule. The model,
loss, callbacks, and Lightning Trainer wiring are identical to the
prior exercises so you can A/B the IO path while keeping everything
else fixed.

Workflow::

    # 1. Pack once (offline)
    python exercises/09_webdataset/pack_webdataset.py --max-frames 8 --val-frames 2

    # 2. Train from the shards
    python exercises/09_webdataset/train.py \
        --train-shards 'runs/09_webdataset/shards/nuscenes-train-{000000..000001}.tar' \
        --val-shards   'runs/09_webdataset/shards/nuscenes-val-000000.tar' \
        --epochs 5

Shard URLs accept brace expansion (``{000000..000123}``) AND plain globs
(``nuscenes-train-*.tar``); the DataModule expands them for you.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

EXERCISE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXERCISE_DIR.parents[1]
sys.path.insert(0, str(EXERCISE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from _lit import LitBEVDetector, LitBEVWebDataModule
from nusc_det.voxelize import BEVConfig


@dataclass
class Config:
    train_shards: str = "runs/09_webdataset/shards/nuscenes-train-{000000..000001}.tar"
    val_shards: str | None = "runs/09_webdataset/shards/nuscenes-val-000000.tar"

    bev: BEVConfig = field(default_factory=BEVConfig)

    num_classes: int = 1
    base_channels: int = 32
    max_points: int = 60_000

    epochs: int = 5
    batch_size: int = 1
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # IterableDataset has no inherent length. These two numbers define what
    # "one epoch" means to Lightning + the progress bar. Set them roughly to
    # how many samples you want to see per epoch.
    train_samples_per_epoch: int = 32
    val_samples_per_epoch: int = 8

    output_dir: str = "runs/09_webdataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-shards", type=str, default=Config.train_shards)
    parser.add_argument("--val-shards", type=str, default=Config.val_shards)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--max-points", type=int, default=Config.max_points)
    parser.add_argument("--train-samples", type=int, default=Config.train_samples_per_epoch)
    parser.add_argument("--val-samples", type=int, default=Config.val_samples_per_epoch)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    args = parser.parse_args()

    cfg = Config(
        train_shards=args.train_shards,
        val_shards=args.val_shards,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        max_points=args.max_points,
        train_samples_per_epoch=args.train_samples,
        val_samples_per_epoch=args.val_samples,
        output_dir=args.output_dir,
    )

    out_dir = REPO_ROOT / cfg.output_dir
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    datamodule = LitBEVWebDataModule(
        train_shards=cfg.train_shards,
        val_shards=cfg.val_shards,
        bev_cfg=cfg.bev,
        num_classes=cfg.num_classes,
        max_points=cfg.max_points,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_samples_per_epoch=cfg.train_samples_per_epoch,
        val_samples_per_epoch=cfg.val_samples_per_epoch,
    )

    model = LitBEVDetector(
        in_channels=5,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
    )
    print(f"[model] LitBEVDetector  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"[data] train_shards={cfg.train_shards}")
    print(f"[data] val_shards  ={cfg.val_shards}")
    print(f"[data] epoch defined as: {cfg.train_samples_per_epoch} train / "
          f"{cfg.val_samples_per_epoch} val samples")

    monitor = "val/loss" if cfg.val_shards else "train/loss"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:03d}",
            auto_insert_metric_name=False,
            monitor=monitor,
            mode="min",
            save_top_k=1,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        default_root_dir=str(out_dir),
        logger=CSVLogger(save_dir=str(out_dir), name="csv"),
        callbacks=callbacks,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=datamodule)

    print(f"[done] checkpoints + metrics under: {out_dir}")


if __name__ == "__main__":
    main()
