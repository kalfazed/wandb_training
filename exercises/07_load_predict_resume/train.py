#!/usr/bin/env python3
"""
Exercise 07 (part 1/2) — Train + Resume.

This script trains a fresh model OR resumes from a checkpoint. The Module
/ DataModule / callbacks are identical in *role* to ex04–ex06; the new
piece is the ``--resume-from`` flag and its plumbing.

Two distinct ways to "load a checkpoint" — pay attention to the difference:

  trainer.fit(model, ckpt_path=...)
      Full restore: weights + optimizer state + lr_scheduler step + epoch
      counter + global_step + callback states (e.g. ModelCheckpoint best
      score). Use this to *resume training*.

  LitBEVDetector.load_from_checkpoint(path)
      Only model weights + hyperparameters. Optimizer / scheduler / epoch
      counter are NOT restored. Use this for *inference* or *fine-tuning*
      where you want a fresh optimizer.

predict.py demonstrates the second form.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger

from _lit import LitBEVDataModule, LitBEVDetector
from nusc_det.voxelize import BEVConfig


@dataclass
class Config:
    data_root: str = (
        "/mnt/data_archive/test/j6gen2/e0305816-afe6-4c89-9b5d-1b8aaab1f8b1"
    )
    categories: tuple[str, ...] = ("car",)
    bev: BEVConfig = field(
        default_factory=lambda: BEVConfig(
            x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0,
            z_min=-3.0, z_max=3.0, voxel_size=0.4,
        )
    )

    base_channels: int = 32
    num_classes: int = 1

    epochs: int = 50           # <-- intentionally short, so resuming is meaningful
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
    output_dir: str = "runs/07_load_predict_resume"

    wandb_project: str = "wandb_training"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def build_callbacks(ckpt_dir: Path, monitor: str) -> list[pl.Callback]:
    return [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:03d}-{" + monitor.replace("/", "_") + ":.4f}",
            monitor=monitor,
            mode="min",
            save_top_k=1,
            save_last=True,           # <-- this is what makes --resume-from work
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def summarize_resume(resume_from: Path | None) -> None:
    """Friendly banner so the user can tell which mode they're in."""
    if resume_from is None:
        print("[mode] training from scratch")
        return

    if not resume_from.is_file():
        raise FileNotFoundError(f"--resume-from path does not exist: {resume_from}")

    # Peek at the checkpoint to print which epoch we'll resume from. This
    # is purely informational — Lightning re-reads the file itself.
    state = torch.load(resume_from, map_location="cpu", weights_only=False)
    saved_epoch = state.get("epoch", "?")
    global_step = state.get("global_step", "?")
    print(f"[mode] resuming from {resume_from}")
    print(f"       checkpoint epoch={saved_epoch}  global_step={global_step}")
    print("       tip: bump --epochs above the saved epoch, otherwise fit() exits immediately.")


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
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a .ckpt to resume training from (typically last.ckpt).",
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
    )

    out_dir = REPO_ROOT / cfg.output_dir
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_from = Path(args.resume_from) if args.resume_from else None
    summarize_resume(resume_from)

    use_wandb = not args.no_wandb
    monitor = "val/loss" if cfg.val_frames > 0 else "train/loss"

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
    print(f"[checkpoint] monitor={monitor}  dir={ckpt_dir}")

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=10.0,
        log_every_n_steps=cfg.log_every,
        default_root_dir=str(out_dir),
        logger=build_logger(cfg, out_dir, use_wandb=use_wandb),
        callbacks=build_callbacks(ckpt_dir, monitor=monitor),
        enable_progress_bar=True,
    )

    # The one-line difference between "train fresh" and "resume":
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=str(resume_from) if resume_from else None,
    )

    print(f"[done] checkpoints under: {ckpt_dir}")
    print("       try:  python exercises/07_load_predict_resume/predict.py "
          f"--ckpt-path {ckpt_dir}/last.ckpt")


if __name__ == "__main__":
    # Make `from _lit import ...` work regardless of cwd.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
