#!/usr/bin/env python3
"""
Exercise 02 — Same BEV detector, refactored into a LightningModule.

The research code in ``nusc_det/`` (model / loss / dataset / voxelize / targets)
is reused **byte-for-byte** from exercise 01. The only thing that changes is
the engineering layer:

  * ex01 boilerplate that GOES AWAY here (compare to exercises/01_pure_pytorch/train.py)
      ① manual device selection            -> Trainer(accelerator="auto")
      ② model.to(device)                   -> Lightning moves the module
      ③ for epoch / for batch loops        -> Trainer.fit()
      ④ per-tensor .to(device)             -> auto transfer hook
      ⑥ zero_grad / backward / clip / step -> Lightning runs them around training_step
      ⑦ scheduler.step()                   -> returned from configure_optimizers
      ⑧ print + torch.save(...)            -> self.log(...) + Lightning checkpoints

  * ex01 code that STAYS the same
      ⑤ forward + loss          -> body of training_step  (unchanged logic)
        the entire ``nusc_det`` package      -> imported untouched

If you read this file with exercise 01 open in another pane, you will see
that every line in this script either (a) survives unchanged, or (b) maps to
exactly one Lightning hook. That mapping is the whole point of this exercise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

# Allow running as: python exercises/02_lightning_module/train.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nusc_det.dataset import NuScenesLidarDetDataset
from nusc_det.losses import detection_loss
from nusc_det.model import BEVDetector
from nusc_det.targets import build_center_targets
from nusc_det.voxelize import BEVConfig, points_to_bev


# ---------------------------------------------------------------------------
# Config — identical fields to ex01, kept as a plain dataclass for the CLI.
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
    output_dir: str = "runs/02_lightning_module"


def collate_single(batch):
    """Same as ex01: batch_size=1, return the sole sample dict unchanged."""
    return batch[0]


# ---------------------------------------------------------------------------
# LightningModule — the "research code" container.
#
# Three hooks make up the contract Lightning expects:
#   __init__ / forward     — model definition (same as a plain nn.Module)
#   training_step          — what to do with one batch (must return loss)
#   configure_optimizers   — optimizer + (optional) lr scheduler
#
# Everything else is either a defaulted hook (no-op) or an *opt-in* hook we
# choose to override. In this file we also override:
#   on_before_batch_transfer — CPU-side prep that should run BEFORE the
#                              auto device transfer (voxelize, build targets).
# ---------------------------------------------------------------------------
class LitBEVDetector(pl.LightningModule):
    def __init__(
        self,
        bev_cfg: BEVConfig,
        in_channels: int = 5,
        num_classes: int = 1,
        base_channels: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        hm_weight: float = 1.0,
        reg_weight: float = 0.1,
        epochs: int = 200,
        max_points: int = 60_000,
    ):
        super().__init__()
        # save_hyperparameters() stashes the __init__ args on self.hparams AND
        # bakes them into every checkpoint Lightning saves. We exclude bev_cfg
        # because dataclasses are awkward for the YAML serializer; we keep it
        # on the module as a plain attribute instead.
        self.save_hyperparameters(ignore=["bev_cfg"])
        self.bev_cfg = bev_cfg

        # The actual nn.Module — Lightning will discover its parameters
        # automatically because we assign it as a child module.
        self.detector = BEVDetector(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    # forward() is invoked by `self(bev)` below. Lightning does not call this
    # itself during training; it's here for inference / predict_step parity.
    def forward(self, bev: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.detector(bev)

    # -----------------------------------------------------------------------
    # Hook order inside Trainer.fit() for ONE batch (simplified):
    #
    #   DataLoader.__iter__ ──▶ collate_single  (returns CPU sample dict)
    #                            │
    #                            ▼
    #                    on_before_batch_transfer   ← we override this
    #                            │   (voxelize, build targets, all on CPU)
    #                            ▼
    #                    transfer_batch_to_device   ← Lightning's default
    #                            │   (recursively .to(device) every tensor)
    #                            ▼
    #                    on_after_batch_transfer    (no-op here)
    #                            │
    #                            ▼
    #                       training_step           ← we override this
    #                            │   (forward + loss + self.log)
    #                            ▼
    #          (auto) optimizer.zero_grad
    #          (auto) loss.backward
    #          (auto) gradient clipping  (gradient_clip_val on Trainer)
    #          (auto) optimizer.step
    #                            │
    #                            ▼  end of batch
    #
    # And once per epoch, AFTER the last batch:
    #          (auto) lr_scheduler.step()
    #
    # Knowing this order is 80% of what makes "messy" Lightning code readable.
    # -----------------------------------------------------------------------

    def on_before_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        """CPU-side prep.

        The dataset returns ~240k raw points per frame; voxelizing on CPU here
        means the subsequent device transfer carries only a small (5, H, W)
        BEV tensor plus the targets dict, not the whole point cloud.

        We intentionally do this in a hook (not inside training_step) so the
        training_step itself reads as "forward + loss" with no plumbing.
        """
        points = batch["points"]
        if points.shape[0] > self.hparams.max_points:
            idx = torch.randperm(points.shape[0])[: self.hparams.max_points]
            points = points[idx]

        bev = points_to_bev(points, self.bev_cfg).unsqueeze(0)  # (1, 5, H, W)
        targets = build_center_targets(
            batch["boxes"], self.bev_cfg, num_classes=self.hparams.num_classes
        )
        return {"bev": bev, "targets": targets, "meta": batch.get("meta", {})}

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # `batch["bev"]` and tensors inside `batch["targets"]` were moved to
        # self.device by Lightning's auto transfer hook between hooks.
        outputs = self(batch["bev"])
        losses = detection_loss(
            outputs,
            batch["targets"],
            hm_weight=self.hparams.hm_weight,
            reg_weight=self.hparams.reg_weight,
        )

        # self.log replaces every print/tensorboard/wandb call you'd write by
        # hand in ex01. By default it aggregates across the epoch and writes
        # to whatever logger(s) Trainer is configured with.
        self.log("train/loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_hm", losses["loss_hm"], on_step=False, on_epoch=True)
        self.log("train/loss_reg", losses["loss_reg"], on_step=False, on_epoch=True)
        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
        )

        # Returning the scalar loss tells Lightning what to backprop. You
        # *can* return a dict (e.g. {"loss": ..., "extras": ...}) but the
        # "loss" key is what gets backwarded.
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.epochs
        )
        # Returning a dict (instead of just optimizer) is the canonical way
        # to also attach a scheduler. ``interval="epoch"`` is the default,
        # making this 1:1 with ex01's ``scheduler.step()`` per epoch.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# main — almost entirely "config + Trainer construction". The training loop
# itself is gone.
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
        log_every=args.log_every,
    )

    out_dir = REPO_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Dataset / DataLoader: identical to ex01 -----
    point_range = (
        cfg.bev.x_min,
        cfg.bev.y_min,
        cfg.bev.z_min,
        cfg.bev.x_max,
        cfg.bev.y_max,
        cfg.bev.z_max,
    )
    dataset = NuScenesLidarDetDataset(
        cfg.data_root,
        categories=cfg.categories,
        point_range=point_range,
    )
    if cfg.max_frames is not None:
        dataset.frames = dataset.frames[: cfg.max_frames]
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_single,
    )

    print(f"[data] {len(dataset)} lidar frames available")
    for i in range(len(dataset)):
        s = dataset[i]
        print(
            f"  frame {i}: points={s['points'].shape[0]:6d}  "
            f"cars={len(s['boxes']):3d}  file={Path(s['meta']['pcd_path']).name}"
        )

    print(
        f"[bev] grid: {cfg.bev.grid_w} x {cfg.bev.grid_h} "
        f"(voxel_size={cfg.bev.voxel_size}m)"
    )

    # ----- LightningModule -----
    model = LitBEVDetector(
        bev_cfg=cfg.bev,
        in_channels=5,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        hm_weight=cfg.hm_weight,
        reg_weight=cfg.reg_weight,
        epochs=cfg.epochs,
        max_points=cfg.max_points,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] LitBEVDetector  params={n_params:.2f}M")

    # ----- Trainer: replaces the entire ex01 training loop -----
    # Each flag below is a one-liner for something that was a manual loop /
    # if-branch / call in ex01. The mapping is annotated in the README.
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",         # ① replaces cuda-availability fallback
        devices=1,
        gradient_clip_val=10.0,     # ⑥ replaces clip_grad_norm_ inside the loop
        log_every_n_steps=cfg.log_every,
        default_root_dir=str(out_dir),
        logger=CSVLogger(save_dir=str(out_dir), name="csv"),
        enable_progress_bar=True,
        enable_checkpointing=True,  # ⑧ Lightning saves last.ckpt by default
    )

    trainer.fit(model, train_dataloaders=loader)

    print(f"[done] checkpoints + metrics under: {out_dir}")


if __name__ == "__main__":
    main()
