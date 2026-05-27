#!/usr/bin/env python3
"""
Exercise 07 (part 2/2) — Load a checkpoint and run predictions.

Two things this script demonstrates:

1. ``LitBEVDetector.load_from_checkpoint(...)``
   Builds the Module from a .ckpt file. Because we called
   ``self.save_hyperparameters()`` in __init__, the file already contains
   the arguments needed to reconstruct the Module — you don't have to
   remember them.

2. ``trainer.predict(model, dataloaders=loader)``
   Drives the prediction loop. Lightning will:
     * put the model in eval mode
     * disable gradients
     * iterate the loader
     * call ``model.predict_step(batch, batch_idx)`` per batch
     * collect the per-batch return values into a Python list

The output is then dumped as PNGs (one panel image per frame).

Notes:
  * Resume training (``trainer.fit(ckpt_path=...)``) is a DIFFERENT API:
    it restores optimizer / scheduler / epoch counter too. See train.py.
  * For fine-tuning a pretrained checkpoint you'd combine the two:
        model = LitBEVDetector.load_from_checkpoint(ckpt, lr=1e-4)
        trainer.fit(model, datamodule=dm)  # fresh optimizer with new lr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

from _lit import LitBEVDataModule, LitBEVDetector
from nusc_det.voxelize import BEVConfig


def save_heatmap_png(path: Path, heatmap: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, heatmap, cmap="inferno", vmin=0.0, vmax=1.0)
    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to .ckpt (e.g. runs/07_load_predict_resume/checkpoints/best-*.ckpt)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data_archive/test/j6gen2/e0305816-afe6-4c89-9b5d-1b8aaab1f8b1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/07_load_predict_resume/predictions",
    )
    parser.add_argument("--max-points", type=int, default=60_000)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Limit number of frames to predict on (default: all)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    out_dir = REPO_ROOT / args.output_dir / ckpt_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # (1) Reconstruct the Module from the .ckpt — no need to pass any
    # architecture kwargs; they live inside the file thanks to
    # save_hyperparameters() in LitBEVDetector.__init__.
    # ------------------------------------------------------------------
    print(f"[load] {ckpt_path}")
    model = LitBEVDetector.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    print(
        f"[load] LitBEVDetector ready  "
        f"(hparams: base_channels={model.hparams.base_channels}, "
        f"num_classes={model.hparams.num_classes})"
    )

    # ------------------------------------------------------------------
    # (2) Build a DataModule whose `predict_dataloader` covers all frames.
    # bev_cfg is the only thing we have to pass explicitly — it's a
    # dataclass, so it isn't auto-persisted by save_hyperparameters.
    # ------------------------------------------------------------------
    bev_cfg = BEVConfig(
        x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0,
        z_min=-3.0, z_max=3.0, voxel_size=0.4,
    )
    max_frames = None if args.max_frames < 0 else args.max_frames
    datamodule = LitBEVDataModule(
        data_root=args.data_root,
        bev_cfg=bev_cfg,
        max_points=args.max_points,
        max_frames=max_frames,
        val_frames=0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # (3) Drive the predict loop. No optimizer, no logger, no callbacks.
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    with torch.inference_mode():
        outputs = trainer.predict(model, datamodule=datamodule)
    print(f"[predict] {len(outputs)} batch(es) produced")

    # ------------------------------------------------------------------
    # (4) Dump heatmaps. Each `outputs[i]` is whatever predict_step returned.
    # ------------------------------------------------------------------
    n_saved = 0
    for b_idx, batch_out in enumerate(outputs):
        pred = batch_out["pred_heatmap"]
        gt = batch_out["gt_heatmap"]
        meta = batch_out["meta"]
        bsz = pred.shape[0]
        for i in range(bsz):
            stem = Path(meta[i].get("pcd_path", f"sample{i}")).stem
            tag = f"{b_idx:03d}_{stem}"

            pred_hm = pred[i, 0].numpy()
            gt_hm = gt[i, 0].numpy()
            panel = np.concatenate([pred_hm, gt_hm], axis=1)

            save_heatmap_png(out_dir / f"{tag}_pred.png", pred_hm)
            save_heatmap_png(out_dir / f"{tag}_gt.png", gt_hm)
            save_heatmap_png(out_dir / f"{tag}_panel.png", panel)
            n_saved += 1

    print(f"[done] saved {n_saved} frame(s) under: {out_dir}")


if __name__ == "__main__":
    main()
