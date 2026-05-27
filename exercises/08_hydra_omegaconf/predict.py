#!/usr/bin/env python3
"""
Exercise 08 — Predict, also driven by Hydra.

The model itself is loaded via Lightning's own ``load_from_checkpoint`` (the
ckpt remembers the constructor kwargs). Hydra is only used to assemble
``bev_cfg`` + ``datamodule`` + the output dir.

Run as:
  python predict.py ckpt_path=runs/08_hydra_omegaconf/.../checkpoints/last.ckpt

You can swap data sources from the CLI just like in train.py:
  python predict.py data=full ckpt_path=...
"""

from __future__ import annotations

import sys
from pathlib import Path

EXERCISE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXERCISE_DIR.parents[1]
sys.path.insert(0, str(EXERCISE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from _lit import LitBEVDetector


def save_heatmap_png(path: Path, heatmap: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, heatmap, cmap="inferno", vmin=0.0, vmax=1.0)
    plt.close("all")


@hydra.main(version_base="1.3", config_path="conf", config_name="predict")
def main(cfg: DictConfig) -> None:
    print("[hydra] resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.is_file():
        # `???` in YAML guarantees the key exists; this just guards against
        # the user supplying a wrong path.
        raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")

    bev_cfg = instantiate(cfg.bev)
    datamodule = instantiate(cfg.data, bev_cfg=bev_cfg)

    print(f"[load] {ckpt_path}")
    model = LitBEVDetector.load_from_checkpoint(str(ckpt_path), map_location="cpu")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    with torch.inference_mode():
        outputs = trainer.predict(model, datamodule=datamodule)

    # Always resolve output paths against REPO_ROOT (cfg.output_dir is relative).
    out_dir = REPO_ROOT / Path(cfg.output_dir) / ckpt_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for b_idx, batch_out in enumerate(outputs):
        pred, gt, meta = batch_out["pred_heatmap"], batch_out["gt_heatmap"], batch_out["meta"]
        for i in range(pred.shape[0]):
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
