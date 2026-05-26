#!/usr/bin/env python3
"""
Exercise 01 — Pure PyTorch training loop for BEV 3D car detection.

This script intentionally keeps ALL engineering boilerplate visible:
  * manual device placement
  * explicit epoch / batch loops
  * optimizer.zero_grad() -> loss.backward() -> optimizer.step()
  * a hand-rolled learning-rate schedule

In exercise 02 we will move the *same* model/loss/data code into a
LightningModule and delete most of this file.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow running as: python exercises/01_pure_pytorch/train.py
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

    # BEV range in ego/LIDAR frame (meters)
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
    max_points: int = 60_000  # random subsample — full sweeps are ~240k pts
    max_frames: int | None = None  # set e.g. 8 for a quick smoke test
    device: str = "cuda"
    output_dir: str = "runs/01_pure_pytorch"


def collate_single(batch):
    """Batch size is 1; return the sole sample."""
    return batch[0]


def subsample_points(points: torch.Tensor, max_points: int) -> torch.Tensor:
    n = points.shape[0]
    if n <= max_points:
        return points
    idx = torch.randperm(n)[:max_points]
    return points[idx]


def train_one_epoch(
    model: BEVDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "loss_hm": 0.0, "loss_reg": 0.0}
    n_steps = 0

    for sample in loader:
        points = subsample_points(sample["points"], cfg.max_points)
        boxes = sample["boxes"]

        # Voxelize on CPU (simple scatter); move the BEV tensor to the device.
        bev = points_to_bev(points, cfg.bev).unsqueeze(0).to(device)
        targets = build_center_targets(boxes, cfg.bev, num_classes=cfg.num_classes)
        targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}

        outputs = model(bev)
        losses = detection_loss(
            outputs,
            targets,
            hm_weight=cfg.hm_weight,
            reg_weight=cfg.reg_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        totals["loss"] += float(losses["loss"].item())
        totals["loss_hm"] += float(losses["loss_hm"].item())
        totals["loss_reg"] += float(losses["loss_reg"].item())
        n_steps += 1

    return {k: v / max(n_steps, 1) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--device", type=str, default=Config.device)
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
        device=args.device,
        output_dir=args.output_dir,
        max_points=args.max_points,
        max_frames=max_frames,
        log_every=args.log_every,
    )

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        cfg = Config(**{**cfg.__dict__, "device": "cpu"})

    device = torch.device(cfg.device)
    out_dir = REPO_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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

    model = BEVDetector(
        in_channels=5,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] BEVDetector  params={n_params:.2f}M  device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    t0 = time.time()
    for epoch in range(cfg.epochs):
        metrics = train_one_epoch(model, loader, optimizer, cfg, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            elapsed = time.time() - t0
            print(
                f"[epoch {epoch:4d}] loss={metrics['loss']:.4f}  "
                f"hm={metrics['loss_hm']:.4f}  reg={metrics['loss_reg']:.4f}  "
                f"lr={lr_now:.2e}  elapsed={elapsed:.1f}s"
            )

    ckpt_path = out_dir / "final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "bev": cfg.bev.__dict__,
        },
        ckpt_path,
    )
    print(f"[done] saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
