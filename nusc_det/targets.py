"""Center-based detection targets on the BEV grid (CenterPoint-style)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from nusc_det.dataset import Box3D
from nusc_det.voxelize import BEVConfig


def _gaussian_2d(radius: int, sigma: float = 1.0) -> np.ndarray:
    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return g.astype(np.float32)


def gaussian_radius(det_size: tuple[float, float], min_overlap: float = 0.5) -> float:
    """Heuristic radius from CornerNet / CenterNet (size in meters)."""
    height, width = det_size
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = math.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))
    r1 = (b1 + sq1) / 2.0

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    sq2 = math.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))
    r2 = (b2 + sq2) / 2.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    sq3 = math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / 2.0
    return float(min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    """Paint a 2D Gaussian onto ``heatmap`` (in-place), clipped to bounds."""
    h, w = heatmap.shape
    diameter = 2 * radius + 1
    g = _gaussian_2d(radius)

    x, y = int(center[0]), int(center[1])
    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)

    masked = g[radius - top : radius + bottom, radius - left : radius + right]
    heatmap[y - top : y + bottom, x - left : x + right] = np.maximum(
        heatmap[y - top : y + bottom, x - left : x + right],
        masked,
    )


def build_center_targets(
    boxes: Sequence[Box3D],
    cfg: BEVConfig,
    num_classes: int = 1,
) -> dict[str, torch.Tensor]:
    """Build dense supervision for a single-class (car) CenterHead.

    Returns
    -------
    heatmap : ``(num_classes, H, W)`` — Gaussian peaks at object centers.
    reg : ``(8, H, W)`` — sub-pixel offset (2), log size wl (2), z (1),
          sin/cos yaw (2), class id (1). Only populated at GT centers.
    reg_mask : ``(H, W)`` float — 1 at centers, 0 elsewhere (reg loss mask).
  ind : ``(max_objs,)`` int — flat indices of GT centers (for gather in loss).
    """
    H, W = cfg.grid_h, cfg.grid_w
    heatmap = np.zeros((num_classes, H, W), dtype=np.float32)
    reg = np.zeros((8, H, W), dtype=np.float32)
    reg_mask = np.zeros((H, W), dtype=np.float32)

    max_objs = 128
    ind = np.zeros((max_objs,), dtype=np.int64)
    obj_count = 0

    for box in boxes:
        if box.x < cfg.x_min or box.x >= cfg.x_max:
            continue
        if box.y < cfg.y_min or box.y >= cfg.y_max:
            continue

        # Continuous BEV coordinates (cell centers at +0.5).
        cx = (box.x - cfg.x_min) / cfg.voxel_size
        cy = (box.y - cfg.y_min) / cfg.voxel_size
        ix = int(cx)
        iy = int(cy)
        if not (0 <= ix < W and 0 <= iy < H):
            continue

        # Radius in grid cells (convert meters -> cells via voxel_size).
        r_m = gaussian_radius((box.l / cfg.voxel_size, box.w / cfg.voxel_size))
        radius = max(0, min(int(r_m), 10))

        cls_id = 0  # single-class car for now
        draw_gaussian(heatmap[cls_id], (ix, iy), radius)

        reg[0, iy, ix] = cx - ix
        reg[1, iy, ix] = cy - iy
        reg[2, iy, ix] = math.log(max(box.w, 1e-3))
        reg[3, iy, ix] = math.log(max(box.l, 1e-3))
        reg[4, iy, ix] = box.z
        reg[5, iy, ix] = math.sin(box.yaw)
        reg[6, iy, ix] = math.cos(box.yaw)
        reg[7, iy, ix] = float(cls_id)

        reg_mask[iy, ix] = 1.0
        if obj_count < max_objs:
            ind[obj_count] = iy * W + ix
            obj_count += 1

    return {
        "heatmap": torch.from_numpy(heatmap),
        "reg": torch.from_numpy(reg),
        "reg_mask": torch.from_numpy(reg_mask),
        "ind": torch.from_numpy(ind),
        "num_objs": torch.tensor(obj_count, dtype=torch.long),
    }
