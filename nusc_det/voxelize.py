"""BEV voxelization: scatter points into a multi-channel bird's-eye grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BEVConfig:
    """Bird's-eye view grid definition in the ego / LIDAR frame."""

    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    z_min: float = -3.0
    z_max: float = 3.0
    voxel_size: float = 0.4

    @property
    def grid_w(self) -> int:
        return int(round((self.x_max - self.x_min) / self.voxel_size))

    @property
    def grid_h(self) -> int:
        return int(round((self.y_max - self.y_min) / self.voxel_size))


def points_to_bev(points: torch.Tensor, cfg: BEVConfig) -> torch.Tensor:
    """Rasterize ``(N, 4)`` points ``[x, y, z, intensity]`` into a BEV tensor.

    Output shape: ``(5, H, W)`` with channels

    0. log1p(point count per cell)
    1. mean height (z)
    2. max height (z)
    3. mean intensity
    4. height standard deviation

    Cells with no points are left at zero. This is a deliberately simple
    hand-crafted BEV encoder so we can focus on the training loop / Lightning
    refactor rather than a learned voxelizer.
    """
    if points.numel() == 0:
        return torch.zeros(5, cfg.grid_h, cfg.grid_w, dtype=torch.float32)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3] if points.shape[1] > 3 else torch.zeros_like(z)

    ix = ((x - cfg.x_min) / cfg.voxel_size).floor().long()
    iy = ((y - cfg.y_min) / cfg.voxel_size).floor().long()

    valid = (
        (ix >= 0)
        & (ix < cfg.grid_w)
        & (iy >= 0)
        & (iy < cfg.grid_h)
        & (z >= cfg.z_min)
        & (z <= cfg.z_max)
    )
    if not valid.any():
        return torch.zeros(5, cfg.grid_h, cfg.grid_w, dtype=torch.float32)

    ix = ix[valid]
    iy = iy[valid]
    z = z[valid]
    intensity = intensity[valid]

    flat = iy * cfg.grid_w + ix
    num_cells = cfg.grid_h * cfg.grid_w

    count = torch.zeros(num_cells, dtype=torch.float32)
    sum_z = torch.zeros(num_cells, dtype=torch.float32)
    sum_z2 = torch.zeros(num_cells, dtype=torch.float32)
    sum_i = torch.zeros(num_cells, dtype=torch.float32)

    count.scatter_add_(0, flat, torch.ones_like(z))
    sum_z.scatter_add_(0, flat, z)
    sum_z2.scatter_add_(0, flat, z * z)
    sum_i.scatter_add_(0, flat, intensity)

    max_z = torch.full((num_cells,), float("-inf"))
    max_z.scatter_reduce_(0, flat, z, reduce="amax", include_self=True)
    max_z[max_z == float("-inf")] = 0.0

    mean_z = sum_z / count.clamp_min(1.0)
    var_z = (sum_z2 / count.clamp_min(1.0)) - mean_z * mean_z
    std_z = var_z.clamp_min(0.0).sqrt()

    bev = torch.stack(
        [
            torch.log1p(count),
            mean_z,
            max_z,
            sum_i / count.clamp_min(1.0),
            std_z,
        ],
        dim=0,
    ).view(5, cfg.grid_h, cfg.grid_w)
    return bev
