"""I/O helpers: read NuScenes-style pcd.bin point clouds, load annotation JSONs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def read_lidar_pcd_bin(path: str | Path) -> np.ndarray:
    """Read a NuScenes-style binary point cloud and return ``(N, 4)`` float32.

    Layout autodetect:
      * float32 buffer of length divisible by 5 → reshape ``(N, 5)`` then keep
        ``[x, y, z, intensity]`` (drops the ring index column).
      * float32 buffer of length divisible by 4 → reshape ``(N, 4)``.
      * Anything else raises ``ValueError``.

    The fallback to 4-column layout is here because the J6Gen2 ``LIDAR_CONCAT``
    we use in the exercises is a multi-LIDAR fusion that may not carry a ring
    column.
    """
    raw = np.fromfile(str(path), dtype=np.float32)
    n = raw.size
    if n == 0:
        raise ValueError(f"empty pcd.bin file: {path}")

    # Prefer 5-column layout (NuScenes default) when it's plausible.
    if n % 5 == 0 and n >= 5:
        pts = raw.reshape(-1, 5)[:, :4]
    elif n % 4 == 0:
        pts = raw.reshape(-1, 4)
    else:
        raise ValueError(
            f"pcd.bin float count {n} not divisible by 4 or 5 (file: {path})"
        )
    return np.ascontiguousarray(pts, dtype=np.float32)


def load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


class NuScenesTables:
    """Token-keyed access to the NuScenes-style annotation JSONs.

    Loads the standard tables (``scene``, ``sample``, ``sample_data``,
    ``sample_annotation``, ``instance``, ``category``, ``sensor``,
    ``calibrated_sensor``, ``ego_pose``) into memory and builds a
    ``token -> row`` index for O(1) lookups.

    Use :py:meth:`get` for a single record and :py:meth:`list` to iterate
    a whole table.
    """

    REQUIRED: tuple[str, ...] = (
        "scene",
        "sample",
        "sample_data",
        "sample_annotation",
        "instance",
        "category",
        "sensor",
        "calibrated_sensor",
        "ego_pose",
    )

    def __init__(self, ann_dir: str | Path, required: Iterable[str] | None = None):
        self.ann_dir = Path(ann_dir)
        names = tuple(required) if required is not None else self.REQUIRED
        self.tables: dict[str, list[dict]] = {}
        self._index: dict[str, dict[str, dict]] = {}
        for name in names:
            rows = load_json(self.ann_dir / f"{name}.json")
            self.tables[name] = rows
            self._index[name] = {row["token"]: row for row in rows}

    def get(self, table: str, token: str) -> dict:
        return self._index[table][token]

    def list(self, table: str) -> list[dict]:
        return self.tables[table]
