"""PyTorch Dataset for NuScenes-format LIDAR + 3D box annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from nusc_det.geometry import transform_global_to_ego
from nusc_det.io import NuScenesTables, read_lidar_pcd_bin


@dataclass(frozen=True)
class Box3D:
    """One 3D box in the ego / LIDAR frame."""

    x: float
    y: float
    z: float
    w: float  # width  (size[0] in NuScenes)
    l: float  # length (size[1])
    h: float  # height (size[2])
    yaw: float
    category: str


class NuScenesLidarDetDataset(Dataset):
    """One sample = one LIDAR sweep with car boxes in ego coordinates.

  1. Scan ``sample_data`` for ``LIDAR_CONCAT`` rows whose ``.pcd.bin`` exists.
  2. For each frame, load points and collect ``sample_annotation`` boxes whose
     ``instance`` maps to a category in ``categories``.
  3. Transform box centers/yaws from global -> ego using the sweep's ``ego_pose``.

    With only two physical LIDAR files in the J6Gen2 mini pack, this dataset is
    intentionally tiny — perfect for overfitting while learning the pipeline.
    """

    def __init__(
        self,
        data_root: str | Path,
        categories: Sequence[str] = ("car",),
        point_range: tuple[float, float, float, float, float, float] | None = None,
    ):
        self.data_root = Path(data_root)
        self.ann_dir = self.data_root / "annotation"
        self.categories = tuple(categories)
        self.point_range = point_range

        self.tables = NuScenesTables(self.ann_dir)

        # category_token -> name
        self.cat_token_to_name: dict[str, str] = {
            row["token"]: row["name"] for row in self.tables.list("category")
        }
        self.target_cat_tokens = {
            tok for tok, name in self.cat_token_to_name.items() if name in self.categories
        }

        # instance_token -> category_token
        self.inst_to_cat: dict[str, str] = {
            row["token"]: row["category_token"] for row in self.tables.list("instance")
        }

        # sample_token -> [sample_annotation rows]
        self.ann_by_sample: dict[str, list[dict]] = {}
        for ann in self.tables.list("sample_annotation"):
            self.ann_by_sample.setdefault(ann["sample_token"], []).append(ann)

        # LIDAR_CONCAT sensor token
        lidar_sensor_token = None
        for s in self.tables.list("sensor"):
            if s.get("channel") == "LIDAR_CONCAT":
                lidar_sensor_token = s["token"]
                break
        if lidar_sensor_token is None:
            raise RuntimeError("sensor.json has no LIDAR_CONCAT channel")

        cal_tokens = {
            row["token"]
            for row in self.tables.list("calibrated_sensor")
            if row["sensor_token"] == lidar_sensor_token
        }

        self.frames: list[dict] = []
        for sd in self.tables.list("sample_data"):
            if sd["calibrated_sensor_token"] not in cal_tokens:
                continue
            if not str(sd.get("filename", "")).endswith(".pcd.bin"):
                continue
            pcd_path = self.data_root / sd["filename"]
            if not pcd_path.is_file():
                continue
            self.frames.append(
                {
                    "sample_data_token": sd["token"],
                    "sample_token": sd["sample_token"],
                    "pcd_path": pcd_path,
                    "ego_pose_token": sd["ego_pose_token"],
                }
            )

        self.frames.sort(key=lambda f: str(f["pcd_path"]))

        if not self.frames:
            raise RuntimeError(
                f"No LIDAR .pcd.bin files found under {self.data_root / 'data'}. "
                "Check data_root or copy the mini pack."
            )

    def __len__(self) -> int:
        return len(self.frames)

    def _boxes_for_sample(self, sample_token: str, ego_pose: dict) -> list[Box3D]:
        boxes: list[Box3D] = []
        ego_t = ego_pose["translation"]
        ego_q = ego_pose["rotation"]

        for ann in self.ann_by_sample.get(sample_token, []):
            inst_tok = ann["instance_token"]
            cat_tok = self.inst_to_cat.get(inst_tok)
            if cat_tok not in self.target_cat_tokens:
                continue

            pos, yaw = transform_global_to_ego(
                ann["translation"], ann["rotation"], ego_t, ego_q
            )
            w, l, h = ann["size"]  # NuScenes: width, length, height
            boxes.append(
                Box3D(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    w=float(w),
                    l=float(l),
                    h=float(h),
                    yaw=float(yaw),
                    category=self.cat_token_to_name[cat_tok],
                )
            )
        return boxes

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        if self.point_range is None:
            return points
        xmin, ymin, zmin, xmax, ymax, zmax = self.point_range
        m = (
            (points[:, 0] >= xmin)
            & (points[:, 0] < xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] < ymax)
            & (points[:, 2] >= zmin)
            & (points[:, 2] < zmax)
        )
        return points[m]

    def __getitem__(self, idx: int) -> dict:
        frame = self.frames[idx]
        points = read_lidar_pcd_bin(frame["pcd_path"])
        points = self._filter_points(points)

        ego_pose = self.tables.get("ego_pose", frame["ego_pose_token"])
        boxes = self._boxes_for_sample(frame["sample_token"], ego_pose)

        return {
            "points": torch.from_numpy(points),  # (N, 4) float32
            "boxes": boxes,
            "meta": {
                "sample_data_token": frame["sample_data_token"],
                "sample_token": frame["sample_token"],
                "pcd_path": str(frame["pcd_path"]),
            },
        }
