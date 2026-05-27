"""Shared LightningModule / LightningDataModule / Dataset for ex08.

Identical to ex07's _lit.py — the whole point of ex08 is that the research
code does NOT change when you adopt a config framework. Only the entry
point (train.py / predict.py) is different.

Why a copy and not an import? Each exercise is self-contained; you can
delete the ``07_load_predict_resume`` folder and ex08 still works.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nusc_det.dataset import NuScenesLidarDetDataset
from nusc_det.losses import detection_loss
from nusc_det.model import BEVDetector
from nusc_det.targets import build_center_targets
from nusc_det.voxelize import BEVConfig, points_to_bev


class NuScenesBEVDataset(Dataset):
    def __init__(
        self,
        base: NuScenesLidarDetDataset,
        bev_cfg: BEVConfig,
        num_classes: int = 1,
        max_points: int = 60_000,
        random_subsample: bool = True,
    ):
        self.base = base
        self.bev_cfg = bev_cfg
        self.num_classes = num_classes
        self.max_points = max_points
        self.random_subsample = random_subsample

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        points = sample["points"]
        if points.shape[0] > self.max_points:
            n = points.shape[0]
            if self.random_subsample:
                sel = torch.randperm(n)[: self.max_points]
            else:
                sel = torch.arange(min(n, self.max_points))
            points = points[sel]

        bev = points_to_bev(points, self.bev_cfg)
        targets = build_center_targets(
            sample["boxes"], self.bev_cfg, num_classes=self.num_classes
        )
        return {"bev": bev, "targets": targets, "meta": sample["meta"]}


def collate_bev_batch(batch: Sequence[dict]) -> dict:
    return {
        "bev": torch.stack([b["bev"] for b in batch], dim=0),
        "targets": {
            k: torch.stack([b["targets"][k] for b in batch], dim=0)
            for k in batch[0]["targets"]
        },
        "meta": [b["meta"] for b in batch],
    }


class LitBEVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        bev_cfg: BEVConfig,
        categories: Sequence[str] = ("car",),
        num_classes: int = 1,
        max_points: int = 60_000,
        max_frames: int | None = None,
        val_frames: int = 2,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["bev_cfg"])
        self.bev_cfg = bev_cfg
        self._train_ds: Subset | None = None
        self._val_ds: Subset | None = None
        self._predict_ds: Subset | None = None

    def setup(self, stage: str | None = None) -> None:
        point_range = (
            self.bev_cfg.x_min,
            self.bev_cfg.y_min,
            self.bev_cfg.z_min,
            self.bev_cfg.x_max,
            self.bev_cfg.y_max,
            self.bev_cfg.z_max,
        )
        base = NuScenesLidarDetDataset(
            self.hparams.data_root,
            categories=tuple(self.hparams.categories),
            point_range=point_range,
        )
        n_total = len(base)
        if self.hparams.max_frames is not None:
            n_total = min(n_total, self.hparams.max_frames)

        full = NuScenesBEVDataset(
            base,
            self.bev_cfg,
            num_classes=self.hparams.num_classes,
            max_points=self.hparams.max_points,
            random_subsample=False,
        )
        self._predict_ds = Subset(full, list(range(n_total)))

        if stage in (None, "fit", "validate"):
            n_val = min(self.hparams.val_frames, max(n_total - 1, 0))
            n_train = n_total - n_val
            train_indices = list(range(0, n_train))
            val_indices = list(range(n_train, n_train + n_val))

            train_full = NuScenesBEVDataset(
                base,
                self.bev_cfg,
                num_classes=self.hparams.num_classes,
                max_points=self.hparams.max_points,
                random_subsample=True,
            )
            self._train_ds = Subset(train_full, train_indices)
            self._val_ds = Subset(full, val_indices) if val_indices else None
            print(
                f"[data] train_frames={len(train_indices)} "
                f"val_frames={len(val_indices) if self._val_ds else 0}"
            )
        else:
            print(f"[data] predict_frames={len(self._predict_ds)}")

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_bev_batch,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._val_ds is None:
            return None
        return DataLoader(
            self._val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_bev_batch,
        )

    def predict_dataloader(self) -> DataLoader:
        assert self._predict_ds is not None
        return DataLoader(
            self._predict_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_bev_batch,
        )


class LitBEVDetector(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 1,
        base_channels: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        hm_weight: float = 1.0,
        reg_weight: float = 0.1,
        epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.detector = BEVDetector(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    def forward(self, bev: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.detector(bev)

    def _shared_step(self, batch: dict) -> dict[str, torch.Tensor]:
        outputs = self(batch["bev"])
        return detection_loss(
            outputs,
            batch["targets"],
            hm_weight=self.hparams.hm_weight,
            reg_weight=self.hparams.reg_weight,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses = self._shared_step(batch)
        self.log("train/loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_hm", losses["loss_hm"], on_step=False, on_epoch=True)
        self.log("train/loss_reg", losses["loss_reg"], on_step=False, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        losses = self._shared_step(batch)
        self.log("val/loss", losses["loss"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/loss_hm", losses["loss_hm"], on_epoch=True, sync_dist=True)
        self.log("val/loss_reg", losses["loss_reg"], on_epoch=True, sync_dist=True)

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        outputs = self(batch["bev"])
        return {
            "pred_heatmap": outputs["heatmap"].sigmoid().cpu(),
            "pred_reg": outputs["reg"].cpu(),
            "gt_heatmap": batch["targets"]["heatmap"].cpu(),
            "meta": batch["meta"],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
