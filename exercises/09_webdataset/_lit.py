"""LightningDataModule / LightningModule for ex09 (WebDataset).

The model is identical to ex07/ex08 — the whole point of this exercise is that
swapping the dataset to a WebDataset pipeline is a DataModule-only change.
We just keep a private copy so the exercise folder is self-contained.

What changed vs ex08's ``_lit.py``:
  * ``NuScenesBEVDataset`` (a map-style ``Dataset``) is gone.
  * ``LitBEVDataModule`` (was: builds ``NuScenesLidarDetDataset`` + Subsets) is
    replaced by ``LitBEVWebDataModule`` (was: builds a ``wds.WebDataset``
    pipeline from pre-packed tar shards).
  * ``collate_bev_batch`` is unchanged — webdataset hands us individual sample
    dicts, the DataLoader still does batch collation.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Sequence

import lightning.pytorch as pl
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nusc_det.dataset import Box3D
from nusc_det.losses import detection_loss
from nusc_det.model import BEVDetector
from nusc_det.targets import build_center_targets
from nusc_det.voxelize import BEVConfig, points_to_bev


# ---------------------------------------------------------------------------
# WebDataset pipeline pieces
# ---------------------------------------------------------------------------
def _decode_sample(sample: dict) -> dict:
    """tar member bytes -> python dict.

    ``sample`` is whatever ``wds.WebDataset`` yields BEFORE we ``.map()``:
    a dict with ``__key__`` (the file stem) and one entry per extension.
    Our shards have a single ``.pkl`` per sample, so we just unpickle it.
    """
    payload = pickle.loads(sample["pkl"])
    payload["__key__"] = sample["__key__"]
    return payload


def _arrays_to_boxes(boxes_arr: np.ndarray, categories: Sequence[str]) -> list[Box3D]:
    """Reconstruct ``list[Box3D]`` from the (M, 7) array we packed.

    We avoid pickling ``Box3D`` itself to keep shards portable; the cost is
    this trivial wrapping step in the training pipeline.
    """
    out: list[Box3D] = []
    for row, cat in zip(boxes_arr, categories):
        x, y, z, w, l, h, yaw = row.tolist()
        out.append(Box3D(x=x, y=y, z=z, w=w, l=l, h=h, yaw=yaw, category=cat))
    return out


class _SampleToBEV:
    """Voxelize + draw targets for one sample (callable so it pickles cleanly).

    Same logic as ex08's ``NuScenesBEVDataset.__getitem__``, just adapted to
    plug into ``wds.WebDataset.map(...)``.
    """

    def __init__(
        self,
        bev_cfg: BEVConfig,
        num_classes: int,
        max_points: int,
        random_subsample: bool,
    ):
        self.bev_cfg = bev_cfg
        self.num_classes = num_classes
        self.max_points = max_points
        self.random_subsample = random_subsample

    def __call__(self, payload: dict) -> dict:
        points = torch.from_numpy(payload["points"])
        if points.shape[0] > self.max_points:
            n = points.shape[0]
            if self.random_subsample:
                sel = torch.randperm(n)[: self.max_points]
            else:
                sel = torch.arange(min(n, self.max_points))
            points = points[sel]

        boxes = _arrays_to_boxes(payload["boxes_arr"], payload["categories"])
        bev = points_to_bev(points, self.bev_cfg)
        targets = build_center_targets(boxes, self.bev_cfg, num_classes=self.num_classes)
        return {
            "bev": bev,
            "targets": targets,
            "meta": {**payload["meta"], "__key__": payload["__key__"]},
        }


def collate_bev_batch(batch: Sequence[dict]) -> dict:
    return {
        "bev": torch.stack([b["bev"] for b in batch], dim=0),
        "targets": {
            k: torch.stack([b["targets"][k] for b in batch], dim=0)
            for k in batch[0]["targets"]
        },
        "meta": [b["meta"] for b in batch],
    }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
def _resolve_shard_urls(pattern: str) -> str:
    """Expand a relative shard pattern against ``REPO_ROOT``.

    Accepts either:
      * a brace pattern like ``runs/.../nuscenes-train-{000000..000003}.tar``
      * a glob-ish pattern like ``runs/.../nuscenes-train-*.tar``
        (webdataset itself does NOT glob; we expand to a brace list here)
    """
    if "{" in pattern and "}" in pattern:
        # Already a brace pattern — webdataset / braceexpand handle this.
        return pattern if Path(pattern).is_absolute() or pattern.startswith("/") else str(REPO_ROOT / pattern)

    p = Path(pattern)
    if not p.is_absolute():
        p = REPO_ROOT / p

    if "*" in p.name or "?" in p.name:
        matches = sorted(p.parent.glob(p.name))
        if not matches:
            raise FileNotFoundError(f"No shards match: {p}")
        # Pass the explicit list as a brace expansion-equivalent.
        return [str(m) for m in matches]
    return str(p)


class LitBEVWebDataModule(pl.LightningDataModule):
    """Streams BEV samples from pre-packed tar shards.

    Important differences from ex08's map-style DataModule:
      * Returns an ``IterableDataset`` — no ``__len__``, no random index access.
      * ``shuffle=`` in DataLoader must be ``False``; we shuffle via the
        WebDataset pipeline (``.shuffle(buf)``) and shard order randomization.
      * ``with_epoch(n)`` defines how many samples make up "one epoch" in
        Lightning's eyes (so progress bar + Trainer.max_epochs behave).
    """

    def __init__(
        self,
        train_shards: str,
        val_shards: str | None = None,
        bev_cfg: BEVConfig | None = None,
        num_classes: int = 1,
        max_points: int = 60_000,
        batch_size: int = 1,
        num_workers: int = 0,
        train_samples_per_epoch: int = 64,
        val_samples_per_epoch: int = 8,
        shuffle_buffer: int = 256,
    ):
        super().__init__()
        if bev_cfg is None:
            bev_cfg = BEVConfig()
        self.bev_cfg = bev_cfg
        self.save_hyperparameters(ignore=["bev_cfg"])

    # ------------------------------------------------------------------ build
    def _build_pipeline(
        self,
        shards: str,
        random_subsample: bool,
        shuffle: bool,
        samples_per_epoch: int,
    ) -> wds.WebDataset:
        urls = _resolve_shard_urls(shards)
        # ``shardshuffle`` is INT (buffer of shards to keep in shuffle pool)
        # or False; passing True/None raises a warning in wds>=1.0.
        shard_shuffle_buf = 100 if shuffle else False
        ds = wds.WebDataset(
            urls,
            shardshuffle=shard_shuffle_buf,                # randomize shard order
            nodesplitter=wds.split_by_node,                # rank-aware (DDP)
            workersplitter=wds.split_by_worker,            # worker-aware (DataLoader)
            handler=wds.warn_and_continue,                 # skip corrupt records
            empty_check=False,
        )
        if shuffle:
            ds = ds.shuffle(self.hparams.shuffle_buffer)   # in-buffer sample shuffle
        ds = (
            ds.map(_decode_sample)                         # bytes -> dict
              .map(_SampleToBEV(                           # voxelize + targets
                  bev_cfg=self.bev_cfg,
                  num_classes=self.hparams.num_classes,
                  max_points=self.hparams.max_points,
                  random_subsample=random_subsample,
              ))
        )
        # ``with_epoch`` tells Lightning "one epoch == this many samples";
        # required because IterableDataset has no inherent length.
        return ds.with_epoch(samples_per_epoch)

    # ------------------------------------------------------------- dataloaders
    def train_dataloader(self) -> DataLoader:
        ds = self._build_pipeline(
            shards=self.hparams.train_shards,
            random_subsample=True,
            shuffle=True,
            samples_per_epoch=self.hparams.train_samples_per_epoch,
        )
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_bev_batch,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader | None:
        if not self.hparams.val_shards:
            return None
        ds = self._build_pipeline(
            shards=self.hparams.val_shards,
            random_subsample=False,
            shuffle=False,
            samples_per_epoch=self.hparams.val_samples_per_epoch,
        )
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_bev_batch,
            persistent_workers=self.hparams.num_workers > 0,
        )


# ---------------------------------------------------------------------------
# LightningModule — verbatim from ex08
# ---------------------------------------------------------------------------
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
