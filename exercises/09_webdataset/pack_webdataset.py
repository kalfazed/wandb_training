#!/usr/bin/env python3
"""
Exercise 09 (part 1/2) — Pack NuScenes into WebDataset tar shards.

This script is an OFFLINE step you run ONCE per dataset. It walks the existing
``NuScenesLidarDetDataset`` (the same one ex01-ex08 use), serializes each frame
as a Python pickle blob, and writes the blobs into ``.tar`` shards via
``webdataset.ShardWriter``.

Layout produced (one shard for every ``--maxcount`` samples):

    out_dir/
      nuscenes-train-000000.tar
      nuscenes-train-000001.tar
      ...
      nuscenes-val-000000.tar
      ...

Each tar contains records like::

    000042.pkl     <- pickle.dumps({"points": np.ndarray, "boxes": np.ndarray,
                                    "categories": list[str], "meta": {...}})
    000043.pkl
    ...

A "sample" in WebDataset = all files in the tar sharing the same prefix
(``000042``). Here we use a single ``.pkl`` field per sample (the team
convention you described). The model side only has to ``pickle.loads`` and
build BEV targets on the fly — see ``_lit.py``.

Why store as numpy + plain dict instead of pickling ``Box3D`` dataclasses
directly? Pickle is bound to the producing class path; if anyone renames or
moves ``nusc_det.dataset.Box3D`` the shards become unreadable. Plain numpy
arrays + JSON-able metadata is the portable choice and is what most teams do
in practice. We reconstruct ``Box3D`` lazily at training time.

Usage::

    python exercises/09_webdataset/pack_webdataset.py \
        --data-root /mnt/.../<scene-id> \
        --out-dir   runs/09_webdataset/shards \
        --max-frames 8 \
        --val-frames 2 \
        --maxcount 4
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import webdataset as wds

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nusc_det.dataset import Box3D, NuScenesLidarDetDataset


def _boxes_to_array(boxes: list[Box3D]) -> tuple[np.ndarray, list[str]]:
    """Convert ``list[Box3D]`` to ``(M, 7) float32 + list[str]``.

    Columns: ``[x, y, z, w, l, h, yaw]``. Class name kept separately so we do
    not pickle the dataclass. Empty inputs return a ``(0, 7)`` array.
    """
    if not boxes:
        return np.zeros((0, 7), dtype=np.float32), []
    arr = np.array(
        [[b.x, b.y, b.z, b.w, b.l, b.h, b.yaw] for b in boxes],
        dtype=np.float32,
    )
    cats = [b.category for b in boxes]
    return arr, cats


def _frame_to_payload(sample: dict) -> bytes:
    """Serialize one ``NuScenesLidarDetDataset`` sample into a single blob.

    The receiver (training pipeline) does ``pickle.loads`` and then maps this
    dict through ``points_to_bev`` + ``build_center_targets``.
    """
    points: np.ndarray = sample["points"].numpy().astype(np.float32, copy=False)
    boxes_arr, categories = _boxes_to_array(sample["boxes"])
    payload = {
        "points": points,
        "boxes_arr": boxes_arr,
        "categories": categories,
        "meta": dict(sample["meta"]),
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _write_split(
    base: NuScenesLidarDetDataset,
    indices: list[int],
    out_dir: Path,
    split_name: str,
    maxcount: int,
    maxsize_bytes: int,
) -> int:
    """Write ``indices`` into ``out_dir/nuscenes-{split}-{000000..}.tar``.

    Returns the number of samples actually written.
    """
    pattern = str(out_dir / f"nuscenes-{split_name}-%06d.tar")
    n_written = 0
    # ``ShardWriter`` rotates to a new tar whenever ``maxcount`` samples or
    # ``maxsize`` bytes have been written, whichever comes first.
    with wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize_bytes) as sink:
        for i, idx in enumerate(indices):
            sample = base[idx]
            payload = _frame_to_payload(sample)
            # The dict key ``"pkl"`` becomes the file extension inside the tar:
            #   000042.pkl  <- pickle bytes
            # Anything you put alongside (e.g. ``"meta.json": json.dumps(...)``)
            # would show up as a sibling file with the same ``__key__``.
            sink.write({"__key__": f"{i:06d}", "pkl": payload})
            n_written += 1
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data_archive/test/j6gen2/e0305816-afe6-4c89-9b5d-1b8aaab1f8b1",
        help="Path to the NuScenes-style scene directory (same as ex01-ex08).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/09_webdataset/shards",
        help="Where to write the .tar shards (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="Cap on total frames to pack (smoke test default).",
    )
    parser.add_argument(
        "--val-frames",
        type=int,
        default=2,
        help="Last N frames go into the val split tar.",
    )
    parser.add_argument(
        "--maxcount",
        type=int,
        default=4,
        help="Max samples per tar shard. Smaller -> more shards; pick to keep "
             "each tar a few hundred MB in real datasets.",
    )
    parser.add_argument(
        "--maxsize-mb",
        type=int,
        default=512,
        help="Max bytes per shard (MB). ShardWriter rotates on whichever limit hits first.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = NuScenesLidarDetDataset(
        args.data_root,
        categories=("car",),
        point_range=(-50.0, -50.0, -3.0, 50.0, 50.0, 3.0),
    )
    n_total = min(len(base), args.max_frames)
    n_val = min(args.val_frames, max(n_total - 1, 0))
    n_train = n_total - n_val

    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_total))
    print(f"[pack] dataset frames: total={len(base)} take={n_total} "
          f"train={n_train} val={n_val}")

    maxsize = args.maxsize_mb * 1024 * 1024
    if train_idx:
        n = _write_split(base, train_idx, out_dir, "train", args.maxcount, maxsize)
        print(f"[pack] wrote {n} train sample(s) -> {out_dir}/nuscenes-train-*.tar")
    if val_idx:
        n = _write_split(base, val_idx, out_dir, "val", args.maxcount, maxsize)
        print(f"[pack] wrote {n} val sample(s)   -> {out_dir}/nuscenes-val-*.tar")

    shards = sorted(out_dir.glob("*.tar"))
    print(f"[pack] done. {len(shards)} shard(s):")
    for s in shards:
        print(f"        {s.name:40s} {s.stat().st_size/1024/1024:7.2f} MB")


if __name__ == "__main__":
    main()
