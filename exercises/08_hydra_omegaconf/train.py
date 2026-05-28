#!/usr/bin/env python3
"""
Exercise 08 — Hydra + OmegaConf driven training.

The Module / DataModule / Dataset are unchanged from ex07 (see ``_lit.py``).
What changed is the *entry point*: instead of building Trainer / Module /
DataModule with argparse + dataclass, we declare them in YAML under ``conf/``
and let Hydra wire everything up at startup.

Three Hydra patterns make up >90% of what you'll see in messy code:

  1. ``@hydra.main(...)`` decorator   ->  the script's true ``main``
  2. ``hydra.utils.instantiate(cfg)`` ->  build object from ``_target_`` + kwargs
  3. ``defaults:`` list in YAML       ->  compose configs from groups

If you see all three, you can read the project. The rest is syntax.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `_lit` importable (used by ``_target_: _lit.LitBEVDetector`` in YAML).
EXERCISE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXERCISE_DIR.parents[1]
sys.path.insert(0, str(EXERCISE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from _hydra_paths import register_repo_root_resolver

register_repo_root_resolver()


# ---------------------------------------------------------------------------
# @hydra.main parses CLI, merges YAML files, builds `cfg`, then calls main().
#   - version_base="1.3" pins Hydra's default behaviours (recommended).
#   - config_path is RELATIVE to this file ("conf/" sits next to train.py).
#   - config_name is the entry yaml WITHOUT the .yaml suffix.
# ---------------------------------------------------------------------------
@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Hydra has already auto-saved the resolved config under
    # ${hydra.run.dir}/.hydra/config.yaml; printing it once at startup is
    # the cheapest way to see what *actually* got merged.
    print("[hydra] resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.get("seed") is not None:
        pl.seed_everything(cfg.seed, workers=True)

    # ---- Step 1: build the BEV grid spec (a plain dataclass) -------------
    # cfg.bev has `_target_: nusc_det.voxelize.BEVConfig` + numeric fields.
    bev_cfg = instantiate(cfg.bev)

    # ---- Step 2: build the DataModule, injecting bev_cfg manually --------
    # `instantiate(cfg.data, foo=bar)` passes extra kwargs alongside what's
    # in the YAML — useful when one object needs another already-built one.
    datamodule = instantiate(cfg.data, bev_cfg=bev_cfg)

    # ---- Step 3: build the LightningModule -------------------------------
    model = instantiate(cfg.model)
    print(
        f"[model] {type(model).__name__}  "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M"
    )

    # ---- Step 4: build callbacks (dict of name -> spec) ------------------
    # We use a dict (not a list) in YAML so individual callbacks can be
    # removed/overridden by key:  python train.py ~callbacks.lr_monitor
    callbacks = [instantiate(spec) for spec in cfg.callbacks.values()]

    # ---- Step 5: build logger(s) -----------------------------------------
    # cfg.logger is a list in YAML. Trainer accepts a single Logger or a list.
    loggers = [instantiate(spec) for spec in cfg.logger] if cfg.logger else False
    if isinstance(loggers, list) and len(loggers) == 1:
        loggers = loggers[0]

    # ---- Step 6: build Trainer, passing in callbacks + loggers -----------
    # `_target_: lightning.pytorch.Trainer` in conf/trainer/default.yaml.
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # ---- Step 7: run (with optional resume) ------------------------------
    ckpt_path = str(cfg.ckpt_path) if cfg.get("ckpt_path") else None
    if ckpt_path:
        print(f"[mode] resuming from {ckpt_path}")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"[done] outputs under: {output_dir}")


if __name__ == "__main__":
    main()
