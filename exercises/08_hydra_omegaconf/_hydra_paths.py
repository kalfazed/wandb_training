"""Repo-root paths for Hydra YAML (${repo_root:}/runs/...)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

EXERCISE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXERCISE_DIR.parents[1]


def register_repo_root_resolver() -> None:
    """Register ``repo_root`` so Hydra output dirs match ex01–ex07 layout."""
    OmegaConf.register_new_resolver("repo_root", lambda: str(REPO_ROOT), replace=True)
