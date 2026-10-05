"""Data files shipped inside the nitid wheel."""

from __future__ import annotations

from importlib.resources import files

COCO_NAMES = files(__name__) / "coco_names.yml"
"""COCO class names used to wrap the official pretrained checkpoints."""
