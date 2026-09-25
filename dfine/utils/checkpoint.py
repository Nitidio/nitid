"""
Checkpoint utilities.

Design decision: the wrapper serialises the model config (YAML) into the
.pth file at save time so users deal with exactly one file.

Checkpoint format:
    {
        "format_version": int,
        "task":    str,
        "model":   state_dict,
        "config":  dict  (formerly separate YAML),
        "names":   {int: str},
        "epoch":   int,
        "metrics": dict,
    }
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from dfine.utils.runs import atomic_output_path

CHECKPOINT_FORMAT_VERSION = 1


def _validate_checkpoint_format(checkpoint: Mapping[str, Any]) -> None:
    version = checkpoint.get("format_version")
    if version is None:  # Legacy wrapped checkpoints predate explicit versioning.
        return
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"Invalid checkpoint format_version: {version!r}")
    if version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format_version {version} is newer than supported version "
            f"{CHECKPOINT_FORMAT_VERSION}"
        )


def _checkpoint_task(checkpoint: Mapping[str, Any]) -> str:
    """Resolve and validate canonical task metadata, including legacy checkpoints."""
    from dfine.tasks import normalize_task

    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise KeyError("Checkpoint has no embedded config mapping")
    config_task = normalize_task(str(config.get("task", "detect")))
    metadata_task = checkpoint.get("task")
    if metadata_task is None:
        return config_task
    resolved_metadata_task = normalize_task(str(metadata_task))
    if resolved_metadata_task != config_task:
        raise ValueError(
            "Checkpoint task metadata does not match its embedded config: "
            f"{resolved_metadata_task!r} != {config_task!r}"
        )
    return config_task


def load_checkpoint(path: str | Path, device: str = "cpu"):
    """
    Load a dfine-wrap checkpoint (.pth with embedded config).

    Returns:
        model:   torch.nn.Module in eval mode
        cfg:     dict (model config)
        names:   dict[int, str]
    """
    from dfine.nn.build import build_model

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = load_checkpoint_state(path)

    if "config" not in ckpt:
        raise KeyError(
            f"'{path}' has no embedded config. "
            "Run tools/convert_checkpoint.py to migrate a raw D-FINE .pth."
        )

    _validate_checkpoint_format(ckpt)
    task = _checkpoint_task(ckpt)
    cfg = deepcopy(ckpt["config"])
    cfg["task"] = task
    names = ckpt.get("names", {})
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg, names


def load_checkpoint_state(path: str | Path) -> dict:
    """Load and return the raw serialized checkpoint dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location="cpu", weights_only=False)


def save_checkpoint(
    path: str | Path,
    model,
    cfg: dict,
    names: dict,
    epoch: int = 0,
    metrics: dict | None = None,
    training_state: dict | None = None,
) -> None:
    """Save a dfine-wrap checkpoint with embedded config."""
    from dfine.tasks import normalize_task

    serialized_config = deepcopy(cfg)
    task = normalize_task(str(serialized_config.get("task", "detect")))
    serialized_config["task"] = task
    with atomic_output_path(path) as temporary:
        torch.save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "task": task,
                "model": model.state_dict(),
                "config": serialized_config,
                "names": names,
                "epoch": epoch,
                "metrics": metrics or {},
                "training_state": training_state or {},
            },
            str(temporary),
        )
