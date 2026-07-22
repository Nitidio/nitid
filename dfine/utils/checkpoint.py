"""
Checkpoint utilities.

Design decision: the wrapper serialises the model config (YAML) into the
.pth file at save time so users deal with exactly one file.

Checkpoint format:
    {
        "model":   state_dict,
        "config":  dict  (formerly separate YAML),
        "names":   {int: str},
        "epoch":   int,
        "metrics": dict,
    }
"""

from __future__ import annotations

from pathlib import Path

import torch

from dfine.utils.runs import atomic_output_path


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

    cfg = ckpt["config"]
    names = ckpt.get("names", {})
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
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
    with atomic_output_path(path) as temporary:
        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg,
                "names": names,
                "epoch": epoch,
                "metrics": metrics or {},
                "training_state": training_state or {},
            },
            str(temporary),
        )
