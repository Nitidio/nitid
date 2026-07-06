"""Device resolution — normalises all valid device strings to torch.device."""

from __future__ import annotations

import torch


def resolve_device(device: str | int | None) -> str:
    """
    Accept "cuda", "cuda:0", 0 (int), "cpu", None → returns canonical string.

    Examples:
        resolve_device("cuda")   → "cuda:0"  (if GPU available)
        resolve_device(0)        → "cuda:0"
        resolve_device("cpu")    → "cpu"
        resolve_device(None)     → "cuda:0" or "cpu" depending on availability
    """
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(device, int):
        return f"cuda:{device}"
    device = str(device).lower().strip()
    if device == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device
