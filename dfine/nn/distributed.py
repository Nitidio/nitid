"""Small distributed helpers used by the native D-FINE model core."""

from __future__ import annotations

import torch.distributed as dist


def is_dist_available_and_initialized() -> bool:
    """Return whether a torch distributed process group is ready."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Return the process-group size, or one outside distributed execution."""
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Return the current process rank, or zero outside distributed execution."""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def synchronize() -> None:
    """Synchronize all processes when a multi-process group is active."""
    if get_world_size() > 1:
        dist.barrier()
