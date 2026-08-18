"""Native D-FINE criterion construction for wrapped checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from dfine.nn.native_build import build_native_criterion_from_config


def build_criterion(cfg: Mapping[str, Any]) -> nn.Module:
    """Build the task-specific criterion from checkpoint config."""
    return build_native_criterion_from_config(cfg)
