"""
build_criterion() — wraps D-FINE's DFINECriterion for use in the training loop.
Requires the same extern/dfine submodule as build_model().
"""

from __future__ import annotations

import torch.nn as nn

from dfine.nn.build import _ensure_dfine_on_path, _register_dfine_components


def build_criterion(cfg: dict) -> nn.Module:
    """
    Returns DFINECriterion configured from the checkpoint config dict.
    """
    _ensure_dfine_on_path()

    _register_dfine_components()
    from src.core.workspace import create
    from src.core.yaml_utils import merge_config

    cfg = dict(cfg)
    global_cfg = merge_config(cfg, inplace=False, overwrite=False)
    return create(cfg["criterion"], global_cfg)
