"""
build_model() — constructs a D-FINE nn.Module from a config dict.
Thin wrapper around D-FINE's own model factory.
"""
from __future__ import annotations
import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    """
    Instantiate D-FINE model architecture from config dict.

    This delegates to D-FINE's own build system.
    cfg must contain at minimum: model.type, model.num_classes.

    TODO: import from D-FINE source (added as submodule or installed package).
    """
    raise NotImplementedError(
        "Requires D-FINE source. Add as git submodule: "
        "git submodule add https://github.com/Peterande/D-FINE extern/dfine"
    )
