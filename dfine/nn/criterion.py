"""
build_criterion() — wraps D-FINE's DFINECriterion for use in training loop.
"""
from __future__ import annotations


def build_criterion(cfg: dict):
    """
    Returns a callable: criterion(model, batch) → scalar loss.
    Wraps D-FINE's own DFINECriterion.
    """
    raise NotImplementedError(
        "Requires D-FINE source. See dfine/nn/build.py."
    )
