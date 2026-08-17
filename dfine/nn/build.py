"""Native model and postprocessor construction for wrapped D-FINE checkpoints."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from dfine.nn.native_build import build_native_model_from_config
from dfine.nn.postprocessor import DFINEPostProcessor
from dfine.tasks import normalize_task


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"D-FINE config {key} must be a positive integer")
    return value


def build_postprocessor(cfg: Mapping[str, Any]) -> nn.Module:
    """Build the native detection postprocessor from checkpoint configuration."""
    if normalize_task(str(cfg.get("task", "detect"))) == "semantic":
        raise NotImplementedError(
            "Semantic segmentation is a recognized task, but its postprocessor "
            "is not integrated yet"
        )
    postprocessor_value = cfg.get("DFINEPostProcessor", {})
    if not isinstance(postprocessor_value, Mapping):
        raise ValueError("D-FINE config DFINEPostProcessor must be a mapping")
    postprocessor_config = copy.deepcopy(dict(postprocessor_value))
    postprocessor_config.pop("type", None)
    postprocessor_config.pop("remap_mscoco_category", None)
    postprocessor_config.setdefault("num_top_queries", 300)
    num_top_queries = _positive_int(postprocessor_config, "num_top_queries")

    decoder_value = cfg.get("DFINETransformer")
    if not isinstance(decoder_value, Mapping):
        raise ValueError("D-FINE config DFINETransformer must be a mapping")
    num_queries = _positive_int(decoder_value, "num_queries")
    num_classes = _positive_int(cfg, "num_classes")
    if bool(cfg.get("use_focal_loss", True)):
        postprocessor_config["num_top_queries"] = min(num_top_queries, num_queries * num_classes)

    return DFINEPostProcessor(
        num_classes=num_classes,
        use_focal_loss=bool(cfg.get("use_focal_loss", True)),
        **postprocessor_config,
    ).eval()


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    """Build the integrated model from a current or legacy checkpoint config."""
    return build_native_model_from_config(cfg)
