"""Construct the integrated D-FINE detection and instance-segmentation core.

It accepts both native model-size settings and the legacy self-contained YAML
configuration embedded in existing nitid checkpoints.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from dfine.tasks import Task, normalize_task

from .architecture import DFINEModel, DFINETransformer, HGNetv2, HybridEncoder
from .configs import get_model_config
from .losses import DFINECriterion, HungarianMatcher


def _checkpoint_task(config: Mapping[str, Any]) -> Task:
    return normalize_task(str(config.get("task", "detect")))


def _require_native_runtime(task: Task) -> None:
    if task == "semantic":
        raise NotImplementedError(
            "Semantic segmentation is a recognized task, but its native model "
            "and criterion are not integrated yet"
        )


def _component_config(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"D-FINE config is missing the {key!r} component mapping")
    component = copy.deepcopy(dict(value))
    component.pop("type", None)
    return component


def _validated_image_size(config: Mapping[str, Any]) -> tuple[int, int] | None:
    value = config.get("eval_spatial_size")
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("eval_spatial_size must contain exactly [height, width]")
    height, width = value
    if not isinstance(height, int) or not isinstance(width, int) or height < 1 or width < 1:
        raise ValueError("eval_spatial_size height and width must be positive integers")
    return height, width


def _compose_native_model(
    *,
    backbone_config: dict[str, Any],
    encoder_config: dict[str, Any],
    decoder_config: dict[str, Any],
    num_classes: int,
    task: Task,
    image_size: tuple[int, int] | None,
    in_channels: int,
    device: str | torch.device | None,
) -> DFINEModel:
    _require_native_runtime(task)
    enable_mask_head = task == "segment"
    backbone_config["pretrained"] = False
    encoder_config["eval_spatial_size"] = image_size
    decoder_config["eval_spatial_size"] = image_size
    decoder_config["enable_mask_head"] = enable_mask_head

    encoder_strides = encoder_config["feat_strides"]
    if enable_mask_head and 8 not in encoder_strides:
        return_indices = backbone_config["return_idx"]
        if 1 not in return_indices:
            backbone_config["return_idx"] = [1, *return_indices]
        backbone_name = backbone_config["name"]
        stage_channels = HGNetv2.arch_configs[backbone_name]["stage_config"]["stage2"][2]
        decoder_config["mask_low_level_ch"] = stage_channels

    backbone = HGNetv2(in_channels=in_channels, **backbone_config)
    encoder = HybridEncoder(**encoder_config)
    decoder = DFINETransformer(num_classes=num_classes, **decoder_config)
    native_model = DFINEModel(backbone=backbone, encoder=encoder, decoder=decoder)
    return native_model.to(device) if device is not None else native_model


def build_native_model(
    model: str,
    *,
    num_classes: int,
    task: str,
    image_size: tuple[int, int] | None = None,
    in_channels: int = 3,
    device: str | torch.device | None = None,
) -> DFINEModel:
    """Build a native D-FINE model without loading pretrained weights."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if in_channels not in (3, 4):
        raise ValueError(f"in_channels must be 3 or 4, got {in_channels}")

    resolved_task = normalize_task(task)
    _require_native_runtime(resolved_task)
    config = get_model_config(model)

    return _compose_native_model(
        backbone_config=config["HGNetv2"],
        encoder_config=config["HybridEncoder"],
        decoder_config=config["DFINETransformer"],
        num_classes=num_classes,
        task=resolved_task,
        image_size=image_size,
        in_channels=in_channels,
        device=device,
    )


def build_native_model_from_config(
    config: Mapping[str, Any],
    *,
    device: str | torch.device | None = None,
) -> DFINEModel:
    """Build from a current or legacy self-contained checkpoint configuration."""
    num_classes = config.get("num_classes")
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("D-FINE config num_classes must be a positive integer")
    in_channels = config.get("in_channels", 3)
    if not isinstance(in_channels, int) or in_channels not in (3, 4):
        raise ValueError("D-FINE config in_channels must be 3 or 4")

    return _compose_native_model(
        backbone_config=_component_config(config, "HGNetv2"),
        encoder_config=_component_config(config, "HybridEncoder"),
        decoder_config=_component_config(config, "DFINETransformer"),
        num_classes=num_classes,
        task=_checkpoint_task(config),
        image_size=_validated_image_size(config),
        in_channels=in_channels,
        device=device,
    )


def build_native_criterion(
    model: str,
    *,
    num_classes: int,
    task: str,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Build the task-aware D-FINE detection/instance-segmentation criterion."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    resolved_task = normalize_task(task)
    _require_native_runtime(resolved_task)
    config = get_model_config(model)

    criterion_config = config["DFINECriterion"]
    if resolved_task == "segment" and "masks" not in criterion_config["losses"]:
        criterion_config["losses"].append("masks")

    matcher = HungarianMatcher(**config["matcher"])
    return DFINECriterion(
        matcher=matcher,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        **criterion_config,
    )


def build_native_criterion_from_config(config: Mapping[str, Any]) -> nn.Module:
    """Build a criterion from a current or legacy checkpoint configuration."""
    num_classes = config.get("num_classes")
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("D-FINE config num_classes must be a positive integer")

    task = _checkpoint_task(config)
    _require_native_runtime(task)
    criterion_config = _component_config(config, "DFINECriterion")
    matcher_value = criterion_config.pop("matcher", config.get("matcher"))
    if not isinstance(matcher_value, Mapping):
        raise ValueError("D-FINE criterion config is missing its matcher mapping")
    matcher_config = copy.deepcopy(dict(matcher_value))
    matcher_config.pop("type", None)
    matcher_config.setdefault("use_focal_loss", bool(config.get("use_focal_loss", True)))

    if task == "segment" and "masks" not in criterion_config["losses"]:
        criterion_config["losses"].append("masks")
    criterion_config.setdefault("label_smoothing", 0.0)

    return DFINECriterion(
        matcher=HungarianMatcher(**matcher_config),
        num_classes=num_classes,
        **criterion_config,
    )
