"""Construct the integrated D-FINE detection and instance-segmentation core.

This module is intentionally separate from :mod:`dfine.nn.build` during the
native-core migration. The public loader continues to use the reference
submodule until Phase 2's parity gate is complete.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn as nn

from .architecture import DFINEModel, DFINETransformer, HGNetv2, HybridEncoder
from .configs import get_model_config
from .losses import DFINECriterion, HungarianMatcher

Task = Literal["detect", "segment"]
SUPPORTED_TASKS: tuple[Task, ...] = ("detect", "segment")


def normalize_task(task: str) -> Task:
    """Validate and normalize a public D-FINE task name."""
    normalized = task.lower().strip()
    if normalized not in SUPPORTED_TASKS:
        supported = ", ".join(SUPPORTED_TASKS)
        raise ValueError(f"Unsupported task {task!r}. Choose: {supported}")
    return cast(Task, normalized)


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
    config = get_model_config(model)
    enable_mask_head = resolved_task == "segment"

    config["HGNetv2"]["pretrained"] = False
    config["HybridEncoder"]["eval_spatial_size"] = image_size
    config["DFINETransformer"]["eval_spatial_size"] = image_size
    config["DFINETransformer"]["enable_mask_head"] = enable_mask_head

    # D-FINE-N detection encodes stride-16/32 features. Instance segmentation
    # additionally exposes the backbone's stride-8 feature to MaskDecoder.
    encoder_strides = config["HybridEncoder"]["feat_strides"]
    if enable_mask_head and 8 not in encoder_strides:
        return_indices = config["HGNetv2"]["return_idx"]
        if 1 not in return_indices:
            config["HGNetv2"]["return_idx"] = [1, *return_indices]
        backbone_name = config["HGNetv2"]["name"]
        stage_channels = HGNetv2.arch_configs[backbone_name]["stage_config"]["stage2"][2]
        config["DFINETransformer"]["mask_low_level_ch"] = stage_channels

    backbone = HGNetv2(in_channels=in_channels, **config["HGNetv2"])
    encoder = HybridEncoder(**config["HybridEncoder"])
    decoder = DFINETransformer(num_classes=num_classes, **config["DFINETransformer"])
    native_model = DFINEModel(backbone=backbone, encoder=encoder, decoder=decoder)
    return native_model.to(device) if device is not None else native_model


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
