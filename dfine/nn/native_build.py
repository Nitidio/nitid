"""Construct the integrated D-FINE model core for every supported vision task.

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

from .architecture import (
    DETRPoseDecoder,
    DFINEModel,
    DFINETransformer,
    HGNetv2,
    HybridEncoder,
    SemSegDecoder,
)
from .configs import get_model_config, make_pose_config
from .losses import DFINECriterion, HungarianMatcher, SemSegCriterion
from .rio import (
    HungarianMatcherOBB,
    RioOBBModel,
    RTDETRTransformerv2OBB,
    RTv4OBBCriterion,
    get_rio_obb_config,
    make_rio_obb_config,
)


def _checkpoint_task(config: Mapping[str, Any]) -> Task:
    return normalize_task(str(config.get("task", "detect")))


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
    enable_mask_head = task == "segment"
    if image_size is not None:
        encoder_config["eval_spatial_size"] = image_size
        if task != "semantic":
            decoder_config["eval_spatial_size"] = image_size
            if task == "segment":
                decoder_config["enable_mask_head"] = True
    elif task == "pose":
        encoder_config["eval_spatial_size"] = (640, 640)
        decoder_config["eval_spatial_size"] = (640, 640)
    elif task == "segment":
        decoder_config["enable_mask_head"] = True

    encoder_strides = encoder_config["feat_strides"]
    if (enable_mask_head or task == "semantic") and 8 not in encoder_strides:
        return_indices = backbone_config["return_idx"]
        if 1 not in return_indices:
            backbone_config["return_idx"] = [1, *return_indices]
        backbone_name = backbone_config["name"]
        stage_channels = HGNetv2.arch_configs[backbone_name]["stage_config"]["stage2"][2]
        decoder_config["mask_low_level_ch"] = stage_channels

    backbone = HGNetv2(in_channels=in_channels, **backbone_config)
    encoder = HybridEncoder(**encoder_config)
    decoder_kwargs = copy.deepcopy(decoder_config)
    dec_num_classes = decoder_kwargs.pop("num_classes", num_classes)
    decoder: nn.Module
    if task == "semantic":
        decoder = SemSegDecoder(num_classes=dec_num_classes, **decoder_kwargs)
    elif task == "pose":
        decoder = DETRPoseDecoder(num_classes=dec_num_classes, **decoder_kwargs)
    else:
        decoder = DFINETransformer(num_classes=dec_num_classes, **decoder_kwargs)
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
) -> nn.Module:
    """Build a native D-FINE model without loading pretrained weights."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if in_channels not in (3, 4):
        raise ValueError(f"in_channels must be 3 or 4, got {in_channels}")

    resolved_task = normalize_task(task)
    if resolved_task == "obb":
        config = make_rio_obb_config(
            model,
            num_classes=num_classes,
            image_size=image_size or (1024, 1024),
        )
        backbone = HGNetv2(in_channels=in_channels, **config["HGNetv2"])
        encoder = HybridEncoder(**config["HybridEncoder"])
        decoder = RTDETRTransformerv2OBB(
            num_classes=num_classes,
            **config["RTDETRTransformerv2OBB"],
        )
        native_model = RioOBBModel(backbone=backbone, encoder=encoder, decoder=decoder)
        return native_model.to(device) if device is not None else native_model

    if resolved_task == "pose":
        config = make_pose_config(model, image_size=image_size or (640, 640))
        return _compose_native_model(
            backbone_config=config["HGNetv2"],
            encoder_config=config["HybridEncoder"],
            decoder_config=config["DETRPoseDecoder"],
            num_classes=config["num_classes"],
            task=resolved_task,
            image_size=image_size,
            in_channels=in_channels,
            device=device,
        )

    config = get_model_config(model)
    decoder_config = config["DFINETransformer"]
    if resolved_task == "semantic":
        decoder_config = {
            "feat_channels": decoder_config["feat_channels"],
            "mask_dim": decoder_config["mask_dim"],
        }

    return _compose_native_model(
        backbone_config=config["HGNetv2"],
        encoder_config=config["HybridEncoder"],
        decoder_config=decoder_config,
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
) -> nn.Module:
    """Build from a current or legacy self-contained checkpoint configuration."""
    num_classes = config.get("num_classes")
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("D-FINE config num_classes must be a positive integer")
    in_channels = config.get("in_channels", 3)
    if not isinstance(in_channels, int) or in_channels not in (3, 4):
        raise ValueError("D-FINE config in_channels must be 3 or 4")

    task = _checkpoint_task(config)
    if task == "obb":
        backbone = HGNetv2(in_channels=in_channels, **_component_config(config, "HGNetv2"))
        encoder = HybridEncoder(**_component_config(config, "HybridEncoder"))
        decoder = RTDETRTransformerv2OBB(
            num_classes=num_classes,
            **_component_config(config, "RTDETRTransformerv2OBB"),
        )
        native_model = RioOBBModel(backbone=backbone, encoder=encoder, decoder=decoder)
        return native_model.to(device) if device is not None else native_model
    if task == "pose":
        decoder_config = _component_config(config, "DETRPoseDecoder")
    elif task == "semantic":
        decoder_key = "SemSegDecoder" if "SemSegDecoder" in config else "DFINETransformer"
        decoder_config = _component_config(config, decoder_key)
        if decoder_key == "DFINETransformer":
            decoder_config = {
                key: decoder_config[key]
                for key in ("feat_channels", "mask_dim", "mask_low_level_ch")
                if key in decoder_config
            }
    else:
        decoder_config = _component_config(config, "DFINETransformer")

    return _compose_native_model(
        backbone_config=_component_config(config, "HGNetv2"),
        encoder_config=_component_config(config, "HybridEncoder"),
        decoder_config=decoder_config,
        num_classes=num_classes,
        task=task,
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
    ignore_index: int = 255,
    class_weights: list[float] | None = None,
) -> nn.Module:
    """Build the task-aware D-FINE criterion."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    resolved_task = normalize_task(task)
    if resolved_task == "obb":
        config = get_rio_obb_config(model)
        criterion_config = copy.deepcopy(config["RTv4OBBCriterion"])
        matcher_config = copy.deepcopy(criterion_config.pop("matcher"))
        matcher_config.pop("type", None)
        matcher = HungarianMatcherOBB(
            use_focal_loss=bool(config.get("use_focal_loss", True)),
            **matcher_config,
        )
        return RTv4OBBCriterion(
            matcher=matcher,
            num_classes=num_classes,
            **criterion_config,
        )

    config = get_model_config(model)

    if resolved_task == "semantic":
        return SemSegCriterion(
            config["SemSegCriterion"]["weight_dict"],
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

    criterion_config = config["DFINECriterion"]
    if resolved_task == "segment" and "masks" not in criterion_config["losses"]:
        criterion_config["losses"].append("masks")

    dfine_matcher = HungarianMatcher(**config["matcher"])
    return DFINECriterion(
        matcher=dfine_matcher,
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
    if task == "obb":
        criterion_config = _component_config(config, "RTv4OBBCriterion")
        matcher_config = criterion_config.pop("matcher", None)
        if not isinstance(matcher_config, Mapping):
            raise ValueError("RiO-DETR OBB criterion config is missing its matcher mapping")
        matcher_kwargs = copy.deepcopy(dict(matcher_config))
        matcher_kwargs.pop("type", None)
        matcher = HungarianMatcherOBB(
            use_focal_loss=bool(config.get("use_focal_loss", True)),
            **matcher_kwargs,
        )
        return RTv4OBBCriterion(
            matcher=matcher,
            num_classes=num_classes,
            **criterion_config,
        )

    if task == "semantic":
        criterion_config = _component_config(config, "SemSegCriterion")
        return SemSegCriterion(num_classes=num_classes, **criterion_config)
    if task == "pose":
        decoder_config = _component_config(config, "DETRPoseDecoder")
        decoder_num_classes = decoder_config.get("num_classes")
        if not isinstance(decoder_num_classes, int) or decoder_num_classes < 1:
            raise ValueError("D-FINE pose decoder num_classes must be a positive integer")
        num_classes = decoder_num_classes

    criterion_config = _component_config(config, "DFINECriterion")
    matcher_value = criterion_config.pop("matcher", config.get("matcher"))
    if not isinstance(matcher_value, Mapping):
        raise ValueError("D-FINE criterion config is missing its matcher mapping")
    matcher_config = copy.deepcopy(dict(matcher_value))
    matcher_config.pop("type", None)
    matcher_config.setdefault("use_focal_loss", bool(config.get("use_focal_loss", True)))

    if task == "segment" and "masks" not in criterion_config["losses"]:
        criterion_config["losses"].append("masks")
    elif task == "pose" and "keypoints" not in criterion_config["losses"]:
        criterion_config["losses"].append("keypoints")

    num_body_points = config.get("num_body_points", criterion_config.get("num_body_points", 17))
    matcher_config.setdefault("num_body_points", num_body_points)
    criterion_config.setdefault("num_body_points", num_body_points)
    criterion_config.setdefault("label_smoothing", 0.0)

    return DFINECriterion(
        matcher=HungarianMatcher(**matcher_config),
        num_classes=num_classes,
        **criterion_config,
    )
