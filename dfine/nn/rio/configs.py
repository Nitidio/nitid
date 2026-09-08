"""Self-contained RiO-DETR / RT-DETRv2-OBB model configs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_BASE_OBB_CONFIG: dict[str, Any] = {
    "task": "obb",
    "model": "RioOBB",
    "criterion": "RTv4OBBCriterion",
    "postprocessor": "PostProcessorOBB",
    "num_classes": 15,
    "use_focal_loss": True,
    "eval_spatial_size": [1024, 1024],
    "RioOBB": {
        "backbone": "HGNetv2",
        "encoder": "HybridEncoder",
        "decoder": "RTDETRTransformerv2OBB",
    },
    "RTDETRTransformerv2OBB": {
        "feat_strides": [8, 16, 32],
        "num_levels": 3,
        "num_points": [4, 4, 4],
        "cross_attn_method": "default",
        "query_select_method": "default",
        "query_pos_method": "default",
        "anchor_aspect_ratio": 1.0,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "eval_idx": -1,
    },
    "PostProcessorOBB": {
        "num_top_queries": 300,
        "input_shape": [1024, 1024],
    },
    "RTv4OBBCriterion": {
        "weight_dict": {"loss_focal": 1, "loss_l1": 5, "loss_kld": 5},
        "losses": ["focal", "l1", "kld"],
        "alpha": 0.75,
        "gamma": 2.0,
        "use_uni_set": True,
        "matcher": {
            "type": "HungarianMatcherOBB",
            "weight_dict": {
                "cost_class": 2,
                "cost_bbox": 0,
                "cost_hausdorff": 5,
                "cost_giou": 5,
            },
            "alpha": 0.25,
            "gamma": 2.0,
        },
    },
}

_SIZE_CONFIGS: dict[str, dict[str, Any]] = {
    "n": {
        "HGNetv2": {
            "name": "B0",
            "return_idx": [2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
            "pretrained": False,
            "local_model_dir": "./pretrain/hgnetv2/",
        },
        "HybridEncoder": {
            "in_channels": [512, 1024],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "use_encoder_idx": [1],
            "expansion": 0.34,
            "depth_mult": 0.5,
        },
        "RTDETRTransformerv2OBB": {
            "feat_channels": [128, 128],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "num_layers": 3,
            "num_queries": 300,
            "num_denoising": 100,
            "learn_query_content": False,
        },
    },
    "s": {
        "HGNetv2": {
            "name": "B0",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
            "pretrained": False,
            "local_model_dir": "./pretrain/hgnetv2/",
        },
        "HybridEncoder": {
            "in_channels": [256, 512, 1024],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 224,
            "use_encoder_idx": [2],
            "expansion": 0.5,
            "depth_mult": 0.34,
        },
        "RTDETRTransformerv2OBB": {
            "feat_channels": [224, 224, 224],
            "hidden_dim": 224,
            "num_layers": 3,
            "num_queries": 300,
            "num_denoising": 100,
            "learn_query_content": False,
        },
    },
    "m": {
        "HGNetv2": {
            "name": "B2",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
            "pretrained": False,
            "local_model_dir": "./pretrain/hgnetv2/",
        },
        "HybridEncoder": {
            "in_channels": [384, 768, 1536],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "expansion": 1,
            "depth_mult": 0.67,
        },
        "RTDETRTransformerv2OBB": {
            "feat_channels": [256, 256, 256],
            "hidden_dim": 256,
            "num_layers": 3,
            "num_queries": 300,
            "num_denoising": 100,
            "learn_query_content": False,
        },
    },
    "l": {
        "HGNetv2": {
            "name": "B4",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "pretrained": False,
            "local_model_dir": "./pretrain/hgnetv2/",
        },
        "HybridEncoder": {
            "in_channels": [512, 1024, 2048],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "expansion": 1,
            "depth_mult": 0.67,
        },
        "RTDETRTransformerv2OBB": {
            "feat_channels": [256, 256, 256],
            "hidden_dim": 256,
            "num_layers": 4,
            "num_queries": 300,
            "num_denoising": 100,
            "learn_query_content": False,
        },
    },
    "x": {
        "HGNetv2": {
            "name": "B5",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "pretrained": False,
            "local_model_dir": "./pretrain/hgnetv2/",
        },
        "HybridEncoder": {
            "in_channels": [512, 1024, 2048],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 384,
            "use_encoder_idx": [2],
            "expansion": 1,
            "depth_mult": 0.67,
        },
        "RTDETRTransformerv2OBB": {
            "feat_channels": [384, 384, 384],
            "hidden_dim": 384,
            "num_layers": 4,
            "num_queries": 300,
            "num_denoising": 100,
            "learn_query_content": False,
        },
    },
}

RIO_OBB_MODEL_SIZES = tuple(_SIZE_CONFIGS)
DOTA_OBB_NAMES: tuple[str, ...] = (
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
)


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_rio_obb_config(model_size: str) -> dict[str, Any]:
    """Return a RiO-DETR OBB config for a model size or public model name."""
    normalized = model_size.lower().strip()
    if normalized.startswith("nitid1"):
        normalized = normalized.removeprefix("nitid1")
    normalized = normalized.removeprefix("rio_").removeprefix("rtdetrv2_obb_")
    try:
        return _merge(_BASE_OBB_CONFIG, _SIZE_CONFIGS[normalized])
    except KeyError as error:
        supported = ", ".join(f"nitid1{size}" for size in RIO_OBB_MODEL_SIZES)
        raise ValueError(
            f"Unsupported RiO-DETR OBB model {model_size!r}. Choose: {supported}"
        ) from error


def make_rio_obb_config(
    model_size: str,
    *,
    num_classes: int = 15,
    image_size: tuple[int, int] = (1024, 1024),
) -> dict[str, Any]:
    """Create the self-contained runtime config for a RiO-DETR OBB checkpoint."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    config = get_rio_obb_config(model_size)
    config["num_classes"] = num_classes
    config["eval_spatial_size"] = list(image_size)
    config["PostProcessorOBB"]["input_shape"] = list(image_size)
    config["RTDETRTransformerv2OBB"]["eval_spatial_size"] = list(image_size)
    return config
