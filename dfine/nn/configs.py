"""Native model-size configuration for D-FINE tasks.

Derived from D-FINE and D-FINE-seg; modified for integration into nitid in 2026.
"""

from copy import deepcopy
from typing import Any

BASE_CONFIG = {
    "HGNetv2": {
        "pretrained": False,
        "local_model_dir": "weight/hgnetv2/",
        "freeze_stem_only": True,
    },
    "HybridEncoder": {
        "num_encoder_layers": 1,
        "nhead": 8,
        "dropout": 0.0,
        "enc_act": "gelu",
        "act": "silu",
    },
    "DFINETransformer": {
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "layer_scale": 1,
        "cross_attn_method": "default",
        "query_select_method": "default",
    },
    "DFINECriterion": {
        "weight_dict": {
            "loss_vfl": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
            "loss_mask_bce": 1,  # only for mask head
            "loss_mask_dice": 1,  # only for mask head
        },
        "losses": ["vfl", "boxes", "local"],  #  "masks" will be added if training with segment task
        "alpha": 0.75,
        "gamma": 2.0,
        "reg_max": 32,
    },
    "SemSegCriterion": {
        "weight_dict": {"loss_ce": 1, "loss_dice": 1, "loss_aux": 0.4},
    },
    "matcher": {
        "weight_dict": {
            "cost_class": 2,
            "cost_bbox": 5,
            "cost_giou": 2,
            "cost_mask": 1,  # focal mask cost for segmentation matching
            "cost_mask_dice": 1,  # dice mask cost for segmentation matching
        },
        "alpha": 0.25,
        "gamma": 2.0,
        "use_focal_loss": True,
    },
}

SIZE_CONFIGS = {
    "n": {
        "HGNetv2": {
            "name": "B0",
            "return_idx": [2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
        },
        "HybridEncoder": {
            "in_channels": [512, 1024],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "use_encoder_idx": [1],
            "dim_feedforward": 512,
            "expansion": 0.34,
            "depth_mult": 0.5,
        },
        "DFINETransformer": {
            "feat_channels": [128, 128],
            "feat_strides": [16, 32],
            "hidden_dim": 128,
            "num_levels": 2,
            "num_layers": 3,
            "reg_scale": 4,
            "num_points": [6, 6],
            "dim_feedforward": 512,
            "mask_dim": 128,
        },
    },
    "s": {
        "HGNetv2": {
            "name": "B0",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
        },
        "HybridEncoder": {
            "in_channels": [256, 512, 1024],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "dim_feedforward": 1024,
            "expansion": 0.5,
            "depth_mult": 0.34,
        },
        "DFINETransformer": {
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "num_levels": 3,
            "num_layers": 3,
            "reg_scale": 4,
            "num_points": [3, 6, 3],
            "mask_dim": 256,
        },
    },
    "m": {
        "HGNetv2": {
            "name": "B2",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
        },
        "HybridEncoder": {
            "in_channels": [384, 768, 1536],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "dim_feedforward": 1024,
            "expansion": 1.0,
            "depth_mult": 0.67,
        },
        "DFINETransformer": {
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "dim_feedforward": 1024,
            "num_levels": 3,
            "num_layers": 4,
            "reg_scale": 4,
            "num_points": [3, 6, 3],
            "enable_mask_head": False,
            "mask_dim": 256,
        },
    },
    "l": {
        "HGNetv2": {
            "name": "B4",
            "return_idx": [1, 2, 3],
            "freeze_at": 0,
            "freeze_norm": True,
            "use_lab": False,
        },
        "HybridEncoder": {
            "in_channels": [512, 1024, 2048],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "dim_feedforward": 1024,
            "expansion": 1.0,
            "depth_mult": 1.0,
        },
        "DFINETransformer": {
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "dim_feedforward": 1024,
            "num_levels": 3,
            "num_layers": 6,
            "reg_scale": 4,
            "num_points": [3, 6, 3],
            "mask_dim": 256,
        },
    },
    "x": {
        "HGNetv2": {
            "name": "B5",
            "return_idx": [1, 2, 3],
            "freeze_at": 0,
            "freeze_norm": True,
            "use_lab": False,
        },
        "HybridEncoder": {
            "in_channels": [512, 1024, 2048],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 384,
            "use_encoder_idx": [2],
            "dim_feedforward": 2048,
            "expansion": 1.0,
            "depth_mult": 1.0,
        },
        "DFINETransformer": {
            "feat_channels": [384, 384, 384],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "dim_feedforward": 1024,
            "num_levels": 3,
            "num_layers": 6,
            "reg_scale": 8,
            "num_points": [3, 6, 3],
            "mask_dim": 256,
        },
    },
}


def merge_configs(base: dict[str, Any], size_specific: dict[str, Any]) -> dict[str, Any]:
    # deepcopy so list/dict values (e.g. DFINECriterion.losses) are not shared
    # between model sizes — a shared list lets one size's mutation leak into all.
    result = deepcopy(base)
    for key, value in size_specific.items():
        if key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


MODEL_CONFIGS = {size: merge_configs(BASE_CONFIG, config) for size, config in SIZE_CONFIGS.items()}
MODEL_SIZES = tuple(MODEL_CONFIGS)


def get_model_config(model_size: str) -> dict[str, Any]:
    """Return an isolated native configuration for a supported model size."""
    normalized = model_size.lower().removeprefix("dfine_")
    try:
        return deepcopy(MODEL_CONFIGS[normalized])
    except KeyError as error:
        supported = ", ".join(f"dfine_{size}" for size in MODEL_SIZES)
        raise ValueError(f"Unsupported D-FINE model {model_size!r}. Choose: {supported}") from error


def make_model_config(
    model_size: str,
    *,
    task: str = "detect",
    num_classes: int = 80,
    image_size: tuple[int, int] = (640, 640),
    ignore_index: int = 255,
) -> dict[str, Any]:
    """Create the self-contained runtime config stored in a wrapped checkpoint."""
    from dfine.tasks import normalize_task

    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int) or ignore_index < 0:
        raise ValueError(f"ignore_index must be a non-negative integer, got {ignore_index!r}")
    resolved_task = normalize_task(task)

    model_config = get_model_config(model_size)
    criterion = model_config.pop("DFINECriterion")
    semantic_criterion = model_config.pop("SemSegCriterion")
    matcher = model_config.pop("matcher")
    if resolved_task == "semantic":
        transformer_config = model_config.pop("DFINETransformer")
        semantic_decoder = {
            "feat_channels": transformer_config["feat_channels"],
            "mask_dim": transformer_config["mask_dim"],
            "neck_dim": 128,
            "dropout": 0.1,
            "aux": True,
        }
        semantic_criterion["ignore_index"] = ignore_index
        return {
            "task": resolved_task,
            "model": "DFINE",
            "criterion": "SemSegCriterion",
            "postprocessor": "SemanticPostProcessor",
            "num_classes": num_classes,
            "eval_spatial_size": list(image_size),
            "DFINE": {
                "backbone": "HGNetv2",
                "encoder": "HybridEncoder",
                "decoder": "SemSegDecoder",
            },
            **model_config,
            "SemSegDecoder": semantic_decoder,
            "SemSegCriterion": semantic_criterion,
            "SemanticSegmentation": {
                "ignore_index": ignore_index,
                "output": "semantic_mask",
                "pretrained_source_task": "segment",
            },
            "SemanticPostProcessor": {"output": "semantic_mask"},
        }

    if resolved_task == "segment":
        criterion["losses"].append("masks")
    matcher["type"] = "HungarianMatcher"
    criterion["matcher"] = matcher

    return {
        "task": resolved_task,
        "model": "DFINE",
        "criterion": "DFINECriterion",
        "postprocessor": "DFINEPostProcessor",
        "num_classes": num_classes,
        "use_focal_loss": True,
        "eval_spatial_size": list(image_size),
        "DFINE": {
            "backbone": "HGNetv2",
            "encoder": "HybridEncoder",
            "decoder": "DFINETransformer",
        },
        **model_config,
        "DFINECriterion": criterion,
        "DFINEPostProcessor": {"num_top_queries": 300},
    }


def make_detection_config(
    model_size: str,
    *,
    num_classes: int = 80,
    image_size: tuple[int, int] = (640, 640),
) -> dict[str, Any]:
    """Create a detection checkpoint config."""
    return make_model_config(
        model_size,
        task="detect",
        num_classes=num_classes,
        image_size=image_size,
    )


def make_pose_config(
    model_size: str,
    *,
    dataset: str = "coco",
    image_size: tuple[int, int] = (640, 640),
) -> dict[str, Any]:
    """Create a self-contained runtime config for DETRPose models."""
    from dfine.pose_contract import get_pose_model_spec, get_pose_schema

    spec = get_pose_model_spec(model_size)
    schema = get_pose_schema(dataset)

    return {
        "task": "pose",
        "model": "DFINE",
        "num_classes": spec.public_num_classes,
        "eval_spatial_size": list(image_size),
        "dataset": schema.name,
        "num_body_points": schema.num_keypoints,
        "DFINE": {
            "backbone": "HGNetv2",
            "encoder": "HybridEncoder",
            "decoder": "DETRPoseDecoder",
        },
        "HGNetv2": {
            "name": spec.backbone,
            "return_idx": list(spec.backbone_return_idx),
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
            "pretrained": False,
        },
        "HybridEncoder": {
            "in_channels": list(spec.encoder_in_channels),
            "feat_strides": list(spec.feature_strides),
            "hidden_dim": spec.hidden_dim,
            "use_encoder_idx": [len(spec.encoder_in_channels) - 1],
            "dim_feedforward": spec.encoder_feedforward_dim,
            "expansion": spec.encoder_expansion,
            "depth_mult": spec.encoder_depth_mult,
            "pe_mode": "sinehw",
            "pe_temperature_h": 20.0,
            "pe_temperature_w": 20.0,
        },
        "DETRPoseDecoder": {
            "hidden_dim": spec.hidden_dim,
            "nhead": 8,
            "num_queries": spec.num_queries,
            "num_decoder_layers": spec.decoder_layers,
            "dim_feedforward": spec.decoder_feedforward_dim,
            "num_feature_levels": len(spec.feature_strides),
            "dec_n_points": spec.decoder_points,
            "num_classes": spec.internal_num_classes,
            "num_body_points": schema.num_keypoints,
            "feat_strides": list(spec.feature_strides),
            "eval_spatial_size": list(image_size),
            "reg_max": spec.reg_max,
            "reg_scale": spec.reg_scale,
        },
        "matcher": {
            "weight_dict": {
                "cost_class": 2.0,
                "cost_keypoints": 5.0,
                "cost_oks": 2.0,
            },
            "num_body_points": schema.num_keypoints,
            "use_focal_loss": True,
        },
        "DFINECriterion": {
            "weight_dict": {
                "loss_vfl": 1.0,
                "loss_keypoints": 5.0,
                "loss_oks": 2.0,
                "loss_fgl": 1.0,
                "loss_ddf": 1.5,
            },
            "losses": ["vfl", "keypoints", "local"],
            "reg_max": spec.reg_max,
            "num_body_points": schema.num_keypoints,
        },
    }
