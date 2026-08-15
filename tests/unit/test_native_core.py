"""Tests for the integrated D-FINE model core and reference compatibility."""

from __future__ import annotations

import gc

import pytest
import torch

from dfine.nn.build import build_model
from dfine.nn.configs import get_model_config, make_model_config
from dfine.nn.native_build import build_native_criterion, build_native_model


@pytest.mark.parametrize("model_size", ["n", "s", "m", "l", "x"])
@pytest.mark.parametrize("task", ["detect", "segment"])
def test_native_model_builds_for_every_supported_size_and_task(model_size, task):
    model = build_native_model(
        f"dfine_{model_size}",
        num_classes=3,
        task=task,
        image_size=(640, 640),
    )

    state = model.state_dict()
    mask_keys = {key for key in state if key.startswith("decoder.mask_")}
    if task == "segment":
        assert mask_keys
    else:
        assert not mask_keys

    del model, state
    gc.collect()


def test_native_nano_forward_contracts():
    # Nano selects 300 queries from stride-16/32 features, so the synthetic
    # image must expose at least 300 feature locations.
    images = torch.randn(1, 3, 256, 256)

    detector = build_native_model("dfine_n", num_classes=3, task="detect").eval()
    with torch.inference_mode():
        detection = detector(images)
    assert detection["pred_logits"].shape == (1, 300, 3)
    assert detection["pred_boxes"].shape == (1, 300, 4)
    assert "pred_masks" not in detection

    segmenter = build_native_model("dfine_n", num_classes=3, task="segment").eval()
    with torch.inference_mode():
        segmentation = segmenter(images)
    assert segmentation["pred_logits"].shape == (1, 300, 3)
    assert segmentation["pred_boxes"].shape == (1, 300, 4)
    assert segmentation["pred_masks"].shape == (1, 300, 64, 64)
    assert torch.all((0 <= segmentation["pred_masks"]) & (segmentation["pred_masks"] <= 1))


def test_task_selects_mask_losses_without_mutating_shared_config():
    detection = build_native_criterion("dfine_s", num_classes=3, task="detect")
    segmentation = build_native_criterion("dfine_s", num_classes=3, task="segment")
    detection_again = build_native_criterion("dfine_s", num_classes=3, task="detect")

    assert "masks" not in detection.losses
    assert "masks" in segmentation.losses
    assert "masks" not in detection_again.losses


def test_segment_checkpoint_config_builds_mask_model_and_criterion():
    from dfine.nn.criterion import build_criterion

    config = make_model_config("dfine_s", task="segment", num_classes=3)
    config["DFINETransformer"]["num_layers"] = 1
    model = build_model(config)
    criterion = build_criterion(config)

    assert config["task"] == "segment"
    assert any(key.startswith("decoder.mask_") for key in model.state_dict())
    assert "masks" in criterion.losses


def test_model_configs_are_isolated():
    first = get_model_config("dfine_s")
    first["DFINETransformer"]["num_layers"] = 99
    second = get_model_config("s")
    assert second["DFINETransformer"]["num_layers"] == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model": "dfine_unknown", "task": "detect"}, "Unsupported D-FINE model"),
        ({"model": "dfine_s", "task": "semantic"}, "Unsupported task"),
        ({"model": "dfine_s", "task": "detect", "num_classes": 0}, "num_classes"),
        ({"model": "dfine_s", "task": "detect", "in_channels": 5}, "in_channels"),
    ],
)
def test_native_builder_rejects_invalid_configuration(kwargs, match):
    defaults = {"model": "dfine_s", "num_classes": 3, "task": "detect"}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        build_native_model(**defaults)
