"""Tests for the integrated D-FINE model core and legacy parity."""

from __future__ import annotations

import gc
import sys
import types
from pathlib import Path

import pytest
import torch

from dfine.nn.build import build_model
from dfine.nn.configs import get_model_config
from dfine.nn.native_build import build_native_criterion, build_native_model


def _build_reference_model(config):
    """Build directly from the submodule, which is retained only as a Phase 2 oracle."""
    reference_root = Path(__file__).parents[2] / "extern/dfine"
    sys.path.insert(0, str(reference_root))

    source = types.ModuleType("src")
    source.__path__ = [str(reference_root / "src")]
    source.__package__ = "src"
    sys.modules["src"] = source

    from torch.utils.data import DataLoader

    data = types.ModuleType("src.data")
    data.DataLoader = DataLoader
    sys.modules["src.data"] = data

    misc = types.ModuleType("src.misc")
    misc.__path__ = [str(reference_root / "src/misc")]
    misc.__package__ = "src.misc"
    sys.modules["src.misc"] = misc

    import src.nn  # noqa: F401
    import src.optim  # noqa: F401
    import src.zoo  # noqa: F401
    from src.core.workspace import create
    from src.core.yaml_utils import merge_config

    reference_config = dict(config)
    reference_config["HGNetv2"] = {**reference_config["HGNetv2"], "pretrained": False}
    return create(reference_config["model"], merge_config(reference_config, inplace=False))


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


def test_native_detection_state_schema_matches_reference_dfine():
    """Keep the submodule as a migration oracle until Phase 2 is accepted."""
    from tools.convert_checkpoint import _load_config

    root = Path(__file__).parents[2]
    config_path = root / "extern/dfine/configs/dfine/dfine_hgnetv2_s_coco.yml"
    config = _load_config(config_path)
    reference = _build_reference_model(config)
    native = build_model(config)

    reference_state = reference.state_dict()
    native_state = native.state_dict()
    assert set(reference_state) == set(native_state)
    assert all(reference_state[key].shape == native_state[key].shape for key in reference_state)

    native.load_state_dict(reference_state, strict=True)
    reference.eval()
    native.eval()
    image = torch.zeros(1, 3, 640, 640)
    with torch.inference_mode():
        reference_output = reference(image)
        native_output = native(image)
    assert torch.equal(reference_output["pred_logits"], native_output["pred_logits"])
    assert torch.equal(reference_output["pred_boxes"], native_output["pred_boxes"])


def test_task_selects_mask_losses_without_mutating_shared_config():
    detection = build_native_criterion("dfine_s", num_classes=3, task="detect")
    segmentation = build_native_criterion("dfine_s", num_classes=3, task="segment")
    detection_again = build_native_criterion("dfine_s", num_classes=3, task="detect")

    assert "masks" not in detection.losses
    assert "masks" in segmentation.losses
    assert "masks" not in detection_again.losses


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
