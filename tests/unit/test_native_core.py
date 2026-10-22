"""Tests for the integrated D-FINE model core and reference compatibility."""

from __future__ import annotations

import gc

import pytest
import torch

from dfine.nn.build import build_model, build_postprocessor
from dfine.nn.configs import get_model_config, make_model_config
from dfine.nn.losses import SemSegCriterion
from dfine.nn.native_build import (
    build_native_criterion,
    build_native_model,
    build_native_model_from_config,
)
from dfine.training_recipes import apply_training_recipe_to_config, resolve_training_recipe


@pytest.mark.parametrize("model_size", ["n", "s", "m", "l", "x"])
@pytest.mark.parametrize("task", ["detect", "segment", "semantic"])
def test_native_model_builds_for_every_supported_size_and_task(model_size, task):
    model = build_native_model(
        f"dfine_{model_size}",
        num_classes=3,
        task=task,
        image_size=(640, 640),
    )

    state = model.state_dict()
    mask_keys = {key for key in state if key.startswith("decoder.mask_decoder")}
    if task in {"segment", "semantic"}:
        assert mask_keys
    else:
        assert not mask_keys
    if task == "semantic":
        assert any(key.startswith("decoder.classifier") for key in state)
        assert not any(key.startswith("decoder.dec_score_head") for key in state)

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

    semantic = build_native_model("dfine_n", num_classes=3, task="semantic")
    semantic.train()
    semantic_outputs = semantic(images)
    assert semantic_outputs["sem_seg_logits"].shape == (1, 3, 256, 256)
    assert semantic_outputs["sem_seg_logits_aux"].shape == (1, 3, 256, 256)
    semantic.eval()
    with torch.inference_mode():
        semantic_outputs = semantic(images)
    assert semantic_outputs["sem_seg_logits"].shape == (1, 3, 256, 256)
    assert "sem_seg_logits_aux" not in semantic_outputs


def test_task_selects_mask_losses_without_mutating_shared_config():
    detection = build_native_criterion("dfine_s", num_classes=3, task="detect")
    segmentation = build_native_criterion("dfine_s", num_classes=3, task="segment")
    detection_again = build_native_criterion("dfine_s", num_classes=3, task="detect")

    assert "masks" not in detection.losses
    assert "masks" in segmentation.losses
    assert "masks" not in detection_again.losses


def test_deim_detection_config_builds_mal_criterion():
    from dfine.nn.criterion import build_criterion

    config = make_model_config("dfine_n", task="detect", num_classes=2)
    recipe = resolve_training_recipe("deim", task="detect")
    config = apply_training_recipe_to_config(config, recipe=recipe)
    criterion = build_criterion(config)
    outputs = {
        "pred_logits": torch.tensor([[[0.2, -0.1], [-0.4, 0.7]]], requires_grad=True),
        "pred_boxes": torch.tensor(
            [[[0.5, 0.5, 0.25, 0.25], [0.25, 0.25, 0.2, 0.2]]],
            requires_grad=True,
        ),
    }
    targets = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[0.25, 0.25, 0.2, 0.2]]),
        }
    ]

    losses = criterion(outputs, targets)

    assert "loss_mal" in losses
    assert "loss_vfl" not in losses
    assert torch.isfinite(losses["loss_mal"])


def test_semantic_criterion_is_finite_weighted_and_differentiable():
    criterion = build_native_criterion(
        "dfine_s",
        num_classes=3,
        task="sem_seg",
        label_smoothing=0.05,
        class_weights=[1.0, 2.0, 1.0],
    )
    assert isinstance(criterion, SemSegCriterion)
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    auxiliary = torch.randn(2, 3, 16, 16, requires_grad=True)
    targets = [{"sem_mask": torch.randint(0, 3, (16, 16))} for _ in range(2)]

    losses = criterion(
        {"sem_seg_logits": logits, "sem_seg_logits_aux": auxiliary},
        targets,
    )

    assert set(losses) == {"loss_ce", "loss_dice", "loss_aux"}
    assert all(torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert auxiliary.grad is not None and torch.isfinite(auxiliary.grad).all()


def test_semantic_criterion_respects_ignore_index_and_all_ignore_batches():
    criterion = build_native_criterion("dfine_s", num_classes=3, task="semantic", ignore_index=255)
    logits = torch.randn(1, 3, 16, 16, requires_grad=True)
    target = torch.randint(0, 3, (16, 16))
    target[:8] = 255
    masked = criterion({"sem_seg_logits": logits}, [{"sem_mask": target}])
    cropped = criterion(
        {"sem_seg_logits": logits[..., 8:, :]},
        [{"sem_mask": target[8:]}],
    )
    assert masked.keys() == cropped.keys()
    for name in masked:
        assert torch.allclose(masked[name], cropped[name], atol=1e-6)

    ignored_target = torch.full((16, 16), 255, dtype=torch.long)
    ignored = criterion(
        {"sem_seg_logits": logits, "sem_seg_logits_aux": logits},
        [{"sem_mask": ignored_target}],
    )
    total = sum(ignored.values())
    assert total.item() == 0.0
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_segment_checkpoint_config_builds_mask_model_and_criterion():
    from dfine.nn.criterion import build_criterion

    config = make_model_config("dfine_s", task="segment", num_classes=3)
    config["DFINETransformer"]["num_layers"] = 1
    model = build_model(config)
    criterion = build_criterion(config)

    assert config["task"] == "segment"
    assert any(key.startswith("decoder.mask_") for key in model.state_dict())
    assert "masks" in criterion.losses


def test_semantic_checkpoint_config_declares_dense_output_and_instance_initialization():
    config = make_model_config(
        "dfine_s",
        task="sem_seg",
        num_classes=19,
        ignore_index=255,
    )

    assert config["task"] == "semantic"
    assert config["criterion"] == "SemSegCriterion"
    assert config["postprocessor"] == "SemanticPostProcessor"
    assert config["DFINE"]["decoder"] == "SemSegDecoder"
    assert config["SemSegCriterion"]["ignore_index"] == 255
    assert config["SemSegDecoder"] == {
        "feat_channels": [256, 256, 256],
        "mask_dim": 256,
        "neck_dim": 128,
        "dropout": 0.1,
        "aux": True,
    }
    assert "DFINETransformer" not in config
    assert config["SemanticSegmentation"] == {
        "ignore_index": 255,
        "output": "semantic_mask",
        "pretrained_source_task": "segment",
    }


def test_semantic_checkpoint_builds_model_criterion_and_postprocessor():
    from dfine.nn.criterion import build_criterion

    config = make_model_config("dfine_s", task="semantic", num_classes=19)
    model = build_model(config)
    criterion = build_criterion(config)

    assert model.decoder.__class__.__name__ == "SemSegDecoder"
    assert isinstance(criterion, SemSegCriterion)
    postprocessor = build_postprocessor(config)
    outputs = model(torch.zeros(1, 3, 64, 64))
    processed = postprocessor(outputs, torch.tensor([[48.0, 32.0]]))

    assert isinstance(processed, list)
    assert processed[0]["semantic_logits"].shape == (19, 32, 48)


@pytest.mark.parametrize("model_size", ["dfine_n", "dfine_s"])
def test_instance_checkpoint_initializes_semantic_feature_fuser(model_size):
    instance_model = build_native_model(model_size, num_classes=80, task="segment")
    semantic_model = build_native_model(model_size, num_classes=19, task="semantic")
    instance_state = instance_model.state_dict()
    semantic_state = semantic_model.state_dict()
    transferable = {
        name: value
        for name, value in instance_state.items()
        if name in semantic_state and value.shape == semantic_state[name].shape
    }

    mask_fuser_keys = {name for name in semantic_state if name.startswith("decoder.mask_decoder.")}
    assert mask_fuser_keys
    assert mask_fuser_keys.issubset(transferable)
    assert not any(name.startswith("decoder.neck.") for name in transferable)
    assert not any(name.startswith("decoder.classifier.") for name in transferable)

    load_result = semantic_model.load_state_dict(transferable, strict=False)
    assert not load_result.unexpected_keys
    assert all(
        name.startswith(("decoder.neck.", "decoder.classifier.", "decoder.aux_head."))
        for name in load_result.missing_keys
    )


def test_phase_one_semantic_config_remains_constructible():
    config = make_model_config("dfine_n", task="semantic", num_classes=3)
    semantic_decoder = config.pop("SemSegDecoder")
    source = get_model_config("dfine_n")["DFINETransformer"]
    config["DFINETransformer"] = {
        **source,
        "feat_channels": semantic_decoder["feat_channels"],
        "mask_dim": semantic_decoder["mask_dim"],
    }

    model = build_native_model_from_config(config)

    assert model.decoder.__class__.__name__ == "SemSegDecoder"


def test_model_configs_are_isolated():
    first = get_model_config("dfine_s")
    first["DFINETransformer"]["num_layers"] = 99
    second = get_model_config("s")
    assert second["DFINETransformer"]["num_layers"] == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model": "dfine_unknown", "task": "detect"}, "Unsupported D-FINE model"),
        ({"model": "dfine_s", "task": "panoptic"}, "Unsupported task"),
        ({"model": "dfine_s", "task": "detect", "num_classes": 0}, "num_classes"),
        ({"model": "dfine_s", "task": "detect", "in_channels": 5}, "in_channels"),
    ],
)
def test_native_builder_rejects_invalid_configuration(kwargs, match):
    defaults = {"model": "dfine_s", "num_classes": 3, "task": "detect"}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        build_native_model(**defaults)
