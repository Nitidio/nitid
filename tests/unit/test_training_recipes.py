from __future__ import annotations

import pytest

from dfine.training_recipes import (
    default_train_options,
    infer_dfine_model_size,
    normalize_training_recipe,
    resolve_training_recipe,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("default", "default"),
        ("Auto", "default"),
        (" deim ", "deim"),
    ],
)
def test_normalize_training_recipe(raw: str, expected: str) -> None:
    assert normalize_training_recipe(raw) == expected


def test_normalize_training_recipe_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown training recipe"):
        normalize_training_recipe("experimental")


def test_normalize_training_recipe_rejects_dfine_alias() -> None:
    with pytest.raises(ValueError, match="Supported recipes: default, deim"):
        normalize_training_recipe("dfine")


def test_resolve_training_recipe_defaults_to_task_native_path() -> None:
    recipe = resolve_training_recipe("default", task="pose")

    assert recipe.name == "default"
    assert recipe.task == "pose"
    assert recipe.implemented is True


def test_resolve_training_recipe_recognizes_deim_detection_contract() -> None:
    recipe = resolve_training_recipe("deim", task="detect")

    assert recipe.name == "deim"
    assert recipe.task == "detect"
    assert recipe.implemented is False


def test_resolve_training_recipe_rejects_deim_for_non_detection_tasks() -> None:
    with pytest.raises(ValueError, match="detection-only"):
        resolve_training_recipe("deim", task="pose")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"HGNetv2": {"name": "B0"}, "HybridEncoder": {"hidden_dim": 128}}, "n"),
        ({"HGNetv2": {"name": "B0"}, "HybridEncoder": {"hidden_dim": 256}}, "s"),
        ({"HGNetv2": {"name": "B2"}, "HybridEncoder": {"hidden_dim": 256}}, "m"),
        ({"HGNetv2": {"name": "B4"}, "HybridEncoder": {"hidden_dim": 256}}, "l"),
        ({"HGNetv2": {"name": "B5"}, "HybridEncoder": {"hidden_dim": 384}}, "x"),
    ],
)
def test_infer_dfine_model_size_from_native_config(config: dict, expected: str) -> None:
    assert infer_dfine_model_size(config) == expected


def test_default_train_options_uses_official_detection_size_defaults() -> None:
    recipe = resolve_training_recipe("default", task="detect")
    defaults = default_train_options(
        recipe=recipe,
        config={"HGNetv2": {"name": "B0"}, "HybridEncoder": {"hidden_dim": 256}},
    )

    assert defaults.epochs == 132
    assert defaults.batch == 32
    assert defaults.lr0 == pytest.approx(2e-4)
    assert defaults.backbone_lr == pytest.approx(1e-4)
    assert defaults.lrf == pytest.approx(1.0)
    assert defaults.amp is True
    assert defaults.ema is True


def test_default_train_options_keeps_legacy_defaults_for_non_detection_tasks() -> None:
    recipe = resolve_training_recipe("default", task="pose")
    defaults = default_train_options(recipe=recipe, config={})

    assert defaults.epochs == 50
    assert defaults.batch == 16
    assert defaults.lr0 == pytest.approx(1e-4)
    assert defaults.backbone_lr is None
    assert defaults.amp is False
    assert defaults.ema is False
