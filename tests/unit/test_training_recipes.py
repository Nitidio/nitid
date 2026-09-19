from __future__ import annotations

import pytest

from dfine.training_recipes import normalize_training_recipe, resolve_training_recipe


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
