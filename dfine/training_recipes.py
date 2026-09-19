"""Training recipe resolution for nitid's public training API."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from dfine.tasks import Task, normalize_task

TrainingRecipeName = Literal["default", "deim"]


@dataclass(frozen=True)
class TrainingRecipe:
    """Resolved training recipe metadata.

    Phase 2 introduces the public API contract and validation surface. Later
    phases will attach concrete defaults and policy objects to this metadata.
    """

    name: TrainingRecipeName
    task: Task
    implemented: bool


@dataclass(frozen=True)
class TrainOptionDefaults:
    """Resolved public ``train()`` defaults for a task and recipe."""

    epochs: int = 50
    batch: int = 16
    lr0: float = 1e-4
    backbone_lr: float | None = None
    lrf: float = 0.01
    cos_lr: bool = False
    warmup_epochs: float = 0.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    optimizer: str = "AdamW"
    momentum: float = 0.9
    weight_decay: float = 1e-4
    clip_grad: float = 0.1
    amp: bool = False
    ema: bool = False
    ema_decay: float = 0.9999


_OFFICIAL_DFINE_DETECTION_DEFAULTS: dict[str, TrainOptionDefaults] = {
    "n": TrainOptionDefaults(
        epochs=160,
        batch=32,
        lr0=8e-4,
        backbone_lr=4e-4,
        lrf=1.0,
        amp=True,
        ema=True,
    ),
    "s": TrainOptionDefaults(
        epochs=132,
        batch=32,
        lr0=2e-4,
        backbone_lr=1e-4,
        lrf=1.0,
        amp=True,
        ema=True,
    ),
    "m": TrainOptionDefaults(
        epochs=132,
        batch=32,
        lr0=2e-4,
        backbone_lr=2e-5,
        lrf=1.0,
        amp=True,
        ema=True,
    ),
    "l": TrainOptionDefaults(
        epochs=80,
        batch=32,
        lr0=2.5e-4,
        backbone_lr=1.25e-5,
        lrf=1.0,
        weight_decay=1.25e-4,
        amp=True,
        ema=True,
    ),
    "x": TrainOptionDefaults(
        epochs=80,
        batch=32,
        lr0=2.5e-4,
        backbone_lr=2.5e-6,
        lrf=1.0,
        weight_decay=1.25e-4,
        amp=True,
        ema=True,
        ema_decay=0.9998,
    ),
}


def normalize_training_recipe(recipe: str) -> TrainingRecipeName:
    """Normalize and validate a public training recipe name."""
    normalized = recipe.strip().lower().replace("-", "_")
    if normalized in {"default", "auto"}:
        return "default"
    if normalized == "deim":
        return "deim"
    raise ValueError(
        "Unknown training recipe {!r}. Supported recipes: default, deim".format(recipe)
    )


def resolve_training_recipe(recipe: str, *, task: str) -> TrainingRecipe:
    """Resolve a recipe for the given task.

    DEIM is intentionally detection-only. Until the later implementation
    phases land, it is recognized but marked as not implemented so callers can
    fail with a clear message instead of silently running the D-FINE path.
    """
    recipe_name = normalize_training_recipe(recipe)
    resolved_task = normalize_task(task)
    if recipe_name == "deim" and resolved_task != "detect":
        raise ValueError("recipe='deim' is detection-only; use recipe='default' for this task")
    return TrainingRecipe(
        name=recipe_name,
        task=resolved_task,
        implemented=recipe_name == "default",
    )


def infer_dfine_model_size(config: Mapping[str, Any]) -> str | None:
    """Infer the D-FINE model size key from a self-contained nitid config."""
    backbone = config.get("HGNetv2")
    encoder = config.get("HybridEncoder")
    transformer = config.get("DFINETransformer")
    if not isinstance(backbone, Mapping) or not isinstance(encoder, Mapping):
        return None

    backbone_name = str(backbone.get("name", "")).upper()
    hidden_dim = encoder.get("hidden_dim")
    in_channels = encoder.get("in_channels")
    if backbone_name == "B0" and hidden_dim == 128:
        return "n"
    if backbone_name == "B0" and hidden_dim == 256:
        return "s"
    if backbone_name == "B2":
        return "m"
    if backbone_name == "B4":
        return "l"
    if backbone_name == "B5":
        return "x"

    if isinstance(transformer, Mapping):
        feat_channels = transformer.get("feat_channels")
        if in_channels == [512, 1024] and hidden_dim == 128:
            return "n"
        if feat_channels == [384, 384, 384]:
            return "x"
    return None


def default_train_options(
    *,
    recipe: TrainingRecipe,
    config: Mapping[str, Any],
) -> TrainOptionDefaults:
    """Return task-native public train defaults for the resolved recipe."""
    if recipe.name != "default" or recipe.task != "detect":
        return TrainOptionDefaults()

    model_size = infer_dfine_model_size(config)
    if model_size is None:
        return TrainOptionDefaults(
            epochs=72,
            batch=32,
            lr0=2.5e-4,
            backbone_lr=1.25e-5,
            lrf=1.0,
            weight_decay=1.25e-4,
            amp=True,
            ema=True,
        )
    return _OFFICIAL_DFINE_DETECTION_DEFAULTS[model_size]


def apply_training_recipe_to_config(
    config: Mapping[str, Any],
    *,
    recipe: TrainingRecipe,
) -> dict[str, Any]:
    """Return a config copy with recipe-specific criterion settings applied."""
    resolved = copy.deepcopy(dict(config))
    if recipe.name != "deim":
        return resolved

    criterion = resolved.get("DFINECriterion")
    if recipe.task != "detect" or not isinstance(criterion, dict):
        return resolved

    weight_dict = dict(criterion.get("weight_dict", {}))
    weight_dict.pop("loss_vfl", None)
    weight_dict["loss_mal"] = 1
    criterion["weight_dict"] = weight_dict
    criterion["losses"] = ["mal" if loss == "vfl" else loss for loss in criterion.get("losses", [])]
    if "mal" not in criterion["losses"]:
        criterion["losses"].insert(0, "mal")
    criterion["gamma"] = 1.5
    criterion.setdefault("mal_alpha", None)
    return resolved
