"""Training recipe resolution for nitid's public training API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
