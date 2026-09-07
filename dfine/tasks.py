"""Canonical task names and capability contracts for D-FINE models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

Task = Literal["detect", "segment", "semantic", "pose", "obb"]
SUPPORTED_TASKS: tuple[Task, ...] = ("detect", "segment", "semantic", "pose", "obb")


@dataclass(frozen=True)
class TaskContract:
    """Stable public behavior associated with a task name."""

    name: Task
    aliases: tuple[str, ...]
    result_fields: tuple[str, ...]
    instance_level: bool
    pretrained_source_task: Task


TASK_CONTRACTS: dict[Task, TaskContract] = {
    "detect": TaskContract(
        name="detect",
        aliases=("detection",),
        result_fields=("boxes",),
        instance_level=True,
        pretrained_source_task="detect",
    ),
    "segment": TaskContract(
        name="segment",
        aliases=(),
        result_fields=("boxes", "masks"),
        instance_level=True,
        pretrained_source_task="segment",
    ),
    "semantic": TaskContract(
        name="semantic",
        aliases=("sem_seg",),
        result_fields=("semantic_mask",),
        instance_level=False,
        pretrained_source_task="segment",
    ),
    "pose": TaskContract(
        name="pose",
        aliases=("keypoint", "keypoints", "pose_estimation"),
        result_fields=("keypoints",),
        instance_level=True,
        pretrained_source_task="pose",
    ),
    "obb": TaskContract(
        name="obb",
        aliases=("oriented", "oriented_detection", "rotated", "rotated_detection"),
        result_fields=("obb",),
        instance_level=True,
        pretrained_source_task="obb",
    ),
}

_TASK_ALIASES = {
    alias: contract.name for contract in TASK_CONTRACTS.values() for alias in contract.aliases
}


def normalize_task(task: str) -> Task:
    """Return the canonical public task name, accepting documented aliases."""
    if not isinstance(task, str):
        raise TypeError(f"task must be a string, got {type(task).__name__}")
    normalized = task.lower().strip().replace("-", "_")
    normalized = _TASK_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_TASKS:
        supported = ", ".join(SUPPORTED_TASKS)
        raise ValueError(f"Unsupported task {task!r}. Choose: {supported}")
    return cast(Task, normalized)


def get_task_contract(task: str) -> TaskContract:
    """Return the immutable capability contract for ``task``."""
    return TASK_CONTRACTS[normalize_task(task)]
