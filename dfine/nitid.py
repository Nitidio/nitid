"""Public model-family-neutral NITID API."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dfine.model import DFINE
from dfine.tasks import normalize_task

_NITID_MODEL_RE = re.compile(r"^model(?P<version>\d+)(?P<size>[nsmxl])$")
_SUPPORTED_VERSION = 1


@dataclass(frozen=True)
class NitidModelSpec:
    """Parsed public nitid model identifier."""

    name: str
    version: int
    size: str


def parse_nitid_model_name(model: str | Path) -> NitidModelSpec:
    """Parse a public model name such as ``model1s``."""
    if not isinstance(model, str):
        raise TypeError("NITID model names must be strings such as 'model1s'")

    normalized = model.lower().strip().replace("-", "").replace("_", "")
    match = _NITID_MODEL_RE.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"Unsupported NITID model {model!r}. Expected names like "
            "'model1n', 'model1s', 'model1m', 'model1l', or 'model1x'."
        )

    version = int(match.group("version"))
    if version != _SUPPORTED_VERSION:
        raise ValueError(
            f"Unsupported NITID model version {version}. "
            f"Currently supported version: {_SUPPORTED_VERSION}."
        )

    size = match.group("size")
    return NitidModelSpec(name=f"model{version}{size}", version=version, size=size)


def resolve_nitid_backend_model(model: str | Path, *, task: str) -> str:
    """Resolve a public ``nitid`` model name to the current backend model family."""
    spec = parse_nitid_model_name(model)
    resolved_task = normalize_task(task)

    if resolved_task == "detect" and spec.size == "n":
        raise ValueError(
            "NITID('model1n', task='detect') is not available because no official "
            "DFINE-N detection checkpoint is registered. Use model1s/m/l/x or a "
            "checkpoint path."
        )

    return f"dfine_{spec.size}"


class NITID(DFINE):
    """Model-family-neutral public entry point for nitid models.

    ``NITID`` accepts public model identifiers such as ``model1s`` and maps them
    to the concrete architecture backend for the requested task. Existing
    D-FINE behavior is reused while the public constructor moves away from
    assuming every model family is D-FINE.
    """

    def __init__(
        self,
        model: str | Path = "model1s",
        *,
        task: str = "detect",
        weights: str = "default",
        backend: str = "torch",
        device: str | int | None = None,
        verbose: bool = True,
    ) -> None:
        backend_model = resolve_nitid_backend_model(model, task=task)
        super().__init__(
            backend_model,
            task=task,
            weights=weights,
            backend=backend,
            device=device,
            verbose=verbose,
        )
        self._nitid_model = parse_nitid_model_name(model)

    @property
    def nitid_model(self) -> str:
        """Public nitid model identifier, for example ``'model1s'``."""
        return self._nitid_model.name

    @property
    def nitid_version(self) -> int:
        """Public nitid model-family version."""
        return self._nitid_model.version

    @property
    def size(self) -> str:
        """Model size suffix: ``n``, ``s``, ``m``, ``l``, or ``x``."""
        return self._nitid_model.size
