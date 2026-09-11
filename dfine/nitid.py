"""Public model-family-neutral NITID API."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

from dfine.model import DFINE, ModelCallback
from dfine.tasks import normalize_task
from dfine.utils.device import resolve_device

_NITID_MODEL_RE = re.compile(r"^nitid(?P<version>\d+)(?P<size>[nsmxl])$")
_SUPPORTED_VERSION = 1


@dataclass(frozen=True)
class NitidModelSpec:
    """Parsed public nitid model identifier."""

    name: str
    version: int
    size: str


def parse_nitid_model_name(model: str | Path) -> NitidModelSpec:
    """Parse a public model name such as ``nitid1s``."""
    if not isinstance(model, str):
        raise TypeError("NITID model names must be strings such as 'nitid1s'")

    normalized = model.lower().strip().replace("-", "").replace("_", "")
    match = _NITID_MODEL_RE.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"Unsupported NITID model {model!r}. Expected names like "
            "'nitid1n', 'nitid1s', 'nitid1m', 'nitid1l', or 'nitid1x'."
        )

    version = int(match.group("version"))
    if version != _SUPPORTED_VERSION:
        raise ValueError(
            f"Unsupported NITID model version {version}. "
            f"Currently supported version: {_SUPPORTED_VERSION}."
        )

    size = match.group("size")
    return NitidModelSpec(name=f"nitid{version}{size}", version=version, size=size)


def resolve_nitid_backend_model(model: str | Path, *, task: str) -> str:
    """Resolve a public ``nitid`` model name to the current backend model family."""
    spec = parse_nitid_model_name(model)
    resolved_task = normalize_task(task)

    if resolved_task == "obb":
        return spec.name

    if resolved_task == "pose":
        return f"detrpose_{spec.size}"

    if resolved_task == "detect" and spec.size == "n":
        raise ValueError(
            "NITID('nitid1n', task='detect') is not available because no official "
            "DFINE-N detection checkpoint is registered. Use nitid1s/m/l/x or a "
            "checkpoint path."
        )

    return f"dfine_{spec.size}"


class NITID(DFINE):
    """Model-family-neutral public entry point for nitid models.

    ``NITID`` accepts public model identifiers such as ``nitid1s`` and maps them
    to the concrete architecture backend for the requested task. Existing
    D-FINE/DETRPose behavior is reused while the public constructor moves away
    from assuming every model family is D-FINE.
    """

    def __init__(
        self,
        model: str | Path = "nitid1s",
        *,
        task: str = "detect",
        weights: str | None = "default",
        backend: str = "torch",
        device: str | int | None = None,
        verbose: bool = True,
    ) -> None:
        backend_model = resolve_nitid_backend_model(model, task=task)
        if normalize_task(task) == "obb":
            if weights is None or (
                isinstance(weights, str) and weights.lower().replace("-", "_") in {"none", "random"}
            ):
                if backend == "openvino":
                    raise ValueError(
                        "backend='openvino' is not supported for randomly-initialized OBB "
                        "models — there is no pretrained checkpoint to trace"
                    )
                self._init_random_obb(
                    model=model,
                    backend_model=backend_model,
                    weights=weights,
                    device=device,
                    verbose=verbose,
                )
            else:
                super().__init__(
                    backend_model,
                    task=task,
                    weights=weights,
                    backend=backend,
                    device=device,
                    verbose=verbose,
                )
                self._nitid_model = parse_nitid_model_name(model)
            return
        if weights is None:
            raise ValueError("weights=None is currently only supported for task='obb'")
        super().__init__(
            backend_model,
            task=task,
            weights=weights,
            backend=backend,
            device=device,
            verbose=verbose,
        )
        self._nitid_model = parse_nitid_model_name(model)

    def _init_random_obb(
        self,
        *,
        model: str | Path,
        backend_model: str,
        weights: str | None,
        device: str | int | None,
        verbose: bool,
    ) -> None:
        from dfine.nn.native_build import build_native_model
        from dfine.nn.rio import DOTA_OBB_NAMES, make_rio_obb_config

        self._backend = "torch"
        self._openvino_device: str | None = None
        self._openvino_cache: dict[int, Any] = {}
        self._model_lock = threading.RLock()
        self._device_str: str = resolve_device(device)
        self.verbose = verbose
        self._cfg: dict[str, Any] = make_rio_obb_config(
            backend_model,
            num_classes=len(DOTA_OBB_NAMES),
        )
        self._model: nn.Module = build_native_model(
            backend_model,
            num_classes=len(DOTA_OBB_NAMES),
            task="obb",
            image_size=tuple(self._cfg["eval_spatial_size"]),
            device=self._device_str,
        )
        self._names: dict[int, str] = dict(enumerate(DOTA_OBB_NAMES))
        self._path = backend_model
        self._weights = None
        self._deployed_model: nn.Module | None = None
        self._deployed_model_device: str | None = None
        self._task = "obb"
        self._callbacks: dict[str, list[ModelCallback]] = {}
        self._nitid_model = parse_nitid_model_name(model)
        self._model.eval()
        if self.verbose:
            n_params = sum(p.numel() for p in self._model.parameters())
            print(
                f"[NITID] Built '{self._nitid_model.name}' task='obb' "
                f"— {n_params / 1e6:.1f}M params on {self._device_str}"
            )

    @property
    def nitid_model(self) -> str:
        """Public nitid model identifier, for example ``'nitid1s'``."""
        return self._nitid_model.name

    @property
    def nitid_version(self) -> int:
        """Public nitid model-family version."""
        return self._nitid_model.version

    @property
    def size(self) -> str:
        """Model size suffix: ``n``, ``s``, ``m``, ``l``, or ``x``."""
        return self._nitid_model.size
