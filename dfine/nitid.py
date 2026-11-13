"""Public model-family-neutral NITID API."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dfine.model import DFINE
from dfine.tasks import normalize_task

# Accepted spellings, after lower-casing and dropping "-" and "_":
#   nitid1s   canonical public name (``nitid<version><size>``)
#   model1s   legacy alias used up to 0.9.4; resolves to ``nitid1s``
#   dfines    technical alias (``d-fine-s``, ``dfine_s``); resolves to the
#             current model generation of that size
_NITID_MODEL_RE = re.compile(r"^(?:(?:nitid|model)(?P<version>\d+)|dfine)(?P<size>[nsmxl])$")
_LEGACY_NAME_RE = re.compile(r"^model\d+[nsmxl]$")
_SUPPORTED_VERSION = 1
PUBLIC_NAME_PREFIXES = ("nitid", "model", "d-fine")


@dataclass(frozen=True)
class NitidModelSpec:
    """Parsed public nitid model identifier."""

    name: str
    version: int
    size: str


def is_public_model_name(model: str | Path) -> bool:
    """Return whether ``model`` is spelled like a public nitid name or alias.

    This only looks at the prefix so callers can route public names through
    :func:`parse_nitid_model_name` (which validates them) and pass everything
    else, such as ``dfine_s`` backend names and checkpoint paths, through
    unchanged.
    """
    return isinstance(model, str) and model.lower().strip().startswith(PUBLIC_NAME_PREFIXES)


def parse_nitid_model_name(model: str | Path) -> NitidModelSpec:
    """Parse a public model name such as ``nitid1s``.

    ``model1s`` (legacy) and ``d-fine-s`` / ``dfine_s`` (technical alias) are
    accepted and resolve to the same canonical ``nitid1s``.
    """
    if not isinstance(model, str):
        raise TypeError("NITID model names must be strings such as 'nitid1s'")

    normalized = model.lower().strip().replace("-", "").replace("_", "")
    match = _NITID_MODEL_RE.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"Unsupported NITID model {model!r}. Expected names like "
            "'nitid1n', 'nitid1s', 'nitid1m', 'nitid1l', or 'nitid1x' "
            "(aliases: 'd-fine-s', ...)."
        )

    version_text = match.group("version")
    version = int(version_text) if version_text is not None else _SUPPORTED_VERSION
    if version != _SUPPORTED_VERSION:
        raise ValueError(
            f"Unsupported NITID model version {version}. "
            f"Currently supported version: {_SUPPORTED_VERSION}."
        )

    size = match.group("size")
    return NitidModelSpec(name=f"nitid{version}{size}", version=version, size=size)


def _warn_if_legacy_name(model: str | Path, *, stacklevel: int) -> None:
    """Warn when ``model`` uses a legacy ``model<version><size>`` name (<= 0.9.4).

    The technical aliases (``d-fine-s``) and the hyphenated canonical spelling
    (``nitid1-s``) are not deprecated and never warn. ``stacklevel`` is counted
    from this function, so ``3`` attributes the warning to the caller of the
    function that calls it (here, the user's ``NITID(...)`` call). That keeps
    it visible under Python's default filter, which only shows
    ``DeprecationWarning`` raised from ``__main__``.
    """
    if not isinstance(model, str):
        return
    normalized = model.lower().strip().replace("-", "").replace("_", "")
    if _LEGACY_NAME_RE.fullmatch(normalized) is None:
        return
    canonical = parse_nitid_model_name(model).name
    warnings.warn(
        f"{model!r} is a legacy model name and will be removed in a later minor "
        f"release; use {canonical!r} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def resolve_nitid_backend_model(model: str | Path, *, task: str) -> str:
    """Resolve a public ``nitid`` model name to the current backend model family."""
    spec = parse_nitid_model_name(model)
    resolved_task = normalize_task(task)

    if resolved_task == "detect" and spec.size == "n":
        raise ValueError(
            f"NITID({spec.name!r}, task='detect') is not available because no official "
            "DFINE-N detection checkpoint is registered. Use nitid1s/m/l/x or a "
            "checkpoint path."
        )

    return f"dfine_{spec.size}"


def _is_checkpoint_path(model: str | Path) -> bool:
    """Return whether ``model`` names a checkpoint file rather than a registry model."""
    if isinstance(model, Path):
        return True
    path = Path(model)
    return path.exists() or path.suffix == ".pth" or len(path.parts) > 1


def infer_nitid_model_spec(config: dict[str, Any]) -> NitidModelSpec | None:
    """Infer the public model identifier from a checkpoint's embedded config.

    Sizes are told apart by their backbone and encoder width, which training
    overrides such as decoder depth or query count leave untouched. Returns
    ``None`` when the config does not match any supported size.
    """
    from dfine.nn.configs import MODEL_CONFIGS

    def signature(cfg: dict[str, Any]) -> tuple[Any, Any] | None:
        backbone = cfg.get("HGNetv2")
        encoder = cfg.get("HybridEncoder")
        if not isinstance(backbone, dict) or not isinstance(encoder, dict):
            return None
        return backbone.get("name"), encoder.get("hidden_dim")

    checkpoint_signature = signature(config)
    if checkpoint_signature is None:
        return None
    for size, size_config in MODEL_CONFIGS.items():
        if signature(size_config) == checkpoint_signature:
            return NitidModelSpec(
                name=f"nitid{_SUPPORTED_VERSION}{size}", version=_SUPPORTED_VERSION, size=size
            )
    return None


class NITID(DFINE):
    """Model-family-neutral public entry point for nitid models.

    ``NITID`` accepts public model identifiers such as ``nitid1s`` and maps them
    to the concrete architecture backend for the requested task. It also
    accepts a path to a self-contained checkpoint, such as one written by
    ``train()``; the checkpoint's embedded task must match ``task``. Existing
    D-FINE behavior is reused while the public constructor moves away from
    assuming every model family is D-FINE.
    """

    def __init__(
        self,
        model: str | Path = "nitid1s",
        *,
        task: str = "detect",
        weights: str = "default",
        backend: str = "torch",
        device: str | int | None = None,
        verbose: bool = True,
    ) -> None:
        from_checkpoint = _is_checkpoint_path(model)
        backend_model = (
            str(model) if from_checkpoint else resolve_nitid_backend_model(model, task=task)
        )
        if not from_checkpoint:
            _warn_if_legacy_name(model, stacklevel=3)
        super().__init__(
            backend_model,
            task=task,
            weights=weights,
            backend=backend,
            device=device,
            verbose=verbose,
        )
        self._nitid_model: NitidModelSpec | None = (
            infer_nitid_model_spec(self._cfg) if from_checkpoint else parse_nitid_model_name(model)
        )

    @property
    def nitid_model(self) -> str | None:
        """Public nitid model identifier, for example ``'nitid1s'``.

        ``None`` for a checkpoint whose architecture matches no supported size.
        """
        return self._nitid_model.name if self._nitid_model else None

    @property
    def nitid_version(self) -> int | None:
        """Public nitid model-family version, or ``None`` if it cannot be inferred."""
        return self._nitid_model.version if self._nitid_model else None

    @property
    def size(self) -> str | None:
        """Model size suffix (``n``, ``s``, ``m``, ``l``, ``x``), or ``None`` if unknown."""
        return self._nitid_model.size if self._nitid_model else None
