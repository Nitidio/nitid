"""Model catalogue for ``GET /api/v1/models`` and registry downloads.

Two sources feed the catalogue:

* **registry** -- official pretrained weights known to ``dfine.utils.downloads``
  (``MODEL_REGISTRY`` for detection, ``SEGMENT_MODEL_REGISTRY`` for instance
  segmentation). One entry per ``(id, task)`` pair, ``id`` being the public
  ``nitid1{n,s,m,l,x}`` name.
* **local** -- wrapped ``*.pth`` checkpoints found (non-recursively) in
  ``settings.models_dir``.

Listing never touches the network and never loads weights: ``downloaded`` is
decided by whether the checkpoint file exists in ``models_dir``, and local files
are inspected through ``torch.load(weights_only=True, mmap=True)``, which reads
the pickled metadata but memory-maps (does not read) the tensors.

Decision D8 (2026-11-19): ``nitid1n`` for ``detect`` is registered with a "COCO only"
notice (``dfine_n_coco.pth``), so it is listed like every other size.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError

logger = logging.getLogger(__name__)

#: Tasks the Studio API exposes (``Task`` in the contract). ``semantic`` exists in
#: ``dfine`` but is not part of API v1.
STUDIO_TASKS: tuple[str, ...] = ("detect", "segment")

#: Local checkpoints larger than this are not inspected (and are not listed).
MAX_LOCAL_CHECKPOINT_BYTES = 2 * 1024**3
#: At most this many ``*.pth`` files are inspected per listing.
MAX_LOCAL_SCAN_FILES = 500

_SIZE_ORDER = ("n", "s", "m", "l", "x")
_PROGRESS = Callable[[float], None]


@dataclass(frozen=True)
class RegistryEntry:
    """One official ``(id, task)`` pair."""

    id: str
    task: str
    size: str
    backend_model: str  # name inside dfine.utils.downloads, e.g. "dfine_s"


@dataclass(frozen=True)
class LocalModel:
    """A user checkpoint discovered in the models directory."""

    id: str
    task: str
    size: str | None
    path: Path


# --------------------------------------------------------------------------- registry


def _registry_sizes(task: str) -> list[str]:
    """Sizes that have official weights for ``task``, read from ``downloads.py``."""
    from dfine.utils.downloads import MODEL_REGISTRY, SEGMENT_MODEL_REGISTRY

    source = MODEL_REGISTRY if task == "detect" else SEGMENT_MODEL_REGISTRY
    sizes = {name.removeprefix("dfine_") for name in source}
    return [size for size in _SIZE_ORDER if size in sizes]


def registry_entries() -> list[RegistryEntry]:
    """All exposed registry pairs, ordered by task (detect first) then size."""
    from dfine.nitid import parse_nitid_model_name

    entries: list[RegistryEntry] = []
    for task in STUDIO_TASKS:
        for size in _registry_sizes(task):
            public = parse_nitid_model_name(f"dfine_{size}").name
            entries.append(RegistryEntry(public, task, size, f"dfine_{size}"))
    return entries


def registry_ids() -> frozenset[str]:
    """Every public id that has at least one exposed task."""
    return frozenset(entry.id for entry in registry_entries())


def find_registry_entry(model_id: str, task: str) -> RegistryEntry | None:
    """Exact match on the canonical id (no aliases, no normalisation) and task."""
    for entry in registry_entries():
        if entry.id == model_id and entry.task == task:
            return entry
    return None


def weights_path(models_dir: Path, entry: RegistryEntry) -> Path | None:
    """Where the checkpoint for ``entry`` lives, or ``None`` if the registry has no asset."""
    from dfine.utils.downloads import get_model_asset

    try:
        asset = get_model_asset(entry.backend_model, "default", task=entry.task)
    except ValueError:
        return None
    return models_dir / asset.filename


def is_downloaded(models_dir: Path, entry: RegistryEntry) -> bool:
    path = weights_path(models_dir, entry)
    return path is not None and path.is_file()


def _registry_filenames() -> frozenset[str]:
    """File names that official downloads produce (used to avoid listing them twice)."""
    from dfine.utils.downloads import (
        MODEL_REGISTRY,
        SEGMENT_MODEL_REGISTRY,
        SEMANTIC_MODEL_REGISTRY,
    )

    names: set[str] = set()
    for registry in (MODEL_REGISTRY, SEGMENT_MODEL_REGISTRY, SEMANTIC_MODEL_REGISTRY):
        for variants in registry.values():
            names.update(asset.filename for asset in variants.values())
    return frozenset(names)


# --------------------------------------------------------------------------- local scan

_META_CACHE: dict[tuple[str, int, int], tuple[str, str | None] | None] = {}
_META_LOCK = threading.Lock()


def _read_checkpoint_meta(path: Path) -> tuple[str, str | None] | None:
    """Return ``(task, size)`` for a wrapped checkpoint, or ``None`` if it is not one.

    Only ``torch.load(..., weights_only=True)`` is used, so unpickling cannot execute
    code. ``mmap=True`` keeps the tensors on disk. Any failure means "not a usable
    checkpoint" and is logged, never raised.
    """
    import torch

    from dfine.nitid import infer_nitid_model_spec
    from dfine.tasks import normalize_task

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(ckpt, Mapping) or "model" not in ckpt:
            raise ValueError("not a wrapped nitid checkpoint (no 'model')")
        config = ckpt.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("no embedded config")
        names = ckpt.get("names", {})
        if not isinstance(names, (Mapping, list, tuple)):
            raise ValueError("names has an unexpected type")
        config_task = normalize_task(str(config.get("task", "detect")))
        metadata_task = ckpt.get("task")
        if metadata_task is not None and normalize_task(str(metadata_task)) != config_task:
            raise ValueError("task metadata does not match the embedded config")
        spec = infer_nitid_model_spec(dict(config))
        size = spec.size if spec is not None else None
        del ckpt
        return config_task, size
    except Exception as exc:
        logger.warning("Skipping unreadable checkpoint %s (%s)", path.name, type(exc).__name__)
        return None


def scan_local_models(models_dir: Path) -> list[LocalModel]:
    """List usable ``*.pth`` checkpoints directly inside ``models_dir``.

    Skipped, with a log line and without failing the listing: unreadable or
    corrupt files, files over :data:`MAX_LOCAL_CHECKPOINT_BYTES`, tasks outside
    :data:`STUDIO_TASKS`, files that official downloads produced (they appear as
    ``registry`` entries) and files whose stem equals an official id (the id
    would be ambiguous).
    """
    if not models_dir.is_dir():
        return []
    official_files = _registry_filenames()
    official_ids = registry_ids()
    found: list[LocalModel] = []
    seen_keys: set[tuple[str, int, int]] = set()

    try:
        candidates = sorted(p for p in models_dir.iterdir() if p.suffix == ".pth")
    except OSError:
        logger.warning("Cannot read the models directory", exc_info=True)
        return []

    for path in candidates[:MAX_LOCAL_SCAN_FILES]:
        if path.name in official_files:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        if stat.st_size > MAX_LOCAL_CHECKPOINT_BYTES:
            logger.warning(
                "Skipping %s: larger than %d bytes", path.name, MAX_LOCAL_CHECKPOINT_BYTES
            )
            continue
        if path.stem in official_ids:
            logger.warning("Skipping %s: its name collides with an official model id", path.name)
            continue

        key = (str(path), stat.st_mtime_ns, stat.st_size)
        seen_keys.add(key)
        with _META_LOCK:
            cached = key in _META_CACHE
            meta = _META_CACHE.get(key)
        if not cached:
            meta = _read_checkpoint_meta(path)
            with _META_LOCK:
                _META_CACHE[key] = meta
        if meta is None:
            continue
        task, size = meta
        if task not in STUDIO_TASKS:
            logger.info("Skipping %s: task %r is not exposed by Studio", path.name, task)
            continue
        found.append(LocalModel(id=path.stem, task=task, size=size, path=path.resolve()))

    with _META_LOCK:  # forget entries for files that changed or disappeared
        for stale in [
            k for k in _META_CACHE if k[0].startswith(str(models_dir)) and k not in seen_keys
        ]:
            del _META_CACHE[stale]
    return found


class ModelNotAvailable(LookupError):
    """The model is unknown or its checkpoint is not on disk (HTTP 404 ``not_found``)."""


def locate_checkpoint(models_dir: Path, model_id: str, task: str) -> Path:
    """Path of the checkpoint for ``(model_id, task)``.

    Raises :class:`ModelNotAvailable` with a client-safe message (no file system paths)
    when the id is unknown, has no weights for ``task``, or is not downloaded yet.
    """
    entry = find_registry_entry(model_id, task)
    if entry is not None:
        path = weights_path(models_dir, entry)
        if path is None or not path.is_file():
            raise ModelNotAvailable(
                f"Model {_quote(model_id)} ({task}) is not downloaded; "
                f"POST /api/v1/models/{model_id}/download?task={task} first."
            )
        return path
    for local in scan_local_models(models_dir):
        if local.id == model_id and local.task == task:
            return local.path
    if model_id in registry_ids():
        raise ModelNotAvailable(f"No official weights for {_quote(model_id)} with task {task!r}.")
    raise ModelNotAvailable(f"Model {_quote(model_id)} ({task}) does not exist.")


def _quote(value: str, limit: int = 64) -> str:
    return repr(value if len(value) <= limit else value[:limit] + "...")


def local_model_ids(models_dir: Path) -> frozenset[str]:
    """Stems of ``*.pth`` files in ``models_dir``, without opening any of them."""
    if not models_dir.is_dir():
        return frozenset()
    try:
        return frozenset(p.stem for p in models_dir.iterdir() if p.suffix == ".pth" and p.is_file())
    except OSError:
        return frozenset()


def list_models(models_dir: Path) -> list[dict[str, Any]]:
    """Payload of ``GET /api/v1/models`` (schema ``ModelList`` in the API contract)."""
    models: list[dict[str, Any]] = [
        {
            "id": entry.id,
            "task": entry.task,
            "size": entry.size,
            "downloaded": is_downloaded(models_dir, entry),
            "source": "registry",
        }
        for entry in registry_entries()
    ]
    for local in scan_local_models(models_dir):
        models.append(
            {
                "id": local.id,
                "task": local.task,
                "size": local.size,
                "downloaded": True,
                "source": "local",
                "path": str(local.path),
            }
        )
    return models


# --------------------------------------------------------------------------- download


def download_registry_model(entry: RegistryEntry, models_dir: Path, progress: _PROGRESS) -> Path:
    """Download and wrap the official checkpoint for ``entry`` into ``models_dir``.

    The file is produced in a private staging directory and moved into place with
    ``os.replace``, so a half-written checkpoint is never reported as downloaded.
    ``dfine.utils.downloads.download_model`` offers no progress hook, so progress
    goes 0 -> 1. Nothing here comes from the client except the already-validated
    ``(id, task)`` pair: the URL and file name are taken from the registry.
    """
    from dfine.utils.downloads import download_model

    final = weights_path(models_dir, entry)
    if final is None:
        raise ValueError("no official weights are registered for this model")
    progress(0.0)
    if final.is_file():
        progress(1.0)
        return final

    models_dir.mkdir(parents=True, exist_ok=True)
    staging = models_dir / f".partial-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        produced = Path(
            download_model(
                entry.backend_model, task=entry.task, weights="default", output=staging, force=True
            )
        )
        if not produced.is_file():
            raise RuntimeError("download finished without producing a checkpoint")
        os.replace(produced, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    progress(1.0)
    return final


def describe_download_error(exc: BaseException) -> str:
    """A client-safe message for a failed download (never includes paths or URLs)."""
    if isinstance(exc, URLError | TimeoutError | ConnectionError):
        return "Network error while downloading the checkpoint."
    if isinstance(exc, OSError):
        return "Could not write the checkpoint to the models directory."
    if isinstance(exc, ValueError):
        return "No official weights are registered for this model."
    if isinstance(exc, RuntimeError) and str(exc).startswith("Checksum mismatch"):
        return "Checksum verification of the downloaded checkpoint failed."
    return f"Download failed ({type(exc).__name__})."
