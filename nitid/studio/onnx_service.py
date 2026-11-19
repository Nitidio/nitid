"""On-demand ONNX export and cache behind ``GET /api/v1/onnx/{model}``.

What the endpoints promise (``studio/api-contract/openapi.json``): a *static*
``[1, 3, S, S]`` post-processed ONNX with ``nitid_meta`` embedded, whose strong ETag
is the lowercase sha256 of the body. This module produces and remembers those files;
the HTTP layer lives in :mod:`nitid.studio.routes_onnx`.

Design:

* **Cache key** -- ``(id, task, imgsz, nitid version)`` plus the checkpoint's size and
  mtime, so replacing a local ``*.pth`` never serves a stale export. File names inside
  the cache are derived from a hash of the key; nothing from the client reaches the
  file system.
* **Serialised** -- exports run one at a time (``threading.Lock``). A request that
  waited re-checks the cache once it holds the lock, so N simultaneous requests for
  the same file export it once. Handlers are plain ``def`` (thread pool), so a long
  export never blocks the event loop.
* **Atomic** -- the export is produced in a private staging directory (the exporter
  also writes ``args.yaml``/``environment.yaml`` next to its output, which must never
  reach the cache) and moved into place with ``os.replace``.
* **Hash once** -- the sha256 and the parsed meta are kept in memory per
  ``(path, size, mtime)``; a file replaced on disk is hashed again.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nitid.studio import registry

logger = logging.getLogger(__name__)

#: Used when the checkpoint does not say which size it was trained for.
DEFAULT_IMGSZ = 640
MIN_IMGSZ = 32
MAX_IMGSZ = 4096
IMGSZ_MULTIPLE = 32

#: ``(checkpoint, task, imgsz, destination) -> None``: write the ONNX at ``destination``.
Exporter = Callable[[Path, str, int, Path], None]


class InvalidParams(ValueError):
    """A query parameter is missing or out of range (HTTP 422 ``invalid_params``)."""


class ExportFailed(RuntimeError):
    """The exporter raised; the message is safe to show to the client."""


@dataclass(frozen=True)
class OnnxArtifact:
    """A cached export: where it is, its hash and its parsed meta."""

    path: Path
    sha256: str
    size: int
    meta: dict[str, Any]

    @property
    def etag(self) -> str:
        """Strong validator, quoted as HTTP requires: ``"<sha256 hex>"``."""
        return f'"{self.sha256}"'

    def meta_with_sha256(self) -> dict[str, Any]:
        """Body of ``GET /onnx/{model}/meta``: the embedded meta plus ``sha256``."""
        return {**self.meta, "sha256": self.sha256}


def parse_task(value: str | None) -> str:
    if value is None or value not in registry.STUDIO_TASKS:
        allowed = ", ".join(registry.STUDIO_TASKS)
        raise InvalidParams(f"Query parameter task is required: {allowed}.")
    return value


def parse_imgsz(value: str | None) -> int | None:
    """``None`` means "the model's default size". Raises :class:`InvalidParams`."""
    if value is None:
        return None
    text = value.strip()
    if not text.isascii() or not text.isdigit():
        raise InvalidParams("Query parameter imgsz must be an integer.")
    size = int(text)
    if not MIN_IMGSZ <= size <= MAX_IMGSZ or size % IMGSZ_MULTIPLE:
        raise InvalidParams(
            f"Query parameter imgsz must be a multiple of {IMGSZ_MULTIPLE} "
            f"between {MIN_IMGSZ} and {MAX_IMGSZ}."
        )
    return size


def default_imgsz(checkpoint: Path) -> int:
    """The checkpoint's ``eval_spatial_size`` when usable, else :data:`DEFAULT_IMGSZ`.

    Reads only the pickled metadata (``weights_only`` + ``mmap``), like the catalogue scan.
    """
    import torch

    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        size = ckpt["config"]["eval_spatial_size"][0]
    except Exception:
        return DEFAULT_IMGSZ
    if isinstance(size, int) and MIN_IMGSZ <= size <= MAX_IMGSZ and size % IMGSZ_MULTIPLE == 0:
        return size
    return DEFAULT_IMGSZ


def export_checkpoint(checkpoint: Path, task: str, imgsz: int, destination: Path) -> None:
    """Default exporter: static batch 1, post-processed, simplified, ``nitid_meta`` embedded.

    Runs on CPU so the result does not depend on (or occupy) a GPU. The exporter writes
    run metadata next to its output, hence ``destination`` must be in a private directory.
    """
    from dfine import DFINE

    model = DFINE(checkpoint, task=task, device="cpu", verbose=False)
    model.export(
        format="onnx",
        imgsz=imgsz,
        batch=1,
        dynamic=False,
        simplify=True,
        postprocess=True,
        output=destination,
        exist_ok=True,
        verbose=False,
    )


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OnnxService:
    """Export-on-demand with a disk cache. Thread-safe; one instance per app."""

    def __init__(self, exporter: Exporter | None = None) -> None:
        self._exporter: Exporter = exporter if exporter is not None else export_checkpoint
        self._export_lock = threading.Lock()
        self._info_lock = threading.Lock()
        self._info: dict[tuple[str, int, int], OnnxArtifact] = {}

    # ------------------------------------------------------------------ public

    def get(
        self, models_dir: Path, model_id: str, task: str, imgsz: int | None = None
    ) -> OnnxArtifact:
        """Return the (possibly just exported) ONNX for ``(model_id, task, imgsz)``.

        Raises :class:`nitid.studio.registry.ModelNotAvailable` (404) when the model is
        unknown or not downloaded, and :class:`ExportFailed` when exporting fails.
        """
        checkpoint = registry.locate_checkpoint(models_dir, model_id, task)
        size = imgsz if imgsz is not None else default_imgsz(checkpoint)
        cache_dir = cache_directory(models_dir)
        target = cache_dir / self._file_name(model_id, task, size, checkpoint)

        if not target.is_file():
            with self._export_lock:
                if not target.is_file():  # another request may have exported it meanwhile
                    self._export(checkpoint, task, size, cache_dir, target)
        return self._artifact(target)

    # ----------------------------------------------------------------- private

    @staticmethod
    def _file_name(model_id: str, task: str, imgsz: int, checkpoint: Path) -> str:
        from nitid import __version__

        stat = checkpoint.stat()
        key = "\0".join(
            [model_id, task, str(imgsz), __version__, str(stat.st_size), str(stat.st_mtime_ns)]
        )
        return f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.onnx"

    def _export(
        self, checkpoint: Path, task: str, imgsz: int, cache_dir: Path, target: Path
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        staging = cache_dir / f".staging-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            produced = staging / "model.onnx"
            try:
                self._exporter(checkpoint, task, imgsz, produced)
            except Exception as exc:
                logger.exception("ONNX export failed for %s", checkpoint.name)
                raise ExportFailed(f"ONNX export failed ({type(exc).__name__}).") from exc
            if not produced.is_file():
                raise ExportFailed("ONNX export finished without producing a file.")
            os.replace(produced, target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _artifact(self, path: Path) -> OnnxArtifact:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        with self._info_lock:
            known = self._info.get(key)
        if known is not None:
            return known

        from dfine.export_meta import read_onnx_meta

        try:
            meta = read_onnx_meta(path)
        except ValueError as exc:
            raise ExportFailed("The exported ONNX carries an unreadable nitid_meta.") from exc
        if meta is None:
            raise ExportFailed("The exported ONNX carries no nitid_meta.")
        artifact = OnnxArtifact(path, _sha256_of(path), stat.st_size, meta)
        with self._info_lock:
            for stale in [k for k in self._info if k[0] == key[0]]:
                del self._info[stale]
            self._info[key] = artifact
        return artifact


def cache_directory(models_dir: Path) -> Path:
    """``<models>/.cache/onnx``. Not scanned by the catalogue (it lists ``*.pth`` only)."""
    return models_dir / ".cache" / "onnx"
