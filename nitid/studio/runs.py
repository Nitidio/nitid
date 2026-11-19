"""Runs: validated parameters, image intake, background inference and result serialisation.

This module is the framework-free core of ``/api/v1/runs`` (the HTTP layer lives in
:mod:`nitid.studio.routes_runs`). Design constraints, all deliberate:

* **Local single-user tool**: runs live in process memory, their uploaded images in a
  private temporary directory. Both vanish when the server stops.
* **One inference at a time**: runs are queued on a :class:`~nitid.studio.jobs.JobStore`
  with a single worker thread, so a laptop is never asked to run two models at once.
  A second run stays ``pending`` until the first one finishes.
* **Lazy model loading**: nothing is loaded when the server starts. The worker loads the
  model on the first run that needs it and keeps the most recent
  :data:`MODEL_CACHE_SIZE` models.
* **No leaks**: the ``error_msg`` a client sees always comes from :func:`describe_run_error`
  (fixed, path-free sentences), never from ``str(exception)``.
* **Contract shapes**: :func:`result_to_item` produces the ``RunItem`` schema of
  ``studio/api-contract/openapi.json``: boxes are ``xyxy`` in pixels of the original image,
  confidences are in ``[0, 1]`` and item ``id`` is zero-based.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import secrets
import shutil
import tempfile
import threading
import weakref
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from nitid.studio import registry
from nitid.studio.jobs import JobStore, ProgressCallback
from nitid.studio.settings import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS

logger = logging.getLogger(__name__)

RunStatus = Literal["pending", "running", "done", "failed"]

# --------------------------------------------------------------------------- defaults/limits

#: Defaults for the optional ``params`` fields (the contract leaves them to the backend).
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.85
DEFAULT_IMGSZ = 640
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_FRAME_STEP = 1

#: Files accepted by one ``POST /runs``.
MAX_FILES_PER_RUN = 50
#: Runs kept in memory; creating one more evicts the oldest *finished* run.
MAX_RUNS = 50
#: Models kept loaded by the worker (the most recently used).
MODEL_CACHE_SIZE = 2
#: Longest file name echoed back in ``items[].name``.
MAX_NAME_LENGTH = 128

#: Image formats accepted (PIL format name -> media type, file extension).
ACCEPTED_FORMATS: dict[str, tuple[str, str]] = {
    "JPEG": ("image/jpeg", "jpg"),
    "PNG": ("image/png", "png"),
    "WEBP": ("image/webp", "webp"),
    "BMP": ("image/bmp", "bmp"),
}

_PARAM_KEYS = frozenset(
    {
        "model",
        "task",
        "conf",
        "iou",
        "imgsz",
        "classes",
        "mask_threshold",
        "backend",
        "device",
        "frame_step",
    }
)
_BACKENDS = ("torch", "openvino")


# --------------------------------------------------------------------------- errors


class ParamsError(ValueError):
    """The client sent invalid parameters (HTTP 422, code ``invalid_params``)."""


class ModelNotDownloaded(Exception):
    """A registry model has no weights in the models directory yet (HTTP 409)."""


class PayloadTooLarge(Exception):
    """An image is over the byte or pixel limit (HTTP 413)."""


class UnsupportedImage(Exception):
    """A file is not a supported image (HTTP 415)."""


class RunError(Exception):
    """A failure whose ``message`` is already safe to show to the client."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# --------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class RunParams:
    """Validated ``params`` of ``POST /runs`` with defaults applied."""

    model: str
    task: str
    backend: str
    device: str | None = None
    conf: float = DEFAULT_CONF
    iou: float = DEFAULT_IOU
    imgsz: int = DEFAULT_IMGSZ
    classes: tuple[int, ...] | None = None
    mask_threshold: float = DEFAULT_MASK_THRESHOLD
    frame_step: int = DEFAULT_FRAME_STEP


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unit_interval(data: Mapping[str, Any], key: str, default: float) -> float:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ParamsError(f"{key} must be a number between 0 and 1.")
    if not 0 <= value <= 1:
        raise ParamsError(f"{key} must be between 0 and 1.")
    return float(value)


def parse_run_params(text: str) -> RunParams:
    """Validate the JSON in the ``params`` form field (schema ``RunParams``).

    Unknown keys are rejected so that a typo (``confidence``) is not silently ignored.
    Raises :class:`ParamsError` with a client-safe message.
    """
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        raise ParamsError("params must be valid JSON.") from None
    if not isinstance(data, dict):
        raise ParamsError("params must be a JSON object.")

    unknown = sorted(set(data) - _PARAM_KEYS)
    if unknown:
        raise ParamsError(f"Unknown parameter: {unknown[0]!r}.")

    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise ParamsError("model is required and must be a non-empty string.")
    task = data.get("task")
    if task not in registry.STUDIO_TASKS:
        raise ParamsError(f"task is required: {', '.join(registry.STUDIO_TASKS)}.")
    backend = data.get("backend")
    if backend not in _BACKENDS:
        raise ParamsError(f"backend is required: {', '.join(_BACKENDS)}.")

    conf = _unit_interval(data, "conf", DEFAULT_CONF)
    iou = _unit_interval(data, "iou", DEFAULT_IOU)
    mask_threshold = _unit_interval(data, "mask_threshold", DEFAULT_MASK_THRESHOLD)

    imgsz = data.get("imgsz", DEFAULT_IMGSZ)
    if not _is_int(imgsz) or imgsz < 32 or imgsz % 32:
        raise ParamsError("imgsz must be an integer, at least 32 and a multiple of 32.")

    classes_raw = data.get("classes")
    classes: tuple[int, ...] | None = None
    if classes_raw is not None:
        if not isinstance(classes_raw, list) or not all(_is_int(c) and c >= 0 for c in classes_raw):
            raise ParamsError("classes must be null or a list of non-negative integers.")
        classes = tuple(classes_raw)

    device = data.get("device")
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise ParamsError("device must be null or a non-empty string.")

    frame_step = data.get("frame_step", DEFAULT_FRAME_STEP)
    if not _is_int(frame_step) or frame_step < 1:
        raise ParamsError("frame_step must be an integer >= 1.")

    return RunParams(
        model=model,
        task=task,
        backend=backend,
        device=device.strip() if isinstance(device, str) else None,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=classes,
        mask_threshold=mask_threshold,
        frame_step=frame_step,
    )


def validate_device(params: RunParams, devices: Mapping[str, Any]) -> None:
    """Check ``params.device`` against a ``GET /devices`` payload for ``params.backend``.

    ``None`` means "backend default" and is always accepted for an available backend.
    Raises :class:`ParamsError`.
    """
    if params.backend == "openvino":
        ov = devices.get("openvino", {})
        if not ov.get("available"):
            raise ParamsError("The openvino backend is not available on this machine.")
        names = [str(d) for d in ov.get("devices", [])]
        if params.device is not None and params.device not in names:
            raise ParamsError(
                f"Unknown OpenVINO device {echo(params.device)}. Available: {', '.join(names)}."
            )
        return
    torch_info = devices.get("torch", {})
    allowed = ["cpu"] + [f"cuda:{d['index']}" for d in torch_info.get("cuda", [])]
    if params.device is not None and params.device not in allowed:
        raise ParamsError(
            f"Unknown torch device {echo(params.device)}. Available: {', '.join(allowed)}."
        )


def echo(value: str, limit: int = 64) -> str:
    return repr(value if len(value) <= limit else value[:limit] + "...")


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class ResolvedModel:
    """A model id resolved to a checkpoint file."""

    id: str
    task: str
    path: Path


def resolve_model(models_dir: Path, model_id: str, task: str) -> ResolvedModel:
    """Map ``(model_id, task)`` to a checkpoint file in ``models_dir``.

    Raises :class:`ParamsError` for an unknown id or an id without weights for ``task``,
    and :class:`ModelNotDownloaded` for an official model that is not on disk yet.
    """
    entry = registry.find_registry_entry(model_id, task)
    if entry is not None:
        path = registry.weights_path(models_dir, entry)
        if path is None or not path.is_file():
            raise ModelNotDownloaded(
                f"Model {echo(model_id)} ({task}) is not downloaded yet. "
                f"Download it with POST /api/v1/models/{model_id}/download?task={task}."
            )
        return ResolvedModel(model_id, task, path)

    local = registry.scan_local_models(models_dir)
    for model in local:
        if model.id == model_id and model.task == task:
            return ResolvedModel(model_id, task, model.path)
    if registry_has_id(model_id) or any(m.id == model_id for m in local):
        raise ParamsError(f"Model {echo(model_id)} has no weights for task {task!r}.")
    raise ParamsError(f"Model {echo(model_id)} does not exist.")


def registry_has_id(model_id: str) -> bool:
    return model_id in registry.registry_ids()


ModelLoader = Callable[[ResolvedModel, RunParams], Any]


def load_nitid_model(resolved: ResolvedModel, params: RunParams) -> Any:
    """Default loader: ``NITID(checkpoint, task, backend, device)`` without console output."""
    from dfine.nitid import NITID

    return NITID(
        str(resolved.path),
        task=resolved.task,
        backend=params.backend,
        device=params.device,
        verbose=False,
    )


# --------------------------------------------------------------------------- images


@dataclass(frozen=True)
class ImageInfo:
    """Header facts about an accepted image (nothing was decoded to get them)."""

    format: str
    media_type: str
    extension: str
    width: int
    height: int


def inspect_image(data: bytes) -> ImageInfo:
    """Validate an upload from its bytes without decoding the pixels.

    Raises :class:`PayloadTooLarge` (over :data:`MAX_IMAGE_BYTES` or
    :data:`MAX_IMAGE_PIXELS`) or :class:`UnsupportedImage` (not JPEG/PNG/WEBP/BMP).
    The size checks run first, so an oversized body is never handed to a decoder.
    """
    if len(data) > MAX_IMAGE_BYTES:
        raise PayloadTooLarge(f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise UnsupportedImage("Empty file.")

    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            fmt = image.format or ""
            width, height = image.size
    except Image.DecompressionBombError:
        raise PayloadTooLarge(
            f"Image exceeds {MAX_IMAGE_PIXELS // 1_000_000} megapixels."
        ) from None
    except Exception:
        raise UnsupportedImage(_UNSUPPORTED_MESSAGE) from None
    if fmt not in ACCEPTED_FORMATS:
        raise UnsupportedImage(_UNSUPPORTED_MESSAGE)
    if width * height > MAX_IMAGE_PIXELS:
        raise PayloadTooLarge(f"Image exceeds {MAX_IMAGE_PIXELS // 1_000_000} megapixels.")
    media_type, extension = ACCEPTED_FORMATS[fmt]
    return ImageInfo(fmt, media_type, extension, width, height)


_UNSUPPORTED_MESSAGE = (
    "Unsupported file type. Send a JPEG, PNG, WEBP or BMP image (video is not supported yet)."
)


def safe_name(raw: str | None, index: int) -> str:
    """A display name for an upload: base name only, no control characters, bounded."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    if not name:
        name = f"image-{index}"
    return name[:MAX_NAME_LENGTH]


def decode_image(data: bytes) -> Any:
    """Decode bytes to an HWC BGR ``uint8`` array (EXIF orientation applied)."""
    import cv2
    import numpy as np

    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise RunError("Could not decode the image.")
    return array


# --------------------------------------------------------------------------- serialisation


def _class_name(names: Mapping[Any, str], class_id: int) -> str:
    if class_id in names:
        return str(names[class_id])
    return str(names.get(str(class_id), "unknown"))


def result_to_item(index: int, name: str, result: Any) -> dict[str, Any]:
    """Serialise one ``dfine`` ``Results`` as a ``RunItem`` (contract schema).

    ``index`` is the zero-based item id. Boxes are ``xyxy`` pixels of the original image.
    For ``segment`` results every detection has ``has_mask: true`` and an optional
    ``segments`` polygon ``{"x": [...], "y": [...]}`` (original-image pixels; additive
    to the v1 schema, same shape as ``Results.to_json``).
    """
    height, width = (int(v) for v in result.orig_img.shape[:2])
    detections: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        masks = result.masks
        polygons = masks.xy if masks is not None else []
        for i, (box, conf, cls) in enumerate(zip(xyxy, confs, classes)):
            class_id = int(cls)
            has_mask = masks is not None and i < len(polygons)
            detection: dict[str, Any] = {
                "box": {
                    "x1": round(box[0], 2),
                    "y1": round(box[1], 2),
                    "x2": round(box[2], 2),
                    "y2": round(box[3], 2),
                },
                "confidence": round(min(1.0, max(0.0, float(conf))), 4),
                "class_id": class_id,
                "name": _class_name(result.names, class_id),
                "has_mask": bool(has_mask),
            }
            if has_mask:
                polygon = polygons[i]
                detection["segments"] = {
                    "x": [round(float(v), 2) for v in polygon[:, 0].tolist()],
                    "y": [round(float(v), 2) for v in polygon[:, 1].tolist()],
                }
            detections.append(detection)
    speed = result.speed or {}
    return {
        "id": index,
        "name": name,
        "width": width,
        "height": height,
        "detections": detections,
        "speed": {
            key: round(max(0.0, float(speed.get(key, 0.0))), 3)
            for key in ("preprocess", "inference", "postprocess")
        },
    }


_CSV_COLUMNS = ("item", "image", "x1", "y1", "x2", "y2", "confidence", "class_id", "name")


def _csv_safe(value: str) -> str:
    """Neutralise spreadsheet formula injection (``=``, ``+``, ``-``, ``@`` prefixes)."""
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def items_to_csv(items: Sequence[Mapping[str, Any]]) -> str:
    """All detections of ``items`` as CSV (one row per detection)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for item in items:
        for det in item["detections"]:
            box = det["box"]
            writer.writerow(
                [
                    item["id"],
                    _csv_safe(item["name"]),
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"],
                    det["confidence"],
                    det["class_id"],
                    _csv_safe(det["name"]),
                ]
            )
    return buffer.getvalue()


# --------------------------------------------------------------------------- run store


@dataclass(frozen=True)
class StoredUpload:
    """An uploaded image on disk."""

    index: int
    name: str
    path: Path
    info: ImageInfo


@dataclass
class RunRecord:
    """Mutable state of one run. Only touched under :class:`RunStore`'s lock."""

    id: str
    params: RunParams
    model: ResolvedModel
    uploads: list[StoredUpload]
    status: RunStatus = "pending"
    error_msg: str | None = None
    items: list[dict[str, Any]] = field(default_factory=list)
    deleted: bool = False


@dataclass(frozen=True)
class RunSnapshot:
    """Immutable copy of a run, safe to hand out of the lock."""

    id: str
    params: RunParams
    status: RunStatus
    error_msg: str | None
    item_count: int
    items: tuple[dict[str, Any], ...]
    uploads: tuple[StoredUpload, ...]

    def summary(self) -> dict[str, Any]:
        """Schema ``RunSummary``."""
        return {
            "id": self.id,
            "model": self.params.model,
            "task": self.params.task,
            "backend": self.params.backend,
            "device": self.params.device,
            "status": self.status,
            "error_msg": self.error_msg,
            "item_count": self.item_count,
        }

    def payload(self) -> dict[str, Any]:
        """Schema ``Run`` (summary plus the items processed so far)."""
        return {**self.summary(), "items": list(self.items)}


class RunStore:
    """Thread-safe table of runs plus their uploaded files in a private temp directory."""

    def __init__(self, *, root: Path | None = None, max_runs: int = MAX_RUNS) -> None:
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, RunRecord] = OrderedDict()
        self._max_runs = max_runs
        self._root = root
        self._owns_root = root is None
        self._finalizer: weakref.finalize | None = None

    def _directory(self) -> Path:
        if self._root is None:
            self._root = Path(tempfile.mkdtemp(prefix="nitid-studio-runs-"))
            self._finalizer = weakref.finalize(self, shutil.rmtree, str(self._root), True)
        return self._root

    def create(
        self,
        params: RunParams,
        model: ResolvedModel,
        images: Sequence[tuple[str, bytes, ImageInfo]],
    ) -> str:
        """Persist the uploads and register a ``pending`` run. Returns its id."""
        with self._lock:
            self._evict_locked()
            run_id = f"r_{secrets.token_hex(6)}"
            run_dir = self._directory() / run_id
            run_dir.mkdir(parents=True)
            uploads: list[StoredUpload] = []
            try:
                for index, (name, data, info) in enumerate(images):
                    path = run_dir / f"{index}.{info.extension}"
                    path.write_bytes(data)
                    uploads.append(StoredUpload(index, name, path, info))
            except OSError:
                shutil.rmtree(run_dir, ignore_errors=True)
                raise
            self._runs[run_id] = RunRecord(run_id, params, model, uploads)
        return run_id

    def _evict_locked(self) -> None:
        while len(self._runs) >= self._max_runs:
            victim = next(
                (rid for rid, r in self._runs.items() if r.status in ("done", "failed")), None
            )
            if victim is None:
                return  # everything is active; allow temporary overshoot
            self._drop_locked(victim)

    def _drop_locked(self, run_id: str) -> None:
        record = self._runs.pop(run_id)
        record.deleted = True
        if self._root is not None:
            shutil.rmtree(self._root / run_id, ignore_errors=True)

    def get(self, run_id: str) -> RunSnapshot | None:
        with self._lock:
            record = self._runs.get(run_id)
            return self._snapshot(record) if record is not None else None

    def list(self) -> list[RunSnapshot]:
        """Newest first."""
        with self._lock:
            return [self._snapshot(r) for r in reversed(self._runs.values())]

    def delete(self, run_id: str) -> bool:
        with self._lock:
            if run_id not in self._runs:
                return False
            self._drop_locked(run_id)
            return True

    def close(self) -> None:
        """Forget every run and remove the temporary directory."""
        with self._lock:
            for record in self._runs.values():
                record.deleted = True
            self._runs.clear()
            if self._finalizer is not None:
                self._finalizer()
            self._root = None if self._owns_root else self._root

    # -- worker side ---------------------------------------------------------------------

    def record(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def mark_running(self, record: RunRecord) -> bool:
        """Move to ``running``; ``False`` if the run was deleted while queued."""
        with self._lock:
            if record.deleted:
                return False
            record.status = "running"
            return True

    def add_item(self, record: RunRecord, item: dict[str, Any]) -> None:
        with self._lock:
            if not record.deleted:
                record.items.append(item)

    def finish(self, record: RunRecord, error_msg: str | None) -> None:
        with self._lock:
            record.error_msg = error_msg
            record.status = "failed" if error_msg is not None else "done"

    @staticmethod
    def _snapshot(record: RunRecord) -> RunSnapshot:
        return RunSnapshot(
            id=record.id,
            params=record.params,
            status=record.status,
            error_msg=record.error_msg,
            item_count=len(record.uploads),
            items=tuple(record.items),
            uploads=tuple(record.uploads),
        )


def describe_run_error(exc: BaseException) -> str:
    """A client-safe ``error_msg`` for a failed run (never paths, never ``str(exc)``)."""
    if isinstance(exc, RunError):
        return exc.message
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return "Out of memory while running the model."
    if isinstance(exc, OSError):
        return "Could not read the stored image."
    return f"Inference failed ({type(exc).__name__})."


# --------------------------------------------------------------------------- service


class RunService:
    """Creates runs and executes them one at a time on a background worker."""

    def __init__(
        self,
        *,
        store: RunStore | None = None,
        loader: ModelLoader | None = None,
        jobs: JobStore | None = None,
    ) -> None:
        self.store = store if store is not None else RunStore()
        self._loader: ModelLoader = loader if loader is not None else load_nitid_model
        self._jobs = jobs if jobs is not None else JobStore(max_workers=1)
        self._models: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    # -- request side --------------------------------------------------------------------

    def submit(
        self,
        params: RunParams,
        model: ResolvedModel,
        images: Sequence[tuple[str, bytes, ImageInfo]],
    ) -> str:
        """Store the uploads, queue the run and return its id."""
        run_id = self.store.create(params, model, images)
        record = self.store.record(run_id)
        assert record is not None
        self._jobs.submit(
            run_id,
            lambda progress: self._execute(record, progress),
            describe_error=describe_run_error,
        )
        return run_id

    def close(self) -> None:
        self._jobs.shutdown(wait=False)
        self.store.close()

    # -- worker side ---------------------------------------------------------------------

    def _model_for(self, record: RunRecord) -> Any:
        stat = record.model.path.stat()
        key = (
            str(record.model.path),
            stat.st_mtime_ns,
            record.params.task,
            record.params.backend,
            record.params.device,
        )
        cached = self._models.get(key)
        if cached is not None:
            self._models.move_to_end(key)
            return cached
        try:
            model = self._loader(record.model, record.params)
        except Exception:
            logger.exception("Could not load model %s for run %s", record.model.id, record.id)
            raise RunError(f"Could not load model {record.model.id!r}.") from None
        self._models[key] = model
        while len(self._models) > MODEL_CACHE_SIZE:
            self._models.popitem(last=False)
        return model

    def _execute(self, record: RunRecord, progress: ProgressCallback) -> None:
        store = self.store
        if not store.mark_running(record):
            return
        try:
            model = self._model_for(record)
            params = record.params
            total = len(record.uploads)
            for position, upload in enumerate(record.uploads):
                if record.deleted:
                    return
                image = decode_image(upload.path.read_bytes())
                results = model.predict(
                    image,
                    conf=params.conf,
                    iou=params.iou,
                    imgsz=params.imgsz,
                    classes=list(params.classes) if params.classes is not None else None,
                    mask_threshold=params.mask_threshold,
                    verbose=False,
                )
                store.add_item(record, result_to_item(upload.index, upload.name, results[0]))
                progress((position + 1) / total)
        except Exception as exc:
            logger.exception("Run %s failed", record.id)
            store.finish(record, describe_run_error(exc))
            return
        store.finish(record, None)


# --------------------------------------------------------------------------- exports


def build_zip(snapshot: RunSnapshot) -> bytes:
    """``run.json``, ``detections.csv``, ``items/<n>.json`` and ``images/<n>.<ext>``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("run.json", json.dumps(snapshot.payload(), indent=2))
        archive.writestr("detections.csv", items_to_csv(snapshot.items))
        for item in snapshot.items:
            archive.writestr(f"items/{item['id']}.json", json.dumps(item, indent=2))
        for upload in snapshot.uploads:
            archive.write(upload.path, f"images/{upload.index}.{upload.info.extension}")
    return buffer.getvalue()
