"""Self-describing metadata embedded in exported ONNX files.

An exported ONNX graph carries tensors but not the facts a client needs to use
them: which class each integer label means, how to prepare the input image, or
what the box coordinates are relative to. This module builds that description,
serialises it deterministically, writes it to the ONNX ``metadata_props`` and
reads it back, so a runtime that never imports nitid (for example ONNX Runtime
Web in a browser) can label detections and replicate the preprocessing.

The meta lives under one ``metadata_props`` entry, :data:`META_KEY`, whose value
is compact UTF-8 JSON with sorted keys. Writing it does not touch the graph,
the input/output names or any computed value.

Canonical shape (``schema_version`` 1)::

    {
      "schema_version": 1,
      "task": "detect" | "segment",
      "model": "nitid1s",              # optional: omitted when unknown
      "imgsz": 640,
      "input": {"name": "images", "shape": [1, 3, 640, 640], "dtype": "float32",
                "range": "0-1", "color": "RGB", "resize": "squash"},
      "outputs": [{"name": "labels", "dtype": "int64"}, ...],
      "names": {"0": "person", ...},
      "topk": 300,
      "default_iou": 0.85,
      "nitid_version": "0.9.6"
    }

The file's own ``sha256`` is deliberately absent: a file cannot contain its own
hash. Whoever serves the file can add it next to this meta.

Facts recorded here and where they come from in the code base:

* ``input.range`` / ``color`` / ``resize``: the predictor preprocesses with
  ``PIL`` RGB -> ``T.Resize((imgsz, imgsz))`` -> ``T.ToTensor()``
  (``dfine/utils/sources.py:331-341``). ``Resize`` with an ``(H, W)`` pair does
  not preserve aspect ratio (hence ``"squash"``, no letterbox) and ``ToTensor``
  scales uint8 pixels to float32 in ``[0, 1]`` with no mean/std normalisation.
* ``outputs`` (post-processed): ``labels`` int64, ``boxes`` float32 and
  ``scores`` float32 in that order, plus ``masks`` for ``segment``
  (``dfine/exporter.py`` ``_POSTPROCESSED_OUTPUT_NAMES``). ``boxes`` are
  ``xyxy`` pixels of the *model input* because the deploy wrapper passes the
  input tensor's ``(W, H)`` as the target size
  (``dfine/exporter.py`` ``DeployModel.forward`` and
  ``dfine/nn/postprocessor.py:47-48``).
* ``outputs`` (``postprocess=False``): the raw decoder dict
  (``dfine/exporter.py`` ``_RAW_OUTPUT_NAMES``); ``pred_boxes`` are normalised
  ``cxcywh`` (``dfine/nn/postprocessor.py:47``) and ``pred_masks`` are
  sigmoid probabilities in eval mode (``dfine/nn/architecture/decoder.py:1198-1199``).
* ``default_iou``: the predictor's default ``iou=0.85``
  (``dfine/predictor.py:67``, ``dfine/model.py:136``). nitid only uses it to
  merge flip-augmentation views, never inside the exported graph.
* ``topk``: number of rows the post-processor keeps
  (``dfine/nn/postprocessor.py:53,68-69``); the exporter passes the effective
  value, which can be below the configured 300 for small models.
* ``masks.stride``: mask resolution is ``imgsz / stride``; verified against the
  real ONNX output shape by ``tests/integration/test_export_onnx_meta.py``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import light
    import onnx

#: Key of the ``metadata_props`` entry that holds the JSON document.
META_KEY = "nitid_meta"
#: Version of the document shape; bump on any incompatible change.
META_SCHEMA_VERSION = 1
#: Default NMS IoU used by the predictor (``dfine/predictor.py:67``).
DEFAULT_IOU = 0.85
#: Mask output stride relative to the input (``imgsz / 4``).
MASK_STRIDE = 4

#: Tasks an ONNX meta can describe. ``semantic`` exports are not described.
META_TASKS = ("detect", "segment")

_POSTPROCESSED_OUTPUTS: dict[str, tuple[dict[str, Any], ...]] = {
    "detect": (
        {"name": "labels", "dtype": "int64"},
        {"name": "boxes", "dtype": "float32", "format": "xyxy_input_px"},
        {"name": "scores", "dtype": "float32"},
    ),
    "segment": (
        {"name": "labels", "dtype": "int64"},
        {"name": "boxes", "dtype": "float32", "format": "xyxy_input_px"},
        {"name": "scores", "dtype": "float32"},
        {"name": "masks", "dtype": "float32", "stride": MASK_STRIDE},
    ),
}

_RAW_OUTPUTS: dict[str, tuple[dict[str, Any], ...]] = {
    "detect": (
        {"name": "pred_logits", "dtype": "float32"},
        {"name": "pred_boxes", "dtype": "float32", "format": "cxcywh_norm"},
    ),
    "segment": (
        {"name": "pred_logits", "dtype": "float32"},
        {"name": "pred_boxes", "dtype": "float32", "format": "cxcywh_norm"},
        {"name": "pred_masks", "dtype": "float32", "stride": MASK_STRIDE},
    ),
}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalise_names(names: Mapping[Any, Any]) -> dict[str, str]:
    """Return ``{"0": "person", ...}`` with int-like keys turned into strings."""
    if not isinstance(names, Mapping) or not names:
        raise ValueError("names must be a non-empty mapping of class id to class name")
    result: dict[str, str] = {}
    for key, value in names.items():
        if isinstance(key, bool):
            raise ValueError(f"class id must be a non-negative integer, got {key!r}")
        try:
            class_id = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"class id must be a non-negative integer, got {key!r}") from None
        if class_id < 0 or (isinstance(key, str) and key.strip() != str(class_id)):
            raise ValueError(f"class id must be a non-negative integer, got {key!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"class name for id {class_id} must be a non-empty string")
        if str(class_id) in result:
            raise ValueError(f"duplicate class id {class_id} in names")
        result[str(class_id)] = value
    return result


def build_onnx_meta(
    *,
    task: str,
    imgsz: int,
    names: Mapping[Any, Any],
    topk: int,
    nitid_version: str,
    postprocess: bool = True,
    model: str | None = None,
    batch: int = 1,
    default_iou: float = DEFAULT_IOU,
) -> dict[str, Any]:
    """Build the JSON-ready meta document for one ONNX export.

    Pure function: no I/O and no heavy imports. Raises ``ValueError`` on
    invalid input.

    Args:
        task: ``"detect"`` or ``"segment"``. Semantic exports are not described.
        imgsz: Square model input size in pixels.
        names: Class id -> class name. Keys may be ``int`` or digit strings;
            they are emitted as strings (JSON object keys).
        topk: Number of detections the post-processed graph returns per image
            (or the value a client should keep when ``postprocess`` is false).
        nitid_version: Version of the nitid library that wrote the file.
        postprocess: ``True`` for the decoded ``labels/boxes/scores[/masks]``
            contract, ``False`` for the raw ``pred_logits/pred_boxes[/pred_masks]``.
        model: Public model name such as ``"nitid1s"``. The ``model`` key is
            omitted when this is ``None`` (an empty string is not allowed).
        batch: First input dimension used for the export.
        default_iou: Default IoU threshold of the nitid predictor.
    """
    if task not in META_TASKS:
        raise ValueError(f"task must be one of {META_TASKS}, got {task!r}")
    if not _is_int(imgsz) or imgsz < 1:
        raise ValueError(f"imgsz must be a positive integer, got {imgsz!r}")
    if not _is_int(batch) or batch < 1:
        raise ValueError(f"batch must be a positive integer, got {batch!r}")
    if not _is_int(topk) or topk < 1:
        raise ValueError(f"topk must be a positive integer, got {topk!r}")
    if isinstance(default_iou, bool) or not isinstance(default_iou, (int, float)):
        raise ValueError(f"default_iou must be a number in [0, 1], got {default_iou!r}")
    if not 0.0 <= float(default_iou) <= 1.0:
        raise ValueError(f"default_iou must be in [0, 1], got {default_iou!r}")
    if not isinstance(nitid_version, str) or not nitid_version:
        raise ValueError("nitid_version must be a non-empty string")
    if model is not None and (not isinstance(model, str) or not model):
        raise ValueError("model must be a non-empty string or None")
    if postprocess and imgsz % MASK_STRIDE and task == "segment":
        raise ValueError(f"imgsz must be divisible by {MASK_STRIDE} for segment exports")

    outputs_table = _POSTPROCESSED_OUTPUTS if postprocess else _RAW_OUTPUTS
    meta: dict[str, Any] = {
        "schema_version": META_SCHEMA_VERSION,
        "task": task,
        "imgsz": imgsz,
        "input": {
            "name": "images",
            "shape": [batch, 3, imgsz, imgsz],
            "dtype": "float32",
            "range": "0-1",
            "color": "RGB",
            "resize": "squash",
        },
        "outputs": [dict(entry) for entry in outputs_table[task]],
        "names": _normalise_names(names),
        "topk": topk,
        "default_iou": float(default_iou),
        "nitid_version": nitid_version,
    }
    if model is not None:
        meta["model"] = model
    return meta


def encode_onnx_meta(meta: Mapping[str, Any]) -> str:
    """Serialise ``meta`` as compact JSON with sorted keys (deterministic)."""
    return json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def embed_onnx_meta(model_proto: onnx.ModelProto, meta: Mapping[str, Any]) -> None:
    """Write ``meta`` into ``model_proto.metadata_props`` in place.

    Replaces an existing :data:`META_KEY` entry and leaves every other entry,
    and the graph, untouched.
    """
    value = encode_onnx_meta(meta)
    for entry in model_proto.metadata_props:
        if entry.key == META_KEY:
            entry.value = value
            return
    entry = model_proto.metadata_props.add()
    entry.key = META_KEY
    entry.value = value


def read_onnx_meta(
    path_or_model: str | os.PathLike[str] | onnx.ModelProto,
) -> dict[str, Any] | None:
    """Return the nitid meta of an ONNX file or ``ModelProto``.

    Returns ``None`` when the model carries no :data:`META_KEY` entry (for
    example an export made by an older nitid). Raises ``ValueError`` when the
    entry is present but is not a JSON object.
    """
    import onnx

    if isinstance(path_or_model, (str, os.PathLike)):
        # Weights are irrelevant here; skip external data so reading stays cheap.
        proto = onnx.load(os.fspath(path_or_model), load_external_data=False)
    else:
        proto = path_or_model
    for entry in proto.metadata_props:
        if entry.key == META_KEY:
            try:
                decoded = json.loads(entry.value)
            except json.JSONDecodeError as error:
                raise ValueError(f"{META_KEY} metadata is not valid JSON: {error}") from error
            if not isinstance(decoded, dict):
                raise ValueError(f"{META_KEY} metadata must be a JSON object")
            return decoded
    return None
