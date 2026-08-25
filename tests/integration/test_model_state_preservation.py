"""Regression tests for keeping the trainable model untouched by inference paths."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _as_list(value: Any) -> list[Any]:
    assert isinstance(value, list)
    return value


def _module_signature(model: Any) -> tuple[tuple[str, str], ...]:
    return tuple((name, type(module).__name__) for name, module in model.named_modules())


def _state_signature(model: Any) -> tuple[str, ...]:
    return tuple(model.state_dict().keys())


def _structural_fingerprint(model: Any) -> dict[str, Any]:
    decoder = getattr(model, "decoder", None)
    return {
        "deployed_flag": bool(getattr(model, "_deployed", False)),
        "modules": _module_signature(model),
        "state_keys": _state_signature(model),
        "decoder_layers": len(getattr(decoder, "layers", ())),
        "decoder_lqe_layers": tuple(
            type(layer).__name__ for layer in getattr(decoder, "lqe_layers", ())
        ),
        "decoder_lqe_embed": tuple(
            type(layer).__name__ for layer in getattr(decoder, "lqe_embed", ())
        ),
    }


@pytest.fixture
def frame() -> np.ndarray:
    rng = np.random.default_rng(105)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    ("checkpoint_fixture", "task", "imgsz"),
    [
        ("tiny_checkpoint", "detect", 640),
        ("tiny_segment_checkpoint", "segment", 640),
        ("tiny_semantic_checkpoint", "semantic", 64),
        ("tiny_pose_checkpoint", "pose", 640),
    ],
)
def test_predict_does_not_structurally_modify_trainable_model(
    request: pytest.FixtureRequest,
    checkpoint_fixture: str,
    task: str,
    imgsz: int,
    frame: np.ndarray,
) -> None:
    from dfine import DFINE

    checkpoint = request.getfixturevalue(checkpoint_fixture)
    model = DFINE(checkpoint, task=task, device="cpu", verbose=False)
    before = _structural_fingerprint(model._model)

    model.predict(frame, conf=0.0, imgsz=imgsz, verbose=False)

    assert _structural_fingerprint(model._model) == before


def test_repeated_predict_reuses_one_deployed_copy_without_mutating_training_model(
    tiny_checkpoint: str,
    frame: np.ndarray,
) -> None:
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    before = _structural_fingerprint(model._model)

    first = _as_list(model.predict(frame, conf=0.0, verbose=False))
    after_first = _structural_fingerprint(model._model)
    second = _as_list(model.predict(frame, conf=0.0, verbose=False))

    assert after_first == before
    assert _structural_fingerprint(model._model) == before
    assert len(first[0].boxes) == len(second[0].boxes)


def _export_formats_for_task(task: str) -> Iterator[str]:
    if task in {"semantic", "pose"}:
        yield "onnx"
    else:
        yield "onnx"
        yield "torchscript"


@pytest.mark.parametrize(
    ("checkpoint_fixture", "task", "imgsz"),
    [
        ("tiny_checkpoint", "detect", 640),
        ("tiny_segment_checkpoint", "segment", 640),
        ("tiny_semantic_checkpoint", "semantic", 64),
        ("tiny_pose_checkpoint", "pose", 640),
    ],
)
def test_export_does_not_structurally_modify_trainable_model(
    request: pytest.FixtureRequest,
    checkpoint_fixture: str,
    task: str,
    imgsz: int,
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnx")
    from dfine import DFINE

    checkpoint = request.getfixturevalue(checkpoint_fixture)
    model = DFINE(checkpoint, task=task, device="cpu", verbose=False)
    before = _structural_fingerprint(model._model)

    for export_format in _export_formats_for_task(task):
        output = tmp_path / f"{task}.{export_format}"
        if export_format == "torchscript":
            output = output.with_suffix(".torchscript")
        model.export(
            format=export_format,
            imgsz=imgsz,
            simplify=False,
            output=output,
            verbose=False,
        )
        assert output.exists()

    assert _structural_fingerprint(model._model) == before


def test_predict_then_export_then_trainable_model_remains_complete(
    tiny_checkpoint: str,
    tmp_path: Path,
    frame: np.ndarray,
) -> None:
    pytest.importorskip("onnx")
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    before = _structural_fingerprint(model._model)

    model.predict(frame, conf=0.0, verbose=False)
    model.export(
        format="onnx",
        imgsz=640,
        simplify=False,
        output=tmp_path / "detect.onnx",
        verbose=False,
    )

    assert _structural_fingerprint(model._model) == before
