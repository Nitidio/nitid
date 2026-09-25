"""Regression tests for keeping the trainable model untouched by inference paths."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch


def _as_list(value: Any) -> list[Any]:
    assert isinstance(value, list)
    return value


def _write_test_video(
    path: Path, frame_values: list[int], shape: tuple[int, int] = (64, 64)
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, shape)
    for value in frame_values:
        frame = np.full((shape[1], shape[0], 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()


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
    deployed_after_first = model._deployed_model
    after_first = _structural_fingerprint(model._model)
    second = _as_list(model.predict(frame, conf=0.0, verbose=False))

    assert deployed_after_first is not None
    assert model._deployed_model is deployed_after_first
    assert bool(getattr(deployed_after_first, "_deployed", False))
    assert after_first == before
    assert _structural_fingerprint(model._model) == before
    assert len(first[0].boxes) == len(second[0].boxes)


def test_predict_matches_independent_deployed_copy(
    tiny_checkpoint: str,
    frame: np.ndarray,
) -> None:
    from dfine import DFINE
    from dfine.predictor import DFINEPredictor

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    deployed = copy.deepcopy(model._model)
    deploy = getattr(deployed, "deploy")
    assert callable(deploy)
    deploy()
    setattr(deployed, "_deployed", True)

    expected = _as_list(
        DFINEPredictor(deployed, model._cfg, model.device, model.names).run(
            frame,
            conf=0.0,
            mask_threshold=0.5,
            imgsz=640,
            classes=None,
            stream=False,
            vid_stride=1,
            augment=False,
            save=False,
            project="runs/detect",
            name="exp",
            save_dir=None,
            exist_ok=True,
            verbose=False,
        )
    )
    actual = _as_list(model.predict(frame, conf=0.0, verbose=False))

    assert actual[0].boxes is not None
    assert expected[0].boxes is not None
    assert torch.allclose(actual[0].boxes.data, expected[0].boxes.data, atol=1e-6)


def test_predictor_rejects_undeployed_trainable_model(tiny_checkpoint: str) -> None:
    from dfine import DFINE
    from dfine.predictor import DFINEPredictor

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)

    with pytest.raises(RuntimeError, match="expects a deployed inference model"):
        DFINEPredictor(model._model, model._cfg, model.device, model.names)


def _export_formats_for_task(task: str) -> Iterator[str]:
    if task == "semantic":
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


def test_repeated_export_reuses_one_deployed_copy_without_mutating_training_model(
    tiny_checkpoint: str,
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnx")
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    before = _structural_fingerprint(model._model)

    model.export(
        format="onnx",
        imgsz=640,
        simplify=False,
        output=tmp_path / "first.onnx",
        verbose=False,
    )
    deployed_after_first = model._deployed_model
    model.export(
        format="onnx",
        imgsz=640,
        simplify=False,
        output=tmp_path / "second.onnx",
        verbose=False,
    )

    assert deployed_after_first is not None
    assert model._deployed_model is deployed_after_first
    assert bool(getattr(deployed_after_first, "_deployed", False))
    assert _structural_fingerprint(model._model) == before


def test_track_uses_deployed_copy_without_mutating_training_model(
    tiny_checkpoint: str,
    tmp_path: Path,
) -> None:
    from dfine import DFINE
    from dfine.tracking import ResultTracker

    class PassthroughTracker(ResultTracker):
        def __init__(self) -> None:
            self.calls = 0
            self.reset_calls = 0

        def update(self, result: Any) -> Any:
            self.calls += 1
            return result

        def reset(self) -> None:
            self.reset_calls += 1

    video_path = tmp_path / "tracking.mp4"
    _write_test_video(video_path, frame_values=[25, 75])
    tracker = PassthroughTracker()
    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    before = _structural_fingerprint(model._model)

    results = _as_list(
        model.track(
            str(video_path),
            conf=0.0,
            tracker=tracker,
            verbose=False,
            exist_ok=True,
        )
    )

    assert len(results) == 2
    assert tracker.calls == 2
    assert tracker.reset_calls == 1
    assert model._deployed_model is not None
    assert bool(getattr(model._deployed_model, "_deployed", False))
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
