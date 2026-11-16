"""The Space demo's inference path against real (tiny, randomly initialised) nitid models."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dfine import NITID

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("space_demo_app", REPO_ROOT / "spaces/demo/app.py")
assert _spec is not None and _spec.loader is not None
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_run_inference_with_a_real_model(task, tiny_checkpoint, tiny_segment_checkpoint) -> None:
    checkpoint = tiny_checkpoint if task == "detect" else tiny_segment_checkpoint

    def loader(task_name: str, size: str) -> NITID:
        assert (task_name, size) == (task, "s")
        return NITID(checkpoint, task=task_name, device="cpu", verbose=False)

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)

    # conf=0 keeps every query, so the drawing and table code runs on non-empty results.
    annotated, table, summary = app.run_inference(image, task, "s", 0.0, loader=loader)

    assert annotated.shape == image.shape
    assert annotated.dtype == np.uint8
    assert list(table.columns) == ["x1", "y1", "x2", "y2", "confidence", "class", "name"]
    assert len(table) > 0
    assert summary.startswith(f"{len(table)} object")


def test_run_inference_with_a_real_model_and_no_detections(tiny_checkpoint) -> None:
    model = NITID(tiny_checkpoint, task="detect", device="cpu", verbose=False)

    annotated, table, summary = app.run_inference(
        np.zeros((64, 64, 3), dtype=np.uint8), "detect", "s", 0.999, loader=lambda t, s: model
    )

    assert annotated.shape == (64, 64, 3)
    assert len(table) == 0
    assert summary.startswith("0 objects found")
