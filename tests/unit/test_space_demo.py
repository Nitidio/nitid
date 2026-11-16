"""Unit tests for the Hugging Face Space demo (``spaces/demo/app.py``).

Gradio is not a dependency of nitid, so these tests only exercise the plain
functions; ``build_demo`` is covered by a test that is skipped without Gradio.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dfine.results import Boxes, Results

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("space_demo_app", REPO_ROOT / "spaces/demo/app.py")
assert _spec is not None and _spec.loader is not None
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)

NAMES = {0: "person", 1: "ski"}


class FakeModel:
    """Returns one fixed detection (or none) and records what it was given."""

    def __init__(self, detections: bool = True) -> None:
        self.detections = detections
        self.calls: list[tuple[np.ndarray, float]] = []

    def predict(self, source: np.ndarray, conf: float, verbose: bool = True) -> list[Results]:
        self.calls.append((source, conf))
        height, width = source.shape[:2]
        boxes = None
        if self.detections:
            data = torch.tensor([[10.0, 12.0, 60.0, 70.0, 0.91234, 0.0]])
            boxes = Boxes(data, (height, width))
        result = Results(source, "array", NAMES, boxes=boxes, speed={"inference": 123.4})
        return [result]


def _image(height: int = 100, width: int = 120) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = 200  # red in RGB
    return image


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    app.load_model.cache_clear()


def test_sizes_for_task_hides_nano_for_detection() -> None:
    detect_sizes, _ = app.sizes_for_task("detect")
    segment_sizes, _ = app.sizes_for_task("segment")

    assert detect_sizes == ["s", "m", "l", "x"]
    assert segment_sizes == ["n", "s", "m", "l", "x"]


def test_sizes_for_task_keeps_a_valid_size_and_resets_an_invalid_one() -> None:
    assert app.sizes_for_task("segment", "l")[1] == "l"
    # Switching to detection while "n" is selected falls back to the default.
    assert app.sizes_for_task("detect", "n")[1] == app.DEFAULT_SIZE
    with pytest.raises(ValueError, match="Unknown task"):
        app.sizes_for_task("pose")


def test_run_inference_returns_annotated_rgb_table_and_summary() -> None:
    model = FakeModel()
    image = _image()

    annotated, table, summary = app.run_inference(
        image, "detect", "s", 0.4, loader=lambda t, s: model
    )

    assert annotated.shape == image.shape
    assert annotated.dtype == np.uint8
    assert list(table.columns) == ["x1", "y1", "x2", "y2", "confidence", "class", "name"]
    assert table.loc[0, "name"] == "person"
    assert table.loc[0, "confidence"] == 0.912
    assert summary == "1 object found · model time 123 ms on CPU"
    assert model.calls[0][1] == 0.4


def test_run_inference_passes_bgr_to_the_model_and_returns_rgb() -> None:
    model = FakeModel(detections=False)
    image = _image()  # channel 0 (red) is 200, the others 0

    annotated, _, _ = app.run_inference(image, "detect", "s", 0.5, loader=lambda t, s: model)

    seen = model.calls[0][0]
    assert seen[0, 0].tolist() == [0, 0, 200]  # BGR: red moved to the last channel
    assert annotated[0, 0].tolist() == [200, 0, 0]  # back to RGB for display
    assert seen.flags["C_CONTIGUOUS"]
    assert annotated.flags["C_CONTIGUOUS"]


def test_run_inference_without_detections() -> None:
    annotated, table, summary = app.run_inference(
        _image(), "detect", "s", 0.5, loader=lambda t, s: FakeModel(detections=False)
    )

    assert annotated.shape == (100, 120, 3)
    assert len(table) == 0
    assert summary.startswith("0 objects found")


def test_run_inference_downscales_large_images() -> None:
    model = FakeModel(detections=False)
    big = _image(height=2000, width=3000)

    app.run_inference(big, "detect", "s", 0.5, loader=lambda t, s: model)

    seen = model.calls[0][0]
    assert max(seen.shape[:2]) == app.MAX_SIDE
    assert seen.shape[:2] == (round(2000 * app.MAX_SIDE / 3000), app.MAX_SIDE)


def test_run_inference_keeps_small_images_unchanged() -> None:
    model = FakeModel(detections=False)

    app.run_inference(_image(100, 120), "detect", "s", 0.5, loader=lambda t, s: model)

    assert model.calls[0][0].shape[:2] == (100, 120)


def test_run_inference_rejects_missing_image_and_unavailable_model() -> None:
    with pytest.raises(ValueError, match="Upload an image"):
        app.run_inference(None, "detect", "s", 0.5, loader=lambda t, s: FakeModel())
    with pytest.raises(ValueError, match="size 'n'"):
        app.run_inference(_image(), "detect", "n", 0.5, loader=lambda t, s: FakeModel())


def test_load_model_prefers_the_nitid_name_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    import nitid

    created: list[tuple[str, str]] = []

    class FakeNITID:
        def __init__(self, name: str, *, task: str, device: str, verbose: bool) -> None:
            assert device == "cpu"
            created.append((name, task))

    monkeypatch.setattr(nitid, "NITID", FakeNITID)

    first = app.load_model("segment", "m")
    second = app.load_model("segment", "m")

    assert first is second
    assert created == [("nitid1m", "segment")]


def test_load_model_falls_back_to_legacy_names_on_old_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nitid

    created: list[str] = []

    class OldNITID:  # nitid <= 0.9.4 only understands model1<size>
        def __init__(self, name: str, **_: Any) -> None:
            if not name.startswith("model1"):
                raise ValueError(
                    f"Unsupported NITID model {name!r}. Expected names like 'model1s'."
                )
            created.append(name)

    monkeypatch.setattr(nitid, "NITID", OldNITID)

    app.load_model("detect", "s")

    assert created == ["model1s"]


def test_load_model_does_not_swallow_other_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import nitid

    class BrokenNITID:
        def __init__(self, name: str, **_: Any) -> None:
            raise ValueError("checkpoint download failed")

    monkeypatch.setattr(nitid, "NITID", BrokenNITID)

    with pytest.raises(ValueError, match="download failed"):
        app.load_model("detect", "s")


def test_example_images_exist() -> None:
    assert list(app.EXAMPLES_DIR.glob("*.jpg")), "the Space ships at least one example image"


def test_build_demo_wires_the_interface() -> None:
    gr = pytest.importorskip("gradio")

    demo = app.build_demo()

    assert isinstance(demo, gr.Blocks)
