"""Integration tests for the full predict() pipeline."""

import itertools
import types

import numpy as np
import torch
from PIL import Image


def _random_frame(seed=42, shape=(480, 640, 3)):
    """Non-symmetric input — required for tests that exercise flip-dependent
    logic (all-zero/uniform frames are flip-invariant and can't reveal
    coordinate or dedup bugs in the augment path)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, shape, dtype=np.uint8)


def test_predict_numpy_frame(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = model.predict(frame, conf=0.3)
    assert isinstance(results, list)
    assert len(results) == 1


def test_predict_returns_results_object(tiny_checkpoint):
    from dfine import DFINE
    from dfine.results import Results

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = model.predict(frame, conf=0.0)
    assert isinstance(results[0], Results)


def test_predict_returns_speed_timings(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = model.predict(frame, conf=0.0)[0]

    assert set(result.speed) == {"preprocess", "inference", "postprocess"}
    assert all(isinstance(value, float) for value in result.speed.values())
    assert all(value >= 0 for value in result.speed.values())


def test_predict_stream_is_generator(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gen = model.predict(frame, stream=True)
    assert isinstance(gen, types.GeneratorType)


def test_predict_conf_filter(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    all_dets = model.predict(frame, conf=0.0)[0]
    none_dets = model.predict(frame, conf=1.0)[0]
    assert len(all_dets) >= len(none_dets)
    assert len(none_dets) == 0


def test_predict_names_populated(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    assert len(model.names) == 80
    assert model.names[0] == "class_0"


def test_predict_boxes_within_image(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = model.predict(frame, conf=0.0)[0]
    if len(results) > 0:
        boxes = results.boxes.xyxy
        assert (boxes[:, 0] >= 0).all()
        assert (boxes[:, 1] >= 0).all()
        assert (boxes[:, 2] <= 640).all()
        assert (boxes[:, 3] <= 480).all()


def test_predict_save_writes_annotated_image(tiny_checkpoint, tmp_path):
    from dfine import DFINE

    image_path = tmp_path / "input.jpg"
    Image.fromarray(_random_frame()).save(image_path)

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    results = model.predict(
        str(image_path),
        conf=0.0,
        save=True,
        project=str(tmp_path / "runs"),
        name="detect-test",
    )

    save_path = tmp_path / "runs" / "detect-test" / "input.jpg"
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    assert results[0].save_path == str(save_path)


def test_predict_save_generates_name_for_numpy_frame(tiny_checkpoint, tmp_path):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    frame = _random_frame()
    results = model.predict(
        frame,
        conf=0.0,
        save=True,
        project=str(tmp_path / "runs"),
        name="array-test",
    )

    save_path = tmp_path / "runs" / "array-test" / "image_000001.jpg"
    assert save_path.exists()
    assert results[0].save_path == str(save_path)


# ---------------------------------------------------------------------------
# augment=True (TTA) tests
# ---------------------------------------------------------------------------


def test_predict_augment_runs_and_returns_valid_boxes(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    frame = _random_frame()

    results = model.predict(frame, conf=0.0, augment=True)
    assert isinstance(results, list)
    assert len(results) == 1

    boxes = results[0].boxes.xyxy
    if len(boxes) > 0:
        assert (boxes[:, 0] >= 0).all()
        assert (boxes[:, 1] >= 0).all()
        assert (boxes[:, 2] <= 640).all()
        assert (boxes[:, 3] <= 480).all()
        # x1 < x2, y1 < y2 sanity — catches flip-coordinate bugs
        assert (boxes[:, 2] > boxes[:, 0]).all()
        assert (boxes[:, 3] > boxes[:, 1]).all()


def test_predict_augment_close_to_baseline_count(tiny_checkpoint):
    """augment=True shouldn't wildly over- or under-detect vs. augment=False.
    Loose bound since a tiny/lightly-trained checkpoint is noisy on random
    input — this is a regression guard, not a precision check."""
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    frame = _random_frame()

    baseline = model.predict(frame, conf=0.3, augment=False)[0]
    augmented = model.predict(frame, conf=0.3, augment=True)[0]

    n_base, n_aug = len(baseline.boxes), len(augmented.boxes)
    assert abs(n_aug - n_base) <= max(2, n_base * 0.5), (
        f"augment=True detection count ({n_aug}) diverges too far from "
        f"baseline ({n_base}) — check cross-view NMS threshold/logic"
    )


def test_predict_augment_respects_iou_kwarg(tiny_checkpoint):
    """A very low iou threshold should suppress more than a very high one,
    proving the iou kwarg actually reaches the cross-view NMS step."""
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    frame = _random_frame()

    loose = model.predict(frame, conf=0.3, augment=True, iou=0.99)[0]
    tight = model.predict(frame, conf=0.3, augment=True, iou=0.01)[0]

    assert len(tight.boxes) <= len(loose.boxes)


# ---------------------------------------------------------------------------
# Unit-level tests on _postprocess's cross-view NMS (deterministic, no model)
# ---------------------------------------------------------------------------


def _fake_orig_img(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_postprocess_dedupes_same_object_across_views(tiny_checkpoint):
    """Two near-identical boxes from *different* views (original vs. flipped)
    representing the same object should collapse to one after NMS."""
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    predictor = model.predictor  # adjust attribute name if different

    det = {
        "labels": torch.tensor([0, 0]),
        "boxes": torch.tensor(
            [
                [100.0, 100.0, 200.0, 200.0],
                [101.0, 101.0, 201.0, 201.0],  # near-duplicate, other view
            ]
        ),
        "scores": torch.tensor([0.9, 0.85]),
        "num_orig": 1,  # first box = original view, second = flipped view
    }
    results = predictor._postprocess(
        det,
        _fake_orig_img(),
        path="fake",
        conf_thr=0.0,
        classes=None,
        augment=True,
        iou=0.5,
    )
    assert len(results.boxes) == 1


def test_postprocess_keeps_same_view_close_objects(tiny_checkpoint):
    """Two close boxes from the *same* view (e.g. two adjacent real objects)
    must NOT be suppressed by cross-view NMS — D-FINE's set prediction
    already guarantees no same-view duplicates."""
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    predictor = model.predictor

    det = {
        "labels": torch.tensor([0, 0]),
        "boxes": torch.tensor(
            [
                [100.0, 100.0, 200.0, 200.0],
                [101.0, 101.0, 201.0, 201.0],  # close, but SAME view
            ]
        ),
        "scores": torch.tensor([0.9, 0.85]),
        "num_orig": 2,  # both boxes are from the original view
    }
    results = predictor._postprocess(
        det,
        _fake_orig_img(),
        path="fake",
        conf_thr=0.0,
        classes=None,
        augment=True,
        iou=0.5,
    )
    assert len(results.boxes) == 2


def test_predict_screen_source_streams_frames(tiny_checkpoint, monkeypatch):
    from dfine import DFINE
    from dfine.utils.sources import LoadSource

    frames = iter(
        [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640, 3), dtype=np.uint8) * 255,
            None,
        ]
    )

    def fake_capture(self):
        return next(frames)

    monkeypatch.setattr(LoadSource, "_capture_screen_frame", fake_capture)

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    gen = model.predict("screen", conf=0.0, stream=True)
    results = list(itertools.islice(gen, 2))

    assert len(results) == 2
    assert results[0].path == "<screen>"
    assert results[1].path == "<screen>"
