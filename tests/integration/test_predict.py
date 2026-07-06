"""Integration tests for the full predict() pipeline."""

import itertools
import types

import numpy as np


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
