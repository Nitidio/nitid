"""Unit tests for tracker adapters and tracked result semantics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dfine.media import FrameMetadata
from dfine.results import Boxes, Results
from dfine.tracking import ByteTrack, ResultTracker, create_tracker


def make_result(source_id="video.mp4", frame_index=0, *, empty=False):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    data = torch.empty((0, 6)) if empty else torch.tensor([[10.0, 10.0, 30.0, 30.0, 0.9, 0.0]])
    return Results(
        image,
        source_id,
        {0: "person"},
        Boxes(data, image.shape[:2]),
        frame_metadata=FrameMetadata(source_id, frame_index, fps=20.0, frame_stride=2),
    )


class FakeDetections:
    def __init__(self, *, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = None

    def __len__(self):
        return len(self.xyxy)


class FakeBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = 0

    def update(self, detections, frame=None):
        self.calls += 1
        detections.tracker_id = np.full(len(detections), 17, dtype=np.int64)
        return detections


def test_bytetrack_converts_results_and_uses_effective_frame_rate():
    backends = []

    def backend_factory(**kwargs):
        backend = FakeBackend(**kwargs)
        backends.append(backend)
        return backend

    tracker = ByteTrack(
        backend_factory=backend_factory,
        detections_factory=FakeDetections,
    )
    result = tracker.update(make_result())

    assert result.boxes is not None
    assert result.boxes.is_track
    assert result.boxes.data.shape == (1, 7)
    assert result.boxes.id.tolist() == [17]
    assert result.boxes.conf.tolist() == pytest.approx([0.9])
    assert result.boxes.cls.tolist() == [0]
    assert backends[0].kwargs["frame_rate"] == pytest.approx(10.0)


def test_bytetrack_advances_with_empty_detections():
    backend = FakeBackend()
    tracker = ByteTrack(
        backend_factory=lambda **kwargs: backend,
        detections_factory=FakeDetections,
    )
    result = tracker.update(make_result(empty=True))

    assert backend.calls == 1
    assert result.boxes is not None
    assert result.boxes.data.shape == (0, 7)
    assert result.boxes.id is not None


def test_bytetrack_resets_backend_when_source_changes_or_discontinues():
    backends = []

    def backend_factory(**kwargs):
        backend = FakeBackend(**kwargs)
        backends.append(backend)
        return backend

    tracker = ByteTrack(
        backend_factory=backend_factory,
        detections_factory=FakeDetections,
    )
    tracker.update(make_result("one", 0))
    tracker.update(make_result("one", 1))
    tracker.update(make_result("two", 0))
    discontinuous = make_result("two", 1)
    discontinuous.frame_metadata = FrameMetadata(
        "two", 1, fps=20.0, frame_stride=2, discontinuity=True
    )
    tracker.update(discontinuous)

    assert len(backends) == 3
    assert backends[0].calls == 2


def test_create_tracker_validation():
    class CustomTracker(ResultTracker):
        def update(self, result):
            return result

        def reset(self):
            pass

    custom = CustomTracker()
    assert create_tracker(custom) is custom
    assert isinstance(create_tracker("byte-track"), ByteTrack)
    with pytest.raises(ValueError, match="unsupported tracker"):
        create_tracker("unknown")
    with pytest.raises(ValueError, match="tracker_kwargs"):
        create_tracker(custom, frame_rate=30)


def test_tracked_boxes_serialize_and_tabulate_ids():
    data = torch.tensor([[1.0, 2.0, 10.0, 12.0, 8.0, 0.75, 0.0]])
    result = Results(
        np.zeros((20, 20, 3), dtype=np.uint8),
        "video.mp4",
        {0: "person"},
        Boxes(data, (20, 20)),
    )

    assert result.boxes.id.tolist() == [8]
    assert result.to_json()[0]["track_id"] == 8
    assert result.to_df().to_dict(orient="records")[0]["track_id"] == 8
    assert result.crop()[0]["track_id"] == 8
    assert np.any(result.plot() != 0)


def test_tracked_boxes_save_txt_appends_id(tmp_path):
    data = torch.tensor([[1.0, 2.0, 10.0, 12.0, 8.0, 0.75, 0.0]])
    result = Results(
        np.zeros((20, 20, 3), dtype=np.uint8),
        "video.mp4",
        {0: "person"},
        Boxes(data, (20, 20)),
    )
    output = tmp_path / "tracked.txt"

    result.save_txt(output, save_conf=True)

    assert output.read_text(encoding="utf-8").strip().endswith("0.750000 8")


def test_boxes_reject_invalid_column_count():
    with pytest.raises(ValueError, match=r"\[N, 6\].*\[N, 7\]"):
        Boxes(torch.empty((1, 5)), (10, 10))


def test_real_bytetrack_assigns_persistent_id_when_extra_installed():
    pytest.importorskip("trackers")
    pytest.importorskip("supervision")
    tracker = ByteTrack(
        minimum_consecutive_frames=0,
        track_activation_threshold=0.25,
    )

    results = [tracker.update(make_result(frame_index=index)) for index in range(3)]
    ids = [int(result.boxes.id[0]) for result in results]

    assert ids[1] >= 0
    assert ids[1] == ids[2]
