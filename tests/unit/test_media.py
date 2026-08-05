"""Unit tests for frame/source/sink media contracts."""

from __future__ import annotations

import numpy as np
import pytest

from dfine.media import Frame, FrameMetadata, FrameSource, OpenCVVideoSink
from dfine.utils.sources import LoadSource, OpenCVFrameSource


def test_frame_metadata_validates_timeline_fields():
    with pytest.raises(ValueError, match="frame_index"):
        FrameMetadata(source_id="camera", frame_index=-1)
    with pytest.raises(ValueError, match="fps"):
        FrameMetadata(source_id="camera", frame_index=0, fps=0)
    with pytest.raises(ValueError, match="frame_stride"):
        FrameMetadata(source_id="camera", frame_index=0, frame_stride=0)


def test_frame_validates_bgr_shape():
    metadata = FrameMetadata(source_id="camera", frame_index=0)
    with pytest.raises(ValueError, match=r"\[H, W, 3\]"):
        Frame(np.zeros((10, 10), dtype=np.uint8), metadata)


def test_custom_frame_source_flows_through_loader_with_metadata():
    class CustomSource(FrameSource):
        mode = "custom"
        fps = 12.5
        vid_stride = 2

        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield Frame(
                np.zeros((8, 8, 3), dtype=np.uint8),
                FrameMetadata("custom://one", 4, timestamp=0.32, fps=self.fps, frame_stride=2),
            )

        def close(self):
            self.closed = True

    source = CustomSource()
    loader = LoadSource(source, imgsz=16, device="cpu")
    samples = list(loader.iter_samples())

    assert source.closed
    assert loader.mode == "custom"
    assert loader.video_fps == 12.5
    assert samples[0].tensor.shape == (1, 3, 16, 16)
    assert samples[0].frame.metadata.frame_index == 4
    assert samples[0].frame.metadata.timestamp == pytest.approx(0.32)


def test_opencv_source_stride_metadata_and_release(monkeypatch):
    frames = [np.full((4, 6, 3), value, dtype=np.uint8) for value in (0, 10, 20, 30, 40)]

    class FakeCapture:
        def __init__(self, source):
            self.frames = iter(frames)
            self.open = True
            self.released = False
            self.position = 0

        def isOpened(self):
            return self.open

        def read(self):
            try:
                frame = next(self.frames)
            except StopIteration:
                return False, None
            self.position += 1
            return True, frame

        def get(self, prop):
            import cv2

            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            if prop == cv2.CAP_PROP_POS_MSEC:
                return self.position * 100.0
            return 0.0

        def release(self):
            self.released = True
            self.open = False

    captures = []

    def make_capture(source):
        capture = FakeCapture(source)
        captures.append(capture)
        return capture

    monkeypatch.setattr("dfine.utils.sources.cv2.VideoCapture", make_capture)
    source = OpenCVFrameSource("clip.mp4", mode="video", vid_stride=2)
    decoded = list(source)

    assert [frame.metadata.frame_index for frame in decoded] == [0, 2, 4]
    assert all(frame.metadata.frame_stride == 2 for frame in decoded)
    assert all(frame.metadata.fps == 10.0 for frame in decoded)
    assert decoded[1].metadata.timestamp == pytest.approx(0.3)
    assert all(capture.released for capture in captures)


def test_opencv_source_releases_when_iteration_stops_early(monkeypatch):
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    class FakeCapture:
        def __init__(self, source):
            self.released = False

        def isOpened(self):
            return not self.released

        def read(self):
            return True, frame

        def get(self, prop):
            return 30.0

        def release(self):
            self.released = True

    capture = FakeCapture(0)
    monkeypatch.setattr("dfine.utils.sources.cv2.VideoCapture", lambda source: capture)
    source = OpenCVFrameSource(0, mode="webcam")
    iterator = iter(source)
    next(iterator)
    iterator.close()

    assert capture.released


def test_video_sink_validates_size_and_is_idempotently_closeable(monkeypatch, tmp_path):
    class FakeWriter:
        def __init__(self, *args):
            self.frames = []
            self.releases = 0

        def isOpened(self):
            return True

        def write(self, image):
            self.frames.append(image.copy())

        def release(self):
            self.releases += 1

    writer = FakeWriter()
    monkeypatch.setattr("dfine.media.cv2.VideoWriter", lambda *args: writer)
    sink = OpenCVVideoSink(tmp_path / "out.mp4", frame_size=(6, 4), fps=5.0)
    metadata = FrameMetadata("source", 0, fps=5.0)
    sink.write(Frame(np.zeros((4, 6, 3), dtype=np.uint8), metadata))

    with pytest.raises(ValueError, match="does not match"):
        sink.write(Frame(np.zeros((5, 6, 3), dtype=np.uint8), metadata))

    sink.close()
    sink.close()
    assert len(writer.frames) == 1
    assert writer.releases == 1
    with pytest.raises(RuntimeError, match="closed"):
        sink.write(Frame(np.zeros((4, 6, 3), dtype=np.uint8), metadata))
