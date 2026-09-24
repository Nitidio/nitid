"""Unit tests for frame/source/sink media contracts."""

from __future__ import annotations

import numpy as np
import pytest

from dfine.media import (
    Frame,
    FrameMetadata,
    FrameSource,
    GStreamerVideoSink,
    OpenCVVideoSink,
    build_gstreamer_output_pipeline,
)
from dfine.utils.sources import (
    GStreamerFrameSource,
    LoadSource,
    OpenCVFrameSource,
    build_gstreamer_pipeline,
    gstreamer_available,
)


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


def test_gstreamer_pipeline_builder_handles_rtsp_files_and_explicit_pipelines(tmp_path):
    rtsp = build_gstreamer_pipeline(
        'rtsp://user:p"ass@camera/live', rtsp_latency=350, rtsp_transport="udp"
    )
    assert 'location="rtsp://user:p\\"ass@camera/live"' in rtsp
    assert "latency=350" in rtsp
    assert "protocols=udp" in rtsp
    assert "video/x-raw,format=BGR" in rtsp
    assert "drop=true max-buffers=1 sync=false" in rtsp

    video = tmp_path / "clip.mp4"
    video.touch()
    file_pipeline = build_gstreamer_pipeline(video)
    assert video.resolve().as_uri() in file_pipeline
    assert "appsink sync=false" in file_pipeline
    assert "drop=true" not in file_pipeline

    explicit = "videotestsrc num-buffers=1 ! appsink sync=false"
    assert build_gstreamer_pipeline("ignored", pipeline=explicit) == explicit

    finite_pipeline = build_gstreamer_pipeline(
        "ignored",
        pipeline="videotestsrc num-buffers=1",
        live=False,
    )
    assert "appsink sync=false" in finite_pipeline
    assert "drop=true" not in finite_pipeline


def test_gstreamer_pipeline_builder_validates_configuration(tmp_path):
    with pytest.raises(ValueError, match="rtsp_latency"):
        build_gstreamer_pipeline("rtsp://camera/live", rtsp_latency=-1)
    with pytest.raises(ValueError, match="rtsp_transport"):
        build_gstreamer_pipeline("rtsp://camera/live", rtsp_transport="http")
    with pytest.raises(ValueError, match="cannot be empty"):
        build_gstreamer_pipeline("ignored", pipeline="  ")
    with pytest.raises(FileNotFoundError, match="not found"):
        build_gstreamer_pipeline(tmp_path / "missing.mp4")


def test_gstreamer_source_ends_when_a_live_stream_drops():
    image = np.zeros((4, 6, 3), dtype=np.uint8)

    class FakeCapture:
        def __init__(self, reads):
            self.reads = iter(reads)
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            return next(self.reads, (False, None))

        def get(self, prop):
            return 25.0

        def release(self):
            self.released = True

    capture = FakeCapture([(True, image), (False, None), (True, image)])
    source = GStreamerFrameSource("rtsp://camera/live", _capture_factory=lambda pipeline: capture)

    frames = list(source)

    assert [frame.metadata.frame_index for frame in frames] == [0]
    assert capture.released


def test_gstreamer_source_reports_a_source_that_cannot_open():
    class ClosedCapture:
        def isOpened(self):
            return False

        def release(self):
            pass

    source = GStreamerFrameSource(
        "rtsp://camera/live", _capture_factory=lambda pipeline: ClosedCapture()
    )

    with pytest.raises(RuntimeError, match="Failed to open GStreamer source"):
        list(source)


@pytest.mark.skipif(not gstreamer_available(), reason="OpenCV was built without GStreamer")
def test_real_gstreamer_videotestsrc_pipeline():
    source = GStreamerFrameSource(
        "videotestsrc num-buffers=3 ! video/x-raw,width=32,height=24,framerate=5/1",
        mode="stream",
    )

    frames = list(source)

    assert len(frames) == 3
    assert all(frame.image.shape == (24, 32, 3) for frame in frames)
    assert [frame.metadata.frame_index for frame in frames] == [0, 1, 2]


def test_gstreamer_output_pipeline_builder_supports_rtsp_segments_and_files(tmp_path):
    rtsp = build_gstreamer_output_pipeline('rtsp://server/publish"here', rtsp_transport="udp")
    assert "appsrc format=time" in rtsp
    assert "queue leaky=downstream max-size-buffers=4" in rtsp
    assert "rtph264pay config-interval=1" in rtsp
    assert 'location="rtsp://server/publish\\"here"' in rtsp
    assert "protocols=udp" in rtsp

    segments = build_gstreamer_output_pipeline(tmp_path / "segments", segment_duration=2.5)
    assert "splitmuxsink" in segments
    assert "appsrc format=time ! queue !" in segments
    assert "leaky=" not in segments
    assert "segment_%05d.mp4" in segments
    assert "max-size-time=2500000000" in segments

    named_segments = build_gstreamer_output_pipeline(tmp_path / "camera.mp4", segment_duration=60)
    assert "camera_%05d.mp4" in named_segments
    assert "max-size-time=60000000000" in named_segments

    output_file = build_gstreamer_output_pipeline(tmp_path / "annotated.mp4")
    assert "mp4mux faststart=true" in output_file
    assert "appsrc format=time ! queue !" in output_file
    assert "leaky=" not in output_file
    assert str((tmp_path / "annotated.mp4").resolve()) in output_file


def test_gstreamer_output_pipeline_builder_validates_options():
    with pytest.raises(ValueError, match="destination or pipeline"):
        build_gstreamer_output_pipeline()
    with pytest.raises(ValueError, match="segment_duration"):
        build_gstreamer_output_pipeline("segments", segment_duration=0)
    with pytest.raises(ValueError, match="RTSP destination"):
        build_gstreamer_output_pipeline("rtsp://server/live", segment_duration=10)
    with pytest.raises(ValueError, match="custom output pipeline"):
        build_gstreamer_output_pipeline("out.mp4", pipeline="fakesink")
    with pytest.raises(ValueError, match="RTSP URL or local path"):
        build_gstreamer_output_pipeline("udp://127.0.0.1:5000")
    with pytest.raises(ValueError, match="must end in .mp4"):
        build_gstreamer_output_pipeline("runs/annotated")
    with pytest.raises(ValueError, match="rtsp_transport"):
        build_gstreamer_output_pipeline("rtsp://server/live", rtsp_transport="http")

    custom = build_gstreamer_output_pipeline(pipeline="videoconvert ! fakesink")
    assert custom.startswith("appsrc format=time ! videoconvert")
    with_appsrc = "appsrc ! videoconvert ! fakesink"
    assert build_gstreamer_output_pipeline(pipeline=with_appsrc) == with_appsrc


def test_gstreamer_video_sink_opens_lazily_uses_effective_fps_and_closes(tmp_path):
    class FakeWriter:
        def __init__(self):
            self.frames = []
            self.releases = 0

        def isOpened(self):
            return True

        def write(self, image):
            self.frames.append(image.copy())

        def release(self):
            self.releases += 1

    observed = {}
    writer = FakeWriter()

    def make_writer(pipeline, fps, frame_size):
        observed.update(pipeline=pipeline, fps=fps, frame_size=frame_size)
        return writer

    destination = tmp_path / "nested" / "segments"
    sink = GStreamerVideoSink(
        destination,
        segment_duration=10,
        _writer_factory=make_writer,
    )
    assert sink.pipeline is None
    metadata = FrameMetadata("camera", 0, fps=20.0, frame_stride=2)
    sink.write(Frame(np.zeros((4, 6, 3), dtype=np.uint8), metadata))

    assert destination.is_dir()
    assert observed["fps"] == 10.0
    assert observed["frame_size"] == (6, 4)
    assert "splitmuxsink" in observed["pipeline"]
    assert len(writer.frames) == 1

    with pytest.raises(ValueError, match="does not match"):
        sink.write(Frame(np.zeros((5, 6, 3), dtype=np.uint8), metadata))

    sink.close()
    sink.close()
    assert writer.releases == 1
    with pytest.raises(RuntimeError, match="closed"):
        sink.write(Frame(np.zeros((4, 6, 3), dtype=np.uint8), metadata))


def test_gstreamer_video_sink_releases_failed_writer(tmp_path):
    class ClosedWriter:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    writer = ClosedWriter()
    sink = GStreamerVideoSink(
        tmp_path / "out.mp4",
        _writer_factory=lambda pipeline, fps, frame_size: writer,
    )
    frame = Frame(np.zeros((4, 6, 3), dtype=np.uint8), FrameMetadata("source", 0))

    with pytest.raises(RuntimeError, match="Failed to open"):
        sink.write(frame)

    assert writer.released
    sink.close()


@pytest.mark.skipif(not gstreamer_available(), reason="OpenCV was built without GStreamer")
def test_real_gstreamer_video_sink(tmp_path):
    output = tmp_path / "output.mp4"
    sink = GStreamerVideoSink(output, fps=5)
    metadata = FrameMetadata("test", 0, fps=5)
    for index in range(3):
        image = np.full((24, 32, 3), index * 30, dtype=np.uint8)
        sink.write(Frame(image, metadata))
    sink.close()

    assert output.is_file()
    assert output.stat().st_size > 0


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
