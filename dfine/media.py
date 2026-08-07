"""Media contracts shared by source, inference, tracking, and output backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from dfine.gstreamer import resolve_hardware_fragment

DEFAULT_GSTREAMER_ENCODER = "x264enc tune=zerolatency speed-preset=veryfast key-int-max=30"


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Identity and timing information for one decoded frame.

    ``frame_index`` is zero-based and refers to the original source, including
    frames skipped by a stride. ``timestamp`` is expressed in seconds on the
    source timeline for files and from a monotonic clock for live sources.
    """

    source_id: str
    frame_index: int
    timestamp: float | None = None
    fps: float | None = None
    frame_stride: int = 1
    discontinuity: bool = False

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be > 0 when provided")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")


@dataclass(frozen=True, slots=True)
class Frame:
    """A decoded BGR image and its source metadata."""

    image: np.ndarray
    metadata: FrameMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray):
            raise TypeError("frame image must be a numpy array")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("frame image must have shape [H, W, 3]")


class FrameSource(ABC):
    """Ordered, explicitly closeable source of decoded frames."""

    mode: str
    fps: float | None = None
    vid_stride: int = 1

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        """Yield frames in source order."""

    def close(self) -> None:
        """Release source resources. Implementations may override."""

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class FrameSink(ABC):
    """Explicitly closeable destination for decoded or annotated frames."""

    @abstractmethod
    def write(self, frame: Frame) -> None:
        """Write one frame."""

    def close(self) -> None:
        """Flush and release sink resources. Implementations may override."""

    def __enter__(self) -> FrameSink:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class OpenCVVideoSink(FrameSink):
    """MP4 output sink backed by :class:`cv2.VideoWriter`."""

    def __init__(
        self,
        path: str | Path,
        *,
        frame_size: tuple[int, int],
        fps: float,
        fourcc: str = "mp4v",
    ) -> None:
        if fps <= 0:
            raise ValueError("sink fps must be > 0")
        if len(fourcc) != 4:
            raise ValueError("fourcc must contain exactly four characters")

        self.path = Path(path)
        self.frame_size = frame_size
        self.fps = float(fps)
        self._closed = False
        codec = cv2.VideoWriter_fourcc(*fourcc)  # type: ignore[attr-defined]
        self._writer = cv2.VideoWriter(str(self.path), codec, self.fps, frame_size)
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(f"Failed to open video writer for '{self.path}'")

    def write(self, frame: Frame) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed video sink")
        height, width = frame.image.shape[:2]
        if (width, height) != self.frame_size:
            raise ValueError(
                f"frame size {(width, height)} does not match sink size {self.frame_size}"
            )
        self._writer.write(frame.image)

    def close(self) -> None:
        if not self._closed:
            self._writer.release()
            self._closed = True


def _gst_property_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _segment_template(destination: str | Path) -> Path:
    path = Path(destination).expanduser()
    if path.suffix.lower() == ".mp4":
        return path.with_name(f"{path.stem}_%05d.mp4").resolve()
    return (path / "segment_%05d.mp4").resolve()


def build_gstreamer_output_pipeline(
    destination: str | Path | None = None,
    *,
    pipeline: str | None = None,
    segment_duration: float | None = None,
    encoder: str | None = None,
    rtsp_transport: str = "tcp",
    hardware_profile: str | None = None,
    _element_available: Callable[[str], bool] | None = None,
) -> str:
    """Build an appsrc pipeline for MP4, segmented MP4, or RTSP publishing."""
    if rtsp_transport not in {"tcp", "udp"}:
        raise ValueError("rtsp_transport must be 'tcp' or 'udp'")
    if segment_duration is not None and segment_duration <= 0:
        raise ValueError("segment_duration must be > 0 when provided")

    if pipeline is not None:
        if (
            destination is not None
            or segment_duration is not None
            or hardware_profile is not None
            or encoder is not None
        ):
            raise ValueError(
                "a custom output pipeline cannot be combined with destination, "
                "segment_duration, encoder, or hardware_profile"
            )
        description = pipeline.strip()
        if not description:
            raise ValueError("output pipeline cannot be empty")
        if "appsrc" not in description.lower():
            description = f"appsrc format=time ! {description}"
        return description

    if destination is None:
        raise ValueError("destination or pipeline is required for a GStreamer output sink")
    if encoder is not None and not encoder.strip():
        raise ValueError("encoder cannot be empty")
    if encoder is not None and hardware_profile is not None:
        raise ValueError("encoder cannot be combined with hardware_profile")

    destination_text = str(destination)
    if hardware_profile is not None:
        resolver_kwargs: dict[str, Any] = {}
        if _element_available is not None:
            resolver_kwargs.update(
                element_available=_element_available,
                require_opencv=False,
            )
        encoder_fragment = resolve_hardware_fragment(
            hardware_profile,
            "encode",
            **resolver_kwargs,
        )
    else:
        selected_encoder = encoder or DEFAULT_GSTREAMER_ENCODER
        encoder_fragment = f"videoconvert ! video/x-raw,format=I420 ! {selected_encoder.strip()}"
    if destination_text.lower().startswith("rtsp://"):
        if segment_duration is not None:
            raise ValueError("segment_duration cannot be used with an RTSP destination")
        base = (
            f"appsrc format=time ! queue leaky=downstream max-size-buffers=4 ! {encoder_fragment}"
        )
        return (
            f"{base} ! h264parse ! rtph264pay config-interval=1 pt=96 "
            f"! rtspclientsink location={_gst_property_quote(destination_text)} "
            f"protocols={rtsp_transport}"
        )

    if "://" in destination_text:
        raise ValueError("output destination must be an RTSP URL or local path")
    base = f"appsrc format=time ! queue ! {encoder_fragment}"
    if segment_duration is not None:
        template = _segment_template(destination)
        duration_ns = round(segment_duration * 1_000_000_000)
        return (
            f"{base} ! h264parse ! splitmuxsink "
            f"location={_gst_property_quote(str(template))} "
            f"max-size-time={duration_ns} muxer-factory=mp4mux"
        )

    path = Path(destination).expanduser().resolve()
    if path.suffix.lower() != ".mp4":
        raise ValueError("local output destination must end in .mp4 unless segment_duration is set")
    return (
        f"{base} ! h264parse ! mp4mux faststart=true "
        f"! filesink location={_gst_property_quote(str(path))}"
    )


class GStreamerVideoSink(FrameSink):
    """Lazy annotated-video sink backed by an OpenCV GStreamer appsrc pipeline."""

    def __init__(
        self,
        destination: str | Path | None = None,
        *,
        pipeline: str | None = None,
        fps: float | None = None,
        segment_duration: float | None = None,
        encoder: str | None = None,
        rtsp_transport: str = "tcp",
        hardware_profile: str | None = None,
        _writer_factory: Callable[[str, float, tuple[int, int]], Any] | None = None,
        _element_available: Callable[[str], bool] | None = None,
    ) -> None:
        if fps is not None and fps <= 0:
            raise ValueError("sink fps must be > 0 when provided")
        self.destination = destination
        self.requested_pipeline = pipeline
        self.requested_fps = float(fps) if fps is not None else None
        self.segment_duration = segment_duration
        self.encoder = encoder
        self.rtsp_transport = rtsp_transport
        self.hardware_profile = hardware_profile
        self._element_available = _element_available
        self.pipeline: str | None = None
        self.fps: float | None = None
        self.frame_size: tuple[int, int] | None = None
        self._writer_factory = _writer_factory or self._open_opencv_writer
        self._writer: Any | None = None
        self._closed = False

        # Validate configuration before inference starts.
        build_gstreamer_output_pipeline(
            destination,
            pipeline=pipeline,
            segment_duration=segment_duration,
            encoder=encoder,
            rtsp_transport=rtsp_transport,
            hardware_profile=hardware_profile,
            _element_available=_element_available,
        )

    @staticmethod
    def _open_opencv_writer(pipeline: str, fps: float, frame_size: tuple[int, int]):
        from dfine.utils.sources import gstreamer_available

        if not gstreamer_available():
            raise RuntimeError(
                "GStreamer is not enabled in this OpenCV build. Install an OpenCV build "
                "compiled with GStreamer and verify cv2.getBuildInformation()."
            )
        return cv2.VideoWriter(
            pipeline,
            cv2.CAP_GSTREAMER,
            0,
            fps,
            frame_size,
            True,
        )

    def _prepare_local_destination(self) -> None:
        if self.requested_pipeline is not None or self.destination is None:
            return
        destination_text = str(self.destination)
        if destination_text.lower().startswith("rtsp://"):
            return
        if self.segment_duration is not None:
            _segment_template(self.destination).parent.mkdir(parents=True, exist_ok=True)
        else:
            Path(self.destination).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    def _open(self, frame: Frame) -> None:
        height, width = frame.image.shape[:2]
        self.frame_size = (width, height)
        source_fps = frame.metadata.fps
        if self.requested_fps is not None:
            self.fps = self.requested_fps
        elif source_fps is not None:
            self.fps = source_fps / frame.metadata.frame_stride
        else:
            self.fps = 30.0
        self.fps = max(self.fps, 1.0)
        self._prepare_local_destination()
        self.pipeline = build_gstreamer_output_pipeline(
            self.destination,
            pipeline=self.requested_pipeline,
            segment_duration=self.segment_duration,
            encoder=self.encoder,
            rtsp_transport=self.rtsp_transport,
            hardware_profile=self.hardware_profile,
            _element_available=self._element_available,
        )
        writer = self._writer_factory(self.pipeline, self.fps, self.frame_size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("Failed to open GStreamer output pipeline")
        self._writer = writer

    def write(self, frame: Frame) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed video sink")
        if self._writer is None:
            self._open(frame)
        assert self.frame_size is not None
        height, width = frame.image.shape[:2]
        if (width, height) != self.frame_size:
            raise ValueError(
                f"frame size {(width, height)} does not match sink size {self.frame_size}"
            )
        writer = self._writer
        assert writer is not None
        writer.write(frame.image)

    def close(self) -> None:
        if not self._closed:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            self._closed = True
