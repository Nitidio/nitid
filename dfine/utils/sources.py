"""
LoadSource — unified iterator over any valid input source.
Yields (tensor [1,3,H,W], orig_img [HWC BGR], path_str) tuples.

Supported sources:
    str / Path  — image file, video file, directory, glob pattern, URL
    int         — webcam index
    np.ndarray  — single frame (GStreamer pipeline entry point)
    list        — list of any of the above
    "screen"    — screen capture (uses mss)
    rtsp://...  — RTSP / RTMP stream
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterator

import cv2
import numpy as np

from dfine.gstreamer import gstreamer_available, resolve_hardware_fragment
from dfine.media import Frame, FrameMetadata, FrameSource

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v"}


def _gst_quote(value: str) -> str:
    """Quote a value for a GStreamer pipeline property."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_gstreamer_pipeline(
    source: str | Path | int,
    *,
    pipeline: str | None = None,
    live: bool | None = None,
    rtsp_latency: int = 200,
    rtsp_transport: str = "tcp",
    hardware_profile: str | None = None,
    rtsp_username: str | None = None,
    rtsp_password: str | None = None,
    _element_available: Callable[[str], bool] | None = None,
) -> str:
    """Build an appsink pipeline for a URI, video file, webcam, or explicit pipeline."""
    if rtsp_latency < 0:
        raise ValueError("rtsp_latency must be >= 0")
    if rtsp_transport not in {"tcp", "udp"}:
        raise ValueError("rtsp_transport must be 'tcp' or 'udp'")
    if pipeline is not None and hardware_profile is not None:
        raise ValueError("hardware_profile cannot be combined with gst_pipeline")
    if pipeline is not None and rtsp_username is not None:
        raise ValueError("RTSP credentials cannot be combined with gst_pipeline")
    if rtsp_password is not None and rtsp_username is None:
        raise ValueError("rtsp_password requires rtsp_username")

    if live is None:
        source_text = str(source).strip().lower()
        live = (
            pipeline is not None
            or isinstance(source, int)
            or "!" in source_text
            or source_text.startswith(("rtsp://", "rtmp://", "http://", "https://"))
        )

    if pipeline is not None:
        description = pipeline.strip()
    elif isinstance(source, int):
        if rtsp_username is not None:
            raise ValueError("RTSP credentials require an RTSP source")
        if hardware_profile is not None:
            raise ValueError(
                "hardware_profile currently supports H.264 RTSP sources; "
                "use gst_pipeline for webcams"
            )
        if source < 0:
            raise ValueError("webcam index must be >= 0")
        if sys.platform.startswith("linux"):
            description = f"v4l2src device=/dev/video{source}"
        elif sys.platform == "darwin":
            description = f"avfvideosrc device-index={source}"
        elif sys.platform == "win32":
            description = f"ksvideosrc device-index={source}"
        else:
            raise RuntimeError(
                "automatic GStreamer webcam pipelines are not supported on this platform; "
                "pass gst_pipeline= explicitly"
            )
    else:
        source_text = str(source).strip()
        if "!" in source_text:
            if rtsp_username is not None:
                raise ValueError("RTSP credentials cannot be combined with an explicit pipeline")
            if hardware_profile is not None:
                raise ValueError("hardware_profile cannot be combined with an explicit pipeline")
            description = source_text
        elif source_text.lower().startswith("rtsp://"):
            source_element = (
                f"rtspsrc location={_gst_quote(source_text)} latency={rtsp_latency} "
                f"protocols={rtsp_transport}"
            )
            if rtsp_username is not None:
                source_element += f" user-id={_gst_quote(rtsp_username)}"
                source_element += f" user-pw={_gst_quote(rtsp_password or '')}"
            if hardware_profile is None:
                description = f"{source_element} ! decodebin"
            else:
                resolver_kwargs: dict[str, Any] = {}
                if _element_available is not None:
                    resolver_kwargs.update(
                        element_available=_element_available,
                        require_opencv=False,
                    )
                fragment = resolve_hardware_fragment(
                    hardware_profile,
                    "decode",
                    **resolver_kwargs,
                )
                description = f"{source_element} ! rtph264depay ! h264parse ! {fragment}"
        else:
            if rtsp_username is not None:
                raise ValueError("RTSP credentials require an RTSP source")
            if hardware_profile is not None:
                raise ValueError(
                    "hardware_profile currently supports H.264 RTSP sources; "
                    "use gst_pipeline for files or other codecs"
                )
            if "://" in source_text:
                uri = source_text
            else:
                path = Path(source_text).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(
                        f"GStreamer video source was not found: '{source_text}'"
                    )
                uri = path.as_uri()
            description = f"uridecodebin uri={_gst_quote(uri)}"

    if not description:
        raise ValueError("gst_pipeline cannot be empty")
    if "appsink" not in description.lower():
        appsink = "appsink drop=true max-buffers=1 sync=false" if live else "appsink sync=false"
        description += f" ! videoconvert ! video/x-raw,format=BGR ! {appsink}"
    return description


@dataclass(frozen=True, slots=True)
class SourceSample:
    """A preprocessed tensor paired with its original decoded frame."""

    tensor: Any
    frame: Frame

    def as_legacy_tuple(self) -> tuple:
        """Return the historical ``(tensor, image, path)`` representation."""
        return self.tensor, self.frame.image, self.frame.metadata.source_id


class OpenCVFrameSource(FrameSource):
    """Video, webcam, or network stream source backed by OpenCV."""

    def __init__(self, source: str | int, *, mode: str, vid_stride: int = 1) -> None:
        if mode not in {"video", "webcam", "stream"}:
            raise ValueError(f"unsupported OpenCV source mode: {mode}")
        if vid_stride < 1:
            raise ValueError("vid_stride must be >= 1")
        self.source = source
        self.mode = mode
        self.vid_stride = vid_stride
        self.fps = self._probe_fps() if mode == "video" else None
        self._capture: cv2.VideoCapture | None = None

    @property
    def source_id(self) -> str:
        return str(self.source)

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open {self.mode} source '{self.source_id}'")
        return capture

    def _probe_fps(self) -> float | None:
        capture = cv2.VideoCapture(self.source)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        return fps if fps > 0 else None

    def __iter__(self) -> Iterator[Frame]:
        if self._capture is not None:
            raise RuntimeError("an OpenCV source cannot be iterated more than once concurrently")

        self._capture = self._open()
        capture = self._capture
        if self.fps is None:
            reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
            self.fps = reported_fps if reported_fps > 0 else None
        frame_index = 0
        try:
            while capture.isOpened():
                ok, image = capture.read()
                if not ok:
                    break

                current_index = frame_index
                frame_index += 1
                if current_index % self.vid_stride != 0:
                    continue

                if self.mode == "video":
                    position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    timestamp = position_ms / 1000.0 if position_ms >= 0 else None
                    if (timestamp is None or timestamp == 0.0) and self.fps:
                        timestamp = current_index / self.fps
                else:
                    timestamp = time.monotonic()

                yield Frame(
                    image=image,
                    metadata=FrameMetadata(
                        source_id=self.source_id,
                        frame_index=current_index,
                        timestamp=timestamp,
                        fps=self.fps,
                        frame_stride=self.vid_stride,
                    ),
                )
        finally:
            self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class GStreamerFrameSource(FrameSource):
    """Video or live-stream source using an OpenCV GStreamer appsink pipeline.

    Reconnection applies only to live sources. A recovered stream marks the
    first emitted frame as discontinuous so stateful consumers can reset.
    """

    def __init__(
        self,
        source: str | Path | int,
        *,
        pipeline: str | None = None,
        mode: str | None = None,
        vid_stride: int = 1,
        reconnect: bool = False,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        reconnect_attempts: int | None = None,
        rtsp_latency: int = 200,
        rtsp_transport: str = "tcp",
        hardware_profile: str | None = None,
        rtsp_username: str | None = None,
        rtsp_password: str | None = None,
        _capture_factory: Callable[[str], Any] | None = None,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if vid_stride < 1:
            raise ValueError("vid_stride must be >= 1")
        if reconnect_initial_delay < 0:
            raise ValueError("reconnect_initial_delay must be >= 0")
        if reconnect_max_delay < reconnect_initial_delay:
            raise ValueError("reconnect_max_delay must be >= reconnect_initial_delay")
        if reconnect_attempts is not None and reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be >= 0 when provided")

        inferred_mode = self._infer_mode(source, pipeline)
        self.mode = mode or inferred_mode
        if self.mode not in {"video", "webcam", "stream"}:
            raise ValueError(f"unsupported GStreamer source mode: {self.mode}")

        self.source = source
        self.source_id = str(source)
        self.vid_stride = vid_stride
        self.reconnect = reconnect
        self.reconnect_initial_delay = float(reconnect_initial_delay)
        self.reconnect_max_delay = float(reconnect_max_delay)
        self.reconnect_attempts = reconnect_attempts
        self.pipeline = build_gstreamer_pipeline(
            source,
            pipeline=pipeline,
            live=self.mode != "video",
            rtsp_latency=rtsp_latency,
            rtsp_transport=rtsp_transport,
            hardware_profile=hardware_profile,
            rtsp_username=rtsp_username,
            rtsp_password=rtsp_password,
        )
        self.fps: float | None = None
        self._capture_factory = _capture_factory or self._open_opencv_capture
        self._sleep = _sleep
        self._capture: Any | None = None
        self._stop_requested = False
        self._stop_event = threading.Event()

    @staticmethod
    def _infer_mode(source: str | Path | int, pipeline: str | None) -> str:
        if isinstance(source, int):
            return "webcam"
        source_text = str(source).lower()
        if pipeline is not None or "!" in source_text:
            return "stream"
        if source_text.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return "stream"
        return "video"

    @staticmethod
    def _open_opencv_capture(pipeline: str):
        if not gstreamer_available():
            raise RuntimeError(
                "GStreamer is not enabled in this OpenCV build. Install an OpenCV build "
                "compiled with GStreamer and verify cv2.getBuildInformation()."
            )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    def _open(self):
        capture = self._capture_factory(self.pipeline)
        if not capture.isOpened():
            capture.release()
            return None
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if reported_fps > 0:
            self.fps = reported_fps
        return capture

    def _reopen(self):
        delay = self.reconnect_initial_delay
        attempts = 0
        while not self._stop_requested:
            if self.reconnect_attempts is not None and attempts >= self.reconnect_attempts:
                raise RuntimeError(
                    f"Failed to reconnect GStreamer source '{self.source_id}' "
                    f"after {attempts} attempt(s)"
                )
            if delay:
                if self._sleep is time.sleep:
                    self._stop_event.wait(delay)
                else:
                    self._sleep(delay)
                if self._stop_requested:
                    return None
            attempts += 1
            capture = self._open()
            if capture is not None:
                return capture
            delay = min(max(delay * 2, 0.001), self.reconnect_max_delay)
        return None

    def __iter__(self) -> Iterator[Frame]:
        if self._capture is not None:
            raise RuntimeError("a GStreamer source cannot be iterated more than once concurrently")

        self._stop_requested = False
        self._stop_event.clear()
        capture = self._open()
        if capture is None:
            if not self.reconnect or self.reconnect_attempts == 0:
                raise RuntimeError(f"Failed to open GStreamer source '{self.source_id}'")
            capture = self._reopen()
        if capture is None:
            return

        self._capture = capture
        frame_index = 0
        has_emitted_frame = False
        pending_discontinuity = False
        try:
            while not self._stop_requested:
                ok, image = capture.read()
                if not ok:
                    capture.release()
                    self._capture = None
                    if self.mode == "video" or not self.reconnect:
                        break
                    capture = self._reopen()
                    if capture is None:
                        break
                    self._capture = capture
                    pending_discontinuity = has_emitted_frame
                    continue

                current_index = frame_index
                frame_index += 1
                if current_index % self.vid_stride != 0:
                    continue

                if self.mode == "video":
                    position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    timestamp = position_ms / 1000.0 if position_ms >= 0 else None
                    if (timestamp is None or timestamp == 0.0) and self.fps:
                        timestamp = current_index / self.fps
                else:
                    timestamp = time.monotonic()

                yield Frame(
                    image=image,
                    metadata=FrameMetadata(
                        source_id=self.source_id,
                        frame_index=current_index,
                        timestamp=timestamp,
                        fps=self.fps,
                        frame_stride=self.vid_stride,
                        discontinuity=pending_discontinuity,
                    ),
                )
                has_emitted_frame = True
                pending_discontinuity = False
        finally:
            self.close()

    def close(self) -> None:
        self._stop_requested = True
        self._stop_event.set()
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class LoadSource:
    """
    Wraps any source and yields preprocessed batches.

    Args:
        source:  Any valid source (see module docstring).
        imgsz:   Target inference size (square).
        device:  Target torch device string.
    """

    def __init__(
        self,
        source,
        imgsz: int = 640,
        device: str = "cuda:0",
        vid_stride: int = 1,
        backend: str = "opencv",
        gst_pipeline: str | None = None,
        reconnect: bool = False,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        reconnect_attempts: int | None = None,
        rtsp_latency: int = 200,
        rtsp_transport: str = "tcp",
        hardware_profile: str | None = None,
        rtsp_username: str | None = None,
        rtsp_password: str | None = None,
    ) -> None:
        if vid_stride < 1:
            raise ValueError("vid_stride must be >= 1")
        self.source = source
        self.imgsz = imgsz
        self.device = device
        self.vid_stride = vid_stride
        self._frame_source: FrameSource | None = None
        self.video_fps: float | None = None
        backend = backend.lower()
        if backend not in {"opencv", "gstreamer"}:
            raise ValueError("backend must be 'opencv' or 'gstreamer'")
        if isinstance(source, FrameSource):
            if (
                backend != "opencv"
                or gst_pipeline is not None
                or reconnect
                or hardware_profile is not None
                or rtsp_username is not None
                or rtsp_password is not None
            ):
                raise ValueError("backend options cannot be combined with a FrameSource instance")
            self._mode = source.mode
            self._frame_source = source
            self.video_fps = source.fps
            self.vid_stride = source.vid_stride
        elif backend == "gstreamer":
            if not isinstance(source, (str, Path, int)):
                raise TypeError(
                    "the GStreamer backend requires a path, URI, webcam index, or pipeline"
                )
            gstreamer_source = GStreamerFrameSource(
                source,
                pipeline=gst_pipeline,
                vid_stride=vid_stride,
                reconnect=reconnect,
                reconnect_initial_delay=reconnect_initial_delay,
                reconnect_max_delay=reconnect_max_delay,
                reconnect_attempts=reconnect_attempts,
                rtsp_latency=rtsp_latency,
                rtsp_transport=rtsp_transport,
                hardware_profile=hardware_profile,
                rtsp_username=rtsp_username,
                rtsp_password=rtsp_password,
            )
            self._mode = gstreamer_source.mode
            self._frame_source = gstreamer_source
        else:
            if (
                gst_pipeline is not None
                or reconnect
                or hardware_profile is not None
                or rtsp_username is not None
                or rtsp_password is not None
            ):
                raise ValueError("GStreamer source options require backend='gstreamer'")
            self._mode = self._detect_mode(source)
            if self._mode in ("video", "webcam", "stream"):
                opencv_source = OpenCVFrameSource(
                    source if self._mode == "webcam" else str(source),
                    mode=self._mode,
                    vid_stride=vid_stride,
                )
                self._frame_source = opencv_source
                self.video_fps = opencv_source.fps

    def _detect_mode(self, source) -> str:
        if isinstance(source, FrameSource):
            return source.mode
        if isinstance(source, np.ndarray):
            return "array"
        if isinstance(source, int):
            return "webcam"
        if isinstance(source, (str, Path)):
            s = str(source)
            if s.startswith(("rtsp://", "rtmp://", "http://", "https://")):
                return "stream"
            if s == "screen":
                return "screen"
            p = Path(s)
            if p.is_dir():
                return "directory"
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                return "image"
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                return "video"
        if isinstance(source, list):
            return "list"
        raise ValueError(f"Unrecognised source type: {type(source)}")

    @property
    def mode(self) -> str:
        return self._mode

    def __iter__(self) -> Generator:
        """Yield historical ``(tensor, image, path)`` tuples for compatibility."""
        for sample in self.iter_samples():
            yield sample.as_legacy_tuple()

    def iter_samples(self) -> Generator[SourceSample, None, None]:
        """Yield preprocessed samples while preserving frame metadata."""
        try:
            for frame in self.iter_frames():
                yield SourceSample(tensor=self._preprocess(frame.image), frame=frame)
        finally:
            self.close()

    def iter_frames(self) -> Generator[Frame, None, None]:
        """Yield decoded frames before model preprocessing."""
        if self._frame_source is not None:
            yield from self._frame_source
        elif self._mode == "array":
            yield self._make_frame(self.source, source_id="<ndarray>", frame_index=0)
        elif self._mode == "image":
            from PIL import Image as _PILImage

            pil_img = _PILImage.open(str(self.source)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            yield self._make_frame(img, source_id=str(self.source), frame_index=0)
        elif self._mode == "screen":
            frame_index = 0
            try:
                while True:
                    screen_frame = self._capture_screen_frame()
                    if screen_frame is None:
                        break
                    yield self._make_frame(
                        screen_frame,
                        source_id="<screen>",
                        frame_index=frame_index,
                        timestamp=time.monotonic(),
                    )
                    frame_index += 1
            finally:
                if hasattr(self, "_sct"):
                    self._sct.close()
                    del self._sct
        elif self._mode == "directory":
            frame_index = 0
            for p in sorted(Path(self.source).iterdir()):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    image_frame: np.ndarray | None = cv2.imread(str(p))
                    if image_frame is None:
                        continue
                    yield self._make_frame(image_frame, source_id=str(p), frame_index=frame_index)
                    frame_index += 1
        elif self._mode == "list":
            for item in self.source:
                child = LoadSource(item, self.imgsz, self.device, self.vid_stride)
                yield from child.iter_frames()

    def _make_frame(
        self,
        image: np.ndarray,
        *,
        source_id: str,
        frame_index: int,
        timestamp: float | None = None,
    ) -> Frame:
        return Frame(
            image=image.copy(),
            metadata=FrameMetadata(
                source_id=source_id,
                frame_index=frame_index,
                timestamp=timestamp,
                fps=self.video_fps,
                frame_stride=self.vid_stride,
            ),
        )

    def _preprocess(self, img: np.ndarray):
        """Preprocess a BGR numpy frame into a model input tensor.

        img is BGR numpy (HWC). The tensor is built with PIL+torchvision to
        exactly match D-FINE's own preprocessing (T.Resize → T.ToTensor).
        """
        import torchvision.transforms as T
        from PIL import Image as _PILImage

        pil_img = _PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        transform = T.Compose([T.Resize((self.imgsz, self.imgsz)), T.ToTensor()])
        return transform(pil_img).unsqueeze(0).to(self.device)

    def close(self) -> None:
        """Release any source resources owned by this loader."""
        if self._frame_source is not None:
            self._frame_source.close()
        if hasattr(self, "_sct"):
            self._sct.close()
            del self._sct

    def _capture_screen_frame(self) -> np.ndarray | None:
        """Capture the current desktop as a BGR frame.

        Lazily creates and caches a single mss instance across calls.
        """
        if not hasattr(self, "_sct"):
            try:
                from mss import mss
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError("Screen capture requires mss") from exc

            try:
                self._sct = mss()
                self._monitor = self._sct.monitors[0]
            except (OSError, ValueError, IndexError) as exc:
                raise RuntimeError("Screen capture is not available in this environment") from exc

        screen = self._sct.grab(self._monitor)
        return cv2.cvtColor(np.array(screen), cv2.COLOR_BGRA2BGR)

    def __len__(self) -> int:
        """Returns -1 for live streams and webcams."""
        if self._mode in ("stream", "webcam", "screen"):
            return -1
        if self._mode == "image":
            return 1
        if self._mode == "array":
            return 1
        if self._mode == "directory":
            return sum(
                1 for p in Path(self.source).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
        return -1
