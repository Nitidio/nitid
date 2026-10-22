"""
LoadSource — unified iterator over any valid input source.
Yields (tensor [1,3,H,W], orig_img [HWC BGR], path_str) tuples.

Supported sources:
    str / Path  — image file, video file, directory, glob pattern, URL
    int         — webcam index
    np.ndarray  — single frame
    list        — list of any of the above
    "screen"    — screen capture (uses mss)
    rtsp://...  — RTSP / RTMP / HTTP stream (OpenCV's default FFmpeg backend)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterator
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np

from dfine.media import Frame, FrameMetadata, FrameSource

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v"}
RTSP_SCHEMES = {"rtsp", "rtsps"}


def rtsp_url_with_credentials(url: str, username: str, password: str | None = None) -> str:
    """Return ``url`` with percent-encoded RTSP credentials in its userinfo.

    OpenCV's FFmpeg backend only accepts RTSP credentials as part of the URL, so
    they are injected here instead of being written into user-facing source
    strings. The returned URL contains the password and must never be logged;
    use the original ``url`` as the source identifier.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in RTSP_SCHEMES:
        raise ValueError("RTSP credentials require an rtsp:// or rtsps:// source")
    if not parts.hostname:
        raise ValueError("RTSP source URL must include a host")
    if "@" in parts.netloc:
        raise ValueError(
            "RTSP source URL already contains credentials; pass them either in the URL "
            "or through rtsp_username/rtsp_password, not both"
        )
    userinfo = quote(username, safe="")
    if password is not None:
        userinfo += ":" + quote(password, safe="")
    return urlunsplit(parts._replace(netloc=f"{userinfo}@{parts.netloc}"))


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

    def __init__(
        self,
        source: str | int,
        *,
        mode: str,
        vid_stride: int = 1,
        rtsp_username: str | None = None,
        rtsp_password: str | None = None,
    ) -> None:
        if mode not in {"video", "webcam", "stream"}:
            raise ValueError(f"unsupported OpenCV source mode: {mode}")
        if vid_stride < 1:
            raise ValueError("vid_stride must be >= 1")
        if rtsp_password is not None and rtsp_username is None:
            raise ValueError("rtsp_password requires rtsp_username")
        self.source = source
        self.mode = mode
        self.vid_stride = vid_stride
        # The capture target may carry credentials; ``source`` and ``source_id``
        # stay credential-free so they can be logged and stored in results.
        self._capture_target: str | int = source
        if rtsp_username is not None:
            if not isinstance(source, str):
                raise ValueError("RTSP credentials require an rtsp:// or rtsps:// source")
            self._capture_target = rtsp_url_with_credentials(source, rtsp_username, rtsp_password)
        self.fps = self._probe_fps() if mode == "video" else None
        self._capture: cv2.VideoCapture | None = None

    @property
    def source_id(self) -> str:
        return str(self.source)

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self._capture_target)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open {self.mode} source '{self.source_id}'")
        return capture

    def _probe_fps(self) -> float | None:
        capture = cv2.VideoCapture(self._capture_target)
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
        if isinstance(source, FrameSource):
            if rtsp_username is not None or rtsp_password is not None:
                raise ValueError("RTSP credentials cannot be combined with a FrameSource instance")
            self._mode = source.mode
            self._frame_source = source
            self.video_fps = source.fps
            self.vid_stride = source.vid_stride
        else:
            self._mode = self._detect_mode(source)
            if (rtsp_username is not None or rtsp_password is not None) and self._mode != "stream":
                raise ValueError("RTSP credentials require an rtsp:// or rtsps:// source")
            if self._mode in ("video", "webcam", "stream"):
                opencv_source = OpenCVFrameSource(
                    source if self._mode == "webcam" else str(source),
                    mode=self._mode,
                    vid_stride=vid_stride,
                    rtsp_username=rtsp_username,
                    rtsp_password=rtsp_password,
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
            if s.startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")):
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
        # Close the generator we delegate to rather than letting it fall out of
        # scope. Abandoning this iterator early raises GeneratorExit here, and
        # the traceback that unwinds keeps this frame — and so the generator
        # below — alive in a reference cycle, leaving the capture handle open
        # until the cycle collector happens to run.
        samples = self.iter_samples()
        try:
            for sample in samples:
                yield sample.as_legacy_tuple()
        finally:
            samples.close()

    def iter_samples(self) -> Generator[SourceSample, None, None]:
        """Yield preprocessed samples while preserving frame metadata."""
        frames = self.iter_frames()
        try:
            for frame in frames:
                yield SourceSample(tensor=self._preprocess(frame.image), frame=frame)
        finally:
            frames.close()
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
