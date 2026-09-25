"""Media contracts shared by source, inference, tracking, and output backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


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
