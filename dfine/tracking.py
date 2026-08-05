"""Object-tracking processors for sequential detection results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from dfine.results import Boxes, Results


class ResultTracker(ABC):
    """Stateful processor that assigns identities to sequential results."""

    @abstractmethod
    def update(self, result: Results) -> Results:
        """Update tracker state and return the tracked result."""

    @abstractmethod
    def reset(self) -> None:
        """Discard all active tracks."""


class ByteTrack(ResultTracker):
    """Adapter from nitid ``Results`` to the optional ByteTrack backend.

    The backend is constructed lazily so importing and using ordinary detection
    does not require the ``track`` optional dependency.
    """

    def __init__(
        self,
        *,
        frame_rate: float | None = None,
        lost_track_buffer: int = 30,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 1,
        minimum_iou_threshold: float = 0.1,
        high_conf_det_threshold: float = 0.6,
        backend_factory: Callable[..., Any] | None = None,
        detections_factory: Callable[..., Any] | None = None,
    ) -> None:
        if frame_rate is not None and frame_rate <= 0:
            raise ValueError("frame_rate must be > 0")
        if lost_track_buffer < 0:
            raise ValueError("lost_track_buffer must be >= 0")
        if minimum_consecutive_frames < 0:
            raise ValueError("minimum_consecutive_frames must be >= 0")
        for name, value in {
            "track_activation_threshold": track_activation_threshold,
            "minimum_iou_threshold": minimum_iou_threshold,
            "high_conf_det_threshold": high_conf_det_threshold,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        self._configured_frame_rate = frame_rate
        self._backend_factory = backend_factory
        self._detections_factory = detections_factory
        self._backend_kwargs = {
            "lost_track_buffer": lost_track_buffer,
            "track_activation_threshold": track_activation_threshold,
            "minimum_consecutive_frames": minimum_consecutive_frames,
            "minimum_iou_threshold": minimum_iou_threshold,
            "high_conf_det_threshold": high_conf_det_threshold,
        }
        self._backend: Any | None = None
        self._source_id: str | None = None

    def _load_backend_factory(self) -> Callable[..., Any]:
        if self._backend_factory is not None:
            return self._backend_factory
        try:
            from trackers import ByteTrackTracker
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise RuntimeError(
                "ByteTrack requires the optional tracking dependencies. "
                "Install them with: pip install 'nitid[track]'"
            ) from exc
        return ByteTrackTracker

    def _effective_frame_rate(self, result: Results) -> float:
        if self._configured_frame_rate is not None:
            return self._configured_frame_rate
        metadata = result.frame_metadata
        if metadata is not None and metadata.fps:
            return metadata.fps / metadata.frame_stride
        return 30.0

    def _ensure_backend(self, result: Results) -> Any:
        metadata = result.frame_metadata
        source_id = metadata.source_id if metadata is not None else result.path
        discontinuity = bool(metadata and metadata.discontinuity)
        if self._backend is None or source_id != self._source_id or discontinuity:
            factory = self._load_backend_factory()
            self._backend = factory(
                frame_rate=self._effective_frame_rate(result),
                **self._backend_kwargs,
            )
            self._source_id = source_id
        return self._backend

    def update(self, result: Results) -> Results:
        backend = self._ensure_backend(result)
        detections_factory: Callable[..., Any]
        if self._detections_factory is None:
            try:
                from supervision import Detections
            except ImportError as exc:  # pragma: no cover - depends on installation
                raise RuntimeError(
                    "ByteTrack requires the optional tracking dependencies. "
                    "Install them with: pip install 'nitid[track]'"
                ) from exc
            detections_factory = Detections
        else:
            detections_factory = self._detections_factory

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty((0,), dtype=np.float32)
            class_id = np.empty((0,), dtype=np.int64)
            device = torch.device("cpu")
            dtype = torch.float32
            orig_shape = result.orig_img.shape[:2]
        else:
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
            confidence = boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
            class_id = boxes.cls.detach().cpu().numpy().astype(np.int64, copy=False)
            device = boxes.data.device
            dtype = boxes.data.dtype
            orig_shape = boxes.orig_shape

        detections = detections_factory(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
        tracked = backend.update(detections)

        count = len(tracked)
        if count:
            tracked_confidence = (
                tracked.confidence
                if tracked.confidence is not None
                else np.ones(count, dtype=np.float32)
            )
            tracked_class_id = (
                tracked.class_id
                if tracked.class_id is not None
                else np.zeros(count, dtype=np.int64)
            )
            tracked_id = (
                tracked.tracker_id
                if tracked.tracker_id is not None
                else np.full(count, -1, dtype=np.int64)
            )
            data_np = np.column_stack(
                [
                    tracked.xyxy,
                    tracked_id,
                    tracked_confidence,
                    tracked_class_id,
                ]
            )
            data = torch.as_tensor(data_np, device=device, dtype=dtype)
        else:
            data = torch.empty((0, 7), device=device, dtype=dtype)

        result.boxes = Boxes(data, orig_shape=orig_shape)
        return result

    def reset(self) -> None:
        self._backend = None
        self._source_id = None


def create_tracker(tracker: str | ResultTracker, **kwargs: Any) -> ResultTracker:
    """Resolve a public tracker selection into a result processor."""
    if isinstance(tracker, ResultTracker):
        if kwargs:
            raise ValueError("tracker_kwargs cannot be used with a tracker instance")
        return tracker
    if not isinstance(tracker, str):
        raise TypeError("tracker must be a tracker name or ResultTracker instance")
    if tracker.lower().replace("-", "") != "bytetrack":
        raise ValueError(f"unsupported tracker '{tracker}'; expected 'bytetrack'")
    return ByteTrack(**kwargs)


class DFINETracker:
    """Runs D-FINE detection with a stateful result tracker in the pipeline."""

    def __init__(self, predictor) -> None:
        self.predictor = predictor

    def run(
        self,
        source,
        *,
        tracker: str | ResultTracker,
        tracker_kwargs: dict[str, Any] | None,
        **predict_kwargs: Any,
    ):
        processor = create_tracker(tracker, **(tracker_kwargs or {}))
        tracker_name = tracker if isinstance(tracker, str) else type(tracker).__name__
        return self.predictor.run(
            source,
            result_processor=processor.update,
            run_mode="track",
            run_metadata={
                "tracker": tracker_name,
                "tracker_kwargs": tracker_kwargs or {},
            },
            **predict_kwargs,
        )
