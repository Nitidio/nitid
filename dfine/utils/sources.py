"""
LoadSource — unified iterator over any valid input source.
Yields (tensor [1,3,H,W], orig_img [HWC BGR], path_str) tuples.

Supported sources:
    str / Path  — image file, video file, directory, glob pattern, URL
    int         — webcam index
    np.ndarray  — single frame (GStreamer pipeline entry point)
    list        — list of any of the above
    "screen"    — screen capture (requires mss)
    rtsp://...  — RTSP / RTMP stream
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v"}


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
    ) -> None:
        self.source = source
        self.imgsz = imgsz
        self.device = device
        self._mode = self._detect_mode(source)

    def _detect_mode(self, source) -> str:
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

    def __iter__(self) -> Generator:
        if self._mode == "array":
            yield self._process_frame(self.source, path="<ndarray>")
        elif self._mode == "image":
            from PIL import Image as _PILImage

            pil_img = _PILImage.open(str(self.source)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            yield self._process_frame(img, path=str(self.source))
        elif self._mode in ("video", "webcam", "stream"):
            cap = cv2.VideoCapture(self.source if self._mode == "webcam" else str(self.source))
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                yield self._process_frame(frame, path=str(self.source))
            cap.release()
        elif self._mode == "directory":
            for p in sorted(Path(self.source).iterdir()):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    image_frame: np.ndarray | None = cv2.imread(str(p))
                    if image_frame is None:
                        continue
                    yield self._process_frame(image_frame, path=str(p))
        elif self._mode == "list":
            for item in self.source:
                yield from LoadSource(item, self.imgsz, self.device)

    def _process_frame(self, img: np.ndarray, path: str):
        """Preprocess a BGR numpy frame → (tensor, orig_img, path).

        img is BGR numpy (HWC). The tensor is built with PIL+torchvision to
        exactly match D-FINE's own preprocessing (T.Resize → T.ToTensor).
        """
        import torchvision.transforms as T
        from PIL import Image as _PILImage

        orig_img = img.copy()
        pil_img = _PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        transform = T.Compose([T.Resize((self.imgsz, self.imgsz)), T.ToTensor()])
        tensor = transform(pil_img).unsqueeze(0).to(self.device)
        return tensor, orig_img, path

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
