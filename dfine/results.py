"""
Results and Boxes — return types from predict().
Mirrors ultralytics.engine.results.Results / Boxes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class Results:
    """
    Single-image detection result.

    Attributes:
        orig_img:  Original image as HWC BGR numpy array.
        path:      Source path or descriptor.
        names:     {class_id: class_name} dict.
        boxes:     Boxes object (None if no detections).
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: dict[int, str],
        boxes=None,
        save_path: str | None = None,
    ) -> None:
        self.orig_img = orig_img
        self.path = path
        self.names = names
        self.boxes = boxes
        self.save_path = save_path

    def plot(
        self,
        conf: bool = True,
        labels: bool = True,
        line_width: int | None = None,
        font_size: int | None = None,
    ) -> np.ndarray:
        """Draw boxes on image. Returns HWC BGR numpy array."""
        from dfine.plotting import plot_results

        return plot_results(
            self, conf=conf, labels=labels, line_width=line_width, font_size=font_size
        )

    def save(self, filename: str) -> None:
        """Save plotted image to disk."""
        cv2.imwrite(str(filename), self.plot())

    def save_txt(self, path: str | Path, save_conf: bool = False) -> None:
        """Save detections as YOLO-format text labels."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        if self.boxes is not None:
            xywhn = self.boxes.xywhn
            for i in range(len(self)):
                cls = int(self.boxes.cls[i])
                coords = [f"{float(x):.6f}" for x in xywhn[i].tolist()]
                values = [str(cls), *coords]
                if save_conf:
                    values.append(f"{float(self.boxes.conf[i]):.6f}")
                lines.append(" ".join(values))

        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def show(self) -> None:
        """Display image in a window (blocks until key press)."""
        cv2.imshow(str(self.path), self.plot())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def to_json(self) -> list[dict]:
        """Serialise detections to a list of dicts."""
        out: list[dict[str, object]] = []
        if self.boxes is None:
            return out
        for i in range(len(self)):
            xyxy = self.boxes.xyxy[i].tolist()
            out.append(
                {
                    "box": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
                    "confidence": round(float(self.boxes.conf[i]), 4),
                    "class": int(self.boxes.cls[i]),
                    "name": self.names.get(int(self.boxes.cls[i]), "unknown"),
                }
            )
        return out

    def __len__(self) -> int:
        return 0 if self.boxes is None else len(self.boxes)

    def __repr__(self) -> str:
        return f"Results(path={self.path!r}, detections={len(self)})"


class Boxes:
    """
    Bounding box container for one image.

    Args:
        data:  Tensor [N, 6] — columns: x1 y1 x2 y2 conf cls
        orig_shape: (H, W) of the original image (for normalised coords).
    """

    def __init__(self, data, orig_shape: tuple[int, int]) -> None:
        self._data = data  # torch.Tensor [N, 6]
        self.orig_shape = orig_shape  # (H, W)

    @property
    def data(self):
        """Raw [N, 6] tensor: xyxy + conf + cls."""
        return self._data

    @property
    def xyxy(self):
        """Absolute pixel coords [N, 4]."""
        return self._data[:, :4]

    @property
    def xyxyn(self):
        """Normalised 0-1 coords [N, 4]."""
        h, w = self.orig_shape
        norm = self._data[:, :4].clone()
        norm[:, [0, 2]] /= w
        norm[:, [1, 3]] /= h
        return norm

    @property
    def xywh(self):
        """cx, cy, w, h in absolute pixels [N, 4]."""
        x1, y1, x2, y2 = self.xyxy.unbind(1)
        return __import__("torch").stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=1)

    @property
    def xywhn(self):
        """cx, cy, w, h normalised [N, 4]."""
        h, w = self.orig_shape
        xywh = self.xywh.clone()
        xywh[:, [0, 2]] /= w
        xywh[:, [1, 3]] /= h
        return xywh

    @property
    def conf(self):
        """Confidence scores [N]."""
        return self._data[:, 4]

    @property
    def cls(self):
        """Class indices [N] as int."""
        return self._data[:, 5].int()

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Boxes(n={len(self)}, device={self._data.device})"
