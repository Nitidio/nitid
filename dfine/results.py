"""
Results, Boxes, and Masks — return types from predict().
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from dfine.media import FrameMetadata


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
        masks=None,
        save_path: str | None = None,
        speed: dict[str, float] | None = None,
        frame_metadata: FrameMetadata | None = None,
    ) -> None:
        self.orig_img = orig_img
        self.path = path
        self.names = names
        self.boxes = boxes
        self.masks = masks
        self.save_path = save_path
        self.speed = speed or {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        self.frame_metadata = frame_metadata

    def plot(
        self,
        conf: bool = True,
        labels: bool = True,
        line_width: int | None = None,
        font_size: int | None = None,
    ) -> np.ndarray:
        """Draw instance masks and boxes on the image."""
        from dfine.plotting import plot_results

        return plot_results(
            self, conf=conf, labels=labels, line_width=line_width, font_size=font_size
        )

    def save(self, filename: str) -> None:
        """Save plotted image to disk."""
        cv2.imwrite(str(filename), self.plot())

    def save_json(self, path: str | Path) -> None:
        """Save detections as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    def save_txt(self, path: str | Path, save_conf: bool = False) -> None:
        """Save detections as YOLO-format text labels."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        if self.boxes is not None:
            xywhn = self.boxes.xywhn
            for i in range(len(self)):
                cls = int(self.boxes.cls[i])
                if self.masks is not None and i < len(self.masks):
                    polygon = self.masks.xyn[i]
                    coords = [f"{float(x):.6f}" for x in polygon.reshape(-1).tolist()]
                    if not coords:
                        coords = [f"{float(x):.6f}" for x in xywhn[i].tolist()]
                else:
                    coords = [f"{float(x):.6f}" for x in xywhn[i].tolist()]
                values = [str(cls), *coords]
                if save_conf:
                    values.append(f"{float(self.boxes.conf[i]):.6f}")
                if self.boxes.id is not None:
                    values.append(str(int(self.boxes.id[i])))
                lines.append(" ".join(values))

        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def crop(
        self,
        save_dir: str | Path | None = None,
        file_name: str | Path | None = None,
    ) -> list[dict[str, object]]:
        """Return detection crops and optionally save them into class folders."""
        if self.boxes is None:
            return []

        save_root = Path(save_dir) if save_dir is not None else None
        source_name = Path(file_name or self.path or "im.jpg")
        if not source_name.suffix:
            source_name = source_name.with_suffix(".jpg")

        h, w = self.orig_img.shape[:2]
        crops: list[dict[str, object]] = []
        class_totals: dict[str, int] = {}
        class_seen: dict[str, int] = {}

        for cls_id in self.boxes.cls.tolist():
            name = self.names.get(int(cls_id), "unknown")
            class_totals[name] = class_totals.get(name, 0) + 1

        for i in range(len(self)):
            xyxy = self.boxes.xyxy[i].detach().cpu().numpy()
            x1 = max(0, min(w, int(np.floor(xyxy[0]))))
            y1 = max(0, min(h, int(np.floor(xyxy[1]))))
            x2 = max(0, min(w, int(np.ceil(xyxy[2]))))
            y2 = max(0, min(h, int(np.ceil(xyxy[3]))))

            crop_img = self.orig_img[y1:y2, x1:x2].copy()
            cls_id = int(self.boxes.cls[i])
            name = self.names.get(cls_id, "unknown")
            confidence = round(float(self.boxes.conf[i]), 4)
            save_path: str | None = None

            if save_root is not None and crop_img.size:
                class_seen[name] = class_seen.get(name, 0) + 1
                stem = source_name.stem
                suffix = source_name.suffix or ".jpg"
                crop_file = (
                    f"{stem}_{class_seen[name] - 1}{suffix}"
                    if class_totals[name] > 1
                    else f"{stem}{suffix}"
                )
                path = save_root / name / crop_file
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), crop_img)
                save_path = str(path)

            crops.append(
                {
                    "im": crop_img,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "confidence": confidence,
                    "class": cls_id,
                    "name": name,
                    "save_path": save_path,
                    **({"track_id": int(self.boxes.id[i])} if self.boxes.id is not None else {}),
                }
            )

        return crops

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
            item = {
                "box": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
                "confidence": round(float(self.boxes.conf[i]), 4),
                "class": int(self.boxes.cls[i]),
                "name": self.names.get(int(self.boxes.cls[i]), "unknown"),
            }
            if self.boxes.id is not None:
                item["track_id"] = int(self.boxes.id[i])
            if self.masks is not None and i < len(self.masks):
                polygon = self.masks.xy[i]
                item["segments"] = {
                    "x": polygon[:, 0].tolist(),
                    "y": polygon[:, 1].tolist(),
                }
            out.append(item)
        return out

    def pandas(self) -> pd.DataFrame:
        """Return detections as a pandas DataFrame."""
        return self.to_df()

    def to_df(self) -> pd.DataFrame:
        """Return detections as a pandas DataFrame."""
        import pandas as pd

        columns = ["x1", "y1", "x2", "y2", "confidence", "class", "name"]
        if self.boxes is not None and self.boxes.is_track:
            columns.append("track_id")
        return pd.DataFrame(self._tabular_rows(), columns=columns)

    def to_csv(self, filename: str | PathLike[str], index: bool = False) -> None:
        """Save detections as a CSV file."""
        self.to_df().to_csv(filename, index=index)

    def _tabular_rows(self) -> list[dict[str, object]]:
        """Return detections in a tabular row format for DataFrame/CSV export."""
        rows: list[dict[str, object]] = []
        if self.boxes is None:
            return rows

        for i in range(len(self)):
            xyxy = self.boxes.xyxy[i].tolist()
            cls_id = int(self.boxes.cls[i])
            row = {
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3],
                "confidence": round(float(self.boxes.conf[i]), 4),
                "class": cls_id,
                "name": self.names.get(cls_id, "unknown"),
            }
            if self.boxes.id is not None:
                row["track_id"] = int(self.boxes.id[i])
            rows.append(row)
        return rows

    def __len__(self) -> int:
        return 0 if self.boxes is None else len(self.boxes)

    def __repr__(self) -> str:
        return f"Results(path={self.path!r}, detections={len(self)}, masks={len(self.masks or [])})"


class Masks:
    """Per-instance binary masks with polygon projections."""

    def __init__(self, data, orig_shape: tuple[int, int]) -> None:
        if data.ndim != 3:
            raise ValueError("masks data must have shape [N, H, W]")
        self._data = data
        self.orig_shape = orig_shape

    @property
    def data(self):
        """Raw mask tensor with shape ``[N, H, W]``."""
        return self._data

    @property
    def xy(self) -> list[np.ndarray]:
        """Largest external contour for each mask in absolute pixel coordinates."""
        polygons: list[np.ndarray] = []
        for mask in self._data.detach().cpu().numpy():
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                polygons.append(np.empty((0, 2), dtype=np.float32))
                continue
            contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
            polygons.append(contour.astype(np.float32, copy=False))
        return polygons

    @property
    def xyn(self) -> list[np.ndarray]:
        """Largest external contours normalized to ``[0, 1]``."""
        height, width = self.orig_shape
        scale = np.array([width, height], dtype=np.float32)
        return [polygon / scale for polygon in self.xy]

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __repr__(self) -> str:
        return f"Masks(n={len(self)}, shape={self.orig_shape}, device={self._data.device})"


class Boxes:
    """
    Bounding box container for one image.

    Args:
        data: Tensor [N, 6] for detections (xyxy, conf, cls) or [N, 7]
              for tracks (xyxy, track_id, conf, cls).
        orig_shape: (H, W) of the original image (for normalised coords).
    """

    def __init__(self, data, orig_shape: tuple[int, int]) -> None:
        if data.ndim != 2 or data.shape[1] not in (6, 7):
            raise ValueError("boxes data must have shape [N, 6] or [N, 7]")
        self._data = data
        self.orig_shape = orig_shape  # (H, W)
        self.is_track = data.shape[1] == 7

    @property
    def data(self):
        """Raw [N, 6] detection or [N, 7] tracking tensor."""
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
        return self._data[:, -2]

    @property
    def cls(self):
        """Class indices [N] as int."""
        return self._data[:, -1].int()

    @property
    def id(self):
        """Persistent tracking IDs [N], or ``None`` for detection results."""
        return self._data[:, 4].int() if self.is_track else None

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Boxes(n={len(self)}, device={self._data.device})"
