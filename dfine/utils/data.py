"""
Dataset utilities for fine-tuning and validation.

Supported data YAML styles:

COCO JSON:
    path: /data/my_dataset
    train: images/train
    val:   images/val
    train_ann: annotations/train.json   # optional
    val_ann:   annotations/val.json     # optional

YOLO txt:
    path: /data/my_dataset
    train: images/train                  # labels/train
    val:   images/val

or split-first:
    path: /data/my_dataset
    train: train/images                 # train/labels
    val:   val/images
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dfine.utils.logging import LOGGER

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DetectionSplitSpec:
    format: Literal["coco", "yolo"]
    img_dir: Path
    ann_file: Path
    label_dir: Path | None = None


def load_data_yaml(path: str | Path) -> dict:
    """Load an ultralytics-style data YAML and return it as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_names(cfg: dict) -> dict[int, str]:
    """Return names as a dense int->str mapping."""
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}

    nc = cfg.get("nc")
    if isinstance(nc, int):
        return {i: f"class_{i}" for i in range(nc)}
    raise ValueError("Data YAML must define either 'names' or 'nc'")


def resolve_detection_split(data: str | Path, split: str) -> DetectionSplitSpec:
    """Resolve one split to either a COCO annotation file or YOLO label directory."""
    cfg = load_data_yaml(data)
    root = Path(cfg["path"])
    img_dir = root / cfg[split]

    ann_file = _find_coco_annotation(root, cfg, split, img_dir)
    if ann_file is not None:
        return DetectionSplitSpec(format="coco", img_dir=img_dir, ann_file=ann_file)

    label_dir = _find_yolo_label_dir(root, img_dir, split)
    if label_dir is not None:
        cache_dir = _dataset_cache_dir(root)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{split}_{_slugify_relpath(Path(cfg[split]))}.coco.json"
        ann_file = cache_dir / cache_name
        if not _yolo_cache_is_fresh(ann_file, img_dir, label_dir):
            convert_yolo_split_to_coco_json(img_dir, label_dir, ann_file, normalize_names(cfg))
        return DetectionSplitSpec(
            format="yolo", img_dir=img_dir, ann_file=ann_file, label_dir=label_dir
        )

    raise FileNotFoundError(
        "Could not resolve dataset format for split "
        f"{split!r}. Expected COCO annotations (e.g. train_ann/val_ann or "
        "annotations/instances_*.json) or YOLO labels beside the images "
        "(e.g. images/train + labels/train, or train/images + train/labels)."
    )


class CocoFinetuneDataset(Dataset):
    """
    COCO-format dataset compatible with DFINECriterion.

    Each item returns ``(tensor [3,H,W], target)`` where target is::

        {
            "labels": LongTensor  [N]     — 0-based class indices,
            "boxes":  FloatTensor [N, 4]  — cxcywh normalised to [0, 1],
            "image_id": LongTensor [1],
        }
    """

    def __init__(
        self,
        img_dir: str | Path,
        ann_file: str | Path,
        imgsz: int,
        cat_id_to_label: dict[int, int] | None = None,
    ) -> None:
        from pycocotools.coco import COCO

        self.coco = COCO(str(ann_file))
        self.img_dir = Path(img_dir)
        self.ids = sorted(self.coco.imgs)
        self.transform = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])

        if cat_id_to_label is None:
            sorted_cat_ids = sorted(self.coco.cats)
            cat_id_to_label = {c: i for i, c in enumerate(sorted_cat_ids)}
        self.cat_id_to_label = cat_id_to_label

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.imgs[img_id]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        width, height = img.size

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([(x + w / 2) / width, (y + h / 2) / height, w / width, h / height])
            labels.append(self.cat_id_to_label.get(ann["category_id"], 0))

        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "image_id": torch.tensor([img_id], dtype=torch.long),
        }
        return self.transform(img), target


def _collate(batch):
    """Stack images into [B,C,H,W]; keep targets as a list of dicts."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def build_detection_dataloader(
    data: str | Path,
    split: str,
    imgsz: int,
    batch_size: int,
    spec: DetectionSplitSpec | None = None,
) -> DataLoader:
    """Build a DataLoader from COCO JSON or YOLO txt labels."""
    cfg = load_data_yaml(data)
    spec = spec or resolve_detection_split(data, split)

    cat_ids_cfg = cfg.get("cat_ids")
    cat_id_to_label = {int(k): int(v) for k, v in cat_ids_cfg.items()} if cat_ids_cfg else None

    dataset = CocoFinetuneDataset(
        img_dir=spec.img_dir,
        ann_file=spec.ann_file,
        imgsz=imgsz,
        cat_id_to_label=cat_id_to_label,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        collate_fn=_collate,
        drop_last=(split == "train"),
    )


def build_coco_dataloader(
    data: str | Path,
    split: str,
    imgsz: int,
    batch_size: int,
    spec: DetectionSplitSpec | None = None,
) -> DataLoader:
    """Backward-compatible alias for the generalized detection dataloader."""
    return build_detection_dataloader(data, split, imgsz, batch_size, spec=spec)


def convert_yolo_split_to_coco_json(
    img_dir: str | Path,
    label_dir: str | Path,
    output_file: str | Path,
    names: dict[int, str],
) -> Path:
    """Convert one YOLO split to COCO JSON for downstream train/val reuse."""
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    output_file = Path(output_file)

    images = []
    annotations = []
    ann_id = 1
    image_id = 1

    for image_path in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        with Image.open(image_path) as image:
            width, height = image.size

        rel_image = image_path.relative_to(img_dir).as_posix()
        images.append({"id": image_id, "file_name": rel_image, "width": width, "height": height})

        label_path = label_dir / image_path.relative_to(img_dir).with_suffix(".txt")
        if label_path.exists():
            for ann in _load_yolo_annotations(label_path, width, height, names):
                ann.update({"id": ann_id, "image_id": image_id, "iscrowd": 0})
                annotations.append(ann)
                ann_id += 1

        image_id += 1

    categories = [{"id": class_id + 1, "name": name} for class_id, name in sorted(names.items())]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    return output_file


def _find_coco_annotation(root: Path, cfg: dict, split: str, img_dir: Path) -> Path | None:
    ann_key = f"{split}_ann"
    if ann_key in cfg:
        ann_file = root / cfg[ann_key]
        if not ann_file.exists():
            raise FileNotFoundError(f"Explicit annotation file not found: {ann_file}")
        return ann_file

    split_name = Path(cfg[split]).name
    rel_img_dir = Path(cfg[split])
    split_like_names = {split, split_name}
    if rel_img_dir.parts and rel_img_dir.parts[0] in {"train", "val", "test"}:
        split_like_names.add(rel_img_dir.parts[0])

    candidates = [
        candidate
        for token in split_like_names
        for candidate in (
            root / "annotations" / f"instances_{token}.json",
            img_dir.parent / "annotations" / f"instances_{token}.json",
            root / f"instances_{token}.json",
            img_dir.parent / f"instances_{token}.json",
            img_dir.parent / "annotations" / f"{token}.json",
            root / "annotations" / f"{token}.json",
        )
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_yolo_label_dir(root: Path, img_dir: Path, split: str) -> Path | None:
    rel_img_dir = img_dir.relative_to(root)
    candidates: list[Path] = []

    parts = list(rel_img_dir.parts)
    if "images" in parts:
        image_index = parts.index("images")
        replaced = parts.copy()
        replaced[image_index] = "labels"
        candidates.append(root / Path(*replaced))

    candidates.extend(
        [
            root / "labels" / rel_img_dir.name,
            root / split / "labels",
            img_dir.parent / "labels",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _load_yolo_annotations(
    label_path: Path,
    width: int,
    height: int,
    names: dict[int, str],
) -> list[dict[str, int | float | list[float]]]:
    annotations: list[dict[str, int | float | list[float]]] = []
    raw = label_path.read_text().splitlines()

    for line_no, line in enumerate(raw, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            LOGGER.warning("Skipping malformed YOLO label row %s:%d", label_path, line_no)
            continue

        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            LOGGER.warning("Skipping non-numeric YOLO label row %s:%d", label_path, line_no)
            continue

        if class_id not in names:
            LOGGER.warning(
                "Skipping out-of-range YOLO class id %s in %s:%d", class_id, label_path, line_no
            )
            continue
        if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
            LOGGER.warning("Skipping non-finite YOLO box in %s:%d", label_path, line_no)
            continue
        if bw <= 0 or bh <= 0:
            LOGGER.warning("Skipping non-positive YOLO box in %s:%d", label_path, line_no)
            continue

        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height

        x1 = min(max(x1, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        x2 = min(max(x2, 0.0), float(width))
        y2 = min(max(y2, 0.0), float(height))

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue

        x1 = round(x1, 6)
        y1 = round(y1, 6)
        box_w = round(box_w, 6)
        box_h = round(box_h, 6)

        annotations.append(
            {
                "category_id": class_id + 1,
                "bbox": [x1, y1, box_w, box_h],
                "area": round(box_w * box_h, 6),
            }
        )

    return annotations


def _slugify_relpath(path: Path) -> str:
    return "__".join(path.parts) or "root"


def _dataset_cache_dir(root: Path) -> Path:
    root_hash = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:12]
    return _nitid_cache_root() / "datasets" / root_hash


def _nitid_cache_root() -> Path:
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "nitid"
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / "nitid"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "nitid"
    return Path.home() / ".cache" / "nitid"


def _yolo_cache_is_fresh(ann_file: Path, img_dir: Path, label_dir: Path) -> bool:
    if not ann_file.exists():
        return False

    cache_mtime = ann_file.stat().st_mtime
    newest_source_mtime = max(
        _latest_tree_mtime(img_dir),
        _latest_tree_mtime(label_dir),
    )
    return cache_mtime >= newest_source_mtime


def _latest_tree_mtime(root: Path) -> float:
    mtimes = [root.stat().st_mtime]
    for path in root.rglob("*"):
        mtimes.append(path.stat().st_mtime)
    return max(mtimes)
