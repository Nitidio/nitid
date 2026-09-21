# DOTA parsing and OBB target conventions adapted from RiO-DETR (Apache-2.0).
# Source: https://github.com/RicePasteM/RiO-DETR
# Modified for native dataset integration into nitid in 2026.
# See THIRD_PARTY_NOTICES.md for upstream attribution.
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

Semantic masks:
    path: /data/my_dataset
    train: images/train
    val:   images/val
    train_masks: labels/train          # optional when inferable from images path
    val_masks: labels/val
    ignore_index: 255

OBB labels:
    # YOLO-OBB normalized polygons: class x1 y1 x2 y2 x3 y3 x4 y4
    path: /data/obb
    train: images/train
    val: images/val

    # DOTA text polygons: x1 y1 x2 y2 x3 y3 x4 y4 class [difficulty]
    path: /data/obb
    train: images/train
    val: images/val
    obb_format: dota
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from dfine.utils.augmentations import (
    AugmentationConfig,
    color_jitter_hsv,
    horizontal_flip,
    horizontal_flip_keypoints,
    horizontal_flip_masks,
    random_crop,
    random_crop_instances,
    random_crop_semantic,
    resize_masks,
    resize_semantic_mask,
    sanitize,
    sanitize_instances,
    scale_translate,
    scale_translate_instances,
    scale_translate_keypoints,
    scale_translate_semantic,
    stretch_resize,
    stretch_resize_keypoints,
    to_tensor,
)
from dfine.utils.logging import LOGGER

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DetectionSplitSpec:
    format: Literal["coco", "yolo"]
    img_dir: Path
    ann_file: Path
    label_dir: Path | None = None


@dataclass(frozen=True)
class SemanticSplitSpec:
    """Resolved image and dense-mask directories for one semantic split."""

    img_dir: Path
    mask_dir: Path


@dataclass(frozen=True)
class OBBSplitSpec:
    """Resolved image/label resources for one oriented-box split."""

    format: Literal["coco", "yolo_obb", "dota"]
    img_dir: Path
    ann_file: Path | None = None
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
    root_value = cfg.get("path")
    if not isinstance(root_value, (str, Path)):
        raise ValueError("Data YAML must define a dataset 'path'")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (Path(data).resolve().parent / root).resolve()
    img_dir = root / cfg[split]

    ann_file = _find_coco_annotation(root, cfg, split, img_dir)
    if ann_file is not None:
        return DetectionSplitSpec(format="coco", img_dir=img_dir, ann_file=ann_file)

    label_dir = _find_yolo_label_dir(root, img_dir, split)
    if label_dir is not None:
        cache_dir = _dataset_cache_dir(root)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{split}_{_slugify_relpath(Path(cfg[split]))}.v2.coco.json"
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


def resolve_semantic_split(data: str | Path, split: str) -> SemanticSplitSpec:
    """Resolve a semantic split using explicit mask paths or images/labels mirroring."""
    cfg = load_data_yaml(data)
    if split not in cfg:
        raise KeyError(f"Data YAML does not define split {split!r}")
    root_value = cfg.get("path")
    if not isinstance(root_value, (str, Path)):
        raise ValueError("Data YAML must define a dataset 'path'")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (Path(data).resolve().parent / root).resolve()

    split_value = cfg[split]
    if not isinstance(split_value, (str, Path)):
        raise ValueError(f"Data YAML {split!r} must be a directory path")
    img_dir = root / split_value
    mask_value = cfg.get(f"{split}_masks")
    if mask_value is not None:
        if not isinstance(mask_value, (str, Path)):
            raise ValueError(f"Data YAML {split}_masks must be a directory path")
        mask_dir = root / mask_value
    else:
        relative = Path(split_value)
        parts = list(relative.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            mask_dir = root / Path(*parts)
        elif relative.name == "images":
            mask_dir = root / relative.parent / "labels"
        else:
            mask_dir = root / "labels" / relative.name

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Semantic image directory not found: {img_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Semantic mask directory not found: {mask_dir}")
    return SemanticSplitSpec(img_dir=img_dir, mask_dir=mask_dir)


def resolve_obb_split(data: str | Path, split: str) -> OBBSplitSpec:
    """Resolve an OBB split to COCO polygons, YOLO-OBB txt, or DOTA txt labels."""
    cfg = load_data_yaml(data)
    if split not in cfg:
        raise KeyError(f"Data YAML does not define split {split!r}")
    root_value = cfg.get("path")
    if not isinstance(root_value, (str, Path)):
        raise ValueError("Data YAML must define a dataset 'path'")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (Path(data).resolve().parent / root).resolve()

    img_dir = root / cfg[split]
    if not img_dir.is_dir():
        raise FileNotFoundError(f"OBB image directory not found: {img_dir}")

    ann_file = _find_coco_annotation(root, cfg, split, img_dir)
    if ann_file is not None:
        return OBBSplitSpec(format="coco", img_dir=img_dir, ann_file=ann_file)

    label_dir = _find_yolo_label_dir(root, img_dir, split)
    if label_dir is None:
        raise FileNotFoundError(
            "Could not resolve OBB labels. Expected COCO polygon annotations, "
            "YOLO-OBB labels beside images, or DOTA text labels in labels/<split>."
        )

    obb_format = str(cfg.get("obb_format", cfg.get("format", "yolo_obb"))).lower()
    if obb_format in {"dota", "dota_txt", "dota-obb"}:
        return OBBSplitSpec(format="dota", img_dir=img_dir, label_dir=label_dir)
    if obb_format in {"yolo", "yolo_obb", "yolo-obb", "ultralytics"}:
        return OBBSplitSpec(format="yolo_obb", img_dir=img_dir, label_dir=label_dir)
    raise ValueError(
        f"Unsupported OBB dataset format {obb_format!r}; choose 'yolo_obb', 'dota', or COCO JSON"
    )


class SemanticSegmentationDataset(Dataset):
    """Image/dense-PNG pairs for semantic segmentation.

    Every mask is a single-channel integer class map. Its relative path mirrors
    the image path and its suffix is always ``.png``.
    """

    def __init__(
        self,
        spec: SemanticSplitSpec,
        imgsz: int,
        num_classes: int,
        ignore_index: int = 255,
        cache: bool | str = False,
        augment: AugmentationConfig | None = None,
        seed: int = 0,
        retain_original_mask: bool = False,
    ) -> None:
        if imgsz < 1:
            raise ValueError(f"imgsz must be positive, got {imgsz}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if 0 <= ignore_index < num_classes:
            raise ValueError(
                f"ignore_index={ignore_index} overlaps valid class IDs [0, {num_classes - 1}]"
            )

        self.spec = spec
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.augment = augment
        self.seed = seed
        self.retain_original_mask = retain_original_mask
        self.epoch = 0
        self.mosaic_enabled = False
        self.image_paths = sorted(
            path for path in spec.img_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No supported images found in semantic split: {spec.img_dir}")
        self.mask_paths = [
            spec.mask_dir / path.relative_to(spec.img_dir).with_suffix(".png")
            for path in self.image_paths
        ]
        missing = [path for path in self.mask_paths if not path.is_file()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            suffix = " ..." if len(missing) > 3 else ""
            raise FileNotFoundError(f"Missing {len(missing)} semantic mask(s): {preview}{suffix}")

        self.cache = cache
        self._cache: dict[int, tuple[Image.Image, torch.Tensor]] = {}
        if cache is True or str(cache).lower() == "ram":
            for index in range(len(self.image_paths)):
                self._cache[index] = self._load_pair(index)
        elif cache not in (False, None, "false"):
            raise ValueError("cache must be False, True, or 'ram'")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cached = self._cache.get(index)
        image, original_mask = self._load_pair(index) if cached is None else cached
        image = image.copy()
        original_mask = original_mask.clone()
        original_height, original_width = original_mask.shape
        mask = resize_semantic_mask(original_mask, self.imgsz)
        image = image.resize((self.imgsz, self.imgsz), Image.Resampling.BILINEAR)

        rng = random.Random(self.seed + self.epoch * max(len(self), 1) + index)
        cfg = self.augment
        if cfg and cfg.enabled:
            if cfg.fliplr and rng.random() < cfg.fliplr:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = torch.flip(mask, dims=[1])
            if cfg.scale or cfg.translate:
                image, mask = scale_translate_semantic(
                    image,
                    mask,
                    cfg.scale,
                    cfg.translate,
                    rng,
                    self.ignore_index,
                )
            if cfg.crop and rng.random() < cfg.crop:
                image, mask = random_crop_semantic(image, mask, cfg.crop, rng)
                image = image.resize((self.imgsz, self.imgsz), Image.Resampling.BILINEAR)
                mask = resize_semantic_mask(mask, self.imgsz)
            image = color_jitter_hsv(image, cfg, rng)

        target = {
            "sem_mask": mask.long(),
            "orig_size": torch.tensor([original_height, original_width], dtype=torch.long),
            "image_id": torch.tensor([index], dtype=torch.long),
        }
        if self.retain_original_mask:
            target["orig_mask"] = original_mask.long()
        return to_tensor(image), target

    def set_epoch(self, epoch: int, mosaic: bool = True) -> None:
        del mosaic
        self.epoch = epoch

    def _load_pair(self, index: int) -> tuple[Image.Image, torch.Tensor]:
        with Image.open(self.image_paths[index]) as image_file:
            image = image_file.convert("RGB").copy()
        with Image.open(self.mask_paths[index]) as mask_file:
            mask_array = np.asarray(mask_file).copy()
        if mask_array.ndim != 2 or not np.issubdtype(mask_array.dtype, np.integer):
            raise ValueError(
                f"Semantic mask must be a single-channel integer image: {self.mask_paths[index]}"
            )
        if mask_array.shape != (image.height, image.width):
            raise ValueError(
                "Semantic image and mask dimensions differ for "
                f"{self.image_paths[index]}: image={(image.height, image.width)}, "
                f"mask={mask_array.shape}"
            )
        invalid = (mask_array != self.ignore_index) & (
            (mask_array < 0) | (mask_array >= self.num_classes)
        )
        if invalid.any():
            invalid_id = int(mask_array[invalid][0])
            raise ValueError(
                f"Semantic mask {self.mask_paths[index]} contains class ID {invalid_id}; "
                f"expected [0, {self.num_classes - 1}] or ignore_index={self.ignore_index}"
            )
        return image, torch.from_numpy(mask_array.astype(np.int64, copy=False))


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
        classes: list[int] | None = None,
        single_cls: bool = False,
        cache: bool | str = False,
        augment: AugmentationConfig | None = None,
        seed: int = 0,
        task: Literal["detect", "segment", "pose"] = "detect",
    ) -> None:
        from pycocotools.coco import COCO

        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(str(ann_file))
        self.img_dir = Path(img_dir)
        self.ids = sorted(self.coco.imgs)
        self.imgsz = imgsz
        self.augment = augment
        self.mosaic_enabled = True
        self.seed = seed
        self.epoch = 0
        self.task = task

        if cat_id_to_label is None:
            sorted_cat_ids = sorted(self.coco.cats)
            cat_id_to_label = {c: i for i, c in enumerate(sorted_cat_ids)}
        self.cat_id_to_label = cat_id_to_label
        self.classes = set(classes) if classes is not None else None
        self.single_cls = single_cls
        self.cache = cache
        self._image_cache: dict[int, Image.Image] = {}
        if self.task == "segment" and not any(
            ann.get("segmentation") for ann in self.coco.anns.values()
        ):
            raise ValueError(
                "task='segment' requires polygon or RLE instance annotations; "
                "the selected split contains bounding boxes only"
            )
        if self.task == "pose" and not any(
            ann.get("keypoints") and int(ann.get("num_keypoints", 0)) > 0
            for ann in self.coco.anns.values()
        ):
            raise ValueError(
                "task='pose' requires COCO keypoint annotations; "
                "the selected split contains no visible keypoints"
            )
        if (
            self.task == "pose"
            and augment
            and augment.enabled
            and (augment.mosaic > 0 or augment.mixup > 0)
        ):
            raise ValueError("Pose training does not support mosaic or mixup augmentations")
        if cache is True or str(cache).lower() == "ram":
            for index in range(len(self.ids)):
                self._image_cache[index] = self._load_image(index)
        elif cache not in (False, None, "false"):
            raise ValueError("cache must be False, True, or 'ram'")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image, boxes, labels, masks, keypoints, areas, img_id = self._load_item(idx)
        rng = random.Random(self.seed + self.epoch * max(len(self), 1) + idx)
        cfg = self.augment
        mosaic_active = bool(
            cfg
            and cfg.enabled
            and self.task != "pose"
            and self.mosaic_enabled
            and cfg.mosaic > 0
            and rng.random() < cfg.mosaic
        )
        if mosaic_active:
            image, boxes, labels, masks = self._mosaic(idx, rng)
            keypoints = torch.zeros((0, 0), dtype=torch.float32)
            areas = torch.zeros((len(labels),), dtype=torch.float32)
        else:
            orig_w, orig_h = image.size
            image, boxes = stretch_resize(image, boxes, self.imgsz)
            if self.task == "pose":
                keypoints = stretch_resize_keypoints(keypoints, orig_w, orig_h, self.imgsz)
                areas = (
                    areas
                    * (self.imgsz / max(1.0, float(orig_w)))
                    * (self.imgsz / max(1.0, float(orig_h)))
                )
            masks = resize_masks(masks, self.imgsz)

        if cfg and cfg.enabled:
            if cfg.fliplr and rng.random() < cfg.fliplr:
                image, boxes = horizontal_flip(image, boxes)
                masks = horizontal_flip_masks(masks)
                if self.task == "pose":
                    keypoints = horizontal_flip_keypoints(keypoints, width=self.imgsz)
            if cfg.scale or cfg.translate:
                if self.task == "segment":
                    image, boxes, masks = scale_translate_instances(
                        image, boxes, masks, cfg.scale, cfg.translate, rng
                    )
                elif self.task == "pose":
                    width, height = image.size
                    factor = rng.uniform(1.0 - cfg.scale, 1.0 + cfg.scale)
                    new_w, new_h = max(1, round(width * factor)), max(1, round(height * factor))
                    tx = round(rng.uniform(-cfg.translate, cfg.translate) * width)
                    ty = round(rng.uniform(-cfg.translate, cfg.translate) * height)
                    left, top = (width - new_w) // 2 + tx, (height - new_h) // 2 + ty
                    image, boxes = scale_translate_with_params(image, boxes, factor, left, top)
                    keypoints = scale_translate_keypoints(keypoints, factor, left, top)
                    areas = areas * factor * factor
                else:
                    image, boxes = scale_translate(image, boxes, cfg.scale, cfg.translate, rng)
            if cfg.crop and rng.random() < cfg.crop:
                if self.task == "segment":
                    image, boxes, masks, keep = random_crop_instances(
                        image, boxes, masks, cfg.crop, rng
                    )
                elif self.task == "pose":
                    image, boxes, keypoints, keep = random_crop_pose(
                        image, boxes, keypoints, cfg.crop, rng
                    )
                else:
                    image, boxes, keep = random_crop(image, boxes, cfg.crop, rng)
                labels = labels[keep]
                areas = areas[keep]
                crop_w, crop_h = image.size
                image, boxes = stretch_resize(image, boxes, self.imgsz)
                if self.task == "pose":
                    keypoints = stretch_resize_keypoints(keypoints, crop_w, crop_h, self.imgsz)
                    areas = (
                        areas
                        * (self.imgsz / max(1.0, float(crop_w)))
                        * (self.imgsz / max(1.0, float(crop_h)))
                    )
                masks = resize_masks(masks, self.imgsz)
            image = color_jitter_hsv(image, cfg, rng)
            if cfg.mixup and rng.random() < cfg.mixup:
                other_idx = rng.randrange(len(self))
                other_image, other_boxes, other_labels, other_masks, _, _, _ = self._load_item(
                    other_idx
                )
                other_image, other_boxes = stretch_resize(other_image, other_boxes, self.imgsz)
                other_masks = resize_masks(other_masks, self.imgsz)
                ratio = rng.betavariate(32.0, 32.0)
                image = Image.blend(image, other_image, 1.0 - ratio)
                boxes = torch.cat((boxes, other_boxes))
                labels = torch.cat((labels, other_labels))
                masks = torch.cat((masks, other_masks))

        if self.task == "segment":
            boxes, labels, masks = sanitize_instances(boxes, labels, masks, self.imgsz, self.imgsz)
        elif self.task == "pose":
            boxes, labels, keypoints, areas = sanitize_pose(
                boxes, labels, keypoints, areas, self.imgsz, self.imgsz
            )
        else:
            boxes, labels = sanitize(boxes, labels, self.imgsz, self.imgsz)
        boxes = self._normalize_boxes(boxes)
        target = {
            "labels": labels,
            "boxes": boxes,
            "image_id": torch.tensor([img_id], dtype=torch.long),
        }
        if self.task == "segment":
            target["masks"] = masks
        if self.task == "pose":
            target["keypoints"] = normalize_keypoints_for_pose_loss(keypoints, self.imgsz)
            target["area"] = (areas / float(self.imgsz * self.imgsz)).clamp(min=1e-6)
        return to_tensor(image), target

    def _load_item(
        self, idx: int
    ) -> tuple[
        Image.Image,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        img_id = self.ids[idx]
        image = self._image_cache.get(idx)
        if image is None:
            image = self._load_image(idx)
        width, height = image.size
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        box_values, label_values, mask_values, keypoint_values, area_values = [], [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            label = self.cat_id_to_label.get(ann["category_id"], 0)
            if self.classes is not None and label not in self.classes:
                continue
            if self.task == "segment" and not ann.get("segmentation"):
                continue
            if self.task == "pose":
                raw_keypoints = ann.get("keypoints")
                if not raw_keypoints or int(ann.get("num_keypoints", 0)) <= 0:
                    continue
                keypoint_values.append(raw_keypoints)
                area_values.append(float(ann.get("area", w * h)))
            box_values.append([x, y, x + w, y + h])
            label_values.append(0 if self.single_cls else label)
            if self.task == "segment":
                mask_values.append(torch.from_numpy(self.coco.annToMask(ann)).to(torch.uint8))
        boxes = torch.tensor(box_values, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(label_values, dtype=torch.long)
        masks = (
            torch.stack(mask_values)
            if mask_values
            else torch.zeros((0, height, width), dtype=torch.uint8)
        )
        keypoints = (
            torch.tensor(keypoint_values, dtype=torch.float32).reshape(len(keypoint_values), -1)
            if keypoint_values
            else torch.zeros((0, 0), dtype=torch.float32)
        )
        areas = torch.tensor(area_values, dtype=torch.float32)
        return image.copy(), boxes, labels, masks, keypoints, areas, img_id

    def _mosaic(
        self, idx: int, rng: random.Random
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor, torch.Tensor]:
        half = self.imgsz // 2
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        indices = [idx, *(rng.randrange(len(self)) for _ in range(3))]
        all_boxes, all_labels, all_masks = [], [], []
        offsets = ((0, 0), (half, 0), (0, half), (half, half))
        for item_idx, (left, top) in zip(indices, offsets):
            image, boxes, labels, masks, _, _, _ = self._load_item(item_idx)
            image, boxes = stretch_resize(image, boxes, half)
            masks = resize_masks(masks, half)
            canvas.paste(image, (left, top))
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            all_boxes.append(boxes)
            all_labels.append(labels)
            placed_masks = torch.zeros((len(masks), self.imgsz, self.imgsz), dtype=torch.uint8)
            placed_masks[:, top : top + half, left : left + half] = masks
            all_masks.append(placed_masks)
        return canvas, torch.cat(all_boxes), torch.cat(all_labels), torch.cat(all_masks)

    def _normalize_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        result = boxes.clone()
        if result.numel():
            xyxy = result.clone()
            result[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2 / self.imgsz
            result[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2 / self.imgsz
            result[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) / self.imgsz
            result[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) / self.imgsz
        return result

    def set_epoch(self, epoch: int, mosaic: bool = True) -> None:
        """Select the deterministic transform stream and optionally disable mosaic."""
        self.epoch = epoch
        self.mosaic_enabled = mosaic

    def _load_image(self, idx: int) -> Image.Image:
        info = self.coco.imgs[self.ids[idx]]
        with Image.open(self.img_dir / info["file_name"]) as image:
            return image.convert("RGB").copy()


class OBBFinetuneDataset(Dataset):
    """Oriented-box dataset normalized to RiO-DETR ``cx, cy, w, h, angle`` targets."""

    def __init__(
        self,
        spec: OBBSplitSpec,
        imgsz: int,
        names: dict[int, str],
        classes: list[int] | None = None,
        single_cls: bool = False,
        cache: bool | str = False,
        augment: AugmentationConfig | None = None,
        seed: int = 0,
    ) -> None:
        if imgsz < 1:
            raise ValueError(f"imgsz must be positive, got {imgsz}")
        self.spec = spec
        self.imgsz = imgsz
        self.names = names
        self.classes = set(classes) if classes is not None else None
        self.single_cls = single_cls
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.mosaic_enabled = False
        self.image_paths = sorted(
            path for path in spec.img_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No supported images found in OBB split: {spec.img_dir}")
        self._coco = None
        self._coco_ids: list[int] = []
        self._cat_id_to_label: dict[int, int] = {}
        if spec.format == "coco":
            if spec.ann_file is None:
                raise ValueError("COCO OBB split requires ann_file")
            from pycocotools.coco import COCO

            with contextlib.redirect_stdout(io.StringIO()):
                self._coco = COCO(str(spec.ann_file))
            self._coco_ids = sorted(self._coco.imgs)
            self._cat_id_to_label = {cat_id: i for i, cat_id in enumerate(sorted(self._coco.cats))}
            self.image_paths = [
                spec.img_dir / self._coco.imgs[img_id]["file_name"] for img_id in self._coco_ids
            ]

        self._image_cache: dict[int, Image.Image] = {}
        if cache is True or str(cache).lower() == "ram":
            for index in range(len(self.image_paths)):
                self._image_cache[index] = self._load_image(index)
        elif cache not in (False, None, "false"):
            raise ValueError("cache must be False, True, or 'ram'")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = self._image_cache.get(idx)
        if image is None:
            image = self._load_image(idx)
        polygons, labels, img_id = self._load_annotations(idx, image.size)
        width, height = image.size
        polygons = polygons.clone()
        if polygons.numel():
            polygons[..., 0] *= self.imgsz / width
            polygons[..., 1] *= self.imgsz / height
        image = image.resize((self.imgsz, self.imgsz), Image.Resampling.BILINEAR)

        rng = random.Random(self.seed + self.epoch * max(len(self), 1) + idx)
        cfg = self.augment
        if cfg and cfg.enabled:
            if cfg.fliplr and rng.random() < cfg.fliplr and polygons.numel():
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                polygons[..., 0] = self.imgsz - polygons[..., 0]
            if cfg.hsv_h or cfg.hsv_s or cfg.hsv_v:
                image = color_jitter_hsv(image, cfg, rng)

        polygons, labels = _sanitize_obb_polygons(polygons, labels, self.imgsz)
        boxes = _polygons_to_xywhr(polygons)
        if boxes.numel():
            boxes[:, :4] /= float(self.imgsz)
            boxes[:, 4] /= math.pi
        target = {
            "labels": labels,
            "boxes": boxes,
            "image_id": torch.tensor([img_id], dtype=torch.long),
        }
        return to_tensor(image), target

    def set_epoch(self, epoch: int, mosaic: bool = True) -> None:
        del mosaic
        self.epoch = epoch

    def _load_image(self, idx: int) -> Image.Image:
        with Image.open(self.image_paths[idx]) as image:
            return image.convert("RGB").copy()

    def _load_annotations(
        self, idx: int, image_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.spec.format == "coco":
            return self._load_coco_annotations(idx)
        if self.spec.label_dir is None:
            raise ValueError("Text OBB split requires label_dir")
        label_path = self.spec.label_dir / self.image_paths[idx].relative_to(
            self.spec.img_dir
        ).with_suffix(".txt")
        if self.spec.format == "dota":
            return self._load_dota_annotations(label_path, idx)
        return self._load_yolo_obb_annotations(label_path, image_size, idx)

    def _load_coco_annotations(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        assert self._coco is not None
        img_id = self._coco_ids[idx]
        anns = self._coco.loadAnns(self._coco.getAnnIds(imgIds=img_id, iscrowd=False))
        polygons: list[list[list[float]]] = []
        labels: list[int] = []
        for ann in anns:
            label = self._cat_id_to_label.get(int(ann["category_id"]), 0)
            if self.classes is not None and label not in self.classes:
                continue
            segmentation = ann.get("segmentation")
            polygon = _first_polygon(segmentation)
            if polygon is None:
                bbox = ann.get("bbox")
                if not bbox:
                    continue
                x, y, w, h = [float(value) for value in bbox]
                polygon = [x, y, x + w, y, x + w, y + h, x, y + h]
            polygons.append([[polygon[i], polygon[i + 1]] for i in range(0, 8, 2)])
            labels.append(0 if self.single_cls else label)
        return _obb_tensor(polygons), torch.tensor(labels, dtype=torch.long), int(img_id)

    def _load_yolo_obb_annotations(
        self, label_path: Path, image_size: tuple[int, int], idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        width, height = image_size
        polygons: list[list[list[float]]] = []
        labels: list[int] = []
        if not label_path.exists():
            return _obb_tensor(polygons), torch.zeros((0,), dtype=torch.long), idx
        for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) != 9:
                LOGGER.warning("Skipping malformed YOLO-OBB label row %s:%d", label_path, line_no)
                continue
            try:
                class_id = int(float(parts[0]))
                values = [float(value) for value in parts[1:]]
            except ValueError:
                LOGGER.warning("Skipping non-numeric YOLO-OBB label row %s:%d", label_path, line_no)
                continue
            if class_id not in self.names or (
                self.classes is not None and class_id not in self.classes
            ):
                continue
            points = [[values[i] * width, values[i + 1] * height] for i in range(0, 8, 2)]
            polygons.append(points)
            labels.append(0 if self.single_cls else class_id)
        return _obb_tensor(polygons), torch.tensor(labels, dtype=torch.long), idx

    def _load_dota_annotations(
        self, label_path: Path, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        name_to_id = {name: class_id for class_id, name in self.names.items()}
        polygons: list[list[list[float]]] = []
        labels: list[int] = []
        if not label_path.exists():
            return _obb_tensor(polygons), torch.zeros((0,), dtype=torch.long), idx
        for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) < 9:
                LOGGER.warning("Skipping malformed DOTA label row %s:%d", label_path, line_no)
                continue
            try:
                values = [float(value) for value in parts[:8]]
            except ValueError:
                LOGGER.warning("Skipping non-numeric DOTA label row %s:%d", label_path, line_no)
                continue
            class_id = name_to_id.get(parts[8])
            if class_id is None or (self.classes is not None and class_id not in self.classes):
                continue
            polygons.append([[values[i], values[i + 1]] for i in range(0, 8, 2)])
            labels.append(0 if self.single_cls else class_id)
        return _obb_tensor(polygons), torch.tensor(labels, dtype=torch.long), idx


def _first_polygon(segmentation: object) -> list[float] | None:
    if not isinstance(segmentation, list) or not segmentation:
        return None
    candidate = segmentation[0]
    if not isinstance(candidate, list) or len(candidate) < 8:
        return None
    values = [float(value) for value in candidate[:8]]
    return values if all(math.isfinite(value) for value in values) else None


def _obb_tensor(polygons: list[list[list[float]]]) -> torch.Tensor:
    if not polygons:
        return torch.zeros((0, 4, 2), dtype=torch.float32)
    return torch.tensor(polygons, dtype=torch.float32).reshape(-1, 4, 2)


def _sanitize_obb_polygons(
    polygons: torch.Tensor,
    labels: torch.Tensor,
    imgsz: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if polygons.numel() == 0:
        return polygons.reshape(0, 4, 2), labels.reshape(0)
    polygons = polygons.clone()
    polygons[..., 0].clamp_(0, imgsz)
    polygons[..., 1].clamp_(0, imgsz)
    finite = torch.isfinite(polygons).flatten(1).all(dim=1)
    widths = polygons[..., 0].max(dim=1).values - polygons[..., 0].min(dim=1).values
    heights = polygons[..., 1].max(dim=1).values - polygons[..., 1].min(dim=1).values
    keep = finite & (widths > 1.0) & (heights > 1.0)
    return polygons[keep], labels[keep]


def _polygons_to_xywhr(polygons: torch.Tensor) -> torch.Tensor:
    """Convert four-point polygons to ``cx, cy, w, h, angle_radians``."""
    if polygons.numel() == 0:
        return torch.zeros((0, 5), dtype=torch.float32)
    boxes: list[list[float]] = []
    polygon_array = cast(
        npt.NDArray[np.float32], polygons.detach().cpu().numpy().astype(np.float32)
    )
    for polygon in polygon_array:
        (cx, cy), (w, h), angle_degrees = cv2.minAreaRect(polygon)
        if w < h:
            w, h = h, w
            angle_degrees += 90.0
        angle = math.radians(angle_degrees)
        angle = ((angle + math.pi) % (2 * math.pi)) - math.pi
        boxes.append([float(cx), float(cy), float(w), float(h), float(angle)])
    return torch.tensor(boxes, dtype=torch.float32)


def _collate(batch):
    """Stack images into [B,C,H,W]; keep targets as a list of dicts."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def scale_translate_with_params(
    image: Image.Image,
    boxes: torch.Tensor,
    factor: float,
    left: int,
    top: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, torch.Tensor]:
    """Apply a fixed scale/translate transform shared by boxes and keypoints."""
    width, height = image.size
    new_w, new_h = max(1, round(width * factor)), max(1, round(height * factor))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(resized, (left, top))
    result = boxes.clone()
    if result.numel():
        result[:, [0, 2]] = result[:, [0, 2]] * factor + left
        result[:, [1, 3]] = result[:, [1, 3]] * factor + top
        result[:, [0, 2]].clamp_(0, width)
        result[:, [1, 3]].clamp_(0, height)
    return canvas, result


def random_crop_pose(
    image: Image.Image,
    boxes: torch.Tensor,
    keypoints: torch.Tensor,
    gain: float,
    rng: random.Random,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop an image while keeping pose boxes and keypoints aligned."""
    width, height = image.size
    left = round(rng.uniform(0.0, gain) * width)
    right = round(rng.uniform(0.0, gain) * width)
    top = round(rng.uniform(0.0, gain) * height)
    bottom = round(rng.uniform(0.0, gain) * height)
    if left + right >= width or top + bottom >= height:
        return image, boxes, keypoints, torch.ones(len(boxes), dtype=torch.bool)

    result_boxes = boxes.clone()
    result_boxes[:, [0, 2]] -= left
    result_boxes[:, [1, 3]] -= top
    crop_w, crop_h = width - left - right, height - top - bottom
    result_boxes[:, [0, 2]].clamp_(0, crop_w)
    result_boxes[:, [1, 3]].clamp_(0, crop_h)
    keep = (result_boxes[:, 2] - result_boxes[:, 0] >= 1.0) & (
        result_boxes[:, 3] - result_boxes[:, 1] >= 1.0
    )

    result_keypoints = keypoints.clone()
    if result_keypoints.numel():
        view = result_keypoints.view(result_keypoints.shape[0], -1, 3)
        view[..., 0] -= left
        view[..., 1] -= top
        outside = (
            (view[..., 0] < 0)
            | (view[..., 0] > crop_w)
            | (view[..., 1] < 0)
            | (view[..., 1] > crop_h)
        )
        view[..., 2] = torch.where(outside, torch.zeros_like(view[..., 2]), view[..., 2])
        view[..., 0].clamp_(0, crop_w)
        view[..., 1].clamp_(0, crop_h)

    return (
        image.crop((left, top, width - right, height - bottom)),
        result_boxes[keep],
        result_keypoints[keep],
        keep,
    )


def sanitize_pose(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    keypoints: torch.Tensor,
    areas: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip boxes and remove invalid pose instances with no visible keypoints."""
    boxes = boxes.clone()
    if boxes.numel():
        boxes[:, [0, 2]].clamp_(0, width)
        boxes[:, [1, 3]].clamp_(0, height)
    keep = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
    keep = keep & torch.isfinite(boxes).all(dim=1)
    if keypoints.numel():
        view = keypoints.view(keypoints.shape[0], -1, 3)
        outside = (
            (view[..., 0] < 0)
            | (view[..., 0] > width)
            | (view[..., 1] < 0)
            | (view[..., 1] > height)
        )
        view[..., 2] = torch.where(outside, torch.zeros_like(view[..., 2]), view[..., 2])
        view[..., 0].clamp_(0, width)
        view[..., 1].clamp_(0, height)
        keep = keep & (view[..., 2] > 0).any(dim=1)
    else:
        keep = keep & torch.zeros_like(keep)
    return boxes[keep], labels[keep], keypoints[keep], areas[keep]


def normalize_keypoints_for_pose_loss(keypoints: torch.Tensor, image_size: int) -> torch.Tensor:
    """Convert COCO interleaved ``x,y,v`` keypoints to DETRPose loss format."""
    if not keypoints.numel():
        return torch.zeros((0, 0), dtype=torch.float32, device=keypoints.device)
    view = keypoints.view(keypoints.shape[0], -1, 3)
    xy = view[..., :2].reshape(keypoints.shape[0], -1) / float(image_size)
    visibility = (view[..., 2] > 0).to(dtype=keypoints.dtype)
    return torch.cat([xy, visibility], dim=1)


def build_detection_dataloader(
    data: str | Path,
    split: str,
    imgsz: int,
    batch_size: int,
    spec: DetectionSplitSpec | None = None,
    workers: int = 0,
    cache: bool | str = False,
    seed: int = 0,
    deterministic: bool = True,
    classes: list[int] | None = None,
    single_cls: bool = False,
    fraction: float = 1.0,
    augment: AugmentationConfig | None = None,
    task: Literal["detect", "segment", "pose"] = "detect",
) -> DataLoader:
    """Build a DataLoader from COCO JSON or YOLO txt labels."""
    cfg = load_data_yaml(data)
    spec = spec or resolve_detection_split(data, split)

    cat_ids_cfg = cfg.get("cat_ids")
    cat_id_to_label = {int(k): int(v) for k, v in cat_ids_cfg.items()} if cat_ids_cfg else None

    base_dataset = CocoFinetuneDataset(
        img_dir=spec.img_dir,
        ann_file=spec.ann_file,
        imgsz=imgsz,
        cat_id_to_label=cat_id_to_label,
        classes=classes,
        single_cls=single_cls,
        cache=cache,
        augment=augment if split == "train" else None,
        seed=seed,
        task=task,
    )

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in the range (0, 1]")
    dataset: Dataset = base_dataset
    dataset_size = len(base_dataset)
    if fraction < 1.0:
        count = max(1, int(len(base_dataset) * fraction))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(base_dataset), generator=generator)[:count].tolist()
        dataset = Subset(base_dataset, indices)
        dataset_size = count

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        collate_fn=_collate,
        drop_last=split == "train" and dataset_size >= batch_size,
        generator=generator,
        worker_init_fn=_seed_worker if workers > 0 and deterministic else None,
    )


def build_obb_dataloader(
    data: str | Path,
    split: str,
    imgsz: int,
    batch_size: int,
    spec: OBBSplitSpec | None = None,
    workers: int = 0,
    cache: bool | str = False,
    seed: int = 0,
    deterministic: bool = True,
    classes: list[int] | None = None,
    single_cls: bool = False,
    fraction: float = 1.0,
    augment: AugmentationConfig | None = None,
) -> DataLoader:
    """Build a DataLoader for YOLO-OBB, DOTA, or COCO polygon OBB datasets."""
    cfg = load_data_yaml(data)
    names = normalize_names(cfg)
    if augment and augment.enabled and (augment.mosaic > 0 or augment.mixup > 0):
        raise ValueError("OBB training does not support mosaic or mixup augmentations")

    base_dataset = OBBFinetuneDataset(
        spec=spec or resolve_obb_split(data, split),
        imgsz=imgsz,
        names=names,
        classes=classes,
        single_cls=single_cls,
        cache=cache,
        augment=augment if split == "train" else None,
        seed=seed,
    )
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in the range (0, 1]")
    dataset: Dataset = base_dataset
    dataset_size = len(base_dataset)
    if fraction < 1.0:
        count = max(1, int(dataset_size * fraction))
        subset_generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(dataset_size, generator=subset_generator)[:count].tolist()
        dataset = Subset(base_dataset, indices)
        dataset_size = count

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        collate_fn=_collate,
        drop_last=split == "train" and dataset_size >= batch_size,
        generator=generator,
        worker_init_fn=_seed_worker if workers > 0 and deterministic else None,
    )


def build_semantic_dataloader(
    data: str | Path,
    split: str,
    imgsz: int,
    batch_size: int,
    spec: SemanticSplitSpec | None = None,
    workers: int = 0,
    cache: bool | str = False,
    seed: int = 0,
    deterministic: bool = True,
    fraction: float = 1.0,
    augment: AugmentationConfig | None = None,
) -> DataLoader:
    """Build a semantic loader that preserves integer class maps."""
    cfg = load_data_yaml(data)
    names = normalize_names(cfg)
    ignore_index = cfg.get("ignore_index", 255)
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise ValueError("Data YAML ignore_index must be an integer")
    if ignore_index < 0:
        raise ValueError("Data YAML ignore_index must be non-negative")
    if augment and augment.enabled and (augment.mosaic > 0 or augment.mixup > 0):
        raise ValueError("Semantic training does not support mosaic or mixup augmentations")

    base_dataset = SemanticSegmentationDataset(
        spec=spec or resolve_semantic_split(data, split),
        imgsz=imgsz,
        num_classes=len(names),
        ignore_index=ignore_index,
        cache=cache,
        augment=augment if split == "train" else None,
        seed=seed,
        retain_original_mask=split != "train",
    )
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in the range (0, 1]")
    dataset: Dataset = base_dataset
    dataset_size = len(base_dataset)
    if fraction < 1.0:
        count = max(1, int(dataset_size * fraction))
        subset_generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(dataset_size, generator=subset_generator)[:count].tolist()
        dataset = Subset(base_dataset, indices)
        dataset_size = count

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        collate_fn=_collate,
        drop_last=split == "train" and dataset_size >= batch_size,
        generator=generator,
        worker_init_fn=_seed_worker if workers > 0 and deterministic else None,
    )


def build_coco_dataloader(
    data: str | Path,
    split: str,
    imgsz: int,
    batch_size: int,
    spec: DetectionSplitSpec | None = None,
    **kwargs,
) -> DataLoader:
    """Backward-compatible alias for the generalized detection dataloader."""
    return build_detection_dataloader(data, split, imgsz, batch_size, spec=spec, **kwargs)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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
) -> list[dict[str, object]]:
    annotations: list[dict[str, object]] = []
    raw = label_path.read_text().splitlines()

    for line_no, line in enumerate(raw, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        is_box = len(parts) == 5
        is_polygon = len(parts) >= 7 and (len(parts) - 1) % 2 == 0
        if not is_box and not is_polygon:
            LOGGER.warning("Skipping malformed YOLO label row %s:%d", label_path, line_no)
            continue

        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            LOGGER.warning("Skipping non-numeric YOLO label row %s:%d", label_path, line_no)
            continue

        if class_id not in names:
            LOGGER.warning(
                "Skipping out-of-range YOLO class id %s in %s:%d", class_id, label_path, line_no
            )
            continue
        if not all(math.isfinite(value) for value in values):
            LOGGER.warning("Skipping non-finite YOLO annotation in %s:%d", label_path, line_no)
            continue

        segmentation: list[list[float]] | None = None
        if is_box:
            cx, cy, bw, bh = values
            if bw <= 0 or bh <= 0:
                LOGGER.warning("Skipping non-positive YOLO box in %s:%d", label_path, line_no)
                continue
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height
        else:
            points = [
                [
                    min(max(values[index] * width, 0.0), float(width)),
                    min(max(values[index + 1] * height, 0.0), float(height)),
                ]
                for index in range(0, len(values), 2)
            ]
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            x1, y1, x2, y2 = min(x_values), min(y_values), max(x_values), max(y_values)
            segmentation = [[coordinate for point in points for coordinate in point]]

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

        area = box_w * box_h
        if segmentation is not None:
            polygon = segmentation[0]
            point_count = len(polygon) // 2
            area = (
                abs(
                    sum(
                        polygon[2 * index] * polygon[2 * ((index + 1) % point_count) + 1]
                        - polygon[2 * ((index + 1) % point_count)] * polygon[2 * index + 1]
                        for index in range(point_count)
                    )
                )
                / 2
            )

        annotation: dict[str, object] = {
            "category_id": class_id + 1,
            "bbox": [x1, y1, box_w, box_h],
            "area": round(area, 6),
        }
        if segmentation is not None:
            annotation["segmentation"] = segmentation
        annotations.append(annotation)

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
