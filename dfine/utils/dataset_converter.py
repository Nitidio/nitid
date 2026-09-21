"""Convert box/polygon detection datasets between YOLO text and COCO JSON layouts."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml

from dfine.utils.data import (
    IMAGE_SUFFIXES,
    _find_coco_annotation,
    _find_yolo_label_dir,
    convert_yolo_split_to_coco_json,
    normalize_names,
)

TargetFormat = Literal["coco", "yolo"]


@dataclass(frozen=True)
class ConversionResult:
    output_dir: Path
    config_path: Path
    target_format: TargetFormat
    splits: dict[str, int]


def _dataset_root(data_path: Path, config: dict) -> Path:
    value = config.get("path")
    if not isinstance(value, (str, Path)):
        raise ValueError("Data YAML must define a dataset 'path'")
    root = Path(value).expanduser()
    return root.resolve() if root.is_absolute() else (data_path.parent / root).resolve()


def _copy_images(source: Path, destination: Path) -> int:
    images = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    for image in images:
        output = destination / image.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, output)
    return len(images)


def _safe_relative_image_path(filename: object) -> Path:
    path = PurePosixPath(str(filename))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe COCO image path: {filename!r}")
    return Path(*path.parts)


def _coco_to_yolo_split(
    annotation_file: Path,
    source_images: Path,
    source_root: Path,
    output_images: Path,
    output_labels: Path,
) -> tuple[int, list[dict]]:
    payload = json.loads(annotation_file.read_text())
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if (
        not isinstance(images, list)
        or not isinstance(annotations, list)
        or not isinstance(categories, list)
    ):
        raise ValueError(f"Invalid COCO document: {annotation_file}")

    ordered_categories = sorted(categories, key=lambda item: int(item["id"]))
    category_index = {int(item["id"]): index for index, item in enumerate(ordered_categories)}
    by_image: dict[int, list[dict]] = {}
    for annotation in annotations:
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    for image in images:
        image_id = int(image["id"])
        width, height = float(image["width"]), float(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Image {image_id} has invalid dimensions")
        relative = _safe_relative_image_path(image["file_name"])
        source = source_images / relative
        if not source.is_file():
            alternate = source_root / relative
            source = alternate if alternate.is_file() else source
        if not source.is_file():
            raise FileNotFoundError(f"COCO image not found: {relative}")

        image_output = output_images / relative
        image_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_output)
        label_output = (output_labels / relative).with_suffix(".txt")
        label_output.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for annotation in by_image.get(image_id, []):
            category_id = int(annotation["category_id"])
            if category_id not in category_index:
                raise ValueError(f"Annotation references unknown category_id={category_id}")
            class_id = category_index[category_id]
            polygons = annotation.get("segmentation")
            polygon = (
                max((part for part in polygons if isinstance(part, list)), key=len, default=None)
                if isinstance(polygons, list)
                else None
            )
            if polygon and len(polygon) >= 6 and len(polygon) % 2 == 0:
                values = [
                    min(max(float(value) / (width if index % 2 == 0 else height), 0.0), 1.0)
                    for index, value in enumerate(polygon)
                ]
            else:
                bbox = annotation.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Annotation {annotation.get('id')} has no valid bbox")
                x, y, box_width, box_height = (float(value) for value in bbox)
                if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
                    raise ValueError(
                        f"Annotation {annotation.get('id')} contains non-finite values"
                    )
                values = [
                    min(max((x + box_width / 2) / width, 0.0), 1.0),
                    min(max((y + box_height / 2) / height, 0.0), 1.0),
                    min(max(box_width / width, 0.0), 1.0),
                    min(max(box_height / height, 0.0), 1.0),
                ]
            rows.append(" ".join([str(class_id), *(f"{value:.8g}" for value in values)]))
        label_output.write_text("\n".join(rows) + ("\n" if rows else ""))

    return len(images), ordered_categories


def convert_dataset(
    data: str | Path,
    output: str | Path,
    target: TargetFormat,
    *,
    exist_ok: bool = False,
) -> ConversionResult:
    """Convert every declared train/val/test split and emit a trainable YAML."""
    data_path = Path(data).expanduser().resolve()
    config = yaml.safe_load(data_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Data YAML must contain a mapping")
    target = str(target).lower()  # type: ignore[assignment]
    if target not in {"coco", "yolo"}:
        raise ValueError("target must be 'coco' or 'yolo'")

    root = _dataset_root(data_path, config)
    output_dir = Path(output).expanduser().resolve()
    if output_dir == root or output_dir in root.parents or root in output_dir.parents:
        raise ValueError("Conversion output must be outside the source dataset")
    for split in ("train", "val", "test"):
        if split in config:
            source_images = (root / str(config[split])).resolve()
            if output_dir == source_images or source_images in output_dir.parents:
                raise ValueError("Conversion output cannot be inside a source image directory")
    if output_dir.exists():
        if not exist_ok:
            raise FileExistsError(f"Conversion output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    names = normalize_names(config)
    generated: dict[str, object] = {"path": "../..", "nc": len(names), "names": names}
    split_counts: dict[str, int] = {}
    converted_categories: list[dict] | None = None

    for split in ("train", "val", "test"):
        if split not in config:
            continue
        source_images = root / str(config[split])
        if not source_images.is_dir():
            raise FileNotFoundError(f"Image directory for {split!r} not found: {source_images}")
        destination_images = output_dir / "images" / split
        generated[split] = f"images/{split}"

        if target == "coco":
            label_dir = _find_yolo_label_dir(root, source_images, split)
            if label_dir is None:
                raise ValueError(f"Split {split!r} does not contain YOLO labels")
            split_counts[split] = _copy_images(source_images, destination_images)
            annotation_path = output_dir / "annotations" / f"instances_{split}.json"
            convert_yolo_split_to_coco_json(source_images, label_dir, annotation_path, names)
            generated[f"{split}_ann"] = f"annotations/instances_{split}.json"
        else:
            annotation_file = _find_coco_annotation(root, config, split, source_images)
            if annotation_file is None:
                raise ValueError(f"Split {split!r} does not contain COCO annotations")
            count, categories = _coco_to_yolo_split(
                annotation_file,
                source_images,
                root,
                destination_images,
                output_dir / "labels" / split,
            )
            split_counts[split] = count
            if converted_categories is None:
                converted_categories = categories
            elif categories != converted_categories:
                raise ValueError("COCO category definitions differ between splits")

    if not split_counts:
        raise ValueError("Data YAML does not define train, val, or test splits")
    if target == "yolo" and converted_categories is not None:
        converted_names = {
            index: str(item["name"]) for index, item in enumerate(converted_categories)
        }
        generated["names"] = converted_names
        generated["nc"] = len(converted_names)

    config_dir = output_dir / "configs" / "datasets"
    config_dir.mkdir(parents=True)
    config_path = config_dir / f"{data_path.stem}-{target}.yml"
    config_path.write_text(yaml.safe_dump(generated, sort_keys=False))
    return ConversionResult(output_dir, config_path, target, split_counts)  # type: ignore[arg-type]
