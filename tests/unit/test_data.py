"""Unit tests for dataset-format detection and YOLO conversion."""

import json

import numpy as np
import yaml
from PIL import Image

from dfine.utils.data import _load_yolo_annotations, normalize_names, resolve_detection_split


def test_normalize_names_accepts_list():
    assert normalize_names({"names": ["person", "car"]}) == {0: "person", 1: "car"}


def test_resolve_detection_split_detects_coco_default_layout(tmp_path):
    root = tmp_path / "coco"
    (root / "images" / "train").mkdir(parents=True)
    (root / "annotations").mkdir()
    ann_file = root / "annotations" / "instances_train.json"
    ann_file.write_text('{"images": [], "annotations": [], "categories": []}')

    data_yaml = root / "data.yml"
    data_yaml.write_text(
        yaml.dump({"path": str(root), "train": "images/train", "names": {0: "person"}})
    )

    spec = resolve_detection_split(data_yaml, "train")
    assert spec.format == "coco"
    assert spec.ann_file == ann_file


def test_resolve_detection_split_detects_yolo_images_first_layout(tmp_path):
    root = tmp_path / "yolo"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)

    data_yaml = root / "data.yml"
    data_yaml.write_text(
        yaml.dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "person"},
            }
        )
    )

    spec = resolve_detection_split(data_yaml, "train")
    assert spec.format == "yolo"
    assert spec.label_dir == root / "labels" / "train"
    assert spec.ann_file.exists()


def test_resolve_detection_split_detects_yolo_split_first_layout(tmp_path):
    root = tmp_path / "yolo_split"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)

    data_yaml = root / "data.yml"
    data_yaml.write_text(
        yaml.dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "val/images",
                "names": {0: "person"},
            }
        )
    )

    spec = resolve_detection_split(data_yaml, "train")
    assert spec.format == "yolo"
    assert spec.label_dir == root / "train" / "labels"


def test_resolve_detection_split_prefers_explicit_coco_ann_over_yolo_dirs(tmp_path):
    root = tmp_path / "mixed"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "annotations").mkdir()
    ann_file = root / "annotations" / "train.json"
    ann_file.write_text('{"images": [], "annotations": [], "categories": []}')

    data_yaml = root / "data.yml"
    data_yaml.write_text(
        yaml.dump(
            {
                "path": str(root),
                "train": "images/train",
                "train_ann": "annotations/train.json",
                "names": {0: "person"},
            }
        )
    )

    spec = resolve_detection_split(data_yaml, "train")
    assert spec.format == "coco"
    assert spec.ann_file == ann_file


def test_yolo_conversion_handles_empty_and_missing_labels(tmp_path):
    root = tmp_path / "yolo_data"
    img_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for idx in range(1, 4):
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img_dir / f"{idx:06d}.jpg")
    (label_dir / "000001.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    (label_dir / "000002.txt").write_text("")
    # 000003.txt intentionally missing

    data_yaml = root / "data.yml"
    data_yaml.write_text(
        yaml.dump({"path": str(root), "train": "images/train", "names": {0: "person"}})
    )

    spec = resolve_detection_split(data_yaml, "train")
    payload = json.loads(spec.ann_file.read_text())

    assert spec.format == "yolo"
    assert len(payload["images"]) == 3
    assert len(payload["annotations"]) == 1


def test_yolo_conversion_rounds_cached_bbox_values(tmp_path):
    label_path = tmp_path / "sample.txt"
    label_path.write_text("20 0.632327 0.626688 0.735347 0.724219\n")

    anns = _load_yolo_annotations(label_path, width=591, height=640, names={20: "elephant"})

    assert anns[0]["bbox"] == [156.410218, 169.33024, 434.589782, 463.50016]
    assert anns[0]["area"] == 201432.433491
