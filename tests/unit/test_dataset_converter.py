import json
from pathlib import Path

import pytest
import yaml

from dfine.utils.data import resolve_detection_split
from dfine.utils.dataset_converter import convert_dataset


def test_yolo_to_coco_preserves_splits_and_emits_config(tiny_yolo_dataset, tmp_path):
    result = convert_dataset(tiny_yolo_dataset, tmp_path / "coco", "coco")

    assert result.splits == {"train": 4, "val": 2}
    config = yaml.safe_load(result.config_path.read_text())
    assert config["path"] == "../.."
    assert config["train_ann"] == "annotations/instances_train.json"
    assert resolve_detection_split(result.config_path, "train").format == "coco"
    payload = json.loads((result.output_dir / config["train_ann"]).read_text())
    assert len(payload["images"]) == 4
    assert {item["name"] for item in payload["categories"]} == {"person", "car"}


def test_coco_to_yolo_preserves_splits_polygons_and_images(tiny_dataset, tmp_path):
    result = convert_dataset(tiny_dataset, tmp_path / "yolo", "yolo")

    assert result.splits == {"train": 4, "val": 2}
    config = yaml.safe_load(result.config_path.read_text())
    assert config["path"] == "../.."
    assert config["names"] == {0: "person", 1: "car"}
    assert resolve_detection_split(result.config_path, "train").format == "yolo"
    labels = (result.output_dir / "labels/train/000001.txt").read_text().split()
    assert labels[0] == "0"
    assert [float(value) for value in labels[1:]] == [
        0.15625,
        0.15625,
        0.46875,
        0.15625,
        0.46875,
        0.46875,
        0.15625,
        0.46875,
    ]
    assert (result.output_dir / "images/val/000001.jpg").is_file()


def test_converter_refuses_to_replace_output_without_permission(tiny_yolo_dataset, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")

    try:
        convert_dataset(tiny_yolo_dataset, output, "coco")
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError")
    assert marker.read_text() == "keep"


def test_converter_refuses_output_inside_source_dataset(tiny_yolo_dataset):
    data_path = Path(tiny_yolo_dataset)
    config = yaml.safe_load(data_path.read_text())
    root = Path(config["path"])
    if not root.is_absolute():
        root = data_path.parent / root

    with pytest.raises(ValueError, match="outside the source dataset"):
        convert_dataset(tiny_yolo_dataset, root / "converted", "coco")
