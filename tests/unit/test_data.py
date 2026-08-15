"""Unit tests for dataset-format detection and YOLO conversion."""

import json

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from dfine.utils.augmentations import (
    AugmentationConfig,
    horizontal_flip,
    letterbox,
    random_crop,
    scale_translate,
    stretch_resize,
)
from dfine.utils.data import (
    _dataset_cache_dir,
    _load_yolo_annotations,
    build_detection_dataloader,
    normalize_names,
    resolve_detection_split,
)


def test_letterbox_preserves_aspect_ratio_and_updates_boxes():
    image = Image.new("RGB", (200, 100))
    boxes = torch.tensor([[20.0, 10.0, 100.0, 50.0]])
    output, transformed = letterbox(image, boxes, 100)
    assert output.size == (100, 100)
    assert torch.allclose(transformed, torch.tensor([[10.0, 30.0, 50.0, 50.0]]))


def test_stretch_resize_matches_dfine_preprocessing_and_updates_boxes():
    image = Image.new("RGB", (200, 100))
    boxes = torch.tensor([[20.0, 10.0, 100.0, 50.0]])
    output, transformed = stretch_resize(image, boxes, 100)
    assert output.size == (100, 100)
    assert torch.allclose(transformed, torch.tensor([[10.0, 10.0, 50.0, 50.0]]))


def test_horizontal_flip_updates_xyxy_boxes():
    image = Image.new("RGB", (100, 50))
    _, boxes = horizontal_flip(image, torch.tensor([[10.0, 5.0, 30.0, 25.0]]))
    assert torch.equal(boxes, torch.tensor([[70.0, 5.0, 90.0, 25.0]]))


def test_scale_translate_is_deterministic_and_keeps_boxes_in_bounds():
    import random

    image = Image.new("RGB", (100, 100))
    source = torch.tensor([[20.0, 20.0, 80.0, 80.0]])
    first = scale_translate(image, source, 0.25, 0.1, random.Random(7))[1]
    second = scale_translate(image, source, 0.25, 0.1, random.Random(7))[1]
    assert torch.equal(first, second)
    assert first.min() >= 0 and first.max() <= 100


def test_random_crop_updates_boxes_and_returns_label_mask():
    import random

    image = Image.new("RGB", (100, 100))
    boxes = torch.tensor([[10.0, 10.0, 60.0, 60.0], [99.5, 99.5, 100.0, 100.0]])
    cropped, transformed, keep = random_crop(image, boxes, 0.1, random.Random(4))
    assert cropped.size[0] <= 100 and cropped.size[1] <= 100
    assert transformed.shape == (1, 4)
    assert keep.tolist() == [True, False]


def test_augmentation_config_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="fliplr"):
        AugmentationConfig(fliplr=1.1).validate()


def test_augmented_loader_is_deterministic_across_worker_counts(tiny_dataset):
    config = AugmentationConfig(crop=0.2, mosaic=1.0, mixup=0.5)
    loaders = [
        build_detection_dataloader(
            tiny_dataset, "train", 64, 2, workers=workers, seed=19, augment=config
        )
        for workers in (0, 2)
    ]
    batches = [next(iter(loader)) for loader in loaders]
    assert torch.equal(batches[0][0], batches[1][0])
    for first, second in zip(batches[0][1], batches[1][1]):
        assert torch.equal(first["boxes"], second["boxes"])
        assert torch.equal(first["labels"], second["labels"])


def test_mosaic_and_mixup_keep_all_boxes_and_labels_aligned(tiny_dataset):
    mosaic = AugmentationConfig(
        fliplr=0.0,
        scale=0.0,
        translate=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=1.0,
        mixup=1.0,
    )
    loader = build_detection_dataloader(tiny_dataset, "train", 64, 1, seed=3, augment=mosaic)
    _, targets = next(iter(loader))
    assert targets[0]["boxes"].shape == (5, 4)
    assert targets[0]["labels"].shape == (5,)
    assert torch.all((targets[0]["boxes"] >= 0) & (targets[0]["boxes"] <= 1))


def test_segment_loader_keeps_masks_aligned_through_augmentations(tiny_dataset):
    augmentation = AugmentationConfig(
        fliplr=1.0,
        scale=0.2,
        translate=0.1,
        crop=0.2,
        mosaic=1.0,
        mixup=1.0,
    )
    loader = build_detection_dataloader(
        tiny_dataset,
        "train",
        64,
        1,
        seed=7,
        augment=augmentation,
        task="segment",
    )
    _, targets = next(iter(loader))
    target = targets[0]

    assert target["masks"].shape[0] == target["boxes"].shape[0] == target["labels"].shape[0]
    assert target["masks"].shape[1:] == (64, 64)
    assert target["masks"].dtype == torch.uint8
    assert target["masks"].sum() > 0


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


def test_yolo_polygon_conversion_preserves_instance_segmentation(tmp_path):
    label_path = tmp_path / "segment.txt"
    label_path.write_text("0 0.1 0.2 0.8 0.2 0.8 0.9 0.1 0.9\n")

    annotations = _load_yolo_annotations(label_path, 100, 50, {0: "object"})

    assert annotations[0]["bbox"] == [10.0, 10.0, 70.0, 35.0]
    assert annotations[0]["area"] == 2450.0
    assert annotations[0]["segmentation"] == [[10.0, 10.0, 80.0, 10.0, 80.0, 45.0, 10.0, 45.0]]


def test_yolo_polygon_conversion_uses_polygon_area(tmp_path):
    label_path = tmp_path / "triangle.txt"
    label_path.write_text("0 0.1 0.2 0.8 0.2 0.8 0.9\n")

    annotations = _load_yolo_annotations(label_path, 100, 50, {0: "object"})

    assert annotations[0]["bbox"] == [10.0, 10.0, 70.0, 35.0]
    assert annotations[0]["area"] == 1225.0


def test_segment_loader_rejects_bbox_only_annotations(tiny_yolo_dataset):
    with pytest.raises(ValueError, match="requires polygon or RLE"):
        build_detection_dataloader(
            tiny_yolo_dataset,
            "train",
            64,
            1,
            task="segment",
        )


def test_load_yolo_annotations_skips_bad_rows_and_logs(tmp_path, caplog):
    label_path = tmp_path / "bad.txt"
    label_path.write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.25",  # malformed
                "0 nope 0.5 0.25 0.25",  # non-numeric
                "7 0.5 0.5 0.25 0.25",  # out of range
                "0 nan 0.5 0.25 0.25",  # non-finite
                "0 0.5 0.5 -0.25 0.25",  # non-positive
                "0 0.5 0.5 0.25 0.25",  # valid
            ]
        )
    )

    with caplog.at_level("WARNING", logger="dfine"):
        anns = _load_yolo_annotations(label_path, width=100, height=100, names={0: "person"})

    assert len(anns) == 1
    messages = [record.message for record in caplog.records]
    assert any("malformed" in message for message in messages)
    assert any("non-numeric" in message for message in messages)
    assert any("out-of-range" in message for message in messages)
    assert any("non-finite" in message for message in messages)
    assert any("non-positive" in message for message in messages)


def test_resolve_detection_split_raises_when_no_format_matches(tmp_path):
    root = tmp_path / "unknown"
    (root / "images" / "train").mkdir(parents=True)

    data_yaml = root / "data.yml"
    data_yaml.write_text(
        yaml.dump({"path": str(root), "train": "images/train", "names": {0: "person"}})
    )

    with pytest.raises(FileNotFoundError, match="Could not resolve dataset format"):
        resolve_detection_split(data_yaml, "train")


def test_resolve_detection_split_uses_system_cache_dir(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache_root"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))

    root = tmp_path / "yolo"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(root / "images" / "train" / "a.jpg")
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.25 0.25\n")

    data_yaml = root / "data.yml"
    data_yaml.write_text(
        yaml.dump({"path": str(root), "train": "images/train", "names": {0: "person"}})
    )

    spec = resolve_detection_split(data_yaml, "train")

    assert spec.ann_file.is_relative_to(cache_root / "nitid")
    assert spec.ann_file.parent == _dataset_cache_dir(root)


def test_resolve_detection_split_reuses_fresh_yolo_cache(tmp_path, monkeypatch):
    import dfine.utils.data as data_mod

    root = tmp_path / "yolo"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(root / "images" / "train" / "a.jpg")
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.25 0.25\n")

    data_yaml = root / "data.yml"
    data_yaml.write_text(
        yaml.dump({"path": str(root), "train": "images/train", "names": {0: "person"}})
    )

    original_convert = data_mod.convert_yolo_split_to_coco_json
    calls = {"count": 0}

    def wrapped_convert(*args, **kwargs):
        calls["count"] += 1
        return original_convert(*args, **kwargs)

    monkeypatch.setattr(data_mod, "convert_yolo_split_to_coco_json", wrapped_convert)

    first = resolve_detection_split(data_yaml, "train")
    second = resolve_detection_split(data_yaml, "train")

    assert first.ann_file == second.ann_file
    assert calls["count"] == 1
