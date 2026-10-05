"""Unit tests for dataset-format detection and YOLO conversion."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import dfine.utils.data as data_utils
from dfine.utils.augmentations import (
    AugmentationConfig,
    horizontal_flip,
    letterbox,
    random_crop,
    random_iou_crop,
    random_zoom_out,
    scale_translate,
    stretch_resize,
)
from dfine.utils.data import (
    DetectionBatchCollate,
    _dataset_cache_dir,
    _load_yolo_annotations,
    build_detection_dataloader,
    build_semantic_dataloader,
    normalize_names,
    resolve_detection_split,
    resolve_semantic_split,
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


def test_random_zoom_out_offsets_boxes_and_expands_canvas():
    import random

    image = Image.new("RGB", (32, 32))
    boxes = torch.tensor([[8.0, 8.0, 24.0, 24.0]])
    output, transformed = random_zoom_out(image, boxes, random.Random(3), p=1.0)

    assert output.size[0] >= 32
    assert output.size[1] >= 32
    assert torch.all(transformed[:, 2:] > transformed[:, :2])


def test_random_zoom_out_short_side_limit_bounds_canvas_and_keeps_boxes_aligned():
    import random

    image = Image.new("RGB", (400, 300))
    image.paste((255, 255, 255), (100, 75, 300, 225))
    boxes = torch.tensor([[100.0, 75.0, 300.0, 225.0]])
    for seed in range(20):
        canvas, transformed = random_zoom_out(
            image, boxes, random.Random(seed), p=1.0, short_side_limit=200
        )

        assert min(canvas.size) <= 201
        assert canvas.size[0] / canvas.size[1] == pytest.approx(400 / 300, rel=0.03)
        # The box still frames the white patch after the downscale and the placement.
        ys, xs = np.nonzero(np.asarray(canvas.convert("L")) > 127)
        white = torch.tensor([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=torch.float32)
        assert torch.allclose(transformed[0], white, atol=1.5)


def test_zoom_out_short_side_limit_never_upsamples_the_smallest_iou_crop():
    from dfine.utils.augmentations import IOU_CROP_MIN_FRACTION, zoom_out_short_side_limit

    for size in (64, 320, 640, 1280):
        assert zoom_out_short_side_limit(size) * IOU_CROP_MIN_FRACTION >= size


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_detection_recipe_bounds_zoom_out_canvas_for_large_images(tmp_path, monkeypatch):
    """Regression for #218: high-resolution images tripped Pillow's decompression-bomb guard."""
    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    Image.new("RGB", (1200, 900), (90, 90, 90)).save(img_dir / "big.jpg")
    ann = tmp_path / "train.json"
    ann.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "big.jpg", "width": 1200, "height": 900}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [500, 400, 200, 150],
                        "area": 30000,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "object"}],
            }
        )
    )
    from dfine.utils.data import CocoFinetuneDataset

    augmentation = AugmentationConfig(profile="deim", photometric=0.0, zoomout=1.0, iou_crop=1.0)
    dataset = CocoFinetuneDataset(img_dir, ann, imgsz=64, augment=augmentation, seed=0)
    # Scale Pillow's guard down so a 1200x900 image zoomed out by up to 4x exceeds it, as a
    # 24 MP photo does against the real 179 MP limit. The bounded canvas stays far below it.
    # Pillow raises above twice the limit and only warns between one and two times it.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 600_000)
    for epoch in range(30):
        dataset.set_epoch(epoch, mosaic=False)
        image, target = dataset[0]
        assert image.shape == (3, 64, 64)
        assert target["boxes"].shape[0] == target["labels"].shape[0]


def test_random_iou_crop_keeps_labels_aligned_and_boxes_valid():
    import random

    image = Image.new("RGB", (100, 100))
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [70.0, 70.0, 95.0, 95.0]])
    labels = torch.tensor([1, 2])
    cropped, transformed, kept_labels = random_iou_crop(
        image, boxes, labels, random.Random(8), p=1.0
    )

    assert cropped.size[0] <= 100
    assert cropped.size[1] <= 100
    assert transformed.shape[0] == kept_labels.shape[0]
    assert torch.all(transformed[:, 2:] > transformed[:, :2])


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
    assert targets[0]["boxes"].shape[0] >= 1
    assert targets[0]["boxes"].shape[0] == targets[0]["labels"].shape[0]
    assert torch.all((targets[0]["boxes"] >= 0) & (targets[0]["boxes"] <= 1))


def test_dfine_profile_detection_loader_keeps_boxes_valid(tiny_dataset):
    augmentation = AugmentationConfig(
        profile="dfine",
        fliplr=1.0,
        scale=0.0,
        translate=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        photometric=1.0,
        zoomout=1.0,
        iou_crop=1.0,
    )
    loader = build_detection_dataloader(
        tiny_dataset,
        "train",
        64,
        2,
        seed=13,
        augment=augmentation,
    )
    images, targets = next(iter(loader))

    assert images.shape == (2, 3, 64, 64)
    for target in targets:
        assert target["boxes"].shape[0] == target["labels"].shape[0]
        assert torch.all((target["boxes"] >= 0) & (target["boxes"] <= 1))


def test_deim_profile_mosaic_policy_can_be_disabled_by_epoch(tiny_dataset):
    augmentation = AugmentationConfig(
        profile="deim",
        fliplr=0.0,
        scale=0.0,
        translate=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=1.0,
        photometric=0.0,
        zoomout=1.0,
        iou_crop=1.0,
    )
    loader = build_detection_dataloader(tiny_dataset, "train", 64, 1, seed=3, augment=augmentation)
    dataset = loader.dataset
    dataset.set_epoch(0, mosaic=False)
    _, target = dataset[0]

    assert target["boxes"].shape == (1, 4)
    assert torch.all((target["boxes"] >= 0) & (target["boxes"] <= 1))


def test_fraction_limits_mosaic_and_mixup_partner_sampling(tiny_dataset):
    augmentation = AugmentationConfig(
        fliplr=0.0,
        scale=0.0,
        translate=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=1.0,
        mixup=1.0,
    )
    loader = build_detection_dataloader(
        tiny_dataset,
        "train",
        64,
        1,
        seed=11,
        fraction=0.5,
        augment=augmentation,
    )
    subset = loader.dataset
    base_dataset = subset.dataset
    allowed_indices = set(subset.indices)
    sampled_indices = []
    original_load_item = base_dataset._load_item

    def recording_load_item(index):
        sampled_indices.append(index)
        return original_load_item(index)

    base_dataset._load_item = recording_load_item

    _ = base_dataset[next(iter(allowed_indices))]

    assert sampled_indices
    assert set(sampled_indices) <= allowed_indices


def test_detection_batch_collate_applies_deim_style_mixup() -> None:
    collate = DetectionBatchCollate(mixup_prob=1.0, mixup_epochs=(0, 2), seed=5)
    collate.set_epoch(1)
    image_a = torch.zeros((3, 4, 4), dtype=torch.float32)
    image_b = torch.ones((3, 4, 4), dtype=torch.float32)
    target_a = {
        "boxes": torch.tensor([[0.1, 0.1, 0.2, 0.2]]),
        "labels": torch.tensor([0]),
    }
    target_b = {
        "boxes": torch.tensor([[0.3, 0.3, 0.4, 0.4]]),
        "labels": torch.tensor([1]),
    }

    images, targets = collate([(image_a, target_a), (image_b, target_b)])

    assert torch.all((images > 0.0) & (images < 1.0))
    assert targets[0]["boxes"].shape == (2, 4)
    assert targets[0]["labels"].tolist() == [0, 1]
    assert targets[1]["labels"].tolist() == [1, 0]
    assert targets[0]["mixup"].shape == (2,)
    assert torch.allclose(target_a["boxes"], torch.tensor([[0.1, 0.1, 0.2, 0.2]]))


def test_detection_batch_collate_skips_mixup_outside_policy_epoch() -> None:
    collate = DetectionBatchCollate(mixup_prob=1.0, mixup_epochs=(1, 2), seed=5)
    collate.set_epoch(2)
    image = torch.zeros((3, 4, 4), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[0.1, 0.1, 0.2, 0.2]]),
        "labels": torch.tensor([0]),
    }

    images, targets = collate([(image, target)])

    assert images.shape == (1, 3, 4, 4)
    assert "mixup" not in targets[0]


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


def test_semantic_loader_preserves_dense_class_ids(tiny_semantic_dataset):
    loader = build_semantic_dataloader(
        tiny_semantic_dataset,
        "train",
        64,
        2,
        seed=4,
        augment=AugmentationConfig(
            fliplr=1.0,
            scale=0.2,
            translate=0.1,
            crop=0.2,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
        ),
    )

    images, targets = next(iter(loader))

    assert images.shape == (2, 3, 64, 64)
    for target in targets:
        assert target["sem_mask"].shape == (64, 64)
        assert target["sem_mask"].dtype == torch.int64
        assert "orig_mask" not in target
        assert set(target["sem_mask"].unique().tolist()) <= {0, 1, 2, 255}

    _, validation_targets = next(
        iter(build_semantic_dataloader(tiny_semantic_dataset, "val", 64, 2))
    )
    assert all(target["orig_mask"].shape == (48, 64) for target in validation_targets)


def test_semantic_split_infers_mirrored_labels_directory(tiny_semantic_dataset):
    spec = resolve_semantic_split(tiny_semantic_dataset, "val")
    assert spec.img_dir.name == "val"
    assert spec.img_dir.parent.name == "images"
    assert spec.mask_dir.parent.name == "labels"


def test_semantic_loader_rejects_instance_only_augmentations(tiny_semantic_dataset):
    with pytest.raises(ValueError, match="mosaic or mixup"):
        build_semantic_dataloader(
            tiny_semantic_dataset,
            "train",
            64,
            1,
            augment=AugmentationConfig(mosaic=1.0),
        )


def test_semantic_loader_rejects_invalid_class_ids(tiny_semantic_dataset, tmp_path):
    import shutil

    source = Path(tiny_semantic_dataset).parent
    copied = tmp_path / "semantic"
    shutil.copytree(source, copied)
    copied_yaml = copied / Path(tiny_semantic_dataset).name
    config = yaml.safe_load(copied_yaml.read_text())
    config["path"] = str(copied)
    copied_yaml.write_text(yaml.safe_dump(config))
    spec = resolve_semantic_split(copied_yaml, "val")
    mask_path = sorted(spec.mask_dir.glob("*.png"))[0]
    mask = np.asarray(Image.open(mask_path)).copy()
    mask[4, 4] = 17
    Image.fromarray(mask).save(mask_path)

    loader = build_semantic_dataloader(copied_yaml, "val", 64, 1)
    with pytest.raises(ValueError, match="class ID 17"):
        next(iter(loader))


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


def test_resolve_detection_split_downloads_declared_missing_dataset(tmp_path, monkeypatch):
    source = tmp_path / "source" / "coco128"
    image_dir = source / "images" / "train2017"
    label_dir = source / "labels" / "train2017"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_dir / "000000000009.jpg")
    (label_dir / "000000000009.txt").write_text("0 0.5 0.5 0.25 0.25\n")

    archive_path = tmp_path / "coco128.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent))

    def fake_urlretrieve(url, filename):
        assert url == "https://example.test/coco128.zip"
        shutil.copyfile(archive_path, filename)
        return filename, None

    monkeypatch.setattr(data_utils, "urlretrieve", fake_urlretrieve)

    config_dir = tmp_path / "configs" / "datasets"
    config_dir.mkdir(parents=True)
    data_yaml = config_dir / "coco128.yml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": "../../datasets/coco128",
                "train": "images/train2017",
                "val": "images/train2017",
                "download": "https://example.test/coco128.zip",
                "names": {0: "person"},
            }
        )
    )

    spec = resolve_detection_split(data_yaml, "train")

    assert spec.format == "yolo"
    assert spec.img_dir == tmp_path / "datasets" / "coco128" / "images" / "train2017"
    assert spec.label_dir == tmp_path / "datasets" / "coco128" / "labels" / "train2017"
    assert spec.ann_file.exists()


def test_segment_loader_downloads_declared_missing_polygon_dataset(tmp_path, monkeypatch):
    source = tmp_path / "source" / "coco128-seg"
    image_dir = source / "images" / "train2017"
    label_dir = source / "labels" / "train2017"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_dir / "000000000009.jpg")
    (label_dir / "000000000009.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")

    archive_path = tmp_path / "coco128-seg.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent))

    def fake_urlretrieve(url, filename):
        assert url == "https://example.test/coco128-seg.zip"
        shutil.copyfile(archive_path, filename)
        return filename, None

    monkeypatch.setattr(data_utils, "urlretrieve", fake_urlretrieve)

    config_dir = tmp_path / "configs" / "datasets"
    config_dir.mkdir(parents=True)
    data_yaml = config_dir / "coco128-seg.yml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": "../../datasets/coco128-seg",
                "train": "images/train2017",
                "val": "images/train2017",
                "download": "https://example.test/coco128-seg.zip",
                "names": {0: "person"},
            }
        )
    )

    loader = build_detection_dataloader(data_yaml, "train", 64, 1, task="segment")
    _, targets = next(iter(loader))

    assert targets[0]["masks"].shape[0] == 1
    assert targets[0]["masks"].shape[1:] == (64, 64)
    assert targets[0]["masks"].sum() > 0


def _write_download_yaml(tmp_path, **extra):
    config_dir = tmp_path / "configs" / "datasets"
    config_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = config_dir / "mini.yml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": "../../datasets/mini",
                "train": "images/train",
                "val": "images/val",
                "download": "https://example.test/mini.zip",
                "names": {0: "person"},
                **extra,
            }
        )
    )
    return data_yaml


def _fake_archive_download(tmp_path, monkeypatch):
    archive_path = tmp_path / "mini.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("mini/images/train/a.jpg", b"not an image")
    calls = []

    def fake_urlretrieve(url, filename):
        calls.append(url)
        shutil.copyfile(archive_path, filename)
        return filename, None

    monkeypatch.setattr(data_utils, "urlretrieve", fake_urlretrieve)
    return calls


def test_dataset_download_rejects_checksum_mismatch_without_partial_dataset(tmp_path, monkeypatch):
    _fake_archive_download(tmp_path, monkeypatch)
    data_yaml = _write_download_yaml(tmp_path, download_sha256="0" * 64)

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        resolve_detection_split(data_yaml, "train")

    assert not (tmp_path / "datasets" / "mini").exists()
    assert list((tmp_path / "datasets").iterdir()) == []


def test_dataset_download_accepts_matching_checksum(tmp_path, monkeypatch):
    _fake_archive_download(tmp_path, monkeypatch)
    digest = hashlib.sha256((tmp_path / "mini.zip").read_bytes()).hexdigest()
    data_yaml = _write_download_yaml(tmp_path, download_sha256=digest)

    data_utils._maybe_download_dataset(data_yaml, yaml.safe_load(data_yaml.read_text()), "train")

    assert (tmp_path / "datasets" / "mini" / "images" / "train" / "a.jpg").exists()


def test_dataset_download_does_not_overwrite_existing_dataset_dir(tmp_path, monkeypatch):
    calls = _fake_archive_download(tmp_path, monkeypatch)
    data_yaml = _write_download_yaml(tmp_path)
    (tmp_path / "datasets" / "mini" / "mine.txt").parent.mkdir(parents=True)
    (tmp_path / "datasets" / "mini" / "mine.txt").write_text("user data")

    with pytest.raises(FileNotFoundError, match="remove it to re-download"):
        resolve_detection_split(data_yaml, "train")

    assert calls == []
    assert (tmp_path / "datasets" / "mini" / "mine.txt").read_text() == "user data"


def test_dataset_download_rejects_non_zip_url_before_downloading(tmp_path, monkeypatch):
    calls = _fake_archive_download(tmp_path, monkeypatch)
    data_yaml = _write_download_yaml(tmp_path, download="https://example.test/mini.tar.gz")

    with pytest.raises(ValueError, match=".zip archive"):
        resolve_detection_split(data_yaml, "train")

    assert calls == []


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
