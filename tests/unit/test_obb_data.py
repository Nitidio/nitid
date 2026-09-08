from __future__ import annotations

import json

import pytest
import torch
import yaml
from PIL import Image


def _write_image(path, size=(64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (0, 0, 0)).save(path)


def _write_yaml(path, payload: dict) -> str:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def test_yolo_obb_dataloader_returns_rotated_targets(tmp_path) -> None:
    from dfine.utils.data import build_obb_dataloader

    root = tmp_path / "dataset"
    _write_image(root / "images" / "train" / "im.jpg")
    label_dir = root / "labels" / "train"
    label_dir.mkdir(parents=True)
    label_dir.joinpath("im.txt").write_text(
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    data = _write_yaml(
        tmp_path / "data.yaml",
        {"path": str(root), "train": "images/train", "names": ["plane"]},
    )

    images, targets = next(iter(build_obb_dataloader(data, "train", 64, 1)))

    assert images.shape == (1, 3, 64, 64)
    assert targets[0]["labels"].tolist() == [0]
    assert targets[0]["boxes"].shape == (1, 5)
    assert targets[0]["boxes"][0, :4].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_dota_obb_dataloader_maps_class_names(tmp_path) -> None:
    from dfine.utils.data import build_obb_dataloader

    root = tmp_path / "dataset"
    _write_image(root / "images" / "train" / "im.jpg")
    label_dir = root / "labels" / "train"
    label_dir.mkdir(parents=True)
    label_dir.joinpath("im.txt").write_text(
        "16 8 48 8 48 24 16 24 ship 0\n",
        encoding="utf-8",
    )
    data = _write_yaml(
        tmp_path / "data.yaml",
        {
            "path": str(root),
            "train": "images/train",
            "names": ["plane", "ship"],
            "obb_format": "dota",
        },
    )

    _, targets = next(iter(build_obb_dataloader(data, "train", 64, 1)))

    assert targets[0]["labels"].tolist() == [1]
    assert targets[0]["boxes"][0, :4].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_coco_polygon_obb_dataloader_uses_segmentation_polygon(tmp_path) -> None:
    from dfine.utils.data import build_obb_dataloader

    root = tmp_path / "dataset"
    _write_image(root / "images" / "train" / "im.jpg")
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True)
    ann_dir.joinpath("instances_train.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "im.jpg", "width": 64, "height": 32}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 3,
                        "bbox": [16, 8, 32, 16],
                        "segmentation": [[16, 8, 48, 8, 48, 24, 16, 24]],
                        "area": 512,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 3, "name": "ship"}],
            }
        ),
        encoding="utf-8",
    )
    data = _write_yaml(
        tmp_path / "data.yaml",
        {"path": str(root), "train": "images/train", "names": ["ship"]},
    )

    _, targets = next(iter(build_obb_dataloader(data, "train", 64, 1)))

    assert targets[0]["labels"].tolist() == [0]
    assert targets[0]["boxes"].shape == (1, 5)


def test_obb_dataloader_rejects_mosaic_and_mixup(tmp_path) -> None:
    from dfine.utils.augmentations import AugmentationConfig
    from dfine.utils.data import build_obb_dataloader

    root = tmp_path / "dataset"
    _write_image(root / "images" / "train" / "im.jpg")
    (root / "labels" / "train").mkdir(parents=True)
    data = _write_yaml(
        tmp_path / "data.yaml",
        {"path": str(root), "train": "images/train", "names": ["plane"]},
    )

    with pytest.raises(ValueError, match="mosaic or mixup"):
        build_obb_dataloader(
            data,
            "train",
            64,
            1,
            augment=AugmentationConfig(enabled=True, mosaic=1.0),
        )


def test_obb_train_rejects_too_small_image_size(tmp_path) -> None:
    from dfine import NITID

    root = tmp_path / "dataset"
    _write_image(root / "images" / "train" / "im.jpg")
    (root / "labels" / "train").mkdir(parents=True)
    data = _write_yaml(
        tmp_path / "data.yaml",
        {"path": str(root), "train": "images/train", "names": ["plane"]},
    )
    model = NITID("nitid1n", task="obb", weights=None, device="cpu", verbose=False)

    with pytest.raises(ValueError, match="imgsz >= 256"):
        model.train(data=data, epochs=1, imgsz=128, batch=1, val=False, verbose=False)


def test_obb_criterion_accepts_dataset_targets(tmp_path) -> None:
    from dfine.nn.native_build import build_native_criterion
    from dfine.utils.data import build_obb_dataloader

    root = tmp_path / "dataset"
    _write_image(root / "images" / "train" / "im.jpg")
    label_dir = root / "labels" / "train"
    label_dir.mkdir(parents=True)
    label_dir.joinpath("im.txt").write_text(
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    data = _write_yaml(
        tmp_path / "data.yaml",
        {"path": str(root), "train": "images/train", "names": ["plane"]},
    )
    _, targets = next(iter(build_obb_dataloader(data, "train", 64, 1)))
    criterion = build_native_criterion("nitid1n", num_classes=1, task="obb")
    outputs = {
        "pred_logits": torch.zeros((1, 4, 1), dtype=torch.float32),
        "pred_boxes": torch.full((1, 4, 5), 0.5, dtype=torch.float32),
        "aux_outputs": [
            {
                "pred_logits": torch.zeros((1, 4, 1), dtype=torch.float32),
                "pred_boxes": torch.full((1, 4, 5), 0.5, dtype=torch.float32),
            }
        ],
        "enc_aux_outputs": [
            {
                "pred_logits": torch.zeros((1, 4, 1), dtype=torch.float32),
                "pred_boxes": torch.full((1, 4, 5), 0.5, dtype=torch.float32),
            }
        ],
        "enc_meta": {"class_agnostic": False},
    }

    losses = criterion(outputs, targets)

    assert "loss_focal" in losses
    assert "loss_l1" in losses
    assert "loss_kld" in losses
