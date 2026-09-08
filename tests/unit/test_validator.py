"""Unit tests for validation geometry handling."""

import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

from dfine.validator import (
    DFINEValidator,
    SemanticConfusionMatrix,
    _dynamic_eval_geometry,
    _restore_original_coordinates,
    rotated_box_iou,
)


class _PerfectBoxGlobalMaskModel(nn.Module):
    """Predict the right box but an unbounded mask to exercise mask cleanup."""

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        logits = torch.tensor([10.0, -10.0], device=images.device).repeat(batch_size, 1, 1)
        boxes = torch.tensor([0.3125, 0.3125, 0.3125, 0.3125], device=images.device).repeat(
            batch_size, 1, 1
        )
        masks = torch.ones((batch_size, 1, 8, 8), device=images.device)
        return {"pred_logits": logits, "pred_boxes": boxes, "pred_masks": masks}


class _SemanticGeometryModel(nn.Module):
    """Record the geometry used while semantic validation runs."""

    def __init__(self) -> None:
        super().__init__()
        self.eval_spatial_size = [640, 640]
        self.forward_geometry = object()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forward_geometry = self.eval_spatial_size
        batch_size, _, height, width = images.shape
        logits = torch.zeros((batch_size, 3, height, width), device=images.device)
        return {"sem_seg_logits": logits}


class _PerfectOBBModel(nn.Module):
    """Predict the same normalized OBB as the tiny validation label."""

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        logits = torch.tensor([10.0], device=images.device).reshape(1, 1, 1)
        boxes = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0], device=images.device).reshape(1, 1, 5)
        return {
            "pred_logits": logits.repeat(batch_size, 1, 1),
            "pred_boxes": boxes.repeat(batch_size, 1, 1),
        }


def test_restore_original_coordinates_reverses_dfine_square_resize():
    resized = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    restored = _restore_original_coordinates(
        resized, original_width=200, original_height=100, imgsz=100
    )
    assert torch.allclose(restored, torch.tensor([[20.0, 10.0, 100.0, 50.0]]))


def test_non_native_eval_size_temporarily_uses_dynamic_geometry():
    model = nn.Sequential(nn.Identity(), nn.Identity())
    model[0].eval_spatial_size = [640, 640]
    model[1].eval_spatial_size = [640, 640]

    with _dynamic_eval_geometry(model, 320):
        assert model[0].eval_spatial_size is None
        assert model[1].eval_spatial_size is None

    assert model[0].eval_spatial_size == [640, 640]
    assert model[1].eval_spatial_size == [640, 640]


def test_native_eval_size_keeps_cached_geometry():
    model = nn.Sequential(nn.Identity())
    model[0].eval_spatial_size = [640, 640]

    with _dynamic_eval_geometry(model, 640):
        assert model[0].eval_spatial_size == [640, 640]


def test_rotated_box_iou_identity_and_no_overlap():
    boxes = torch.tensor([[10.0, 10.0, 4.0, 2.0, 0.25]])
    identical = rotated_box_iou(boxes, boxes)
    far = rotated_box_iou(boxes, torch.tensor([[100.0, 100.0, 4.0, 2.0, 0.25]]))

    assert float(identical[0, 0]) == pytest.approx(1.0)
    assert float(far[0, 0]) == pytest.approx(0.0)


def test_obb_map_perfect_prediction_scores_one():
    validator = DFINEValidator(nn.Identity(), {"task": "obb"}, "cpu", {0: "plane"})
    records = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 4.0, 2.0, 0.25]]),
            "labels": torch.tensor([0]),
            "image_id": 1,
        }
    ]
    predictions = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 4.0, 2.0, 0.25]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]

    map50, map5095, per_class = validator._obb_map(records, predictions)

    assert map50 == pytest.approx(1.0)
    assert map5095 == pytest.approx(1.0)
    assert per_class[0]["ap50"] == pytest.approx(1.0)


def test_segment_validation_crops_masks_to_predicted_boxes(tiny_dataset):
    config = {
        "task": "segment",
        "num_classes": 2,
        "use_focal_loss": True,
        "DFINETransformer": {"num_queries": 1},
        "DFINEPostProcessor": {"num_top_queries": 1},
    }
    validator = DFINEValidator(_PerfectBoxGlobalMaskModel(), config, "cpu", {0: "person", 1: "car"})

    metrics = validator.run(
        data=tiny_dataset,
        imgsz=64,
        batch=2,
        conf=0.001,
        split="val",
        verbose=False,
        plots=False,
    )

    assert metrics["mAP50"] > 0.99
    assert metrics["mask_mAP50"] > 0.99
    assert metrics["mask_mAP50-95"] > 0.99


def test_obb_validation_reports_rotated_map(tmp_path):
    root = tmp_path / "obb"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    Image.new("RGB", (64, 64), (0, 0, 0)).save(root / "images" / "val" / "im.jpg")
    (root / "labels" / "val" / "im.txt").write_text(
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"path": str(root), "val": "images/val", "names": ["plane"]}),
        encoding="utf-8",
    )
    config = {
        "task": "obb",
        "num_classes": 1,
        "PostProcessorOBB": {"num_top_queries": 1},
    }
    validator = DFINEValidator(_PerfectOBBModel(), config, "cpu", {0: "plane"})

    metrics = validator.run(
        data=str(data_yaml),
        imgsz=64,
        batch=1,
        conf=0.001,
        split="val",
        verbose=False,
        plots=False,
    )

    assert metrics["mAP50"] == pytest.approx(1.0)
    assert metrics["mAP50-95"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_semantic_confusion_matrix_ignores_void_and_excludes_absent_classes():
    confusion = SemanticConfusionMatrix(num_classes=3, ignore_index=255)
    target = torch.tensor([[0, 0, 1], [1, 255, 255]])
    prediction = torch.tensor([[0, 0, 2], [1, 1, 2]])

    confusion.update(prediction, target)
    metrics = confusion.compute({0: "background", 1: "road", 2: "vehicle"})

    # IoU(background)=1, IoU(road)=1/2; class 2 is absent in GT and excluded.
    assert metrics["mIoU"] == 0.75
    assert metrics["pixel_accuracy"] == 0.75
    assert metrics["pixels"] == 4
    assert [row["class_id"] for row in metrics["per_class"]] == [0, 1]


def test_semantic_validation_uses_dynamic_geometry_at_non_native_size(tiny_semantic_dataset):
    model = _SemanticGeometryModel()
    validator = DFINEValidator(
        model,
        {"task": "semantic"},
        "cpu",
        {0: "background", 1: "road", 2: "vehicle"},
    )

    metrics = validator.run(
        data=tiny_semantic_dataset,
        imgsz=32,
        batch=2,
        conf=0.001,
        split="val",
        verbose=False,
        plots=False,
    )

    assert metrics["images"] == 2
    assert model.forward_geometry is None
    assert model.eval_spatial_size == [640, 640]
