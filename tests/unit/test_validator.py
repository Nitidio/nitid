"""Unit tests for validation geometry handling."""

import torch
import torch.nn as nn

from dfine.validator import DFINEValidator, _dynamic_eval_geometry, _restore_original_coordinates


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
