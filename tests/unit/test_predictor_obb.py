from __future__ import annotations

import numpy as np
import pytest
import torch


def _as_list(value):
    return list(value) if not isinstance(value, list) else value


def _obb_predictor():
    from dfine.predictor import DFINEPredictor

    predictor = DFINEPredictor.__new__(DFINEPredictor)
    predictor.task = "obb"
    predictor.names = {0: "plane", 1: "ship"}
    return predictor


def test_obb_postprocess_returns_public_obb_results() -> None:
    from dfine.results import OBB

    predictor = _obb_predictor()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    det = {
        "labels": torch.tensor([0, 1]),
        "boxes": torch.tensor(
            [
                [100.0, 50.0, 40.0, 20.0, 0.1],
                [50.0, 25.0, 20.0, 10.0, -0.2],
            ]
        ),
        "scores": torch.tensor([0.9, 0.1]),
    }

    result = predictor._postprocess(det, image, "obb.jpg", conf_thr=0.5, classes=None)

    assert result.boxes is None
    assert isinstance(result.obb, OBB)
    assert len(result.obb) == 1
    torch.testing.assert_close(
        result.obb.data,
        torch.tensor([[100.0, 50.0, 40.0, 20.0, 0.1, 0.9, 0.0]]),
    )


def test_obb_postprocess_filters_classes_and_handles_empty_results() -> None:
    predictor = _obb_predictor()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    det = {
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[100.0, 50.0, 40.0, 20.0, 0.1]]),
        "scores": torch.tensor([0.9]),
    }

    result = predictor._postprocess(det, image, "obb.jpg", conf_thr=0.5, classes=[1])

    assert result.obb is not None
    assert result.obb.data.shape == (0, 7)
    assert result.to_json() == []
    assert result.plot().shape == image.shape


def test_obb_augmented_merge_mirrors_center_and_angle() -> None:
    from dfine.predictor import DFINEPredictor

    merged = {
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[20.0, 30.0, 10.0, 5.0, 0.25]]),
        "scores": torch.tensor([0.8]),
    }
    flipped = {
        "labels": torch.tensor([1]),
        "boxes": torch.tensor([[40.0, 35.0, 12.0, 6.0, 0.5]]),
        "scores": torch.tensor([0.7]),
    }

    DFINEPredictor._merge_augmented_detections(merged, flipped, width=100)

    assert merged["labels"].tolist() == [0, 1]
    torch.testing.assert_close(
        merged["boxes"][1],
        torch.tensor([60.0, 35.0, 12.0, 6.0, torch.pi - 0.5]),
    )


def test_obb_postprocess_rejects_mixed_output_types() -> None:
    predictor = _obb_predictor()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    det = {
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[100.0, 50.0, 40.0, 20.0, 0.1]]),
        "scores": torch.tensor([0.9]),
        "masks": torch.zeros((1, 8, 8)),
    }

    with pytest.raises(RuntimeError, match="OBB prediction cannot include masks"):
        predictor._postprocess(det, image, "obb.jpg", conf_thr=0.5, classes=None)


def test_obb_public_predict_returns_obb_for_non_native_image_size() -> None:
    from dfine import NITID

    model = NITID("nitid1n", task="obb", weights=None, device="cpu", verbose=False)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    result = _as_list(model.predict(frame, imgsz=64, conf=1.0, verbose=False))[0]

    assert result.obb is not None
    assert result.obb.data.shape == (0, 7)
    assert result.boxes is None
