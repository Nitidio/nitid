"""Unit tests for geometric utilities."""

import pytest
import torch

from dfine.utils.ops import (
    clip_boxes,
    crop_masks_to_boxes,
    scale_boxes,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_scale_boxes():
    boxes = torch.tensor([[0.0, 0.0, 320.0, 320.0]])
    scaled = scale_boxes(boxes, from_shape=(640, 640), to_shape=(1280, 1280))
    assert scaled[0, 2].item() == pytest.approx(640.0)


def test_clip_boxes():
    boxes = torch.tensor([[-10.0, -5.0, 700.0, 500.0]])
    clipped = clip_boxes(boxes, shape=(480, 640))
    assert clipped[0, 0].item() == 0.0
    assert clipped[0, 2].item() == 640.0


def test_roundtrip_xyxy_xywh():
    boxes = torch.tensor([[10.0, 20.0, 110.0, 120.0]])
    assert torch.allclose(xywh_to_xyxy(xyxy_to_xywh(boxes)), boxes)


def test_clip_boxes_empty():
    empty = torch.zeros((0, 4))
    result = clip_boxes(empty, shape=(480, 640))
    assert result.shape == (0, 4)


def test_scale_boxes_empty():
    empty = torch.zeros((0, 4))
    result = scale_boxes(empty, from_shape=(640, 640), to_shape=(1280, 1280))
    assert result.shape == (0, 4)


def test_crop_masks_to_boxes_applies_each_instance_box():
    masks = torch.ones((2, 5, 6), dtype=torch.bool)
    boxes = torch.tensor([[1.0, 1.0, 4.0, 4.0], [3.0, 0.0, 6.0, 2.0]])

    cropped = crop_masks_to_boxes(masks, boxes)

    expected = torch.zeros_like(masks)
    expected[0, 1:4, 1:4] = True
    expected[1, 0:2, 3:6] = True
    assert torch.equal(cropped, expected)


def test_crop_masks_to_boxes_preserves_empty_mask_shape_and_dtype():
    masks = torch.zeros((0, 5, 6), dtype=torch.float32)
    cropped = crop_masks_to_boxes(masks, torch.zeros((0, 4)))
    assert cropped.shape == masks.shape
    assert cropped.dtype == masks.dtype
