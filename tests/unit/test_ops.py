"""Unit tests for geometric utilities."""

import pytest
import torch

from dfine.utils.ops import clip_boxes, scale_boxes, xywh_to_xyxy, xyxy_to_xywh


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
