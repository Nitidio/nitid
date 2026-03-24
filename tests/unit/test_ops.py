"""Unit tests for geometric utilities."""
import torch
import pytest
from dfine.utils.ops import scale_boxes, clip_boxes, xyxy_to_xywh, xywh_to_xyxy


def test_scale_boxes():
    boxes = torch.tensor([[0., 0., 320., 320.]])
    scaled = scale_boxes(boxes, from_shape=(640, 640), to_shape=(1280, 1280))
    assert scaled[0, 2].item() == pytest.approx(640.0)


def test_clip_boxes():
    boxes = torch.tensor([[-10., -5., 700., 500.]])
    clipped = clip_boxes(boxes, shape=(480, 640))
    assert clipped[0, 0].item() == 0.
    assert clipped[0, 2].item() == 640.


def test_roundtrip_xyxy_xywh():
    boxes = torch.tensor([[10., 20., 110., 120.]])
    assert torch.allclose(xywh_to_xyxy(xyxy_to_xywh(boxes)), boxes)


def test_clip_boxes_empty():
    empty = torch.zeros((0, 4))
    result = clip_boxes(empty, shape=(480, 640))
    assert result.shape == (0, 4)


def test_scale_boxes_empty():
    empty = torch.zeros((0, 4))
    result = scale_boxes(empty, from_shape=(640, 640), to_shape=(1280, 1280))
    assert result.shape == (0, 4)
