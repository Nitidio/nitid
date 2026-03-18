"""Unit tests for Results and Boxes."""
import torch
import numpy as np
import pytest
from dfine.results import Results, Boxes


@pytest.fixture
def dummy_boxes():
    data = torch.tensor([[10., 20., 100., 200., 0.9, 0.],
                         [50., 60., 150., 250., 0.7, 1.]])
    return Boxes(data, orig_shape=(480, 640))


@pytest.fixture
def dummy_result(dummy_boxes):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return Results(orig_img=img, path="test.jpg",
                   names={0: "person", 1: "car"}, boxes=dummy_boxes)


def test_boxes_xyxy(dummy_boxes):
    assert dummy_boxes.xyxy.shape == (2, 4)


def test_boxes_conf(dummy_boxes):
    assert dummy_boxes.conf.shape == (2,)
    assert float(dummy_boxes.conf[0]) == pytest.approx(0.9)


def test_boxes_cls(dummy_boxes):
    assert int(dummy_boxes.cls[0]) == 0
    assert int(dummy_boxes.cls[1]) == 1


def test_boxes_xyxyn(dummy_boxes):
    n = dummy_boxes.xyxyn
    assert n.shape == (2, 4)
    assert float(n[0, 0]) == pytest.approx(10 / 640)


def test_results_len(dummy_result):
    assert len(dummy_result) == 2


def test_results_empty():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    r = Results(orig_img=img, path="x.jpg", names={}, boxes=None)
    assert len(r) == 0


def test_results_json(dummy_result):
    j = dummy_result.to_json()
    assert len(j) == 2
    assert j[0]["name"] == "person"
    assert "confidence" in j[0]
    assert "box" in j[0]
