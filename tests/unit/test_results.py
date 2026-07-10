"""Unit tests for Results and Boxes."""

import json

import numpy as np
import pytest
import torch

from dfine.results import Boxes, Results


@pytest.fixture
def dummy_boxes():
    data = torch.tensor(
        [[10.0, 20.0, 100.0, 200.0, 0.9, 0.0], [50.0, 60.0, 150.0, 250.0, 0.7, 1.0]]
    )
    return Boxes(data, orig_shape=(480, 640))


@pytest.fixture
def dummy_result(dummy_boxes):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return Results(orig_img=img, path="test.jpg", names={0: "person", 1: "car"}, boxes=dummy_boxes)


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


def test_results_json_unknown_class(dummy_boxes):
    """to_json() falls back to 'unknown' when class id missing from names."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    r = Results(orig_img=img, path="x.jpg", names={}, boxes=dummy_boxes)
    j = r.to_json()
    assert j[0]["name"] == "unknown"


def test_results_json_empty():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    r = Results(orig_img=img, path="x.jpg", names={}, boxes=None)
    assert r.to_json() == []


def test_results_save_json(tmp_path, dummy_result):
    out_file = tmp_path / "predictions" / "result.json"
    dummy_result.save_json(out_file)

    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data == dummy_result.to_json()
    assert data[0]["name"] == "person"


def test_results_save_json_empty(tmp_path):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = Results(orig_img=img, path="x.jpg", names={}, boxes=None)
    out_file = tmp_path / "empty.json"
    result.save_json(out_file)

    assert json.loads(out_file.read_text(encoding="utf-8")) == []


def test_results_plot_returns_ndarray(dummy_result):
    out = dummy_result.plot()
    assert isinstance(out, np.ndarray)
    assert out.shape == dummy_result.orig_img.shape


def test_results_plot_no_boxes():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    r = Results(orig_img=img, path="x.jpg", names={}, boxes=None)
    out = r.plot()
    assert out.shape == img.shape


def test_results_save(tmp_path, dummy_result):
    out_file = tmp_path / "out.jpg"
    dummy_result.save(str(out_file))
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_boxes_xywh(dummy_boxes):
    xywh = dummy_boxes.xywh
    assert xywh.shape == (2, 4)
    # first box: x1=10,y1=20,x2=100,y2=200 → cx=55,cy=110,w=90,h=180
    assert float(xywh[0, 0]) == pytest.approx(55.0)
    assert float(xywh[0, 1]) == pytest.approx(110.0)
    assert float(xywh[0, 2]) == pytest.approx(90.0)
    assert float(xywh[0, 3]) == pytest.approx(180.0)


def test_boxes_xywhn(dummy_boxes):
    xywhn = dummy_boxes.xywhn
    assert xywhn.shape == (2, 4)
    # orig_shape (480, 640): cx=55/640, cy=110/480
    assert float(xywhn[0, 0]) == pytest.approx(55 / 640)
    assert float(xywhn[0, 1]) == pytest.approx(110 / 480)


def test_boxes_data(dummy_boxes):
    assert dummy_boxes.data.shape == (2, 6)
