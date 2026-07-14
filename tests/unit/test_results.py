"""Unit tests for Results and Boxes."""

import json

import cv2
import numpy as np
import pandas as pd
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


@pytest.fixture
def crop_result(dummy_boxes):
    img = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    return Results(
        orig_img=img, path="street.jpg", names={0: "person", 1: "car"}, boxes=dummy_boxes
    )


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


def test_results_pandas_returns_dataframe(dummy_result):
    df = dummy_result.pandas()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x1", "y1", "x2", "y2", "confidence", "class", "name"]
    assert df.to_dict(orient="records") == [
        {
            "x1": 10.0,
            "y1": 20.0,
            "x2": 100.0,
            "y2": 200.0,
            "confidence": 0.9,
            "class": 0,
            "name": "person",
        },
        {
            "x1": 50.0,
            "y1": 60.0,
            "x2": 150.0,
            "y2": 250.0,
            "confidence": 0.7,
            "class": 1,
            "name": "car",
        },
    ]


def test_results_to_df_empty():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    r = Results(orig_img=img, path="x.jpg", names={}, boxes=None)
    df = r.to_df()

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ["x1", "y1", "x2", "y2", "confidence", "class", "name"]


def test_results_to_csv(tmp_path, dummy_result):
    out_file = tmp_path / "detections.csv"

    dummy_result.to_csv(out_file)

    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert list(df.columns) == ["x1", "y1", "x2", "y2", "confidence", "class", "name"]
    assert df.to_dict(orient="records")[0] == {
        "x1": 10.0,
        "y1": 20.0,
        "x2": 100.0,
        "y2": 200.0,
        "confidence": 0.9,
        "class": 0,
        "name": "person",
    }


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


def test_results_save_txt(tmp_path, dummy_result):
    out_file = tmp_path / "labels" / "result.txt"
    dummy_result.save_txt(out_file)

    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8").splitlines() == [
        "0 0.085938 0.229167 0.140625 0.375000",
        "1 0.156250 0.322917 0.156250 0.395833",
    ]


def test_results_save_txt_with_conf(tmp_path, dummy_result):
    out_file = tmp_path / "result.txt"
    dummy_result.save_txt(out_file, save_conf=True)

    assert out_file.read_text(encoding="utf-8").splitlines() == [
        "0 0.085938 0.229167 0.140625 0.375000 0.900000",
        "1 0.156250 0.322917 0.156250 0.395833 0.700000",
    ]


def test_results_save_txt_empty(tmp_path):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = Results(orig_img=img, path="x.jpg", names={}, boxes=None)
    out_file = tmp_path / "empty.txt"
    result.save_txt(out_file)

    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == ""


def test_results_crop_returns_cropped_images(crop_result):
    crops = crop_result.crop()

    assert len(crops) == 2
    assert crops[0]["name"] == "person"
    assert crops[0]["class"] == 0
    assert crops[0]["confidence"] == 0.9
    assert crops[0]["box"] == {"x1": 10, "y1": 20, "x2": 100, "y2": 200}
    np.testing.assert_array_equal(crops[0]["im"], crop_result.orig_img[20:200, 10:100])
    assert crops[0]["im"].shape == (180, 90, 3)
    assert crops[0]["save_path"] is None


def test_results_crop_saves_into_class_folders(tmp_path, crop_result):
    crops = crop_result.crop(save_dir=tmp_path / "crops")

    person_path = tmp_path / "crops" / "person" / "street.jpg"
    car_path = tmp_path / "crops" / "car" / "street.jpg"

    assert person_path.exists()
    assert car_path.exists()
    assert crops[0]["save_path"] == str(person_path)
    assert crops[1]["save_path"] == str(car_path)
    assert cv2.imread(str(person_path)).shape == (180, 90, 3)
    assert cv2.imread(str(car_path)).shape == (190, 100, 3)


def test_results_crop_saves_repeated_classes_without_overwriting(tmp_path):
    data = torch.tensor(
        [[10.0, 20.0, 100.0, 200.0, 0.9, 0.0], [50.0, 60.0, 150.0, 250.0, 0.7, 0.0]]
    )
    boxes = Boxes(data, orig_shape=(480, 640))
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = Results(orig_img=img, path="street.jpg", names={0: "person"}, boxes=boxes)

    crops = result.crop(save_dir=tmp_path / "crops")

    assert (tmp_path / "crops" / "person" / "street_0.jpg").exists()
    assert (tmp_path / "crops" / "person" / "street_1.jpg").exists()
    assert crops[0]["save_path"] == str(tmp_path / "crops" / "person" / "street_0.jpg")
    assert crops[1]["save_path"] == str(tmp_path / "crops" / "person" / "street_1.jpg")


def test_results_crop_clips_boxes_to_image_bounds():
    data = torch.tensor([[-5.2, -3.0, 20.1, 30.9, 0.8, 0.0]])
    boxes = Boxes(data, orig_shape=(40, 50))
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    result = Results(orig_img=img, path="x.jpg", names={0: "person"}, boxes=boxes)

    crops = result.crop()

    assert crops[0]["box"] == {"x1": 0, "y1": 0, "x2": 21, "y2": 31}
    assert crops[0]["im"].shape == (31, 21, 3)


def test_results_crop_empty():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = Results(orig_img=img, path="x.jpg", names={}, boxes=None)

    assert result.crop() == []


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
