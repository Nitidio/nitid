"""Unit tests for LoadSource."""
import numpy as np
import pytest

from dfine.utils.sources import LoadSource


def test_array_source():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    loader = LoadSource(frame, imgsz=640, device="cpu")
    items = list(loader)
    assert len(items) == 1
    tensor, orig, path = items[0]
    assert tensor.shape == (1, 3, 640, 640)
    assert path == "<ndarray>"


def test_array_source_len():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    loader = LoadSource(frame, imgsz=640, device="cpu")
    assert len(loader) == 1


def test_invalid_source():
    with pytest.raises(ValueError):
        LoadSource(object(), imgsz=640, device="cpu")


def test_image_source(tmp_path):
    from PIL import Image as _PILImage
    img_path = tmp_path / "test.jpg"
    _PILImage.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_path)

    loader = LoadSource(str(img_path), imgsz=320, device="cpu")
    assert len(loader) == 1
    items = list(loader)
    assert len(items) == 1
    tensor, orig, path = items[0]
    assert tensor.shape == (1, 3, 320, 320)
    assert path == str(img_path)


def test_directory_source(tmp_path):
    from PIL import Image as _PILImage
    for i in range(3):
        _PILImage.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
            tmp_path / f"{i:03d}.jpg"
        )
    # also write a non-image file that should be ignored
    (tmp_path / "notes.txt").write_text("ignored")

    loader = LoadSource(str(tmp_path), imgsz=32, device="cpu")
    assert len(loader) == 3
    items = list(loader)
    assert len(items) == 3
    for tensor, orig, path in items:
        assert tensor.shape == (1, 3, 32, 32)


def test_list_source():
    frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
    frame2 = np.ones((64, 64, 3), dtype=np.uint8) * 128
    loader = LoadSource([frame1, frame2], imgsz=64, device="cpu")
    items = [item for item in loader]  # avoid list() which rejects __len__ == -1
    assert len(items) == 2
    assert items[0][2] == "<ndarray>"
    assert items[1][2] == "<ndarray>"


def test_stream_len_is_minus_one():
    loader = LoadSource("rtsp://fake", imgsz=640, device="cpu")
    # len() rejects negative values, so call __len__ directly
    assert loader.__len__() == -1
