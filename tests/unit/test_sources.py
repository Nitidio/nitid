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
