"""Unit tests for validation geometry handling."""

import torch
import torch.nn as nn

from dfine.validator import _dynamic_eval_geometry, _restore_original_coordinates


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
