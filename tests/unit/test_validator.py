"""Unit tests for validation resolution handling."""

import torch.nn as nn

from dfine.validator import _dynamic_eval_geometry


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
