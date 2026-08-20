"""
Unit tests for DFINE pose API surface, Keypoints container, plotting, and task dispatch.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from dfine.model import DFINE
from dfine.nn.build import build_model
from dfine.nn.configs import make_pose_config
from dfine.results import Boxes, Keypoints, Results
from dfine.utils.checkpoint import save_checkpoint


def test_keypoints_container_properties_and_validation() -> None:
    data_2d = torch.zeros((3, 17, 2))
    data_2d[..., 0] = torch.rand((3, 17)) * 640
    data_2d[..., 1] = torch.rand((3, 17)) * 480
    orig_shape = (480, 640)
    kpts = Keypoints(data_2d, orig_shape=orig_shape)

    assert len(kpts) == 3
    assert bool(kpts) is True
    assert kpts.conf is None
    assert torch.allclose(kpts.xy, data_2d)
    assert kpts.xyn.shape == (3, 17, 2)
    assert float(kpts.xyn[..., 0].max()) <= 1.0
    assert float(kpts.xyn[..., 1].max()) <= 1.0

    # 3D tensor with confidence
    data_3d = torch.cat([data_2d, torch.rand((3, 17, 1))], dim=-1)
    kpts_3d = Keypoints(data_3d, orig_shape=orig_shape)
    assert kpts_3d.conf is not None
    assert kpts_3d.conf.shape == (3, 17)

    # Invalid shapes / types
    with pytest.raises(TypeError):
        Keypoints(data_2d.numpy(), orig_shape=orig_shape)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        Keypoints(torch.rand((3, 17, 4)), orig_shape=orig_shape)


def test_results_with_keypoints_serialization_and_plotting() -> None:
    orig_img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes_data = torch.tensor([[10.0, 10.0, 100.0, 100.0, 0.9, 0.0]])
    boxes = Boxes(boxes_data, orig_shape=(480, 640))
    kpts_data = torch.rand((1, 17, 3)) * 400
    kpts = Keypoints(kpts_data, orig_shape=(480, 640))

    results = Results(
        orig_img=orig_img,
        path="test.jpg",
        names={0: "person"},
        boxes=boxes,
        keypoints=kpts,
    )

    assert results.keypoints is not None
    assert len(results.keypoints) == 1
    assert "keypoints=1" in repr(results)

    json_data = results.to_json()
    assert len(json_data) == 1
    assert "keypoints" in json_data[0]
    assert len(json_data[0]["keypoints"]["x"]) == 17
    assert len(json_data[0]["keypoints"]["y"]) == 17
    assert len(json_data[0]["keypoints"]["visible"]) == 17

    plotted = results.plot()
    assert plotted.shape == (480, 640, 3)
    assert isinstance(plotted, np.ndarray)


def test_dfine_pose_model_inference() -> None:
    config = make_pose_config("detrpose_n")
    model = build_model(config)
    names = {0: "person"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "detrpose_n_wrapped.pth"
        save_checkpoint(ckpt_path, model, config, names)

        dfine = DFINE(ckpt_path, task="pose")
        assert dfine.task == "pose"

        img = np.zeros((640, 640, 3), dtype=np.uint8)
        res_list = dfine(img, conf=0.1)

        assert isinstance(res_list, list)
        assert len(res_list) == 1
        res = res_list[0]

        assert res.boxes is not None
        assert res.keypoints is not None
        assert res.keypoints.xy.ndim == 3
        assert res.keypoints.xy.shape[1] == 17
