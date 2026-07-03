"""Integration tests for DFINE model properties and meta-methods."""

import numpy as np
import pytest


def test_model_task_property(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    assert model.task == "detect"


def test_model_device_property(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    assert model.device == "cpu"


def test_model_default_device_uses_resolver(tiny_checkpoint):
    from dfine import DFINE
    from dfine.utils.device import resolve_device

    model = DFINE(tiny_checkpoint, verbose=False)

    assert model.device == resolve_device(None)


def test_model_names_is_dict(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    assert isinstance(model.names, dict)
    assert len(model.names) > 0


def test_model_info_returns_expected_keys(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    info = model.info(verbose=False)
    assert {"params", "params_trainable", "gflops", "size_mb"} <= info.keys()
    assert info["params"] > 0
    assert info["params_trainable"] > 0
    assert info["size_mb"] is not None and info["size_mb"] > 0


def test_model_info_gflops(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    info = model.info(verbose=False)
    # gflops may be None if profiling fails, but on a standard install it should work
    if info["gflops"] is not None:
        assert info["gflops"] > 0


def test_model_info_detailed(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    info = model.info(detailed=True, verbose=False)
    assert "layers" in info
    assert len(info["layers"]) > 0


def test_model_call_delegates_to_predict(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    via_call = model(frame, conf=0.0)
    via_predict = model.predict(frame, conf=0.0)
    assert len(via_call) == len(via_predict)


def test_model_load_missing_file():
    from dfine import DFINE

    with pytest.raises(FileNotFoundError):
        DFINE("definitely_does_not_exist.pth", device="cpu", verbose=False)
