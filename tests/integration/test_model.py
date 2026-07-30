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


def test_model_load_trigger_download(monkeypatch, tmp_path):
    from dfine import DFINE
    from dfine.utils import checkpoint, downloads

    download_calls = []
    load_calls = []

    def fake_download_model(model, *, weights="default", output=None, force=False):
        download_calls.append((model, weights, output))
        asset = downloads.get_model_asset(model, weights=weights)
        out_path = downloads.resolve_output_path(asset, output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"dummy_weights")
        return out_path

    def fake_load_checkpoint(path, device="cpu"):
        load_calls.append(str(path))

        class DummyModel:
            def eval(self):
                pass

            def parameters(self):
                return []

        return DummyModel(), {}, {}

    monkeypatch.setattr(downloads, "download_model", fake_download_model)
    monkeypatch.setattr(checkpoint, "load_checkpoint", fake_load_checkpoint)

    # 1. Test specifying registered model directly (e.g. dfine_s)
    # It should download to default filename in current working dir (None)
    _ = DFINE("dfine_s", device="cpu", verbose=False)
    assert len(download_calls) == 1
    assert download_calls[0] == ("dfine_s", "obj2coco", None)
    assert load_calls[0] == "dfine_s_obj2coco_wrapped.pth"
    import os

    if os.path.exists("dfine_s_obj2coco_wrapped.pth"):
        os.remove("dfine_s_obj2coco_wrapped.pth")

    # 2. Test specifying path with .pth suffix (e.g. path/to/dfine_s.pth)
    download_calls.clear()
    load_calls.clear()
    custom_path = tmp_path / "custom_dir" / "dfine_s.pth"
    _ = DFINE(str(custom_path), device="cpu", verbose=False)
    assert len(download_calls) == 1
    assert download_calls[0][0:2] == ("dfine_s", "obj2coco")
    assert str(download_calls[0][2]) == str(custom_path)
    assert load_calls[0] == str(custom_path)


def test_model_weights_selects_official_variant(monkeypatch):
    from dfine import DFINE
    from dfine.utils import checkpoint, downloads

    observed = {}

    def fake_download_model(model, *, weights="default", output=None, force=False):
        observed.update(model=model, weights=weights)
        return downloads.resolve_output_path(downloads.get_model_asset(model, weights), output)

    def fake_load_checkpoint(path, device="cpu"):
        class DummyModel:
            def eval(self):
                pass

            def parameters(self):
                return []

        return DummyModel(), {}, {}

    monkeypatch.setattr(downloads, "download_model", fake_download_model)
    monkeypatch.setattr(checkpoint, "load_checkpoint", fake_load_checkpoint)

    model = DFINE("dfine_s", weights="coco", device="cpu", verbose=False)

    assert observed == {"model": "dfine_s", "weights": "coco"}
    assert model.weights == "coco"


def test_existing_checkpoint_rejects_registry_weights(tiny_checkpoint):
    from dfine import DFINE

    with pytest.raises(ValueError, match="cannot be combined"):
        DFINE(tiny_checkpoint, weights="coco", device="cpu", verbose=False)


def test_existing_checkpoint_accepts_default_weights_sentinel(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, weights="DEFAULT", device="cpu", verbose=False)
    assert model.weights is None
