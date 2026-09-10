"""Integration tests for the OpenVINO inference backend (backend="openvino")."""

import numpy as np
import pytest


def _random_frame(seed=42, shape=(480, 640, 3)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, shape, dtype=np.uint8)


def test_openvino_backend_matches_torch_backend(tiny_checkpoint):
    pytest.importorskip("openvino", reason="openvino not installed")
    from dfine import DFINE

    frame = _random_frame()
    torch_model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    torch_result = torch_model.predict(frame, conf=0.0)[0]

    ov_model = DFINE(tiny_checkpoint, backend="openvino", device="CPU", verbose=False)
    ov_result = ov_model.predict(frame, conf=0.0)[0]

    assert ov_model.backend == "openvino"
    assert ov_model.device == "CPU"
    assert torch_result.boxes.cls.tolist() == ov_result.boxes.cls.tolist()
    np.testing.assert_allclose(
        torch_result.boxes.xyxy.numpy(), ov_result.boxes.xyxy.numpy(), atol=1e-2
    )
    np.testing.assert_allclose(
        torch_result.boxes.conf.numpy(), ov_result.boxes.conf.numpy(), atol=1e-4
    )


def test_openvino_backend_includes_masks_for_segment_task(tiny_segment_checkpoint):
    pytest.importorskip("openvino", reason="openvino not installed")
    from dfine import DFINE

    model = DFINE(
        tiny_segment_checkpoint, task="segment", backend="openvino", device="CPU", verbose=False
    )
    result = model.predict(_random_frame(shape=(64, 96, 3)), conf=0.0)[0]
    assert result.masks is not None


def test_openvino_backend_caches_compiled_model_per_imgsz(tiny_checkpoint):
    """Predicting twice at the same imgsz reuses one compiled model; the cache
    is keyed by imgsz because OpenVINO compiles for a fixed input shape.
    (imgsz must match the checkpoint's eval_spatial_size — a pre-existing
    constraint of the static-anchor decoder shared by both backends, not
    specific to OpenVINO.)"""
    pytest.importorskip("openvino", reason="openvino not installed")
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, backend="openvino", device="CPU", verbose=False)
    model.predict(_random_frame(seed=1), conf=0.0)
    compiled_after_first_call = model._openvino_cache[640]
    model.predict(_random_frame(seed=2), conf=0.0)
    assert set(model._openvino_cache) == {640}
    assert model._openvino_cache[640] is compiled_after_first_call


def test_unknown_backend_raises(tiny_checkpoint):
    from dfine import DFINE

    with pytest.raises(ValueError, match="backend must be"):
        DFINE(tiny_checkpoint, backend="tensorflow", verbose=False)


def test_nitid_rejects_openvino_backend_for_random_obb_init():
    from dfine import NITID

    with pytest.raises(ValueError, match="randomly-initialized OBB"):
        NITID("nitid1s", task="obb", weights=None, backend="openvino", verbose=False)


def test_unavailable_openvino_device_raises_with_available_devices_listed(tiny_checkpoint):
    pytest.importorskip("openvino", reason="openvino not installed")
    from dfine import DFINE

    with pytest.raises(ValueError, match="not available on this machine"):
        DFINE(
            tiny_checkpoint,
            backend="openvino",
            device="NOT-A-REAL-DEVICE",
            verbose=False,
        )
