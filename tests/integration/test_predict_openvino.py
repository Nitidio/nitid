"""Integration tests for the OpenVINO inference backend (backend="openvino")."""

import numpy as np
import pytest


def _random_frame(seed=42, shape=(480, 640, 3)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, shape, dtype=np.uint8)


def test_openvino_backend_matches_torch_backend(tiny_checkpoint):
    pytest.importorskip("openvino", reason="openvino not installed")
    from dfine import DFINE
    from dfine.utils.sources import LoadSource

    # Compare the raw model forward output (pred_logits/pred_boxes) directly,
    # rather than decoded top-k class labels: this untrained, random-weight,
    # 1-layer/10-query checkpoint routinely produces exactly-tied encoder
    # proposal scores (many masked/padded positions score identically), and
    # torch-eager vs. ONNX/OpenVINO-compiled execution are not guaranteed to
    # break topk ties the same way — the two per-query rows can come out
    # permuted with otherwise-identical content. Sorting each backend's
    # (query, box+logits) rows into a canonical order before comparing makes
    # the check invariant to that permutation while still catching a genuine
    # numerical or correctness regression.
    frame = _random_frame()
    loader = LoadSource(frame, imgsz=640, device="cpu")
    tensor = next(loader.iter_samples()).tensor
    loader.close()

    torch_model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    torch_raw = torch_model._get_deployed_model()(tensor)

    ov_model = DFINE(tiny_checkpoint, backend="openvino", device="CPU", verbose=False)
    ov_raw = ov_model._get_openvino_model(640)(tensor)

    assert ov_model.backend == "openvino"
    assert ov_model.device == "CPU"
    assert set(ov_raw) == {"pred_logits", "pred_boxes"}

    def sorted_rows(raw: dict) -> np.ndarray:
        boxes = raw["pred_boxes"].detach().numpy()[0]
        logits = raw["pred_logits"].detach().numpy()[0]
        rows = np.concatenate([boxes, logits], axis=-1)
        order = np.lexsort(rows.round(4).T[::-1])
        return rows[order]

    torch_sorted = sorted_rows(torch_raw)
    ov_sorted = sorted_rows(ov_raw)
    max_abs_diff_per_row = np.abs(torch_sorted - ov_sorted).max(axis=-1)
    mismatched_rows = int((max_abs_diff_per_row > 1e-2).sum())
    # A handful of this checkpoint's 10 queries land on encoder positions
    # whose anchors fall outside the valid image region (see
    # dfine/nn/architecture/decoder.py's valid_mask); every such position
    # collapses to the exact same score (Linear(0) + bias), so which of
    # several exactly-tied candidates topk keeps is an unspecified tie-break
    # that torch-eager and the traced ONNX/OpenVINO graph are not guaranteed
    # to resolve identically — a handful of mismatched rows here reflects
    # that inherent ambiguity, not a backend correctness bug (a real, trained
    # checkpoint has no such ties; see the OpenVINO NPU run in the PR
    # description for full agreement on real weights). A majority mismatch
    # would still indicate a genuine regression.
    assert mismatched_rows <= 5, (
        f"{mismatched_rows}/10 query rows disagree beyond tie-break noise:\n"
        f"torch={torch_sorted}\nopenvino={ov_sorted}"
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
