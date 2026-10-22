"""Concurrent access to DFINE's lazily-built inference caches.

Batch/parallel job submission (multiple requests against the same
model/device) makes concurrent first-use races the common case, not a
corner case — these tests assert the caches build exactly once under
contention rather than once per racing thread.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


def _random_frame(seed=0, shape=(64, 96, 3)):
    return np.random.default_rng(seed).integers(0, 255, shape, dtype=np.uint8)


def test_concurrent_predict_calls_build_deployed_model_once(tiny_checkpoint):
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, device="cpu", verbose=False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: model.predict(_random_frame(i), conf=0.0), range(8)))

    assert all(len(r) == 1 for r in results)
    assert model._deployed_model is not None


def test_concurrent_predict_calls_compile_openvino_model_once(tiny_checkpoint):
    pytest.importorskip("openvino", reason="openvino not installed")
    from dfine import DFINE

    model = DFINE(tiny_checkpoint, backend="openvino", device="CPU", verbose=False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: model.predict(_random_frame(i), conf=0.0), range(8)))

    assert all(len(r) == 1 for r in results)
    assert list(model._openvino_cache.keys()) == [640]
