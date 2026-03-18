"""
Integration test for the full predict() pipeline.
Skipped unless a real checkpoint is present at DFINE_TEST_MODEL env var.
"""
import os
import pytest
import numpy as np

MODEL_PATH = os.environ.get("DFINE_TEST_MODEL", "")


@pytest.mark.skipif(not MODEL_PATH, reason="DFINE_TEST_MODEL not set")
def test_predict_numpy_frame():
    from dfine import DFINE
    model = DFINE(MODEL_PATH, verbose=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = model.predict(frame, conf=0.3)
    assert isinstance(results, list)
    assert len(results) == 1


@pytest.mark.skipif(not MODEL_PATH, reason="DFINE_TEST_MODEL not set")
def test_predict_stream_is_generator():
    from dfine import DFINE
    import types
    model = DFINE(MODEL_PATH, verbose=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gen = model.predict(frame, stream=True)
    assert isinstance(gen, types.GeneratorType)
