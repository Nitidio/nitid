"""Precision, streaming, and model-preservation coverage without downloaded weights."""

from contextlib import contextmanager

import numpy as np
import pytest
import torch
import yaml

from dfine import DFINE


class SmallDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1)
        self.forward_dtypes = []

    def forward(self, x):
        features = self.conv(x).mean(dim=(2, 3)).unsqueeze(1)
        self.forward_dtypes.append(features.dtype)
        return {"pred_logits": features[..., :2], "pred_boxes": features[..., 2:].sigmoid()}


@pytest.fixture
def make_model(monkeypatch):
    def load(self, path, *, task, weights):
        self._model = SmallDetector().eval().to(self._device_str)
        self._cfg = {"task": "detect", "num_classes": 2, "DFINETransformer": {"num_queries": 1}}
        self._names = {0: "first", 1: "second"}
        self._path = path

    monkeypatch.setattr(DFINE, "_load", load)
    return lambda device="cpu": DFINE("test", device=device, verbose=False)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("augment", [False, True])
def test_cpu_half_falls_back_and_preserves_weights(make_model, stream, augment, tmp_path):
    model = make_model()
    original = {key: value.clone() for key, value in model._model.state_dict().items()}
    frame = np.full((32, 32, 3), 80, dtype=np.uint8)
    expected = model.predict(frame, imgsz=32, conf=0, augment=augment)[0]
    results = list(
        model.predict(
            [frame, frame],
            imgsz=32,
            conf=0,
            half=True,
            stream=stream,
            augment=augment,
            save=True,
            save_dir=tmp_path / "run",
        )
    )
    assert len(results) == 2
    for result in results:
        torch.testing.assert_close(result.boxes.data, expected.boxes.data)
        assert result.boxes.data.dtype == torch.float32
    assert set(model._deployed_model.forward_dtypes) == {torch.float32}
    for key, value in model._model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    metadata = yaml.safe_load((tmp_path / "run" / "args.yaml").read_text())
    assert metadata["half"] is True
    assert metadata["half_enabled"] is False


def test_autocast_output_is_promoted_before_postprocessing(make_model, monkeypatch):
    predictor = make_model().predictor
    original_autocast = torch.autocast
    entered = []

    @contextmanager
    def simulated_cuda_autocast(device_type, *, dtype):
        assert device_type == "cuda"
        assert dtype == torch.float16
        entered.append(True)
        # Exercise real lower-precision kernels on CPU, without pretending
        # this substitutes for the optional NVIDIA hardware test below.
        with original_autocast("cpu", dtype=torch.bfloat16):
            yield
        entered.pop()

    monkeypatch.setattr(torch, "autocast", simulated_cuda_autocast)
    with torch.no_grad():
        raw = predictor._forward(torch.ones(1, 3, 32, 32), half=True)
    assert predictor.model.forward_dtypes == [torch.bfloat16]
    assert all(value.dtype == torch.float32 for value in raw.values())
    assert not entered
    assert not torch.is_autocast_enabled("cpu")
    assert all(p.dtype == torch.float32 for p in predictor.model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an NVIDIA CUDA GPU")
def test_cuda_half_streaming_and_fp32_switch(make_model):
    model = make_model("cuda:0")
    frame = np.full((32, 32, 3), 80, dtype=np.uint8)
    stream = model.predict([frame, frame], imgsz=32, conf=0, half=True, stream=True, augment=True)
    result = next(stream)
    assert set(model._deployed_model.forward_dtypes) == {torch.float16}
    assert result.boxes.data.dtype == torch.float32
    assert torch.isfinite(result.boxes.data).all()
    assert not torch.is_autocast_enabled("cuda")
    stream.close()
    model.predict(frame, imgsz=32, half=False)
    assert model._deployed_model.forward_dtypes[-1] == torch.float32
    assert all(p.dtype == torch.float32 for p in model._model.parameters())
    assert all(p.dtype == torch.float32 for p in model._deployed_model.parameters())
