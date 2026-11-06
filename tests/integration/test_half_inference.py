"""Exercise half prediction with native architectures and optional CUDA hardware."""

import numpy as np
import pytest
import torch

from dfine import DFINE


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "checkpoint,task,imgsz",
    [
        ("tiny_checkpoint", "detect", 640),
        ("tiny_segment_checkpoint", "segment", 640),
        ("tiny_semantic_checkpoint", "semantic", 64),
    ],
)
def test_native_half_prediction(request, checkpoint, task, imgsz, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("Requires an NVIDIA CUDA GPU")
    model = DFINE(request.getfixturevalue(checkpoint), task=task, device=device, verbose=False)
    deployed = model._get_deployed_model()
    conv = next(layer for layer in deployed.modules() if isinstance(layer, torch.nn.Conv2d))
    original_dtypes = [p.dtype for p in model._model.parameters()]
    deployed_dtypes = [p.dtype for p in deployed.parameters()]
    dtypes = []
    hook = conv.register_forward_hook(lambda module, args, output: dtypes.append(output.dtype))
    try:
        result = model.predict(
            np.full((64, 64, 3), 80, dtype=np.uint8),
            imgsz=imgsz,
            half=True,
            conf=0,
            return_probs=task == "semantic",
        )[0]
    finally:
        hook.remove()
    assert dtypes and set(dtypes) == {torch.float16 if device.startswith("cuda") else torch.float32}
    values = result.semantic.probs if task == "semantic" else result.boxes.data
    assert values.dtype == torch.float32
    assert torch.isfinite(values).all()
    if task == "segment":
        assert result.masks is not None
        assert len(result.masks) == len(result.boxes)
    assert [p.dtype for p in model._model.parameters()] == original_dtypes
    assert [p.dtype for p in deployed.parameters()] == deployed_dtypes
