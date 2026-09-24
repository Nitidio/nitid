# CLAUDE.md

This file provides repository guidance to coding agents.

## Project overview

nitid provides a unified API for detection, segmentation, pose, and oriented
bounding boxes. The public entry point is `NITID` from the `nitid` package;
model construction, losses, and postprocessing are implemented internally in
the `dfine` package.

```python
from nitid import NITID

detector = NITID("nitid1s", task="detect")
segmenter = NITID("nitid1s", task="segment")
```

## Commands

```bash
uv sync --extra dev --extra train
uv run pytest tests/unit
uv run pytest tests/integration
uv run ruff check dfine nitid tools tests
uv run ruff format --check dfine nitid tools tests
uv run mypy dfine nitid tools
```

## Architecture

- `dfine/model.py`: the internal D-FINE model class and task/checkpoint validation.
- `dfine/nitid.py`: the public `NITID` class and model-name mapping.
- `dfine/predictor.py`: detection and instance-mask inference.
- `dfine/trainer.py`: detection and segmentation fine-tuning.
- `dfine/validator.py`: COCO bounding-box and mask evaluation.
- `dfine/exporter.py`: ONNX, OpenVINO, TorchScript, and TensorRT export.
- `dfine/results.py`: public `Results`, `Boxes`, and `Masks` types.
- `dfine/nn/architecture/`: backbone, encoder, transformer decoder, and mask head.
- `dfine/nn/openvino_runtime.py`: OpenVINO Runtime inference backend (Intel CPU/iGPU/NPU), used when `DFINE`/`NITID` is constructed with `backend="openvino"`.
- `dfine/nn/losses/`: matching and detection/mask criteria.
- `dfine/nn/configs.py`: supported model-size configurations.
- `dfine/utils/data.py`: COCO and YOLO detection/segmentation datasets.

`NITID("nitid1s", task="detect")` and `NITID("nitid1s", task="segment")`
resolve separate pretrained registries. Explicit checkpoint paths must contain
a self-contained `config`, `model`, and `names`, and their embedded task must
match the requested task.

## Checkpoint conversion

Raw checkpoints can be wrapped without a separate model config:

```bash
uv run python -m nitid.convert_checkpoint \
    --weights dfine_l.pth \
    --model dfine_l \
    --task detect \
    --names configs/datasets/coco.yml \
    --output dfine_l_wrapped.pth
```

The converter prefers EMA weights when present. Model weights always come from the checkpoint, and class names come from its `names` mapping.

## Testing

Integration tests build compact detection and segmentation checkpoints at runtime, so tests do not download pretrained weights. Add focused unit coverage for parsing and utilities, and integration coverage for public prediction, training, validation, and export behavior.

Some exporters require platform-specific runtimes. TensorRT needs a compatible NVIDIA environment; OpenVINO tests require the optional OpenVINO dependencies.
