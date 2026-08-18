# CLAUDE.md

This file provides repository guidance to coding agents.

## Project overview

nitid provides an Ultralytics-style API for D-FINE object detection and instance segmentation. The public entry point is `DFINE`; model construction, losses, and postprocessing are implemented in the `dfine` package.

```python
from dfine import DFINE

detector = DFINE("dfine_s", task="detect")
segmenter = DFINE("dfine_s", task="segment")
```

## Commands

```bash
uv sync --extra dev --extra train
uv run pytest tests/unit
uv run pytest tests/integration
uv run ruff check dfine tools tests
uv run ruff format --check dfine tools tests
uv run mypy dfine tools
```

## Architecture

- `dfine/model.py`: the public `DFINE` class and task/checkpoint validation.
- `dfine/predictor.py`: detection and instance-mask inference.
- `dfine/trainer.py`: detection and segmentation fine-tuning.
- `dfine/validator.py`: COCO bounding-box and mask evaluation.
- `dfine/exporter.py`: ONNX, OpenVINO, TorchScript, and TensorRT export.
- `dfine/results.py`: public `Results`, `Boxes`, and `Masks` types.
- `dfine/nn/architecture/`: backbone, encoder, transformer decoder, and mask head.
- `dfine/nn/losses/`: matching and detection/mask criteria.
- `dfine/nn/configs.py`: supported model-size configurations.
- `dfine/utils/data.py`: COCO and YOLO detection/segmentation datasets.

`DFINE("dfine_s", task="detect")` and `DFINE("dfine_s", task="segment")` resolve separate pretrained registries. Explicit checkpoint paths must contain a self-contained `config`, `model`, and `names`, and their embedded task must match the requested task.

## Checkpoint conversion

Raw checkpoints can be wrapped without a separate model config:

```bash
uv run python tools/convert_checkpoint.py \
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
