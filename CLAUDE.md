# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nitid** is an Ultralytics-style wrapper for the [D-FINE](https://github.com/Peterande/D-FINE) real-time object detector. The public API intentionally mirrors `ultralytics.YOLO` so users can swap models with minimal friction.

> **Critical dependency**: D-FINE source is not bundled. `dfine/nn/build.py` and `dfine/nn/criterion.py` both raise `NotImplementedError` until D-FINE is added as a submodule:
> ```
> git submodule add https://github.com/Peterande/D-FINE extern/dfine
> ```
> Unit tests do not require this; integration tests do.

## Commands

```bash
# Install dependencies (includes dev extras: pytest, ruff, mypy)
uv sync --extra dev

# Run unit tests (no checkpoint or D-FINE source needed)
uv run pytest tests/unit

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_results.py

# Lint
uv run ruff check .

# Type-check
uv run mypy dfine/

# CLI (after install)
uv run dfine predict model=dfine_l.pth source=image.jpg conf=0.5
uv run dfine train  model=dfine_l.pth data=configs/datasets/coco.yml epochs=50
uv run dfine val    model=dfine_l.pth data=configs/datasets/coco.yml
uv run dfine export model=dfine_l.pth format=onnx
```

## Architecture

### Entry points
- `dfine/model.py` — `DFINE` class: the single public object. Instantiate with a `.pth` path; call `.predict()`, `.train()`, `.val()`, `.export()`. Each method lazily imports its worker class to keep startup fast.
- `tools/dfine_cli.py` — `dfine` CLI command; parses `key=value` arguments and delegates to `DFINE`.

### Worker classes (internal, not public API)
| File | Class | Role |
|---|---|---|
| `dfine/predictor.py` | `DFINEPredictor` | Inference loop |
| `dfine/trainer.py` | `DFINETrainer` | Fine-tuning loop |
| `dfine/validator.py` | `DFINEValidator` | COCO-style evaluation |
| `dfine/exporter.py` | `DFINEExporter` | ONNX / TorchScript export (TensorRT is Phase 2) |

### Data flow
1. `LoadSource` (`dfine/utils/sources.py`) — unified iterator that accepts image files, video files, directories, URLs, webcam indices, RTSP streams, or raw `np.ndarray`. Yields `(tensor [1,3,H,W], orig_img HWC BGR, path_str)` tuples.
2. `Results` / `Boxes` (`dfine/results.py`) — returned by `predict()`. `Boxes._data` is a `[N, 6]` tensor with columns `x1 y1 x2 y2 conf cls`. Provides `.xyxy`, `.xyxyn`, `.xywh`, `.xywhn`, `.conf`, `.cls` properties.

### Checkpoint format
Checkpoints are self-contained `.pth` files with keys: `model` (state dict), `config` (YAML dict), `names` ({int: str}), `epoch`, `metrics`. Raw D-FINE checkpoints lack `config`; convert them with:
```bash
uv run python tools/convert_checkpoint.py \
    --weights dfine_l.pth \
    --config  configs/models/dfine_l.yml \
    --names   configs/datasets/coco.yml \
    --output  dfine_l_wrapped.pth
```

### Configuration files
- `configs/models/dfine_{s,m,l,x}.yml` — model architecture configs (size variants).
- `configs/datasets/coco.yml` / `configs/datasets/example_custom.yml` — dataset path and class-name definitions.

### nn layer
`dfine/nn/build.py::build_model(cfg)` and `dfine/nn/criterion.py::build_criterion(cfg)` are thin stubs that must delegate to D-FINE's own factory once the submodule is present.

## Testing layout
- `tests/unit/` — pure-Python tests; no GPU, no checkpoint, no D-FINE source required.
- `tests/integration/` — require a wrapped checkpoint and D-FINE source (`test_train`, `test_predict`, `test_export`).
