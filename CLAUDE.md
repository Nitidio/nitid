# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nitid** is an Ultralytics-style wrapper for the [D-FINE](https://github.com/Peterande/D-FINE) real-time object detector. The public API intentionally mirrors `ultralytics.YOLO` so users can swap models with minimal friction. The project was designed in five phases: Design → Core → Fine-tuning → Testing → Docs.

**D-FINE submodule** lives at `extern/dfine` (repo: `https://github.com/Peterande/D-FINE`). It is required for integration tests and any code that loads a model. After cloning run:
```
git submodule update --init
```

## Commands

```bash
# Install dependencies (includes dev extras: pytest, ruff, mypy)
uv sync --extra dev

# Run unit tests only (no D-FINE submodule or checkpoint needed)
uv run pytest tests/unit

# Run all tests (integration tests auto-build a tiny checkpoint via conftest.py)
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
- `dfine/model.py` — `DFINE` class: the single public object. Instantiate with a `.pth` path; call `.predict()`, `.train()`, `.val()`, `.export()`. Each method lazily imports its worker class.
- `tools/dfine_cli.py` — `dfine` CLI; parses `key=value` arguments and delegates to `DFINE`.

### Worker classes (internal, not public API)
| File | Class | Role |
|---|---|---|
| `dfine/predictor.py` | `DFINEPredictor` | Inference loop |
| `dfine/trainer.py` | `DFINETrainer` | Fine-tuning loop (Phase 3, partially implemented) |
| `dfine/validator.py` | `DFINEValidator` | COCO-style evaluation (Phase 3, not implemented) |
| `dfine/exporter.py` | `DFINEExporter` | ONNX / TorchScript export; TensorRT deferred |

### Inference data flow
1. `LoadSource` (`dfine/utils/sources.py`) — unified iterator over any input (image/video/dir/URL/webcam/ndarray). Yields `(tensor [1,3,H,W], orig_img HWC BGR, path_str)`.
2. `DFINE.forward()` → `{"pred_logits": [B,300,C], "pred_boxes": [B,300,4]}` in cxcywh normalised coords.
3. `DFINEPostProcessor(raw, orig_target_sizes)` → `[{labels, boxes, scores}]` per image, boxes in absolute xyxy pixel coords.
4. `Results` / `Boxes` (`dfine/results.py`) — public return type. `Boxes._data` is `[N,6]` (x1 y1 x2 y2 conf cls).

### Checkpoint format
Self-contained `.pth` files with keys: `model` (state_dict), `config` (YAML dict from D-FINE), `names` ({int: str}), `epoch`, `metrics`.

Raw D-FINE checkpoints need conversion — use real D-FINE configs from `extern/dfine/configs/`:
```bash
uv run python tools/convert_checkpoint.py \
    --weights dfine_l.pth \
    --config  extern/dfine/configs/dfine/dfine_hgnetv2_l_coco.yml \
    --names   configs/datasets/coco.yml \
    --output  dfine_l_wrapped.pth
```

`--names` must be a file with a `names:` mapping (nitid's `configs/datasets/coco.yml`, not the
D-FINE dataset config which has no class names). The converter prefers EMA weights
(`ckpt["ema"]["module"]`) when present, matching D-FINE's own inference scripts.

### nn layer (`dfine/nn/`)

`build_model(cfg)`, `build_postprocessor(cfg)`, `build_criterion(cfg)` in `build.py` / `criterion.py`.

**D-FINE import shim** — D-FINE's `src/__init__.py` eagerly imports `src.data` (needs `faster_coco_eval`) and `src.misc` (needs `calflops`, `loguru`). These are training-only deps not required for inference. `_ensure_dfine_on_path()` must be called before any `src.*` import; it adds `extern/dfine` to `sys.path` and pre-registers three stub modules in `sys.modules` so Python never runs those `__init__.py` files:

| Stub | Real thing blocked | What it exposes |
|---|---|---|
| `src` | `src/__init__` (imports data) | namespace package |
| `src.data` | `coco_dataset` → `faster_coco_eval` | `DataLoader` from torch |
| `src.misc` | `profiler_utils` → `calflops` | namespace package (real sub-modules importable) |

`build_model` also forces `HGNetv2.pretrained=False` (weights come from the checkpoint) and `build_postprocessor` forces `remap_mscoco_category=False` (class names come from the checkpoint's `names` dict).

### Export constraint
The model pre-computes positional anchors for `eval_spatial_size` (default `[640, 640]`). Export `imgsz` **must match** this value or the encoder will raise a shape error. The size is stored in `cfg["eval_spatial_size"]`.

### Configuration files
- `extern/dfine/configs/dfine/` — canonical D-FINE model configs (used by `convert_checkpoint.py` and the test fixture).
- `configs/models/` — placeholder YAMLs (kept for reference; prefer the extern configs).
- `configs/datasets/coco.yml` / `example_custom.yml` — dataset path and class-name definitions.

## Testing

- `tests/unit/` — pure Python; no GPU, no checkpoint, no submodule required.
- `tests/integration/` — use a session-scoped `tiny_checkpoint` fixture in `tests/conftest.py` that builds a small D-FINE-S model with random weights at test time (no download). The fixture overrides `num_layers=1`, `num_queries=10`, `num_denoising=0`, `depth_mult=0.1` to keep build time fast.
- `tests/integration/test_train.py` — marked `xfail` (Phase 3 not implemented).
