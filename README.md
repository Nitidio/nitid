# nitid

**Ultralytics-style wrapper for [D-FINE](https://github.com/Peterande/D-FINE) — real-time object detection that feels like YOLO.**

[![CI](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml/badge.svg)](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://Vaelsys.github.io/nitid/)
[![codecov](https://codecov.io/gh/Vaelsys/nitid/graph/badge.svg)](https://codecov.io/gh/Vaelsys/nitid)
[![License](https://img.shields.io/github/license/Vaelsys/nitid)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vaelsys/nitid/blob/main/examples/tutorial.ipynb)

nitid gives D-FINE a single-class API that mirrors `ultralytics.YOLO`. Swap one import and keep all the patterns you already know: predict, train, val, export, stream, CLI.

---

## Table of contents

- [Installation](#installation)
- [Pretrained models](#pretrained-models)
- [Quickstart](#quickstart)
- [CLI](#cli)
- [Training and validation](#training-and-validation)
- [Export](#export)
- [Converting a raw D-FINE checkpoint](#converting-a-raw-d-fine-checkpoint)
- [Web application](#web-application)
- [Documentation](#documentation)
- [Development](#development)
- [Acknowledgements](#acknowledgements)

---

## Installation

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Vaelsys/nitid.git && cd nitid
git submodule update --init        # pulls extern/dfine
uv sync
```

For fine-tuning and validation add the `train` extra:

```bash
uv sync --extra train
```

For TensorRT export:

```bash
uv sync --extra tensorrt
```

For the web application:

```bash
uv sync --extra web
```

---

## Pretrained models

nitid loads **wrapped** `.pth` checkpoints. The table below lists the official upstream D-FINE COCO checkpoints. When using these raw checkpoints, they must be converted first (see [Converting a raw D-FINE checkpoint](#converting-a-raw-d-fine-checkpoint)).

| Model | Size | COCO mAP<sup>val 50-95</sup> <br><sup>*(vs YOLO11)*</sup> | Speed<sup>T4 TRT10 FP16 (ms)</sup> <br><sup>*(vs YOLO11)*</sup> | Params<sup>(M)</sup> <br><sup>*(vs YOLO11)*</sup> | FLOPs<sup>(B)</sup> <br><sup>*(vs YOLO11)*</sup> | Config | Download raw checkpoint | Wrapped output |
|---|---:|---:|---:|---:|---:|---|---|---|
| **D-FINE-N** | 640 | **42.8** <br><sup>*(vs 40.9)*</sup> | **2.12** <br><sup>*(vs 1.70)*</sup> | **4.0** <br><sup>*(vs 2.6)*</sup> | **7** <br><sup>*(vs 6.5)*</sup> | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_n_coco.yml) | [dfine_n_coco.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth) | `dfine_n_wrapped.pth` |
| **D-FINE-S** | 640 | **48.5** <br><sup>*(vs 48.6)*</sup> | **3.49** <br><sup>*(vs 2.50)*</sup> | **10.0** <br><sup>*(vs 9.4)*</sup> | **25** <br><sup>*(vs 21.5)*</sup> | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_s_coco.yml) | [dfine_s_coco.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth) | `dfine_s_wrapped.pth` |
| **D-FINE-M** | 640 | **52.3** <br><sup>*(vs 53.1)*</sup> | **5.62** <br><sup>*(vs 4.70)*</sup> | **19.0** <br><sup>*(vs 20.1)*</sup> | **57** <br><sup>*(vs 68.0)*</sup> | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_m_coco.yml) | [dfine_m_coco.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth) | `dfine_m_wrapped.pth` |
| **D-FINE-L** | 640 | **54.0** <br><sup>*(vs 55.0)*</sup> | **8.07** <br><sup>*(vs 6.20)*</sup> | **31.0** <br><sup>*(vs 25.3)*</sup> | **91** <br><sup>*(vs 86.9)*</sup> | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_l_coco.yml) | [dfine_l_coco.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth) | `dfine_l_wrapped.pth` |
| **D-FINE-X** | 640 | **55.8** <br><sup>*(vs 57.5)*</sup> | **12.89** <br><sup>*(vs 11.80)*</sup> | **62.0** <br><sup>*(vs 56.9)*</sup> | **202** <br><sup>*(vs 194.9)*</sup> | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_x_coco.yml) | [dfine_x_coco.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth) | `dfine_x_wrapped.pth` |

*Note: Metrics in <sup>*(vs YOLO11)*</sup> correspond to the equivalent YOLO11 model variant (YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x) for rough context. Benchmark methods and export formats can differ between projects, so use this as a high-level comparison rather than a strict apples-to-apples speed benchmark.*

---

## Quickstart

nitid loads **wrapped** `.pth` checkpoints—self-contained files that embed the model config and class names alongside the weights. These are handled automatically on the fly when you specify a model name (e.g. `dfine_s`) or when using the download CLI.

### Download a checkpoint

Download an official D-FINE checkpoint and wrap it for nitid automatically:

```bash
uv run dfine download model=dfine_s
```

Supported model names are:

```text
dfine_s, dfine_m, dfine_l, dfine_x
```

By default, the command saves the wrapped checkpoint in the current directory:

```text
dfine_s_wrapped.pth
```

To save checkpoints into a folder:

```bash
uv run dfine download model=dfine_s output=models
```

To overwrite an existing checkpoint:

```bash
uv run dfine download model=dfine_s output=models force=true
```

### Inference

```python
from dfine import DFINE

model = DFINE("dfine_s_wrapped.pth")
results = model.predict("image.jpg", conf=0.5)
results[0].save("out.jpg")
```

### Stream video (memory-efficient)

```python
for r in model.predict("video.mp4", stream=True, conf=0.3):
    annotated = r.plot()   # HWC BGR ndarray
```

### Work with detections

```python
r = results[0]
r.boxes.xyxy    # [N, 4]  absolute pixel coords x1 y1 x2 y2
r.boxes.xyxyn   # [N, 4]  normalised 0–1
r.boxes.xywh    # [N, 4]  cx cy w h absolute
r.boxes.conf    # [N]     confidence scores
r.boxes.cls     # [N]     class indices

r.save("out.jpg")   # write annotated image
r.show()            # display window
r.to_json()         # list of dicts
```

### Drop-in replacement for Ultralytics YOLO

```python
# Before
from ultralytics import YOLO
model = YOLO("yolo11n.pt")

# After
from dfine import DFINE
model = DFINE("dfine_l_wrapped.pth")
```

The interface is intentionally close to `ultralytics.YOLO`. All the patterns you already know work the same way.

### Model info

```python
model.info()
# [D-FINE] 31.4M params (31.4M trainable)  120.3 GFLOPs  98.6 MB
```

---

## CLI

All methods are available from the command line using `key=value` arguments:

```bash
uv run dfine download model=dfine_s
uv run dfine predict model=dfine_l_wrapped.pth source=image.jpg conf=0.5
uv run dfine train  model=dfine_l_wrapped.pth data=my_dataset.yml epochs=50
uv run dfine val    model=dfine_l_wrapped.pth data=my_dataset.yml
uv run dfine export model=dfine_l_wrapped.pth format=onnx
```

---

## Training and validation

Fine-tune on any COCO-format dataset:

```python
model = DFINE("dfine_l_wrapped.pth")
metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    batch=16,
    lr0=1e-4,
    optimizer="AdamW",
    amp=True,    # FP16 mixed precision (CUDA only)
    ema=True,    # EMA weight averaging
)
```

Evaluate:

```python
metrics = model.val(data="configs/datasets/my_dataset.yml")
# {"mAP50-95": ..., "mAP50": ..., "AR1": ..., "AR100": ...}
```

Checkpoints are saved after every epoch as wrapped `.pth` files and can be loaded directly:

```python
model = DFINE("runs/train/exp/epoch50.pth")
```

See [docs/fine_tuning.md](docs/fine_tuning.md) for dataset format, data YAML layout, AMP, EMA, and all training parameters.

---

## Export

```python
model.export(format="onnx")         # → dfine_l_wrapped.onnx
model.export(format="torchscript")  # → dfine_l_wrapped.torchscript
model.export(format="tensorrt")     # → dfine_640.engine  (requires tensorrt extra)
model.export(format="tensorrt", half=True)  # FP16
```

The ONNX model includes the postprocessor in deploy mode and outputs `(labels, boxes, scores)` directly. The export `imgsz` must match the model's `eval_spatial_size` (default 640).

See [docs/export.md](docs/export.md) for all options including dynamic batch axes and FP16 TensorRT.

---

## Converting a raw D-FINE checkpoint

Raw D-FINE checkpoints need to be wrapped before nitid can load them. The converter embeds the model config and class names so you always deal with a single self-contained file.

```bash
uv run python tools/convert_checkpoint.py \
    --weights dfine_l.pth \
    --config  extern/dfine/configs/dfine/dfine_hgnetv2_l_coco.yml \
    --names   configs/datasets/coco.yml \
    --output  dfine_l_wrapped.pth
```

| Argument   | Description |
|------------|-------------|
| `--weights` | Raw D-FINE checkpoint (downloaded from the D-FINE repo) |
| `--config`  | Canonical D-FINE config from `extern/dfine/configs/` |
| `--names`   | YAML file with a `names:` mapping — use `configs/datasets/coco.yml` for COCO models |
| `--output`  | Path for the wrapped checkpoint |

When the raw checkpoint contains EMA weights (`ckpt["ema"]["module"]`), the converter uses them automatically — this matches D-FINE's own inference scripts and gives better accuracy.

---

## Web application

nitid includes a browser-based UI for running detection without writing code. Upload images or videos, pick a model, adjust parameters, and browse annotated results with persistent run history per user.

```bash
# Install web extras
uv sync --extra web

# Place a wrapped checkpoint
mkdir -p models && cp dfine_l_wrapped.pth models/

# Start the API (single worker — inference is not thread-safe)
uv run uvicorn web.api.main:app --workers 1

# Start the frontend (separate terminal)
cd web/frontend && npm install && npm run dev
# → open http://localhost:5173
```

Register an account on first visit. The API is self-documented at `http://localhost:8000/docs`.

See [docs/web_app.md](docs/web_app.md) for the full guide: environment variables, REST API reference, data model, and implementation notes.

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/onboarding.md](docs/onboarding.md) | **Start here if you're a new developer** — architecture, conventions, gotchas |
| [docs/quickstart.md](docs/quickstart.md) | Full quickstart for D-FINE users and Ultralytics users |
| [docs/fine_tuning.md](docs/fine_tuning.md) | Training, validation, AMP, EMA, dataset format |
| [docs/export.md](docs/export.md) | ONNX, TorchScript, TensorRT export |
| [docs/api_reference.md](docs/api_reference.md) | Full `DFINE` class API reference |
| [docs/web_app.md](docs/web_app.md) | Web application: setup, UI guide, REST API, data model |

---

## Development

```bash
uv sync --extra dev

# Unit tests (no GPU, no checkpoint needed)
uv run pytest tests/unit

# All tests (integration tests build a tiny checkpoint automatically)
uv run pytest

# Lint
uv run ruff check .

# Type-check
uv run mypy dfine/
```

Integration tests use a session-scoped fixture in `tests/conftest.py` that builds a small D-FINE-S model with random weights at test time — no download required.

### Key differences from Ultralytics YOLO

| Feature | Ultralytics YOLO | nitid DFINE |
|---------|-----------------|-------------|
| Checkpoint format | `.pt` (architecture inferred from filename) | `.pth` (config embedded inside) |
| Raw weights | Download directly | Run `convert_checkpoint.py` first |
| `model.info()` | Returns param/FLOP stats | Supported — params, GFLOPs, disk size |
| TensorRT export | Supported | Supported (`uv sync --extra tensorrt`) |
| AMP / EMA training | Supported | Supported (`amp=True`, `ema=True`) |
| `model.task` | `"detect"`, `"segment"`, … | Always `"detect"` |

---

## Acknowledgements

nitid wraps [D-FINE](https://github.com/Peterande/D-FINE) by Yansong Peng, Hebei University. D-FINE is licensed under the [Apache 2.0 License](extern/dfine/LICENSE).

```bibtex
@article{peng2024dfine,
  title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
  author={Peng, Yansong and Shi, Hongtao and Li, Shiyu and Wang, Yan and Li, Hongbin and Liu, Guozheng and Li, Bing and Hu, Weiming},
  journal={arXiv preprint arXiv:2410.13842},
  year={2024}
}
```
