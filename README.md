# nitid

**Ultralytics-style wrapper for [D-FINE](https://github.com/Peterande/D-FINE) — real-time object detection that feels like YOLO.**

nitid gives D-FINE a single-class API that mirrors `ultralytics.YOLO`. Swap one import and keep all the patterns you already know: predict, train, val, export, stream, CLI.

---

## Table of contents

- [Installation](#installation)
- [Pretrained models](#pretrained-models)
- [Quickstart](#quickstart)
- [CLI](#cli)
- [Web application](#web-application)
- [Converting a raw D-FINE checkpoint](#converting-a-raw-d-fine-checkpoint)
- [Training and validation](#training-and-validation)
- [Export](#export)
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

nitid loads **wrapped** `.pth` checkpoints. The table below links to the official upstream D-FINE COCO checkpoints and shows the wrapped filename to create with `tools/convert_checkpoint.py`.

> Direct nitid-wrapped release assets are not hosted yet. Until they are published, download the matching upstream D-FINE checkpoint below and convert it using the command in [Converting a raw D-FINE checkpoint](#converting-a-raw-d-fine-checkpoint).

| Model | size<br><sup>(pixels)</sup> | COCO mAP<sup>val<br>50-95</sup> | Speed<br><sup>T4 TensorRT10 FP16<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> | Config | Download raw checkpoint | Wrapped output |
|---|---:|---:|---:|---:|---:|---|---|---|
| D-FINE-N | 640 | 42.8 | 2.12 | 4 | 7 | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_n_coco.yml) | [dfine_n_coco.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth) | `dfine_n_wrapped.pth` |
| D-FINE-S | 640 | 48.5 | 3.49 | 10 | 25 | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_s_coco.yml) | [dfine_s_coco.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth) | `dfine_s_wrapped.pth` |
| D-FINE-M | 640 | 52.3 | 5.62 | 19 | 57 | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_m_coco.yml) | [dfine_m_coco.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth) | `dfine_m_wrapped.pth` |
| D-FINE-L | 640 | 54.0 | 8.07 | 31 | 91 | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_l_coco.yml) | [dfine_l_coco.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth) | `dfine_l_wrapped.pth` |
| D-FINE-X | 640 | 55.8 | 12.89 | 62 | 202 | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_x_coco.yml) | [dfine_x_coco.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth) | `dfine_x_wrapped.pth` |

Metrics are from the official D-FINE COCO model zoo. Latency is reported by D-FINE on a single T4 GPU with batch size 1, FP16, and TensorRT 10.4.0.

### YOLO reference comparison

For rough context, the table below shows published Ultralytics YOLO model metrics. Benchmark methods and export formats can differ between projects, so use this as a high-level comparison rather than a strict apples-to-apples speed benchmark.

| Model | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | mAP<sup>val<br>50-95(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n | 640 | 40.9 | 40.1 | 38.9 +/- 0.7 | 1.7 +/- 0.0 | 2.4 | 5.4 |
| YOLO26s | 640 | 48.6 | 47.8 | 87.2 +/- 0.9 | 2.5 +/- 0.0 | 9.5 | 20.7 |
| YOLO26m | 640 | 53.1 | 52.5 | 220.0 +/- 1.4 | 4.7 +/- 0.1 | 20.4 | 68.2 |
| YOLO26l | 640 | 55.0 | 54.4 | 286.2 +/- 2.0 | 6.2 +/- 0.2 | 24.8 | 86.4 |
| YOLO26x | 640 | 57.5 | 56.9 | 525.8 +/- 4.0 | 11.8 +/- 0.2 | 55.7 | 193.9 |

Example for D-FINE-L:

```bash
curl -L -o dfine_l_coco.pth \
    https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth

uv run python tools/convert_checkpoint.py \
    --weights dfine_l_coco.pth \
    --config  extern/dfine/configs/dfine/dfine_hgnetv2_l_coco.yml \
    --names   configs/datasets/coco.yml \
    --output  dfine_l_wrapped.pth
```

---

## Quickstart

nitid works with **nitid-wrapped** `.pth` checkpoints — self-contained files that embed the model config and class names alongside the weights. If you have a raw D-FINE checkpoint, [convert it first](#converting-a-raw-d-fine-checkpoint).

### Inference

```python
from dfine import DFINE

model = DFINE("dfine_l_wrapped.pth")
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
uv run dfine predict model=dfine_l_wrapped.pth source=image.jpg conf=0.5
uv run dfine train  model=dfine_l_wrapped.pth data=my_dataset.yml epochs=50
uv run dfine val    model=dfine_l_wrapped.pth data=my_dataset.yml
uv run dfine export model=dfine_l_wrapped.pth format=onnx
```

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
