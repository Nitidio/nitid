# nitid

**Ultralytics-style wrapper for [D-FINE](https://github.com/Peterande/D-FINE) — real-time object detection that feels like YOLO.**

[![CI](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml/badge.svg)](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://Vaelsys.github.io/nitid/)
[![codecov](https://codecov.io/gh/Vaelsys/nitid/graph/badge.svg)](https://codecov.io/gh/Vaelsys/nitid)
[![License](https://img.shields.io/github/license/Vaelsys/nitid)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vaelsys/nitid/blob/main/examples/tutorial.ipynb)

nitid gives D-FINE a single-class API that mirrors `ultralytics.YOLO`. Swap one import and keep all the patterns you already know: predict, track, train, val, export, stream, CLI.

![nitid detection demo](docs/assets/nitid-demo.png)

Example prediction using D-FINE-S on a street image.

## Why nitid?

- Familiar Ultralytics-style API
- Automatic download and wrapping of official D-FINE checkpoints
- Python API, CLI and web interface
- ByteTrack object tracking with persistent IDs and annotated video output
- Fine-tuning and validation
- ONNX, TorchScript and TensorRT export

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

For ByteTrack object tracking add the `track` extra:

```bash
uv sync --extra track
```

TensorRT:

``` bash
pip install --extra-index-url https://pypi.nvidia.com tensorrt>=8.6
```
For the web application:

```bash
uv sync --extra web
```


## Quick Start

The canonical quickstart lives in [docs/quickstart.md](docs/quickstart.md); use the docs guide for the full walkthrough.

### Inference

```python
from dfine import DFINE

model = DFINE("dfine_s")
# Equivalent explicit selection: DFINE("dfine_s", weights="obj2coco")
results = model.predict("image.jpg", conf=0.5)
results[0].save("out.jpg")
```

### Training

```python
model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
)
# returns final metrics plus per-epoch history in metrics["history"]
```

### Tracking

```python
for result in model.track("video.mp4", conf=0.5, stream=True, save=True):
    if result.boxes.id is not None:
        track_ids = result.boxes.id
```

The annotated video is saved under `runs/track/exp/`. Tracking dependencies
are optional; install them with `uv sync --extra track`.

### Validation

```python
metrics = model.val(
    data="configs/datasets/my_dataset.yml",
    project="runs/val",
    name="exp",
)
# saves validation plots to runs/val/exp by default
```

### Export

```python
model.export(format="onnx")
model.export(format="torchscript")
model.export(format="tensorrt")
```

Capture Python API output, environment details, and failure tracebacks in one
attachable log:

```python
from dfine import DFINE, bugreport

with bugreport("prediction") as report:
    model = DFINE("dfine_s")
    model.predict("image.jpg")

print(report.path)
```


### Command Line Interface

```bash
uv run dfine predict model=dfine_s source=image.jpg
uv run dfine track model=dfine_s source=video.mp4 conf=0.5 save=true
uv run dfine train model=dfine_s data=my_dataset.yml epochs=50
uv run dfine val model=dfine_s data=my_dataset.yml
uv run dfine export model=dfine_s format=onnx
uv run dfine predict model=dfine_s source=image.jpg --report
uv run dfine bugreport
```

For the full guide:

- Full quickstart: [docs/quickstart.md](docs/quickstart.md)
- Fine-tuning and validation: [docs/fine_tuning.md](docs/fine_tuning.md)
- Export: [docs/export.md](docs/export.md)
- CLI: [docs/cli.md](docs/cli.md)


## Official Models

> 💡 Passing `dfine_s`, `dfine_m`, `dfine_l`, or `dfine_x` downloads the recommended Objects365→COCO weights. Use `weights="coco"` for the COCO-only checkpoint or `weights="obj2coco"` to make the default provenance explicit.

| Model | COCO mAP<sup>50-95</sup> *(vs YOLO11)* | Speed<sup>T4 TRT10 FP16</sup> *(vs YOLO11)* | Params | FLOPs | Config | Official Checkpoint |
|:------|---------------------------------------:|--------------------------------------------:|-------:|------:|:------:|:-------------------:|
| **D-FINE-N** | **42.8** *(40.9)* | **2.12 ms** *(1.70)* | 4.0M | 7B | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_n_coco.yml) | [pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth) |
| **D-FINE-S** | **50.7** *(48.6)* | **3.49 ms** *(2.50)* | 10.0M | 25B | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml) | [pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth) |
| **D-FINE-M** | **55.1** *(53.1)* | **5.62 ms** *(4.70)* | 19.0M | 57B | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/objects365/dfine_hgnetv2_m_obj2coco.yml) | [pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj2coco.pth) |
| **D-FINE-L** | **57.3** *(55.0)* | **8.07 ms** *(6.20)* | 31.0M | 91B | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/objects365/dfine_hgnetv2_l_obj2coco.yml) | [pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth) |
| **D-FINE-X** | **59.3** *(57.5)* | **12.89 ms** *(11.80)* | 62.0M | 202B | [yml](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml) | [pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth) |

*Numbers in parentheses correspond to the equivalent YOLO11 model (YOLO11n/s/m/l/x) for quick reference.*


## Web application

nitid includes a browser-based UI for running detection without writing code. Upload images or videos, pick a model, adjust parameters, and browse annotated results with persistent run history per user.

```bash
# Install web extras
uv sync --extra web

# Download the recommended wrapped checkpoint into models/
uv run dfine download model=dfine_l output=models

# Start the API (single worker — inference is not thread-safe)
uv run uvicorn web.api.main:app --workers 1

# Start the frontend (separate terminal)
cd web/frontend && npm install && npm run dev
# → open http://localhost:5173
```

Register an account on first visit. The API is self-documented at `http://localhost:8000/docs`.

See [docs/web_app.md](docs/web_app.md) for the full guide: environment variables, REST API reference, data model, and implementation notes.


## Converting a raw D-FINE checkpoint

Only needed if you have your own D-FINE checkpoint. If you're using an official model, `DFINE("dfine_s")` downloads and wraps it automatically — skip this section.

```bash
uv run python tools/convert_checkpoint.py \
    --weights dfine_l.pth \
    --config  extern/dfine/configs/dfine/dfine_hgnetv2_l_coco.yml \
    --names   configs/datasets/coco.yml \
    --output  dfine_l_wrapped.pth
```

| Argument | Description |
|---|---|
| `--weights` | Raw D-FINE checkpoint |
| `--config` | Canonical D-FINE config from `extern/dfine/configs/` |
| `--names` | YAML with a `names:` mapping — use `configs/datasets/coco.yml` for COCO models |
| `--output` | Path for the wrapped output |

EMA weights are used automatically when present, matching D-FINE's own inference scripts.

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/onboarding.md](docs/onboarding.md) | **Start here if you're a new developer** — architecture, conventions, gotchas |
| [docs/quickstart.md](docs/quickstart.md) | Full quickstart for D-FINE users and Ultralytics users |
| [docs/fine_tuning.md](docs/fine_tuning.md) | Training, validation, AMP, EMA, dataset format |
| [docs/export.md](docs/export.md) | ONNX, TorchScript, TensorRT export |
| [docs/api_reference.md](docs/api_reference.md) | Full `DFINE` class API reference |
| [docs/web_app.md](docs/web_app.md) | Web application: setup, UI guide, REST API, data model |
| [docs/troubleshooting.md](docs/troubleshooting.md) | FAQ and fixes for common install, model, Docker, CUDA, and CLI problems |


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
| `model.info()` | Returns param/FLOP stats | Supported — params, GFLOPs, disk size |
| TensorRT export | Supported | Supported (see Installation) |
| AMP / EMA training | Supported | Supported (`amp=True`, `ema=True`) |
| `model.task` | `"detect"`, `"segment"`, … | Detection only |

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, commit conventions, and the PR checklist.

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
