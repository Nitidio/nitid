# nitid

**Ultralytics-style D-FINE object detection and instance segmentation.**

[![CI](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml/badge.svg)](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://Vaelsys.github.io/nitid/)
[![codecov](https://codecov.io/gh/Vaelsys/nitid/graph/badge.svg)](https://codecov.io/gh/Vaelsys/nitid)
[![License](https://img.shields.io/github/license/Vaelsys/nitid)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vaelsys/nitid/blob/main/examples/tutorial.ipynb)

nitid gives D-FINE a single-entry-point API that mirrors `ultralytics.YOLO`. Swap one import and keep all the patterns you already know: predict, track, train, val, export, stream, CLI.

![nitid detection demo](docs/assets/nitid-demo.png)

Example prediction using D-FINE-S on a street image.

## Why nitid?

- Familiar Ultralytics-style API
- Automatic download and wrapping of official D-FINE checkpoints
- Python API, CLI and web interface
- ByteTrack, BoT-SORT, and OC-SORT tracking with persistent IDs and annotated video output
- Optional GStreamer video/RTSP ingest, annotated restreaming, and segmented recording
- ONVIF camera discovery, profile selection, and secure RTSP resolution
- Detection and instance segmentation with COCO-pretrained weights
- Fine-tuning and validation for detection, instance masks, and dense semantic masks
- ONNX, TorchScript and TensorRT export

## Installation

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Vaelsys/nitid.git && cd nitid
uv sync
```

For fine-tuning and validation add the `train` extra:

```bash
uv sync --extra train
```

For ByteTrack, BoT-SORT, and OC-SORT object tracking add the `track` extra:

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

Instance segmentation uses the same API and downloads the matching COCO mask checkpoint:

```python
model = DFINE("dfine_s", task="segment")
result = model.predict("image.jpg", conf=0.5)[0]
print(result.masks.data.shape)  # [N, H, W]
result.save("segmented.jpg")
```

### Training

```python
model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
)
# returns final metrics plus per-epoch history in metrics["history"]
```

Semantic training and mIoU validation use dense class-ID PNG masks:

```python
semantic = DFINE("dfine_s", task="semantic")
metrics = semantic.train(data="semantic_dataset.yml", epochs=50)
print(metrics["mIoU"], metrics["pixel_accuracy"])
```

Semantic prediction and export will be enabled in a subsequent release phase.
See the [fine-tuning guide](docs/fine_tuning.md#dense-semantic-masks) for the
dataset layout and `ignore_index` contract.

### Tracking

```python
for result in model.track("video.mp4", conf=0.5, stream=True, save=True):
    if result.boxes.id is not None:
        track_ids = result.boxes.id
```

The annotated video is saved under `runs/track/exp/`. Tracking dependencies
are optional; install them with `uv sync --extra track`.

For resilient RTSP ingest, use an OpenCV build compiled with GStreamer:

```python
for result in model.track(
    "rtsp://camera/live",
    backend="gstreamer",
    reconnect=True,
    conf=0.5,
    stream=True,
):
    ...
```

See [docs/gstreamer.md](docs/gstreamer.md) for system requirements, explicit
pipelines, hardware-decoder examples, and reconnect semantics.

Inspect available codec paths before selecting acceleration:

```bash
dfine gstreamer-info
```

Named profiles are `software`, `vaapi`, `v4l2`, `nvidia`, and `jetson`.

Discover ONVIF cameras and inspect their streams:

```bash
dfine onvif action=discover timeout=3
ONVIF_USERNAME=operator ONVIF_PASSWORD=secret \
  dfine onvif action=profiles host=192.0.2.10
```

Annotated tracking can also be published or segmented without buffering
results in Python:

```bash
dfine track model=dfine_s source=video.mp4 \
  output=runs/segments segment_duration=60
```

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
- GStreamer and RTSP: [docs/gstreamer.md](docs/gstreamer.md)
- ONVIF cameras: [docs/onvif.md](docs/onvif.md)


## Official Models

> 💡 Detection defaults to Objects365→COCO weights for S/M/L/X. `task="segment"` selects COCO-pretrained instance-segmentation weights for N/S/M/L/X. `task="semantic"` initializes its shared feature extractor and mask fuser from the matching instance checkpoint while its dense classifier starts fresh.

Segmentation checkpoints are published in the official [D-FINE-seg model repository](https://huggingface.co/ArgoSA/D-FINE-seg).

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
    --model   dfine_l \
    --task    detect \
    --names   configs/datasets/coco.yml \
    --output  dfine_l_wrapped.pth
```

| Argument | Description |
|---|---|
| `--weights` | Raw D-FINE checkpoint |
| `--model` | Architecture: `dfine_n`, `dfine_s`, `dfine_m`, `dfine_l`, or `dfine_x` |
| `--task` | `detect` or `segment` |
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
| `model.task` | `"detect"`, `"segment"`, … | `"detect"` or `"segment"` |

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, commit conventions, and the PR checklist.

## Acknowledgements

nitid contains code derived from [D-FINE](https://github.com/Peterande/D-FINE) and [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the [Apache 2.0 License](LICENSE).

```bibtex
@article{peng2024dfine,
  title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
  author={Peng, Yansong and Shi, Hongtao and Li, Shiyu and Wang, Yan and Li, Hongbin and Liu, Guozheng and Li, Bing and Hu, Weiming},
  journal={arXiv preprint arXiv:2410.13842},
  year={2024}
}
```
