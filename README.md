# nitid

**Ultralytics-style detection, segmentation, pose, and oriented-box models.**

[![CI](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml/badge.svg)](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://Vaelsys.github.io/nitid/)
[![codecov](https://codecov.io/gh/Vaelsys/nitid/graph/badge.svg)](https://codecov.io/gh/Vaelsys/nitid)
[![License](https://img.shields.io/github/license/Vaelsys/nitid)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vaelsys/nitid/blob/main/examples/tutorial.ipynb)

nitid gives modern DETR-style vision models a single-entry-point API that mirrors `ultralytics.YOLO`: predict, track, train, validate, export, stream, and CLI.

![nitid detection demo](docs/assets/nitid-demo.png)

Example prediction using D-FINE-S on a street image.

## Why nitid?

- Familiar Ultralytics-style API
- Automatic download of supported official checkpoints
- Python API, CLI and web interface
- ByteTrack, BoT-SORT, and OC-SORT tracking with persistent IDs and annotated video output
- Optional GStreamer video/RTSP ingest, annotated restreaming, and segmented recording
- ONVIF camera discovery, profile selection, and secure RTSP resolution
- Detection, instance segmentation, semantic segmentation, pose, and oriented bounding boxes
- Fine-tuning and validation for boxes, masks, dense semantic maps, COCO keypoints, and rotated boxes
- ONNX, OpenVINO, TorchScript, and TensorRT export where supported by task

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

## Public names

Use `nitid` for the Python package and command, and `NITID` for the public
model class. Existing `from dfine import ...` imports and the `dfine` command
remain supported compatibility aliases for the 0.1 release series. New code
should use the Nitid names shown below.

## Quick Start

The canonical quickstart lives in [docs/quickstart.md](docs/quickstart.md); use the docs guide for the full walkthrough.

### Inference

```python
from nitid import NITID

model = NITID("nitid1s", task="detect")
results = model.predict("image.jpg", conf=0.5)
results[0].save("out.jpg")
```

Instance segmentation uses the same API and downloads the matching COCO mask checkpoint:

```python
model = NITID("nitid1s", task="segment")
result = model.predict("image.jpg", conf=0.5)[0]
print(result.masks.data.shape)  # [N, H, W]
result.save("segmented.jpg")
```

Semantic segmentation returns one class ID per pixel:

```python
model = NITID("nitid1s", task="semantic")
result = model.predict("image.jpg", return_probs=True)[0]
print(result.semantic.mask.shape)  # [H, W]
result.save_semantic("class_ids.png")
```

Pose estimation returns COCO-style person keypoints:

```python
pose = NITID("nitid1s", task="pose")
result = pose.predict("person.jpg", conf=0.25)[0]
print(result.keypoints.xy.shape)  # [N, 17, 2]
result.save("pose.jpg")
```

Oriented bounding box detection returns rotated boxes:

```python
obb = NITID("nitid1s", task="obb")
result = obb.predict("aerial.jpg", conf=0.25)[0]
print(result.obb.xywhr.shape)      # [N, 5]: cx, cy, w, h, angle
print(result.obb.xyxyxyxy.shape)   # [N, 8]: four polygon corners
result.save("obb.jpg")
```

### Training

```python
metrics = model.train(data="configs/datasets/my_dataset.yml", epochs=50)
print(metrics["mAP50"], metrics["mAP50-95"])

# Detection-only DEIM recipe for faster convergence experiments.
deim_metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    recipe="deim",
)
```

See [fine-tuning](docs/fine_tuning.md) for COCO, YOLO, semantic-mask, pose, and OBB dataset formats.

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
nitid gstreamer-info
```

Named profiles are `software`, `vaapi`, `v4l2`, `nvidia`, and `jetson`.

Discover ONVIF cameras and inspect their streams:

```bash
nitid onvif action=discover timeout=3
ONVIF_USERNAME=operator ONVIF_PASSWORD=secret \
  nitid onvif action=profiles host=192.0.2.10
```

Annotated tracking can also be published or segmented without buffering
results in Python:

```bash
nitid track model=nitid1s task=detect source=video.mp4 \
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
model.export(format="openvino")
model.export(format="torchscript")
model.export(format="tensorrt")
```

Capture Python API output, environment details, and failure tracebacks in one
attachable log:

```python
from nitid import NITID, bugreport

with bugreport("prediction") as report:
    model = NITID("nitid1s", task="detect")
    model.predict("image.jpg")

print(report.path)
```


### Command Line Interface

```bash
uv run nitid predict model=nitid1s task=detect source=image.jpg
uv run nitid predict model=semantic_best.pth task=semantic source=image.jpg save=true
uv run nitid predict model=nitid1s task=obb source=aerial.jpg conf=0.25
uv run nitid track model=nitid1s source=video.mp4 conf=0.5 save=true
uv run nitid train model=nitid1s task=detect data=my_dataset.yml epochs=50
uv run nitid train model=nitid1s task=detect data=my_dataset.yml epochs=50 recipe=deim
uv run nitid val model=nitid1s task=detect data=my_dataset.yml
uv run nitid export model=nitid1s task=detect format=onnx
uv run nitid predict model=nitid1s source=image.jpg --report
uv run nitid bugreport
```

For the full guide:

- Full quickstart: [docs/quickstart.md](docs/quickstart.md)
- Fine-tuning and validation: [docs/fine_tuning.md](docs/fine_tuning.md)
- Export: [docs/export.md](docs/export.md)
- CLI: [docs/cli.md](docs/cli.md)
- GStreamer and RTSP: [docs/gstreamer.md](docs/gstreamer.md)
- ONVIF cameras: [docs/onvif.md](docs/onvif.md)


## Official Models

> 💡 `NITID("nitid1s", task=...)` is the canonical constructor. The trailing size letter selects the model size, and `nitid1` identifies the model generation. Supported tasks are `detect`, `segment`, `semantic`, `pose`, and `obb`.

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

# Download the recommended checkpoint into models/
uv run nitid download model=nitid1l task=detect output=models

# Start the API (single worker — inference is not thread-safe)
uv run uvicorn web.api.main:app --workers 1

# Start the frontend (separate terminal)
cd web/frontend && npm install && npm run dev
# → open http://localhost:5173
```

Register an account on first visit. The API is self-documented at `http://localhost:8000/docs`.

See [docs/web_app.md](docs/web_app.md) for the full guide: environment variables, REST API reference, data model, and implementation notes.


## Documentation

| Doc | Description |
|-----|-------------|
| [docs/onboarding.md](docs/onboarding.md) | **Start here if you're a new developer** — architecture, conventions, gotchas |
| [docs/quickstart.md](docs/quickstart.md) | Full quickstart |
| [docs/fine_tuning.md](docs/fine_tuning.md) | Training, validation, AMP, EMA, dataset formats |
| [docs/export.md](docs/export.md) | ONNX, OpenVINO, TorchScript, and TensorRT export |
| [docs/api_reference.md](docs/api_reference.md) | Full Python API reference |
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
| `model.task` | `"detect"`, `"segment"`, … | `"detect"`, `"segment"`, `"semantic"`, `"pose"`, or `"obb"` |

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, commit conventions, and the PR checklist.

## Acknowledgements

nitid contains code derived from [D-FINE](https://github.com/Peterande/D-FINE), [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg), [DETRPose](https://github.com/SebastianJanampa/DETRPose), and RiO-DETR OBB. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the [Apache 2.0 License](LICENSE).

```bibtex
@article{peng2024dfine,
  title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
  author={Peng, Yansong and Shi, Hongtao and Li, Shiyu and Wang, Yan and Li, Hongbin and Liu, Guozheng and Li, Bing and Hu, Weiming},
  journal={arXiv preprint arXiv:2410.13842},
  year={2024}
}
```
