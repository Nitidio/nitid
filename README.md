<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/nitid-logo-on-dark.png">
    <img src="docs/assets/brand/nitid-logo-on-light.png" alt="nitid" width="240">
  </picture>
</p>

<p align="center"><strong>Object detection that's actually open source.</strong></p>

[![CI](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml/badge.svg)](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://Vaelsys.github.io/nitid/)
[![codecov](https://codecov.io/gh/Vaelsys/nitid/graph/badge.svg)](https://codecov.io/gh/Vaelsys/nitid)
[![License](https://img.shields.io/github/license/Vaelsys/nitid)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vaelsys/nitid/blob/main/examples/tutorial.ipynb)

Train, validate, export and run vision models. The code and our pretrained
weights are released under the Apache License 2.0, patent grant included, so you
can ship them inside your own product without opening your code or paying for a
license.

*nitid*, from Latin *nitidus*: clear, transparent, precise.

**No AGPL, no surprises.**

![nitid detection demo](docs/assets/nitid-demo.png)

Example prediction using D-FINE-S on a street image.

## Why nitid?

- **Edge first.** The models are designed to run at the edge, on the hardware
  next to your cameras, not only on a datacenter GPU.
- **Self-contained checkpoints.** Every checkpoint carries the config and the
  class names it needs to be reproduced and checked — one file per model.
- **Handles messy datasets.** COCO and YOLO layouts are read directly, with no
  conversion step, and the inconsistencies real datasets have are tolerated.
- **Export anywhere.** ONNX, OpenVINO, TorchScript and TensorRT, from the same
  checkpoint.

## One library, three tasks, five operations

Object detection, instance segmentation and semantic segmentation, through one
API: `predict`, `track`, `train`, `val` and `export`. The DETR family moves fast
— RT-DETR, D-FINE, DEIM — and nitid brings a curated selection into a single
library, so getting from your dataset to an exported model does not mean
following every paper.

What comes with it:

- Python API and command line
- Automatic download of supported official checkpoints
- Fine-tuning and validation for boxes, masks and dense semantic maps
- ByteTrack, BoT-SORT, and OC-SORT tracking with persistent IDs and annotated
  video output
- Optional GStreamer video/RTSP ingest, annotated restreaming, and segmented
  recording

Pose estimation and oriented bounding boxes are supported too, outside the three
headline tasks — see the [quickstart](docs/quickstart.md).

## What "open" means here

The nitid code and our pretrained weights are released under the
[Apache License 2.0](LICENSE), including its patent grant. You can use them
commercially, modify them, and ship them in closed products. The public datasets
used for pretraining keep their own terms: the official checkpoints listed in
[Official Models](#official-models) are trained on COCO and Objects365.

## Installation

Requires Python 3.10, 3.11 or 3.12.

```bash
pip install nitid
```

> nitid is not on PyPI yet — the first release, v0.1.0, is being prepared.
> Until it lands, install straight from GitHub:
> `pip install "nitid @ git+https://github.com/Vaelsys/nitid.git"`

Inference and export work out of the box. Other features are optional extras:

| Extra | Adds |
|:------|:-----|
| `train` | Fine-tuning and COCO-style validation |
| `track` | ByteTrack, BoT-SORT, and OC-SORT object tracking |
| `openvino` | OpenVINO export and runtime for Intel CPU, iGPU, and NPU |
| `wandb`, `mlflow` | Experiment tracking during training |

```bash
pip install "nitid[train,track]"
```

pip installs the default PyTorch build for your platform. If you need a
specific CUDA version, install PyTorch first following
[pytorch.org](https://pytorch.org/get-started/locally/), then install nitid.

TensorRT export needs NVIDIA's package as well:

```bash
pip install --extra-index-url https://pypi.nvidia.com "tensorrt>=8.6"
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

Pose estimation (`task="pose"`) and oriented bounding boxes (`task="obb"`) work
the same way; the [quickstart](docs/quickstart.md) has both.

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
are optional; install them with `pip install "nitid[track]"`.

For RTSP ingest through GStreamer, use an OpenCV build compiled with it:

```python
for result in model.track(
    "rtsp://camera/live",
    backend="gstreamer",
    conf=0.5,
    stream=True,
):
    ...
```

See [docs/gstreamer.md](docs/gstreamer.md) for system requirements and
explicit pipelines. `nitid gstreamer-info` checks whether your OpenCV build
has GStreamer enabled.

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
nitid predict model=nitid1s task=detect source=image.jpg
nitid predict model=semantic_best.pth task=semantic source=image.jpg save=true
nitid predict model=nitid1s task=obb source=aerial.jpg conf=0.25
nitid track model=nitid1s source=video.mp4 conf=0.5 save=true
nitid train model=nitid1s task=detect data=my_dataset.yml epochs=50
nitid train model=nitid1s task=detect data=my_dataset.yml epochs=50 recipe=deim
nitid val model=nitid1s task=detect data=my_dataset.yml
nitid export model=nitid1s task=detect format=onnx
nitid predict model=nitid1s source=image.jpg --report
nitid bugreport
```

For the full guide:

- Full quickstart: [docs/quickstart.md](docs/quickstart.md)
- Fine-tuning and validation: [docs/fine_tuning.md](docs/fine_tuning.md)
- Export: [docs/export.md](docs/export.md)
- CLI: [docs/cli.md](docs/cli.md)
- GStreamer and RTSP: [docs/gstreamer.md](docs/gstreamer.md)


## Migrating from YOLO

If you are coming from a YOLO codebase, the quickstart has a side-by-side
mapping of the API and the behaviour that differs:
[Coming from Ultralytics](docs/quickstart.md#coming-from-ultralytics).

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


## Documentation

| Doc | Description |
|-----|-------------|
| [docs/quickstart.md](docs/quickstart.md) | Full quickstart |
| [docs/fine_tuning.md](docs/fine_tuning.md) | Training, validation, AMP, EMA, dataset formats |
| [docs/export.md](docs/export.md) | ONNX, OpenVINO, TorchScript, and TensorRT export |
| [docs/api_reference.md](docs/api_reference.md) | Full Python API reference |
| [docs/troubleshooting.md](docs/troubleshooting.md) | FAQ and fixes for common install, model, Docker, CUDA, and CLI problems |
| [docs/brand.md](docs/brand.md) | Brand: logo, palette, typography, and what the project announces |


## Development

Everything above is for using nitid. To work on nitid itself, use a source
checkout and [uv](https://github.com/astral-sh/uv), which installs the exact
dependency versions pinned in `uv.lock` — including CUDA 12.1 PyTorch wheels on
Linux and Windows.

```bash
git clone https://github.com/Vaelsys/nitid.git && cd nitid
uv sync --extra dev --extra train
```

Inside the checkout, run the CLI and tools through the project environment:

```bash
uv run nitid predict model=nitid1s task=detect source=image.jpg

# Unit tests (no GPU, no checkpoint needed)
uv run pytest tests/unit

# All tests (integration tests build a tiny checkpoint automatically)
uv run pytest

# Lint
uv run ruff check .

# Type-check
uv run mypy dfine/
```

Integration tests use a session-scoped fixture in `tests/conftest.py` that
builds a small D-FINE-S model with random weights at test time — no download
required.

- [docs/onboarding.md](docs/onboarding.md) — **start here if you're new to the
  codebase**: architecture, conventions, gotchas
- [CONTRIBUTING.md](CONTRIBUTING.md) — branch workflow, commit conventions, and
  the PR checklist

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
