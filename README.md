# nitid

**Ultralytics-style wrapper for [D-FINE](https://github.com/Peterande/D-FINE) — real-time object detection that feels like YOLO.**

[![CI](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml/badge.svg)](https://github.com/Vaelsys/nitid/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://Vaelsys.github.io/nitid/)
[![codecov](https://codecov.io/gh/Vaelsys/nitid/graph/badge.svg)](https://codecov.io/gh/Vaelsys/nitid)
[![License](https://img.shields.io/github/license/Vaelsys/nitid)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vaelsys/nitid/blob/main/examples/tutorial.ipynb)

nitid gives D-FINE a single-class API that mirrors `ultralytics.YOLO`. Swap one import and keep all the patterns you already know: predict, train, val, export, stream, CLI.

- **Ultralytics-style Python API** for zero friction.
- **Simple CLI support** matching YOLO commands.
- **Fine-tuning & validation** on custom datasets.
- **ONNX, TorchScript, and TensorRT export** for deployment.
- **FastAPI web application** with an interactive browser UI.
- **Built on D-FINE** (real-time object detection with fine-grained distribution refinement).

---

## Models

nitid supports auto-downloading and wrapping the official D-FINE weights on the fly. 

| Model | Parameters | COCO mAP | Name | Download (Raw weights) |
|---|---|---|---|---|
| D-FINE-N | 4.0M | 42.8 | `dfine_n` | [Download](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth) |
| D-FINE-S | 10.0M | 48.5 | `dfine_s` | [Download](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth) |
| D-FINE-M | 19.0M | 52.3 | `dfine_m` | [Download](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth) |
| D-FINE-L | 31.0M | 54.0 | `dfine_l` | [Download](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth) |
| D-FINE-X | 62.0M | 55.8 | `dfine_x` | [Download](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth) |

*Note: Passing any of the model names to `DFINE("name")` will automatically download, wrap, and load the weights.*

---

## Installation

Requires Python 3.10 to 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/Vaelsys/nitid && cd nitid

# Pull the D-FINE submodule
git submodule update --init

# Synchronize dependencies
uv sync
```

### Optional extras

```bash
# For fine-tuning and validation
uv sync --extra train

# For the FastAPI web application
uv sync --extra web

# For TensorRT export (NVIDIA GPU only)
pip install --extra-index-url https://pypi.nvidia.com tensorrt>=8.6
```

---

## Quickstart

Run inference in just a few lines of code.

### Python Inference

```python
from dfine import DFINE

# Automatically downloads, wraps, and loads dfine_s
model = DFINE("dfine_s")

# Run prediction
results = model.predict("image.jpg", conf=0.25)

# Save annotated result
results[0].save("result.jpg")
```

### CLI Inference

```bash
uv run dfine predict model=dfine_s source=image.jpg conf=0.25
```

See [docs/quickstart.md](docs/quickstart.md) for detailed Python API usage (streaming, working with bounding box detections, and `model.info()`).

---

## Converting a raw D-FINE checkpoint

If you have a raw D-FINE checkpoint, you can wrap it into a self-contained `.pth` file that embeds the model config and class names alongside the weights:

```bash
uv run python tools/convert_checkpoint.py \
    --weights dfine_l.pth \
    --config  extern/dfine/configs/dfine/dfine_hgnetv2_l_coco.yml \
    --names   configs/datasets/coco.yml \
    --output  dfine_l_wrapped.pth
```

| Argument | Description |
|---|---|
| `--weights` | Raw D-FINE checkpoint (downloaded from the D-FINE repo) |
| `--config` | Canonical D-FINE config from `extern/dfine/configs/` |
| `--names` | YAML file with a `names:` mapping (e.g. `configs/datasets/coco.yml`) |
| `--output` | Path for the output wrapped checkpoint |

---

## Training and validation

Fine-tune on any COCO-format dataset:

```python
model = DFINE("dfine_s") # Load a pre-trained wrapped model
metrics = model.train(
    data="configs/datasets/my_dataset.yml",
    epochs=50,
    batch=16,
    lr0=1e-4,
    optimizer="AdamW",
    amp=True,    # FP16 mixed precision
    ema=True,    # Weight averaging
)
```

Evaluate performance:

```python
metrics = model.val(data="configs/datasets/my_dataset.yml")
```

See [docs/fine_tuning.md](docs/fine_tuning.md) for detailed training options and dataset specifications.

---

## Export

Export trained models to deployment formats:

```python
model.export(format="onnx")         # -> dfine_s.onnx
model.export(format="torchscript")  # -> dfine_s.torchscript
model.export(format="tensorrt")     # -> dfine_640.engine (requires tensorrt extra)
```

See [docs/export.md](docs/export.md) for all export options.

---

## Web application

nitid includes a browser-based UI for running detection without writing code.

```bash
# Start the API
uv run uvicorn web.api.main:app --workers 1

# Start the frontend (in a separate terminal)
cd web/frontend && npm install && npm run dev
# -> open http://localhost:5173
```

See [docs/web_app.md](docs/web_app.md) for environment variables and API documentation.

---

## Documentation

Detailed documentation is available in the `docs/` folder:

- [Quick Start Guide](docs/quickstart.md)
- [Fine-tuning & Training](docs/fine_tuning.md)
- [Model Export](docs/export.md)
- [Web Application](docs/web_app.md)
- [API Reference](docs/api_reference.md)
- [Developer Onboarding](docs/onboarding.md)

---

## Development

Instructions for testing and contributing are available in [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
# Run unit tests
uv sync --extra dev
uv run pytest tests/unit
```

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
