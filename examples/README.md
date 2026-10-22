# Examples

## Notebooks

| Notebook | What it shows | |
| :------- | :------------ | :- |
| [01_inference.ipynb](notebooks/01_inference.ipynb) | Detection and instance segmentation with pretrained COCO models, the CLI, and TorchScript export | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitidio/nitid/blob/main/examples/notebooks/01_inference.ipynb) |
| [02_training.ipynb](notebooks/02_training.ipynb) | Fine-tuning on COCO-mini, validation plots, prediction with the trained model, ONNX and OpenVINO export | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nitidio/nitid/blob/main/examples/notebooks/02_training.ipynb) |

In Colab, the first cell installs nitid from PyPI; the training notebook wants
a GPU runtime. Locally, run them from a checkout:

```bash
uv sync --extra train
uv run --with jupyterlab jupyter lab
```

Both write their outputs under `runs/notebooks/`.

## Scripts

Short, self-contained starting points. Replace the placeholder paths
(`image.jpg`, `my_dataset.yaml`, the RTSP URL) with your own.

| Script | What it shows |
| :----- | :------------ |
| [predict_image.py](predict_image.py) | The smallest prediction: one image, JSON output, annotated image |
| [predict_stream.py](predict_stream.py) | Streaming prediction from an RTSP camera or your own frames |
| [fine_tune.py](fine_tune.py) | Fine-tuning on your own dataset YAML |
| [train_coco_mini.py](train_coco_mini.py) | Fine-tuning on COCO-mini, downloaded on first use (`--task detect` or `--task segment`) |
| [export.py](export.py) | Export to ONNX, OpenVINO, TorchScript and TensorRT |

```bash
uv run python examples/train_coco_mini.py --task detect
```

The full guides are in [docs/](../docs/): [quickstart](../docs/quickstart.md),
[fine-tuning](../docs/fine_tuning.md) and [export](../docs/export.md).
