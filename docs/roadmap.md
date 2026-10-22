# nitid Roadmap

*Last reviewed: 2026-10-22.*

nitid is a focused library for DETR-style vision models, released under Apache 2.0
with its pretrained weights. The project is
focused on a small number of well-supported workflows: load a model by name,
predict, track, train, validate, export, inspect results, and deploy against
real video sources.

## Product direction

The public API should stay compact:

```python
from nitid import NITID

detector = NITID("model1s", task="detect")
segmenter = NITID("model1s", task="segment")
semantic = NITID("model1s", task="semantic")
```

Supported tasks are:

- `detect` — D-FINE object detection.
- `segment` — D-FINE instance segmentation.
- `semantic` — dense semantic segmentation on D-FINE features.

The guiding principle is depth over breadth. A task belongs in nitid when it
has a clear implementation, a normal user API, dataset support,
validation metrics, export coverage where realistic, tests, and documentation.

## In scope

- Python API and CLI for prediction, tracking, training, validation, export,
  download, and model information.
- Automatic download/preparation of supported official checkpoints.
- Prediction on images, videos, folders, streams, screen capture, and numpy
  arrays.
- Result helpers for annotated outputs, masks, semantic maps, JSON, YOLO TXT,
  CSV, pandas DataFrames, timings, and crops.
- Fine-tuning and validation for detection, instance segmentation, and semantic
  segmentation.
- Training features: AMP, EMA, resume, per-epoch metrics, callbacks, W&B, and
  MLflow.
- Export to ONNX/OpenVINO for all supported tasks, and TorchScript/TensorRT for
  detection and instance segmentation.
- RTSP camera input and annotated video output.
- Tests and CI that protect supported user flows.

## Not in scope by default

| Area | Current decision |
|---|---|
| Becoming a general YOLO model zoo | Not planned unless a future optional adapter is justified by measured demand. |
| Automatic batch sizing | Not planned for now. |
| Quantization and mobile runtimes | Important for edge deployment, but each target needs a tested support plan. |
| Annotation tools and dataset doctors | Prefer external specialist tools unless a tight nitid workflow emerges. |
| New tasks | Require an implementation, metrics, export story, tests, docs, and a clear reason to belong here. |

## Current status

nitid is published on PyPI (`pip install nitid`) and supports Python 3.10 to 3.13.
The current release is listed on the [releases page](https://github.com/Nitidio/nitid/releases), and
[`CHANGELOG.md`](https://github.com/Nitidio/nitid/blob/main/CHANGELOG.md) records what changed in each
version.

Available today:

- Detection and instance segmentation with official checkpoint downloads.
- Semantic segmentation training, validation, inference, and ONNX/OpenVINO
  export. There are no pretrained semantic checkpoints: semantic models are
  fine-tuned on your own masks.
- Tracking with ByteTrack, BoT-SORT, and OC-SORT through the optional `track`
  extra.
- Video files, webcams, and RTSP/HTTP streams, including authenticated cameras.
- OpenVINO Runtime inference on Intel CPU, integrated GPU, and NPU.
- Export to ONNX, OpenVINO, TorchScript, and TensorRT.

Known gaps:

- Semantic benchmark parity still needs larger public benchmark runs.
- TensorRT export is not available for semantic segmentation.
- PyTorch inference runs in FP32. FP16 is available through OpenVINO and
  TensorRT export, not yet for direct PyTorch inference.
- Training uses a single GPU.
- Quantized deployment is not implemented.

## Near-term priorities

1. FP16 PyTorch inference on CUDA ([#6](https://github.com/Nitidio/nitid/issues/6)).
2. A `benchmark` command for comparing speed across devices and export formats
   ([#9](https://github.com/Nitidio/nitid/issues/9)).
3. Safer checkpoint loading with `torch.load(weights_only=True)`
   ([#27](https://github.com/Nitidio/nitid/issues/27)).
4. Benchmark-grade validation runs for segmentation on datasets larger than
   smoke tests.
5. An interactive demo on Hugging Face Spaces
   ([#16](https://github.com/Nitidio/nitid/issues/16)).

Further out, and accepted but not yet scheduled: multi-GPU training with
DistributedDataParallel ([#37](https://github.com/Nitidio/nitid/issues/37)),
a model registry on the Hugging Face Hub
([#10](https://github.com/Nitidio/nitid/issues/10)), SAHI sliced inference
([#5](https://github.com/Nitidio/nitid/issues/5)), and a conda package
([#8](https://github.com/Nitidio/nitid/issues/8)).

## Documentation

- [Quick Start](quickstart.md)
- [Command Line](cli.md)
- [Fine-tuning](fine_tuning.md)
- [Export](export.md)
- [API Reference](api_reference.md)
- [Troubleshooting](troubleshooting.md)
- [Onboarding](onboarding.md)
- [Decision records](adr/index.md)
