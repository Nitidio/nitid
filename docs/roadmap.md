# nitid Roadmap

*Last reviewed: 2026-11-13.*

nitid is a focused library for DETR-style vision models, released under Apache 2.0
with its pretrained weights. The project is
focused on a small number of well-supported workflows: load a model by name,
predict, track, train, validate, export, inspect results, and deploy against
real video sources.

## Product direction

The public API should stay compact:

```python
from nitid import NITID

detector = NITID("nitid1s", task="detect")
segmenter = NITID("nitid1s", task="segment")
semantic = NITID("nitid1s", task="semantic")
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
- FP16 prediction with `half=True` on CUDA ([#2](https://github.com/Nitidio/nitid/issues/2),
  0.9.4). On CPU it falls back to FP32.
- `nitid benchmark`, which compares latency and throughput across export formats
  on the current machine ([#5](https://github.com/Nitidio/nitid/issues/5), 0.9.4).

Known gaps:

- Semantic benchmark parity still needs larger public benchmark runs.
- TensorRT export is not available for semantic segmentation.
- Training uses a single GPU (multi-GPU is deferred, see
  [#9](https://github.com/Nitidio/nitid/issues/9)).
- Quantized deployment is not implemented.

## Near-term priorities

1. Safer checkpoint loading with `torch.load(weights_only=True)`
   ([#8](https://github.com/Nitidio/nitid/issues/8)).
2. An interactive demo on Hugging Face Spaces
   ([#7](https://github.com/Nitidio/nitid/issues/7)).
3. Automated GitHub Releases from a version tag
   ([#3](https://github.com/Nitidio/nitid/issues/3)).
4. Benchmark-grade validation runs for segmentation on datasets larger than
   smoke tests.

Further out, and accepted but not yet scheduled: multi-GPU training with
DistributedDataParallel ([#9](https://github.com/Nitidio/nitid/issues/9)),
a model registry on the Hugging Face Hub
([#6](https://github.com/Nitidio/nitid/issues/6)), SAHI sliced inference
([#1](https://github.com/Nitidio/nitid/issues/1)), a conda package
([#4](https://github.com/Nitidio/nitid/issues/4)), and Python 3.14 support
([#21](https://github.com/Nitidio/nitid/issues/21)).

## Documentation

- [Quick Start](quickstart.md)
- [Command Line](cli.md)
- [Fine-tuning](fine_tuning.md)
- [Export](export.md)
- [API Reference](api_reference.md)
- [Troubleshooting](troubleshooting.md)
- [Onboarding](onboarding.md)
- [Decision records](adr/index.md)
