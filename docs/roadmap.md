# nitid Roadmap

*Last reviewed: 2026-09-17.*

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
| Distributed training/autobatch | Valuable but not a differentiator yet. |
| Quantization and mobile runtimes | Important for edge deployment, but each target needs a tested support plan. |
| Annotation tools and dataset doctors | Prefer external specialist tools unless a tight nitid workflow emerges. |
| New tasks | Require an implementation, metrics, export story, tests, docs, and a clear reason to belong here. |

## Current status

Available today:

- Detection and instance segmentation with official checkpoint downloads.
- Semantic segmentation training, validation, inference, and ONNX/OpenVINO
  export.
- Tracking with optional tracker dependencies.
- RTSP camera input, including authenticated cameras.
- Developer tests for model construction, datasets, training, validation,
  export, and result containers.

Known gaps:

- Semantic benchmark parity still needs larger public benchmark runs.
- TensorRT coverage has not been completed for semantic segmentation.
- Distributed training and automatic batch sizing are not implemented.
- Quantized deployment is not implemented.
- Public PyPI release workflow and hosted model registry remain release tasks.

## Path to public release

nitid is not yet public. The remaining work to publish the repository and ship
`pip install nitid` at v0.9.1 is tracked in the
[launch blockers milestone](https://github.com/Nitidio/nitid/milestone/1), which is release and
publication work rather than features: third-party attribution, the published
import/CLI name, wheel namespacing, branch and changelog sync, and the PyPI
pipeline. The known gaps above are documented gaps for v0.9.1, not blockers.

## Near-term priorities

1. Stabilize the task surface: detection, instance segmentation, and semantic
   segmentation.
2. Keep README/API/fine-tuning/export docs aligned with the actual public API.
3. Add benchmark-grade validation runs for segmentation on datasets
   larger than smoke tests.
4. Finish release hygiene: package artifacts, notices, versioning, and PyPI.
5. Harden deployment workflows: reproducible export checks, hardware notes, and
   camera-runtime documentation.

## Documentation

- [Quick Start](quickstart.md)
- [Command Line](cli.md)
- [Fine-tuning](fine_tuning.md)
- [Export](export.md)
- [API Reference](api_reference.md)
- [Troubleshooting](troubleshooting.md)
- [Onboarding](onboarding.md)
- [Decision records](adr/index.md)
