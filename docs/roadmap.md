# nitid Roadmap

*Last reviewed: 2026-08-24.*

nitid is an Ultralytics-style library for D-FINE-family models. The project is
focused on a small number of well-supported workflows: load a model by name,
predict, track, train, validate, export, inspect results, and deploy against
real video sources.

## Product direction

The public API should stay compact:

```python
from dfine import DFINE

detector = DFINE("dfine_s", task="detect")
segmenter = DFINE("dfine_s", task="segment")
semantic = DFINE("dfine_s", task="semantic")
pose = DFINE("detrpose_n", task="pose")
```

Supported tasks are:

- `detect` — D-FINE object detection.
- `segment` — D-FINE instance segmentation.
- `semantic` — dense semantic segmentation on D-FINE features.
- `pose` — DETRPose single-class person keypoints.

The guiding principle is depth over breadth. A task belongs in nitid when it
has a clear D-FINE-family implementation, a normal user API, dataset support,
validation metrics, export coverage where realistic, tests, and documentation.

## In scope

- Python API and CLI for prediction, tracking, training, validation, export,
  download, and model information.
- Automatic download/preparation of supported official checkpoints.
- Prediction on images, videos, folders, streams, screen capture, and numpy
  arrays.
- Result helpers for annotated outputs, masks, semantic maps, keypoints, JSON,
  YOLO TXT, CSV, pandas DataFrames, timings, and crops.
- Fine-tuning and validation for detection, instance segmentation, semantic
  segmentation, and COCO keypoints.
- Training features: AMP, EMA, resume, per-epoch metrics, callbacks, W&B, and
  MLflow.
- Export to ONNX/OpenVINO for all supported tasks, and TorchScript/TensorRT for
  detection and instance segmentation.
- RTSP/GStreamer/ONVIF camera workflows and annotated video output.
- Browser-based web application for model selection and inference runs.
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
- DETRPose model loading, training, validation, inference, visualization, and
  ONNX/OpenVINO export.
- Tracking with optional tracker dependencies.
- GStreamer/RTSP and ONVIF camera helpers.
- Web application for browser-based inference workflows.
- Developer tests for model construction, datasets, training, validation,
  export, and result containers.

Known gaps:

- Pose and semantic benchmark parity still need larger public benchmark runs.
- TensorRT coverage has not been completed for semantic segmentation or pose.
- Distributed training and automatic batch sizing are not implemented.
- Quantized deployment is not implemented.
- Public PyPI release workflow and hosted model registry remain release tasks.

## Near-term priorities

1. Stabilize the new task surface: detection, instance segmentation, semantic
   segmentation, and pose.
2. Keep README/API/fine-tuning/export docs aligned with the actual public API.
3. Add benchmark-grade validation runs for segmentation and pose on datasets
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
- [Web App](web_app.md)
- [Troubleshooting](troubleshooting.md)
- [Onboarding](onboarding.md)
- [Decision records](adr/index.md)
