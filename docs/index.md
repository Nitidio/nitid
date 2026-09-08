# Welcome to nitid

**nitid** is an Ultralytics-style library for DETR-style detection, segmentation, pose, and oriented-box models.

It provides a familiar API for inference, training, validation, export, and deployment while preserving D-FINE's performance.

## Features

- Ultralytics-style Python API
- Simple command-line interface
- Detection, instance segmentation, semantic segmentation, pose, and oriented bounding box inference
- Fine-tuning and validation for boxes, masks, dense semantic maps, COCO keypoints, and rotated boxes
- ONNX, OpenVINO IR, TorchScript, and TensorRT export where supported by task
- FastAPI web application
- Self-contained wrapped checkpoints

## Getting Started

If you're new to nitid, follow these guides in order:

1. [Quick Start](quickstart.md)
2. [Fine-tuning](fine_tuning.md)
3. [Export](export.md)
4. [Troubleshooting](troubleshooting.md)

## Documentation

- [Quick Start](quickstart.md)
- [Fine-tuning](fine_tuning.md)
- [Export](export.md)
- [Web Application](web_app.md)
- [API Reference](api_reference.md)
- [Troubleshooting](troubleshooting.md)

## Project Direction

- [Roadmap](roadmap.md) — what is in scope, what is not, and how the phases are sequenced
- [Decision records](adr/index.md) — why the project is built the way it is

## For Contributors

If you're planning to contribute to nitid, start with the [Onboarding Guide](onboarding.md).
