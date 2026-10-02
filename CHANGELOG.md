# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-20

### Added

- A unified Python API and key-value CLI for downloading models, prediction,
  tracking, training, validation, export, model inspection, and bug reports.
- Automatic download and runtime preparation of supported official
  checkpoints.
- D-FINE object detection and instance segmentation, including COCO and YOLO
  dataset support, mask visualization, fine-tuning, COCO validation, and
  export.
- Dense semantic segmentation with mask training, mIoU validation,
  probability output, overlays, class-ID maps, and ONNX/OpenVINO export.
- ByteTrack, BoT-SORT, and OC-SORT tracking with persistent IDs, class
  filtering, frame sampling, and annotated video output.
- Video file, webcam, and RTSP/HTTP stream input through OpenCV, including
  authenticated RTSP cameras.
- OpenVINO Runtime inference on Intel CPU, integrated GPU, and NPU devices,
  with automatic device selection and compiled-model caching.
- ONNX and OpenVINO export for every supported task, plus TorchScript and
  TensorRT export for detection and instance segmentation.
- Fine-tuning controls for AMP, EMA, checkpoint resume, cosine learning-rate
  warmup, freezing, augmentation, callbacks, per-epoch validation, W&B, and
  MLflow.
- Result helpers for annotated media, masks, semantic maps, JSON, YOLO text,
  pandas DataFrames, timings, and crops.
- Atomic, incremented run directories with saved arguments, environment
  snapshots, optional single-file bug reports, and exact output-path
  overrides.
- MkDocs documentation, a Jupyter/Colab tutorial, contributor and security
  guides, issue and pull-request templates, pre-commit hooks, and CI across
  Python 3.10, 3.11, and 3.12.
- Apache-2.0 licensing.
- Brand assets under `docs/assets/brand/` and `docs/brand.md`, the source of
  truth for the palette, typography, logo usage, voice, and what the project
  announces publicly, plus a matching light/dark theme for the documentation
  site.

### Changed

- Integrated model architectures, losses, postprocessing, and configuration
  into the installable package so supported tasks share a self-contained
  runtime core.
- Expanded prediction sources to images, directories, videos, URLs, webcams,
  streams, screen capture, NumPy arrays, and tensors.
- Hardened training defaults and validation behavior for reproducible,
  convergent fine-tuning.
- Public messaging leads with the Apache 2.0 licensing of the code, and
  announces three tasks: detection, instance segmentation, and
  semantic segmentation.

### Fixed

- Prediction and export preserve the trainable model's structure and state.
- OpenVINO parity checks tolerate valid encoder tie-breaking differences.
- `import nitid` works on servers and slim containers without `libGL`: the
  package depends on `opencv-python-headless`, and `Results.show()` falls back
  to matplotlib, or raises a clear error when no display is available.
- MLflow tracking works with MLflow 3: a local tracking directory such as the
  default `runs/mlflow` is stored as SQLite instead of MLflow's file store,
  which MLflow 3 refuses by default.

[Unreleased]: https://github.com/Nitidio/nitid/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Nitidio/nitid/releases/tag/v0.1.0
