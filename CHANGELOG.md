# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Apache 2.0 LICENSE
- Auto-download and runtime preparation of supported official checkpoints
- D-FINE instance segmentation through `DFINE(..., task="segment")`, including pretrained weights,
  prediction masks, fine-tuning, COCO mask validation, and export
- Semantic segmentation through `DFINE(..., task="semantic")`, including dense-mask training,
  mIoU validation, inference overlays/class-ID maps, and ONNX/OpenVINO export
- DETRPose through `DFINE(..., task="pose")`, including official checkpoint loading, COCO-keypoint
  training/validation, visualization, and ONNX/OpenVINO export
- COCO polygon/RLE and YOLO polygon dataset support for instance-segmentation training
- GitHub Actions CI (lint + unit tests)
- Pre-commit hooks (ruff, mypy)
- CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md
- Issue and PR templates
- README model comparison table and YOLO benchmark
- Jupyter/Colab tutorial notebook

### Changed
- Integrated the model architectures, losses, and postprocessing into the installable package so
  supported tasks use the same self-contained runtime core
