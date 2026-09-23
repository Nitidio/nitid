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
- Brand assets under `docs/assets/brand/` and `docs/brand.md`, the source of truth for the palette,
  typography, logo usage, voice, and what the project announces publicly
- Brand theme for the documentation site, with a light/dark palette toggle

### Changed
- Integrated the model architectures, losses, and postprocessing into the installable package so
  supported tasks use the same self-contained runtime core
- Public messaging now leads with the Apache 2.0 licensing of both code and weights instead of the
  resemblance to Ultralytics, and announces three tasks; pose and OBB stay supported and documented
- The web application uses the nitid palette through `web/frontend/src/styles/tokens.css`
