# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Default detection weights are now the COCO-only `coco` checkpoints** (ADR-0006), not the Objects365+COCO `obj2coco` ones: the D-FINE authors say `obj2coco` checkpoints should not be assumed to be commercially cleared and nitid has not established what the Objects365 terms allow. `NITID("nitid1s")` and `nitid download` now fetch `dfine_<size>_coco_wrapped.pth`. Published COCO mAP of the defaults: S 48.5, M 52.3, L 54.0, X 55.8 (was 50.7, 55.1, 57.3, 59.3). To keep the previous weights pass `weights="obj2coco"` / `weights=obj2coco`; downloading them prints the Objects365 notice. Existing `*_obj2coco_wrapped.pth` files still load by path.

### Added

- Nitid app, hosted deployment (browser only, no Python server): `pnpm run build:hosted`, Cloudflare headers and `wrangler.jsonc` under `studio/hosting/cloudflare/`, `scripts/assemble_hosted.py`, a hosted smoke test without backend (`pnpm run smoke:hosted`) and a manual-only workflow `studio-hosted.yml` (nothing deploys unless `deploy` is ticked). `tools/publish_models.py` and `tools/check_hosted_bundle.py` build and validate the static model bundle (`models.json`, ONNX, SHA-256, licences) from the `hostable` entries of `studio/models.lock.json`. Measured ONNX sizes at 640 px: nitid1n 14.7 MiB, s 39.7, m 74.1, l 118.1, x 236.2; Cloudflare static assets allow 25 MiB per file, so the first hosted release serves only `nitid1n`. See `docs/studio-hosted.md` and `docs/studio-hosted-models.md`.
- Segmentation checkpoints are now downloaded from a pinned commit of `ArgoSA/D-FINE-seg` instead of `resolve/main`, so the files match `studio/models.lock.json`.
- The Studio interface, its PDF report and its start-up messages call the app "Nitid app"; the command, extra and package stay `nitid studio` / `nitid[studio]`.
- Docs: "Migrating to Studio 0.10" (`docs/studio-migration.md`) and `studio/IMPORT.md` record what was taken from the old `nitid-studio` repository and the decisions taken at the acceptance of ADR-0005, which moves to Accepted.
- `NITID("nitid1n", task="detect")` and `nitid download model=nitid1n` now work: the official D-FINE-N COCO checkpoint (`dfine_n_coco.pth`, SHA-256 verified) is registered. It is COCO-only (no Objects365 variant) and the download prints a notice saying so; `weights=default` falls back to `coco` for sizes without `obj2coco`. Nitid Studio lists the pair.
- ONNX exports of detection and instance-segmentation models now embed a `nitid_meta` entry in the ONNX `metadata_props` with class names, input preprocessing, output contract, `topk` and default IoU. It is additive and does not change the graph; read it with `dfine.export_meta.read_onnx_meta` or plain `onnx`/`onnxruntime` (see `docs/export.md`).
- `nitid studio` (and `python -m nitid.studio`) starts the local Nitid Studio backend: a token-protected FastAPI app with `GET /api/v1/health`, Host/Origin guards, security headers and a single-page-app static fallback. It needs the new optional extra `pip install "nitid[studio]"`; the web UI, model management and jobs arrive in later changes.
- Nitid Studio: `GET /api/v1/devices`, `GET /api/v1/models`, `POST /api/v1/models/{id}/download` and `GET /api/v1/jobs/{job_id}` list the installed compute devices (CPU, OpenVINO CPU/iGPU/NPU, CUDA), the available `nitid1*` models and their download state.
- Nitid Studio packaging: `scripts/build_studio.py` builds the UI into the wheel, `scripts/verify_wheel.py` and `scripts/verify_install.py --require-ui` check size, contents and a clean install; CI has a `studio-ui` job and releases build the UI first.
- Licensing for Studio weights: `studio/models.lock.json` pins revision, SHA-256 and SPDX licence per model, `tools/check_models_lock.py` validates it, and `THIRD_PARTY_NOTICES.md` plus `docs/licensing/studio-weights.md` document redistribution terms.
- Nitid Studio in-browser engine: ONNX Runtime Web in a Web Worker with Pillow-exact preprocessing, `nitid_meta` decoding for detection and segmentation, and a Cache API keyed by SHA-256; parity fixtures in `studio/` compare it with the Python predictor.
- Nitid Studio result exports: annotated PNG, JSON, CSV, YOLO TXT (with mask polygons), crops, ZIP bundle and a PDF report, plus an `ExportMenu` component; adds the lazy-loaded `fflate` dependency.
- Nitid Studio Local panel (`/run/local`): model download, backend/device picker, multi-image upload, run polling, canvas results with confidence and class filters, and an `ExportMenu` slot for the unified UI.
- Nitid Studio runs API (`POST/GET/DELETE /api/v1/runs`, per-item JSON and image, CSV and ZIP export, one inference at a time, lazy model loading); `/health` reports `api_version` as an integer, 401/403 use the error envelope, `nitid studio` prints the URL only once the server is listening, and Ctrl+C exits with code 0.
- Nitid Studio ONNX endpoints: `GET /api/v1/onnx/{model}` and `/onnx/{model}/meta` export on demand (cached, serialised, ETag = sha256, 304 on `If-None-Match`).
- Nitid Studio unified UI for both deployments: server mode, in-browser mode with ONNX Runtime Web, server-vs-browser comparison with IoU agreement, and the export menu everywhere; the hosted catalogue is set with `STUDIO_MODELS_BASE_URL`.
- Nitid Studio end-to-end suite (`pytest studio/e2e`, real server and real Chrome) with a server-vs-browser parity test, and an optional non-blocking `studio-e2e` CI job.
- Nitid Studio draws the mask outlines of server segmentation runs and includes them as polygons in the YOLO TXT, JSON and annotated PNG exports (server results have no pixel masks, so CSV `mask_pixels` and crops are unchanged).

### Fixed

- Nitid Studio sends `X-Frame-Options: DENY` and CSP `frame-ancestors 'none'` so it cannot be framed by another site, and no longer logs a favicon 404 on every page load.
- ONNX export with `simplify=True` no longer leaves a stale `graph.value_info`, which made instance-segmentation models fail the full ONNX check and load in onnxruntime-web; outputs are bit-identical.

## [0.9.6] - 2026-11-18

Metadata-only release: PyPI and the repository now present nitid.io as the
project's home, with the tagline "Nitid Object detection that's actually open
source". No code or model changes.

### Changed

- Project metadata: the PyPI `Homepage` now points to <https://nitid.io> (documentation stays on GitHub Pages) and the tagline reads "Nitid Object detection that's actually open source" in the README, docs and MkDocs config.

## [0.9.5] - 2026-11-17

nitid's public models now carry the project's name: use `nitid1n`, `nitid1s`,
`nitid1m`, `nitid1l` and `nitid1x`, for example
`NITID("nitid1s", task="detect")` or `NITID("nitid1s", task="segment")`. Code
written for earlier releases keeps working: the old `model1*` names resolve to
the new ones and emit a `DeprecationWarning` that names the replacement. This
release also adds `nitid.__version__`, and the tooling that publishes releases
from now on.

### Changed

- Public model names are now `nitid1n`, `nitid1s`, `nitid1m`, `nitid1l` and
  `nitid1x` (see ADR-0004). `NITID.nitid_model`, the CLI help, the docs and the
  benchmark tooling use the new names.

### Added

- `tools/check_roadmap.py` checks that every issue link in `docs/roadmap.md`
  exists and that "Near-term priorities" only links to open issues.
- `nitid.__version__` is now available (it was only exposed as
  `dfine.__version__`). A unit test keeps it equal to the version in
  `pyproject.toml` (#24).
- `spaces/demo/` holds the source of an interactive Gradio demo (detection and
  instance segmentation on CPU), with unit and integration tests (#7). It is not
  published yet.
- `d-fine-n/s/m/l/x` (and `dfine_*`) are accepted as technical aliases for the
  current model generation. `nitid1-s` and any letter case are also accepted.
- `dfine.nitid.is_public_model_name` decides which names are routed through the
  public-name resolver.
- Tagging a version now creates the GitHub Release automatically after the PyPI
  upload: notes come from the matching `CHANGELOG.md` section, the wheel and
  sdist are attached, and `a`/`b`/`rc`/`dev` tags are marked as pre-releases
  (#3). `scripts/changelog_notes.py` previews the notes.

### Deprecated

- The `model1n/s/m/l/x` names remain accepted and resolve to `nitid1*`;
  `NITID("model1s")` now emits a `DeprecationWarning` naming the canonical
  replacement. They will be removed in a later minor release.

## [0.9.4] - 2026-11-06

### Added

- `nitid benchmark` exports a model to ONNX, TorchScript and TensorRT and
  compares mean latency, batches per second and image FPS on the current
  machine. Runtimes that are not installed are reported as skipped instead of
  aborting the other formats. See "Benchmark exported formats" in
  `docs/deployment.md`.
- `half=True` in prediction runs the forward pass under CUDA FP16 autocast and
  casts the outputs back to FP32 before postprocessing. On CPU it falls back to
  FP32.

## [0.9.3] - 2026-11-05

### Added

- `tools/benchmark_openvino.py` measures latency and COCO mAP of the exported
  detection models on OpenVINO CPU, GPU and NPU devices, using the same method
  as the D-FINE paper's T4 TensorRT figures, and writes a JSON report stamped
  with the hardware, drivers and library versions. `scripts/benchmark_openvino.sh`
  checks the machine, downloads COCO val2017 and runs it end to end. Runs where
  other processes competed for the CPU are flagged as unreliable, and runs that
  return broken outputs are flagged as invalid and get no latency figure.
- The README's detection table gains the latency measured with that tool on the
  Intel Arc B390 integrated GPU (Core Ultra X7 358H, OpenVINO 2026.4.1, FP16).

### Fixed

- Pretrained detection models reach the D-FINE paper's COCO accuracy again. The
  encoder's lateral convolutions applied a SiLU that D-FINE does not have, which
  cost 4 to 11 mAP points (`model1s` obj2coco: 39.3 → 50.7 mAP50-95 on COCO
  val2017). Segmentation models share the encoder and get the same fix. Models
  fine-tuned with nitid 0.9.x were trained with that activation; re-validate
  them after upgrading.
- `model1m` returns valid detections on Intel GPUs with FP16 precision. The
  `openvino` extra now requires OpenVINO 2026.4.1 or later; 2026.2.1 corrupted
  every inference after the first one on the GPU, which gave 0.0 mAP.

### Changed

- The MLflow integration names the experiment `nitid` when training has no
  `project`; it used to fall back to `/Shared/Ultralytics`. Set `project`,
  `experiment_name`, or `MLFLOW_EXPERIMENT_NAME` to keep the old name.
- The roadmap describes the released 0.9.x line and its current priorities.
- The brand page covers the visual identity only: name, logo, colours,
  typography, and the sources of the documentation images.

## [0.9.2] - 2026-10-09

### Fixed

- nitid installs on Python 3.13, which Google Colab now uses: the package
  declared `Requires-Python <3.13`, so `pip install nitid` found no matching
  version. Every dependency ships Python 3.13 wheels; CI covers 3.10 to 3.13.

## [0.9.1] - 2026-10-09

### Added

- A unified Python API and key-value CLI for downloading models, prediction,
  tracking, training, validation, export, model inspection, and bug reports.
  Models are named `model1n`, `model1s`, `model1m`, `model1l`, and `model1x`.
- Automatic download and runtime preparation of supported official
  checkpoints.
- D-FINE object detection and instance segmentation, including COCO and YOLO
  dataset support, mask visualization, fine-tuning, COCO validation, and
  export.
- Dense semantic segmentation with mask training, mIoU validation,
  probability output, overlays, class-ID maps, and ONNX/OpenVINO export.
  There are no pretrained semantic checkpoints: semantic models start from the
  instance-segmentation checkpoint and are fine-tuned on your own masks.
- ByteTrack, BoT-SORT, and OC-SORT tracking with persistent IDs, class
  filtering, frame sampling, and annotated video output.
- Video file, webcam, and RTSP/HTTP stream input through OpenCV, including
  authenticated RTSP cameras.
- OpenVINO Runtime inference on Intel CPU, integrated GPU, and NPU devices,
  with automatic device selection and compiled-model caching.
- ONNX and OpenVINO export for every supported task, plus TorchScript and
  TensorRT export for detection and instance segmentation.
- `export(..., postprocess=False)` exports the decoder's raw outputs without
  the postprocessor, for ONNX, OpenVINO, and TensorRT.
- Fine-tuning controls for AMP, EMA, checkpoint resume, cosine learning-rate
  warmup, freezing, augmentation, callbacks, per-epoch validation, W&B, and
  MLflow.
- Result helpers for annotated media, masks, semantic maps, JSON, YOLO text,
  pandas DataFrames, timings, and crops.
- Atomic, incremented run directories with saved arguments, environment
  snapshots, optional single-file bug reports, and exact output-path
  overrides.
- MkDocs documentation with a deployment guide for exported models, inference
  and training notebooks that run in Colab, example scripts, contributor and
  security guides, issue and pull-request templates, pre-commit hooks, and CI
  across Python 3.10, 3.11, and 3.12.
- Apache-2.0 licensing for the code. The official pretrained weights are
  published by D-FINE and D-FINE-seg under Apache 2.0 as well.
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
- Public messaging leads with the Apache 2.0 licensing of the code and the
  official weights, and announces three tasks: detection, instance
  segmentation, and semantic segmentation.

### Fixed

- Prediction and export preserve the trainable model's structure and state.
- OpenVINO parity checks tolerate valid encoder tie-breaking differences.
- `import nitid` works on servers and slim containers without `libGL`: the
  package depends on `opencv-python-headless`, and `Results.show()` falls back
  to matplotlib, or raises a clear error when no display is available.
- MLflow tracking works with MLflow 3: a local tracking directory such as the
  default `runs/mlflow` is stored as SQLite instead of MLflow's file store,
  which MLflow 3 refuses by default.

[Unreleased]: https://github.com/Nitidio/nitid/compare/v0.9.5...HEAD
[0.9.5]: https://github.com/Nitidio/nitid/compare/v0.9.4...v0.9.5
[0.9.4]: https://github.com/Nitidio/nitid/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/Nitidio/nitid/releases/tag/v0.9.3
