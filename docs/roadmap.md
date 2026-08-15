# nitid Roadmap

*Last reviewed: 2026-08-06.*

This roadmap explains where nitid is heading, what is currently in scope, and what is intentionally out
of scope. It is the canonical version; the [`ROADMAP.md`](https://github.com/Vaelsys/nitid/blob/develop/ROADMAP.md)
at the repository root points here.

nitid is an Ultralytics-style wrapper around the D-FINE object detector. The goal is not to replace
D-FINE's research code, but to make D-FINE easier to use for common object detection workflows: install,
download a model, predict, train, validate, export, and inspect results.

## Product direction

nitid should feel familiar to users who already know Ultralytics YOLO, while keeping D-FINE as the
underlying detection model. The experience we are building toward:

1. Install nitid.
2. Load a model by name, such as `dfine_s`.
3. Run prediction on images, videos, folders, streams, or screen input.
4. Save annotated outputs and structured detection results.
5. Fine-tune on custom detection datasets.
6. Validate training quality with standard detection metrics.
7. Export models for deployment.
8. Use docs, CLI help, and examples without reading the internals first.

### Where nitid competes

Three bets, in the order they earn their keep:

- **Edge and compliance.** A certified runtime, a published hardware support matrix, and a video
  pipeline built for IP cameras rather than files. **No comparable project contests this ground** — it
  is the clearest reason for nitid to exist independently, and it ranks highest for that reason.
- **D-FINE fidelity and depth.** nitid ships the full architecture in its package, loads published
  checkpoints strictly, provides detection and instance segmentation, and preserves proposal ranking
  across a class-taxonomy change. Depth over breadth.
- **Model breadth.** Covering the YOLO family so nitid can be an entry door for users migrating from
  Ultralytics. Deliberately last, and deliberately conditional — see below.


families and 13 tasks, and it already includes its own native D-FINE implementation. nitid does not
intend to compete with it on breadth and will not fork it; any future multi-family support would wrap
it as an optional dependency behind nitid's single API.

The full comparison — including the parts where nitid loses — is in

## In scope

- D-FINE object detection.
- A YOLO-familiar Python API through the `DFINE` class.
- Command-line workflows for prediction, training, validation, export, download, and model information.
- Automatic download and wrapping of official D-FINE checkpoints.
- Prediction on images, videos, folders, streams, screen capture, and numpy arrays.
- Result helpers for annotated images, JSON, YOLO TXT labels, CSV, pandas DataFrames, timing
  information, and object crops.
- Fine-tuning and validation on COCO-style and YOLO-format detection datasets.
- Training quality features: AMP, EMA, resume, per-epoch metrics, callbacks, experiment tracking.
- Export to ONNX, OpenVINO IR, TorchScript, and TensorRT.
- Video ingest for real deployments — RTSP, ONVIF discovery, hardware-accelerated decode.
- A lightweight web application for browser-based detection workflows.
- Deployment targets that can be tested and documented, with a stated support window.
- Documentation, examples, troubleshooting, and onboarding material.
- Tests and CI that protect the core user flows.

## Out of scope, for now

Each of these has a gate that would change the answer. None is a permanent refusal except where stated.

| Not doing | Why | What would change it |
|---|---|---|
| Reimplementing the Ultralytics platform or HUB | Not the product | Nothing foreseeable |
| Our own SAHI, tracker, or annotation tool | Good MIT implementations exist; reimplementing them is not a differentiator | Nothing foreseeable |
| Replacing the D-FINE research repository | nitid is a wrapper, deliberately | Nothing foreseeable |
| Diverging from D-FINE's training recipe without evidence | Reproducibility is the point | Measured improvement, documented |
| Every deployment runtime at once | Each target must be testable and supportable | A target enters via the hardware matrix, not ad hoc |

!!! question "Open: what happens to #123?"
    [#123](https://github.com/Vaelsys/nitid/issues/123) proposes segmentation, pose and OBB heads that
    share the D-FINE backbone, citing D-FINE-seg and DETRPose as precedent. One reading is that this is
    *depth* — the same network, an incremental head, squarely in nitid's lane. The other is that it is
    *breadth*, chasing a project that already shipped 13 tasks. This roadmap does not decide it. The
    decision belongs with the maintainers and should be recorded as an ADR either way.

## Current status

nitid has a complete first workflow today:

- Install from the repository with `uv` (PyPI publishing is in progress — see Phase 1b).
- Download and wrap official D-FINE checkpoints automatically, across four sizes and two weight variants.
- Predict from Python or the CLI, on any supported source type.
- Export detections as annotated images, JSON, YOLO TXT, CSV, pandas DataFrames, and crops.
- Fine-tune and validate on COCO JSON or YOLO txt datasets, with per-class AP, confusion matrix and
  PR/F1 curves.
- Train with AMP, EMA, resume, callbacks, and W&B or MLflow tracking, with per-run `args.yaml`,
  `environment.yaml`, `results.csv` and `results.png`.
- Export to ONNX, OpenVINO IR, TorchScript, and TensorRT.
- Capture diagnostics with `--report` or `dfine bugreport`.
- Run browser-based detection through the multi-user web application.
- 221 tests in CI across Python 3.10, 3.11 and 3.12, with ruff, mypy and coverage reporting.

## Phases

Effort figures are rough dev-day estimates, not commitments.

### Phase 0 — Decide and unblock (August 2026, ~11 d)

| Work | Issues | Days |
|---|---|---|
| Review and merge the tracking / media / ONVIF / GStreamer work (5,159 LOC — split into separate PRs if review stalls) | [#53](https://github.com/Vaelsys/nitid/issues/53), [#127](https://github.com/Vaelsys/nitid/issues/127) | 4–6 |
| Stop `predict()` and `export()` mutating the trainable model | [#105](https://github.com/Vaelsys/nitid/issues/105) | 2–3 |
| Add `mkdocs build --strict` as a pull-request job | — | 0.5 |

The docs workflow currently builds only on push to `main`, and not strictly, so a broken documentation
link is invisible until after it has merged.


### Phase 1a — CRA readiness minimum (by 2026-09-11, ~6 d)

Runs in parallel and is **deadline-driven**. The EU Cyber Resilience Act brings notification obligations
from **2026-09-11** and CE marking from **2027-12-11**. These are fixed calendar constraints, not
aspirations, and this slice outranks everything else on the list.

- Extend `SECURITY.md` with response-time commitments and a named internal owner.
- Generate an SBOM (CycloneDX) covering `uv.lock` and the packaged model implementation.
- Publish the supported-version window.

First slice of [#125](https://github.com/Vaelsys/nitid/issues/125).

### Phase 1b — Ship v0.1.0 to PyPI (~9 d)

[#33](https://github.com/Vaelsys/nitid/issues/33) is a hard prerequisite for shipping tracking, for
[#56](https://github.com/Vaelsys/nitid/issues/56), and for any optional-dependency approach to breadth.
The model implementation is now included in source and wheel distributions; publishing and release
automation remain.

| Work | Days |
|---|---|
| Verify the packaged D-FINE source and accompanying notices in release artifacts | 1 |
| Test model construction from installed source and wheel distributions | 0.5 |
| Add a `nitid` console script alongside `dfine` — free before the first release, a deprecation cycle after | 0.5 |
| Release workflow: tag → build → TestPyPI → PyPI trusted publishing. Claim the `nitid` name now; it is currently unregistered | 2 |
| Versioning and CHANGELOG policy; pip install path in README and docs | 1 |

**Gate G1.**


desk-research spikes run first, because any one of them can eliminate an option before GPU time is spent.


### Phase 3 — The differentiation bets (Q4 2026 → 2027)

| Work | Issues | Days |
|---|---|---|
| Streaming: hardening, hardware decode matrix, output leg, reconnect and backpressure | [#127](https://github.com/Vaelsys/nitid/issues/127) | 10–15 |
| Certified hardware matrix — OpenVINO/Intel first, then Rockchip RKNN, then Hailo. Jetson explicitly out unless funded separately. Scope depends on spike S5 | [#126](https://github.com/Vaelsys/nitid/issues/126) | 20–30 |
| Certified LTS runtime: pinned stack, CVE tracking, upgrade/backport policy, CE-marking preparation | [#125](https://github.com/Vaelsys/nitid/issues/125) | 25–40 |
| Benchmark CLI across export formats and hardware — the measurement tool the hardware matrix depends on | [#71](https://github.com/Vaelsys/nitid/issues/71) | 5 |
| Deployment guide and published Docker images | [#46](https://github.com/Vaelsys/nitid/issues/46), [#45](https://github.com/Vaelsys/nitid/issues/45) | 4 |


nitid has no type for (3 d). Tests skipped when the extra is absent, plus a CI job that installs it
(2 d). Documentation and migration notes (2 d). Pin-bump validation runbook (1 d).

### Phase 5 — Ecosystem (2027)

| Work | Issue | Days |
|---|---|---|
| "Coming from Ultralytics" migration guide — cheap and high-conversion; pull forward if capacity appears | [#124](https://github.com/Vaelsys/nitid/issues/124) | 3 |
| HuggingFace Hub model registry with dataset, metrics and date provenance | [#121](https://github.com/Vaelsys/nitid/issues/121) | 5 |
| Dataset format converters as a CLI command — the validator already converts YOLO to COCO with a cache, so this is largely surfacing existing code | [#122](https://github.com/Vaelsys/nitid/issues/122) | 4 |
| SAHI sliced inference, via the upstream `sahi` package | [#56](https://github.com/Vaelsys/nitid/issues/56) | 4 |
| Web application tests | [#52](https://github.com/Vaelsys/nitid/issues/52) | 3 |

## Known gaps

Honest about the things that are neither differentiators nor currently scheduled:

- **No distributed training and no autobatch.** nitid trains on one GPU and rejects `batch=-1`. This is
  table stakes rather than differentiation, which is why it sits below the bets — but it is a real
  limitation for anyone training at scale. Estimated 8–12 d, after Phase 1.
- **No quantisation.** Awkward given the edge focus: INT8 is how a detector fits on an edge
  accelerator. Scope will be decided by spike S5 and the hardware matrix.
- **Stale documentation.** `docs/public_release_checklist.md` still lists as missing roughly seventeen
  things that now exist, and `CLAUDE.md` describes the trainer and validator as incomplete. Both need a
  pass; the checklist's Ultralytics-gap framing is now superseded by

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
