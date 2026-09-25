# Developer Onboarding

Welcome to **nitid** — a compact, Apache-2.0 library around the [D-FINE](https://github.com/Peterande/D-FINE) family of real-time detectors. This guide is for developers joining the project. The user-facing docs live in the rest of `docs/`; this page covers the architecture decisions you need to understand before touching the code.

---

## 1. Get the repo running

Prerequisites: Python 3.10+, [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Vaelsys/nitid.git && cd nitid
uv sync --extra dev                # installs runtime + pytest, ruff, mypy
```

Verify everything works:

```bash
uv run pytest tests/unit           # ~seconds, no GPU, no checkpoint required
uv run ruff check .
uv run mypy dfine/
```

---

## 2. Project layout

```
nitid/              Public Python package — import NITID from here
  cli.py                  `nitid` CLI entry point (`dfine` is a back-compat alias)
  convert_checkpoint.py   Maintainer utility for raw upstream checkpoints
dfine/              Internal implementation and compatibility namespace
  nitid.py          NITID public class and model-name mapping
  model.py          D-FINE backend class and checkpoint validation
  predictor.py      Inference worker
  trainer.py        Fine-tuning worker
  validator.py      COCO evaluation worker
  exporter.py       ONNX / OpenVINO / TorchScript / TensorRT export worker
  results.py        Results + Boxes + Masks + SemanticMask return types
  nn/               Integrated detection, segmentation, and semantic architectures/losses
  utils/            sources.py (LoadSource), plotting, misc helpers
tools/
  visualize_augmentations.py  Source-checkout maintainer utility
configs/
  datasets/         coco.yml and example_custom.yml
tests/
  unit/             Pure Python — no GPU or downloaded checkpoint
  integration/      Use tiny_checkpoint fixture (see §5)
  conftest.py       Session-scoped fixture that builds a tiny model at test time
docs/               All documentation lives here
```

---

## 3. The architecture in one paragraph

`NITID` is the recommended class users touch. It accepts supported model names
or self-contained `.pth` checkpoints and delegates `predict`,
`track`, `train`, `val`, and `export` to internal workers. Model construction,
losses, and postprocessing live under `dfine/nn/`. The checkpoint's embedded
`task` selects detection, instance segmentation, or semantic segmentation; an
explicitly requested task must match it.

---

## 4. Model core

`dfine/nn/native_build.py` is the construction boundary for supported tasks.
Keep architecture settings in `dfine/nn/configs.py`, route checkpoint
construction through `build_model`, and route losses through `build_criterion`.
Do not add import-path mutation or runtime source discovery: the installed
package must contain everything required to construct a model.

Detection and instance segmentation share the backbone, encoder, transformer
decoder, boxes, and class logits. `task="segment"` enables the mask head and
mask losses. Semantic segmentation adds a dense decoder on shared features.
New code must preserve strict state-dict compatibility with published
checkpoints.

---

## 5. Wrapped checkpoint format

Runtime checkpoints are self-contained dicts with these keys:

| Key | Content |
|---|---|
| `model` | Model `state_dict` (EMA weights preferred over raw weights) |
| `config` | Full D-FINE config dict (all `__include__` directives resolved) |
| `names` | `{int: str}` class index → name mapping |
| `epoch` | Last saved epoch |
| `metrics` | Metrics dict from the epoch |

Official model names perform the download/preparation step automatically. The
raw-checkpoint conversion tool exists for maintainers and unusual research
workflows; it should not be part of the normal user path.

**Why self-contained files?** So users never have to track a separate config file. One file = one model. This also means `build_model` forces `HGNetv2.pretrained=False` (weights come from the checkpoint, not ImageNet) and `build_postprocessor` forces `remap_mscoco_category=False` (names come from `checkpoint["names"]`, not a hardcoded COCO mapping).

---

## 6. Running the tests

### Unit tests

```bash
uv run pytest tests/unit
```

No GPU or downloaded checkpoint is needed. These test pure Python logic, model construction, `Results`/`Boxes`/`Masks`, export argument validation, and dataset parsing.

### Integration tests

```bash
uv run pytest tests/integration
# or run everything:
uv run pytest
```

Integration tests use the `tiny_checkpoint` session fixture in `tests/conftest.py`. It builds a real D-FINE-S model with random weights entirely in memory — no download required. The fixture overrides `num_layers=1`, `num_queries=10`, `num_denoising=0`, `depth_mult=0.1` to keep build time fast (<10 s).

> `tests/integration/test_train.py` verifies the fine-tuning and validation pipelines using the tiny checkpoint on CPU.

---

## 7. Code conventions

| Setting | Value |
|---|---|
| Line length | 100 (E501 ignored; the formatter enforces it) |
| Python target | 3.10 — use `X \| Y` unions only with `from __future__ import annotations` |
| Linter | `ruff check .` |
| Type checker | `mypy dfine/` |
| Test runner | `pytest` |

Write no comments unless the *why* is genuinely non-obvious. Never add docstrings beyond a single short line. The existing codebase is intentionally sparse — match that style.

---

## 8. Inference data flow (follow the code)

Understanding this path makes the codebase navigable:

```
DFINE.predict()
  └─ DFINEPredictor.predict()
       └─ LoadSource (dfine/utils/sources.py)
            yields (tensor [1,3,H,W], orig_img HWC BGR, path_str)
       └─ DFINE.forward() → logits + boxes, and query masks for segmentation
       └─ DFINEPostProcessor(raw, orig_target_sizes)
            → [{labels, boxes, scores}]  boxes: absolute xyxy pixel coords
       └─ Results(boxes=[N,6], masks=[N,H,W] for segmentation)
```

The 300 queries are D-FINE's fixed-size output head. After postprocessing only the above-threshold detections remain in `Boxes._data`.

---

## 9. Getting a real checkpoint (for manual testing)

The integration tests use synthetic tiny models — they never need to download a
real checkpoint. For manual testing, prefer official model names:

```bash
uv run nitid predict model=nitid1s task=detect source=image.jpg conf=0.5
uv run nitid predict model=nitid1s task=segment source=image.jpg conf=0.5
```

If you are maintaining support for a new upstream checkpoint, use the conversion
tool explicitly:

```bash
uv run python -m nitid.convert_checkpoint \
    --weights upstream_checkpoint.pth \
    --model   nitid1l \
    --task    detect \
    --names   configs/datasets/coco.yml \
    --output  nitid1l_detect.pth
```

---

## 10. Where things can go wrong

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'config'` loading a `.pth` | File is not a nitid runtime checkpoint | Use a supported model name, a training checkpoint, or the maintainer conversion tool |
| `AMP has no effect` warning | Running on CPU with `amp=True` | Expected — silently degrades to FP32 |
| Explicit task does not match checkpoint | Checkpoint opened with the wrong task | Select a matching model/task pair |

---

## Further reading

| Doc | When to read it |
|---|---|
| [`quickstart.md`](quickstart.md) | User-facing getting-started guide (good context for what new users experience) |
| [`api_reference.md`](api_reference.md) | Full `DFINE` class API with all parameters |
| [`fine_tuning.md`](fine_tuning.md) | Dataset format, AMP, EMA, all training parameters |
| [`export.md`](export.md) | ONNX, TorchScript, TensorRT — options and constraints |
| [`macos_docker_setup.md`](macos_docker_setup.md) | Setup docker for linux enviroment in MacOS Apple silicon |
