# Contributing to nitid

Thank you for your interest in contributing to nitid!

## Setting up the development environment

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/vaelsys/nitid.git
cd nitid
git submodule update --init
uv sync --extra dev
```

Install PyTorch separately before syncing — see the [README](README.md) for instructions.

If you are working on training, validation, or COCO-style evaluation locally,
install the training extras as well:

```bash
uv sync --extra dev --extra train
```

If you are working on export support, you may also need format-specific
dependencies such as ONNX or TensorRT extras depending on the target.

## Development workflow

The project follows a lightweight branch-based workflow:

1. Start from `develop`.
2. Create one focused branch per issue or task.
3. Make the smallest code and docs change that fully resolves the issue.
4. Add or update tests for behavior changes.
5. Run the relevant test, lint, and type-check commands locally.
6. Open a PR back to `develop` with a clear summary of what changed and why.

For most changes, the practical loop is:

```bash
git checkout develop
git pull
git checkout -b feat/my-change
uv sync --extra dev
uv run pytest tests/unit
uv run ruff check .
uv run mypy dfine/
```

Then iterate on the narrower tests for the area you touched before running the
full suite.

## Running tests

Use the narrowest command that covers your change first, then the full suite
before opening a PR.

```bash
# Unit tests only (no GPU or checkpoint needed)
uv run pytest tests/unit

# Integration tests only
uv run pytest tests/integration

# All tests (integration tests build a small checkpoint automatically)
uv run pytest
```

Useful targeted examples:

```bash
# One file
uv run pytest tests/integration/test_predict.py

# One test
uv run pytest tests/integration/test_predict.py -k vid_stride
```

Some tests depend on optional extras:

- Training and validation tests require `pycocotools` (`uv sync --extra train`)
- TensorRT export tests only run meaningfully in environments with TensorRT available

## Linting and type checking

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy dfine/
```

Run `uv run ruff format .` if you want Ruff to apply formatting fixes directly.

## Commit message conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add ONNX dynamic batch axis support
fix: correct bounding box normalization in Results
docs: update fine-tuning guide
ci: add GPU test workflow
refactor: simplify checkpoint loading logic
```

## Branch conventions

- Branch from `develop`
- One branch per issue: `feat/`, `fix/`, `docs/`, `ci/`
- Open a PR back to `develop`
- `develop` is merged to `main` at release

## Extending nitid

When you add a new capability, keep the public API, implementation, tests, and
docs aligned.

Some common extension points:

- Prediction behavior: start with [`dfine/model.py`](dfine/model.py),
  [`dfine/predictor.py`](dfine/predictor.py), and the relevant integration
  coverage in `tests/integration/test_predict.py`.
- Training or validation behavior: look at [`dfine/model.py`](dfine/model.py),
  [`dfine/trainer.py`](dfine/trainer.py), [`dfine/validator.py`](dfine/validator.py),
  and the corresponding train/val tests.
- Data loading: keep dataset parsing and path resolution in `dfine/utils/`
  and add focused unit tests for edge cases.
- Export formats: extend [`dfine/exporter.py`](dfine/exporter.py), add
  integration coverage in `tests/integration/test_export.py`, and document any
  optional dependencies or format-specific limitations.

General guidelines:

1. Keep the public `DFINE` interface small and consistent.
2. Prefer extending the existing module for a feature before creating a new
   abstraction.
3. Put optional dependency handling close to the feature that requires it.
4. Add the narrowest useful tests first, then broaden coverage if the change
   affects shared behavior.
5. Update the docs that users or contributors will actually read for that
   feature (`README.md`, `docs/`, or API reference pages as appropriate).

## Pull request checklist

Before opening a PR, make sure:

- [ ] Branch is based on `develop`
- [ ] The change is scoped to a single issue or tightly related set of issues
- [ ] Unit tests pass: `uv run pytest tests/unit`
- [ ] Integration tests pass: `uv run pytest tests/integration`
- [ ] Full suite passes when the change touches shared behavior: `uv run pytest`
- [ ] No lint errors: `uv run ruff check .`
- [ ] No formatting errors: `uv run ruff format --check .`
- [ ] No mypy errors: `uv run mypy dfine/`
- [ ] New behaviour is covered by tests
- [ ] Documentation updated if needed
- [ ] Optional dependency changes are reflected in docs or install instructions
- [ ] PR description explains what changed, why it changed, and how it was verified
- [ ] Screenshots / logs are included when user-visible output changed
