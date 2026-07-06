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

## Running tests

```bash
# Unit tests only (no GPU or checkpoint needed)
uv run pytest tests/unit

# All tests (integration tests build a small checkpoint automatically)
uv run pytest
```

## Linting and type checking

```bash
uv run ruff check .
uv run mypy dfine/
```

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

## Pull request checklist

Before opening a PR, make sure:

- [ ] Unit tests pass: `uv run pytest tests/unit`
- [ ] Integration tests pass: `uv run pytest tests/integration`
- [ ] No lint errors: `uv run ruff check .`
- [ ] No formatting errors: `uv run ruff format --check .`
- [ ] No mypy errors: `uv run mypy dfine/`
- [ ] New behaviour is covered by tests
- [ ] Documentation updated if needed
- [ ] PR description explains what and why