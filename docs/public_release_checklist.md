# Public release checklist

Comparison of nitid against Ultralytics YOLO, prioritised by impact on a first-time user.

---

## P0 — Blockers (repo cannot be used without these)

### 1. LICENSE file
**Status:** missing  
The repository has no `LICENSE` file. Without one, the code is legally "all rights reserved" and nobody can legally use, modify, or redistribute it. Pick a license (Apache 2.0 is the most natural choice given D-FINE is Apache 2.0) and add it to the root.

### 2. Downloadable wrapped checkpoints
**Status:** missing  
The README says "if you have a raw D-FINE checkpoint, convert it first" but there is no hosted `.pth` to download and no link to one. A new user who clones the repo has nothing to run.

Options:
- Release wrapped checkpoints as GitHub Release assets (one per model size: S/M/L/X).
- Or add a `dfine download model=dfine_l` CLI command that fetches them automatically (matches how `ultralytics YOLO("yolo11n.pt")` auto-downloads weights).

### 3. pip install support
**Status:** partial  
Installation currently requires `uv`. This is a high barrier — most users expect `pip install nitid`. The package is structured correctly (`pyproject.toml`, `[project.scripts]`) so publishing to PyPI and adding a plain pip install path to the README is the main missing step. The CUDA index pinning in `[tool.uv.sources]` will need to be documented separately as a uv-specific override (pip users install their own torch).

---

## P1 — High priority (first things a contributor looks for)

### 4. GitHub Actions CI
**Status:** missing (no `.github/` directory)  
There are no automated checks. PRs have no safety net. Minimum viable workflow: run `uv run pytest tests/unit` and `uv run ruff check .` on every push and pull request. Integration tests can be conditional (GPU not available on free runners).

### 5. CONTRIBUTING.md
**Status:** missing  
Ultralytics has a detailed contributing guide. nitid needs at minimum:
- How to set up the dev environment (`uv sync --extra dev`)
- How to run tests and lint
- Commit message conventions
- PR checklist

### 6. Issue and PR templates
**Status:** missing  
`.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`, and `.github/pull_request_template.md` reduce noise and help triage. Without them, bug reports arrive with no reproducible info.

### 7. README: fill in the placeholder repo URL
**Status:** broken  
Line 29 reads `git clone <repo>`. The actual URL must be filled in before the repo goes public.

### 8. README: model download / checkpoint table
**Status:** missing  
Ultralytics' README has a table of model variants with accuracy (mAP), speed (ms), and params, each row linking to a downloadable checkpoint. nitid should have the same: D-FINE-S / M / L / X with their COCO mAP and a download link to the wrapped `.pth`.

---

## P2 — Medium priority (build trust and professionalism)

### 9. CHANGELOG.md
**Status:** missing  
A changelog (or GitHub Releases with release notes) tells users what changed between versions and signals that the project is actively maintained.

### 10. SECURITY.md
**Status:** missing  
A security policy tells researchers how to report vulnerabilities privately instead of opening a public issue. One paragraph and an email address is enough (`security@vaelsys.com` or similar).

### 11. CODE_OF_CONDUCT.md
**Status:** missing  
Standard Contributor Covenant boilerplate. Required by most open-source norms and by GitHub's community health checklist.

### 12. README badges
**Status:** missing  
Ultralytics' README leads with CI, PyPI version, license, and Python version badges. These signal project health at a glance. At minimum: license badge and Python version badge (these do not require PyPI or CI to be set up first).

### 13. .gitignore gaps
**Status:** partial  
`dfine_l_wrapped.pth`, `image.jpg`, `nitid_result.jpg`, `plain_dfine_result.jpg`, and `results.jpg` appear in `git status` as untracked. They should either be committed (if intentional examples) or added to `.gitignore`. The `web/storage/nitid.db`, `web/storage/uploads/`, and `web/storage/results/` runtime dirs also need to be in `.gitignore`.

### 14. Pre-commit config
**Status:** missing  
`.pre-commit-config.yaml` running ruff and mypy ensures contributors never push unformatted code, without requiring CI to catch it.

---

## P3 — Nice to have (matches Ultralytics polish)

### 15. Jupyter notebook / Colab tutorial
**Status:** missing  
Ultralytics ships `examples/tutorial.ipynb` that runs end-to-end in Google Colab without any local setup. A single notebook covering install → predict → fine-tune → export dramatically lowers the barrier for ML practitioners who work in notebooks.

### 16. Performance / benchmark table
**Status:** missing  
The README has no numbers. A table showing D-FINE-S/M/L/X mAP50-95, inference speed (ms on A100 and CPU), and parameter count — compared to YOLO11 equivalents — gives users the reason to choose nitid.

### 17. Docker support
**Status:** missing  
A `Dockerfile` (CPU inference + optional GPU) lets users skip the Python/CUDA setup entirely. Ultralytics maintains Docker images for every release. A single `Dockerfile` in the repo root covers the basic case.

### 18. Docs site (MkDocs / Sphinx)
**Status:** partial  
The `docs/` folder has good Markdown files but they are not rendered as a navigable site. Ultralytics runs docs.ultralytics.com. For nitid, deploying with MkDocs + GitHub Pages costs almost nothing and makes documentation searchable and linkable.

### 19. Dependabot
**Status:** missing  
`.github/dependabot.yml` enables automatic security PRs for pip and GitHub Actions dependencies.

### 20. Codecov / test coverage badge
**Status:** missing  
Add `pytest-cov` (already in dev deps) to the CI workflow and upload to Codecov. The coverage badge in the README signals test quality.

---

## Summary table

| # | Item | Status | Effort |
|---|------|--------|--------|
| 1 | LICENSE file | missing | low |
| 2 | Downloadable wrapped checkpoints | missing | medium |
| 3 | pip install / PyPI | missing | medium |
| 4 | GitHub Actions CI | missing | low |
| 5 | CONTRIBUTING.md | missing | low |
| 6 | Issue & PR templates | missing | low |
| 7 | README repo URL | broken placeholder | low |
| 8 | README model table with download links | missing | low |
| 9 | CHANGELOG.md | missing | low |
| 10 | SECURITY.md | missing | low |
| 11 | CODE_OF_CONDUCT.md | missing | low |
| 12 | README badges | missing | low |
| 13 | .gitignore gaps | partial | low |
| 14 | Pre-commit config | missing | low |
| 15 | Jupyter / Colab notebook | missing | medium |
| 16 | Benchmark / performance table | missing | medium |
| 17 | Docker support | missing | medium |
| 18 | Docs site (MkDocs + GitHub Pages) | partial | medium |
| 19 | Dependabot | missing | low |
| 20 | Codecov / coverage badge | missing | low |
