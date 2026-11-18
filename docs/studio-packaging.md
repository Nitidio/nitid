# Nitid Studio: packaging, wheel verification and CI

This guide is for maintainers. It explains how the Studio web UI gets from
`studio/` (TypeScript sources) into the Python wheel, what is checked on the
way, and what fails in CI and why. Facts that were not verified are marked `[NV]`.

## The flow in one picture

```text
studio/ (Vite + React sources, pnpm-lock.yaml)
   |  pnpm install --frozen-lockfile && pnpm run build
   v
nitid/studio/static/        <- git-ignored, never committed
   |  uv build   (sdist first, then the wheel built from that sdist)
   v
dist/nitid-X.Y.Z-py3-none-any.whl   (+ the sdist)
   |  scripts/verify_wheel.py   (contents + size budget)
   v
install the wheel with the [studio] extra
   |  scripts/verify_install.py --require-ui   (start Studio, probe it, stop it)
   v
publish (release.yml only)
```

The compiled UI is a build product. It is produced just before `uv build`, so a
wheel built from a clean checkout has no UI and `verify_wheel.py` rejects it.

## 1. Build the frontend: `scripts/build_studio.py`

```bash
python scripts/build_studio.py              # install from the lockfile, then build
python scripts/build_studio.py --check-only # only check nitid/studio/static/index.html exists
```

It needs Node.js >= 22.13 and pnpm (`corepack enable` picks the version pinned in
`packageManager` in `studio/package.json`; a different pnpm only prints a warning).
It runs `pnpm install --frozen-lockfile` and `pnpm run build` in `studio/`, with
`STUDIO_OUT_DIR` set to `nitid/studio/static`, then requires `index.html`.

| Exit code | Meaning |
| --- | --- |
| 0 | Built (or, with `--check-only`, `index.html` is present). |
| 1 | A pnpm command failed, or the build did not produce `index.html`. |
| 2 | The environment is unusable: Node too old or missing, pnpm missing, no `studio/`. |
| 3 | `--check-only` and `nitid/studio/static/index.html` is missing. |

## 2. What goes into the wheel

`pyproject.toml` declares `[tool.setuptools.package-data] "nitid.studio" = ["static/**"]`.
Verified locally on 2026-11-18 (setuptools 82.0.1, `uv build`):

- `static/**` (index, hashed assets, nested folders such as `ort/`) enters the
  wheel. Dot-files (for example `.hidden`) do not; Vite emits none today.
- **No `MANIFEST.in` is needed.** The sdist also contains `static/` when it exists at
  build time, because setuptools uses `package-data` for both. The sdist does not
  contain `studio/` (the frontend sources); that is intended.
- `uv build` builds the sdist first and the wheel **from the sdist**. The UI therefore
  has to exist before `uv build`; building it afterwards has no effect.
- Without `static/`, the build succeeds silently and the wheel has no UI. That is why
  `verify_wheel.py` requires `index.html` by default.

`tests/unit/test_studio_package_data.py` pins this by building a small tree with the
real `pyproject.toml` through setuptools (offline).

## 3. Wheel rules: `scripts/verify_wheel.py`

```bash
python scripts/verify_wheel.py dist/nitid-*.whl
python scripts/verify_wheel.py dist/nitid-*.whl --allow-missing-ui   # development only
```

On top of the existing checks (required files, licences, only the `dfine` and
`nitid` top-level packages, no reference source trees), the wheel must:

| Rule | Why |
| --- | --- |
| Contain `nitid/studio/static/index.html` | A wheel without the UI would serve a 503 page. `--allow-missing-ui` is explicit, never the default, and CI/release never use it. |
| Contain no `node_modules` directory | Frontend tooling is a build input. |
| Contain no `*.onnx` or `*.pth` | Weights are downloaded or exported at run time, never shipped. |
| Contain no `*.map`, `*.ts`, `*.tsx` | Source maps and sources are not distributable files (production builds have source maps off). |
| Contain `*.wasm` only under `nitid/studio/static/ort/` | The one expected home of ONNX Runtime Web files. Anywhere else is a mistake. |
| Be at most `MAX_WHEEL_BYTES` = 15 MB (15 * 1024 * 1024) | Size budget of the wheel file as uploaded to PyPI. |

When the budget is exceeded the message names the size, the budget and the constant.
To see what grew: `unzip -lv dist/*.whl | sort -k1 -n | tail`. Raising the budget is a
deliberate edit of the constant in the same pull request that explains why.

Measured today (scaffold UI, 2026-11-18): wheel 282 831 bytes (0.27 MB), of which the
UI is about 266 kB uncompressed.

Rules are tested with synthetic wheels in `tests/unit/test_verify_wheel_studio.py`:
valid, missing index, `node_modules`, `.onnx`, `.pth`, `.map`, `.ts`/`.tsx`, `.wasm`
outside `ort/`, oversized, and the older rules.

## 4. Installed-package check: `scripts/verify_install.py`

After the existing detection/segmentation checks it runs `verify_studio()`:

1. Skips with a message (and does not fail) if FastAPI or uvicorn are not importable,
   that is, if the `studio` extra is not installed.
2. Starts `python -m nitid.studio port=0 --no-browser token=<random>` in a subprocess.
3. Waits up to 120 s for the `Open: http://...` line (a cold start imports torch).
4. `GET /api/v1/health` (no token) must answer 200 with `status: ok`. It retries
   refused connections: the launcher prints its URL after binding the port but a moment
   before uvicorn listens.
5. `GET /` must contain `id="root"`. With `--require-ui` (used in CI and release) anything
   else fails; without it, a build without UI is accepted only if it answers the
   documented 503 page.
6. Stops the server with Ctrl+C (SIGINT; `terminate()` on Windows) and kills it only if it
   does not exit within 15 s. The exit code is printed but not asserted: uvicorn
   re-raises SIGINT after shutting down, so the code is `-2` on Linux.

It uses only the standard library for HTTP, so it works in the slim container without
libGL that the `package` job already uses (the wheel and `opencv-python-headless` need
no GUI libraries).

## 5. Try it locally

```bash
# 1. frontend -> nitid/studio/static
python scripts/build_studio.py

# 2. wheel
rm -rf dist && uv build
python scripts/verify_wheel.py dist/nitid-*.whl

# 3. install the wheel with the extra into a clean environment and probe it
uv venv /tmp/nitid-venv
wheel=$(ls dist/*.whl)
uv pip install --python /tmp/nitid-venv/bin/python "$wheel[studio]"
cd /tmp && /tmp/nitid-venv/bin/python /path/to/repo/scripts/verify_install.py --require-ui
```

Unit tests: `python -m pytest tests/unit/test_build_studio.py tests/unit/test_verify_wheel_studio.py tests/unit/test_verify_install_studio.py tests/unit/test_studio_package_data.py`.

## 6. CI and release

`.github/workflows/ci.yml`:

| Job | What it does | Fails when |
| --- | --- | --- |
| `studio-ui` | Node 22, pnpm via Corepack (version from `packageManager`), `pnpm install --frozen-lockfile`, `pnpm run check` (lint, typecheck, tests, build), `pnpm run check:size`, uploads `nitid/studio/static` as artifact `studio-static`. | The lockfile is out of date, a frontend check fails, a tracked file breaks the size policy, or `index.html` is not produced. |
| `package` | Needs `studio-ui`. Downloads `studio-static` into `nitid/studio/static/`, `uv build`, `verify_wheel.py`, installs the wheel, `verify_install.py --require-ui`, then repeats the install check in `python:3.12-slim` with `"<wheel>[studio]"`. | Any wheel rule above is broken, the UI is not served, or Studio does not start or stop. |
| `test`, `mypy` | Install `--extra studio`, so the Studio tests run and mypy sees the real FastAPI types. | Tests or type checks fail. |

`.github/workflows/release.yml` builds the UI (`corepack enable` and
`python scripts/build_studio.py`) before `uv build`, fails explicitly if `index.html` is
missing, and runs the same wheel and installed-package checks. The publishing steps
(PyPI trusted publishing, TestPyPI, GitHub release) are unchanged.

`.github/dependabot.yml` also tracks the `npm` ecosystem in `/studio`, weekly.

## 7. Maintainer notes and open items

- **`uv.lock` is out of date** after this change (new dev dependencies `jsonschema`,
  `openapi-spec-validator`, `httpx`, plus the earlier `studio` extra). The `lint` job runs
  `uv lock --check` and will fail until someone runs `uv lock` on a machine with access
  to the PyTorch index and commits `uv.lock`. It must not be edited by hand.
- Action versions in the new steps (`actions/setup-node@v4`, `actions/upload-artifact@v4`,
  `actions/download-artifact@v4`) follow the ones already used for artifacts; the latest
  releases were not checked (no GitHub access from the authoring environment) `[NV]`;
  dependabot will propose bumps.
- The workflows were linted with `actionlint` and parsed as YAML; they have not been run
  on GitHub `[NV]`. The slim-container step was not run locally (the Docker daemon is not available there) `[NV]`.
- The pnpm store is not cached in CI; add `cache: pnpm` to `setup-node` if install time
  matters `[NV]`.
- This page is not in the `mkdocs.yml` navigation, like `studio-development.md`
  (the docs build is `--strict` but accepts pages outside the nav).
