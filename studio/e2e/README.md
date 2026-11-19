# Studio end-to-end tests

Real stack, no mocks: the product's own launcher (`python -m nitid.studio port=0 ...`), the built
UI, tiny checkpoints built at run time and real headless Chrome driven by `playwright-core`.
Full description, tolerances and known gaps: [`docs/studio-e2e.md`](../../docs/studio-e2e.md).

```bash
# once: Python extras, frontend dependencies, playwright-core (no browser download)
uv sync --extra dev --extra train --extra studio
(cd studio && pnpm install --frozen-lockfile)
npm install --prefix /tmp/pw playwright-core
export PLAYWRIGHT_CORE_PATH=/tmp/pw/node_modules/playwright-core   # optional if it resolves already
export CHROME_PATH=/usr/bin/google-chrome                          # optional, this is the default

pytest studio/e2e -s -rs        # about 2 minutes on 2 CPUs
pytest studio/e2e/test_studio_e2e.py -k ui         # one scenario
```

`nitid/studio/static` is built with `scripts/build_studio.py` when it is missing. Missing Chrome,
Node, pnpm, `playwright-core` or Python extras are **skips with a reason**, never failures
(`-rs` prints them). The folder is not in `testpaths`, so a plain `pytest` does not run it.

| File | What it is |
| --- | --- |
| `test_studio_e2e.py` | UI flows (`ui.mjs`) and security checks (`security.mjs`) against `nitid studio`. |
| `test_studio_parity.py` | Server (`/runs`, PyTorch) vs browser engine (ORT-web) on the same image. |
| `conftest.py`, `support.py` | Fixtures, process management, prerequisite checks. |
| `make_fixtures.py` | Tiny detect and segment checkpoints plus the test image (nothing is downloaded or committed). |
| `lib.mjs`, `ui.mjs`, `security.mjs`, `parity.mjs` | Playwright drivers; each writes `{checks, data}` JSON that pytest asserts on. |
| `harness/` | A Vite page exposing the real browser engine as `window.__parity` (test tooling, never shipped). |
| `serve_harness.py` | The real Studio app (middleware, headers, API) serving that harness. |

The drivers can be run by hand too: `node studio/e2e/ui.mjs --url 'http://127.0.0.1:PORT/#t=TOKEN'
--image image.png --out result.json`; exit status 77 means "skipped".
