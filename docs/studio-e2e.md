# Nitid Studio: end-to-end tests

This page describes the suite in `studio/e2e/` (issue #47): the **real** Studio server, the **built**
UI and **real** headless Chrome, with nothing mocked. It complements the unit tests (`pnpm run test`),
the API contract tests (`tests/integration/studio/`) and the Chrome smoke test with a mocked API
(`studio/scripts/smoke-local.mjs`). Facts that were not verified are marked `[NV]`.

## How to run

```bash
uv sync --extra dev --extra train --extra studio
(cd studio && pnpm install --frozen-lockfile)
npm install --prefix /tmp/pw playwright-core          # driver only; it uses the installed Chrome
export PLAYWRIGHT_CORE_PATH=/tmp/pw/node_modules/playwright-core
export CHROME_PATH=/usr/bin/google-chrome             # default; set it for Chromium or other paths

pytest studio/e2e -s -rs
```

Requirements: Python extras `dev` and `studio`, Node >= 22.13, pnpm (pinned in `studio/package.json`),
Chrome or Chromium. If a prerequisite is missing the tests **skip with the reason** (`-rs`). If
`nitid/studio/static/index.html` is missing, `scripts/build_studio.py` builds it first. No network
is needed beyond installing dependencies: the models are random-weight networks built at run time
(compact decoder, 2 classes, 10 queries, anchored at 640 px; `studio/e2e/make_fixtures.py`) and no
weight, image or ONNX file is committed. Timing on 2 CPUs: UI flows about 25 s, security about 12 s,
parity about 55 s (the ONNX exports take about 10 s each).

CI: the optional job `studio-e2e` in `.github/workflows/ci.yml` runs the same command. It is
`continue-on-error: true` and nothing depends on it, so it cannot block a merge. `[NV]` it has not run
on GitHub Actions yet (only locally); make it required once it has been stable for a while.

## What is covered

How it works: pytest starts `python -m nitid.studio port=0 models=<dir> --no-browser` (the launcher's own
mechanism: free port, random per-start token, `Open: http://127.0.0.1:PORT/#t=TOKEN` read from stdout), hands
the URL to a Node driver and asserts on the JSON it writes. After the tests the process gets Ctrl+C and
must exit with status 0.

### User flows (`ui.mjs`)

- Open the tokenised URL: local mode detected from `/health`, the token fragment removed from the address
  bar, the token kept per tab only (`sessionStorage`).
- The model picker lists the local `*.pth` files (`tiny-det`, `tiny-seg`, marked "[local file]") next to the
  registry models marked "not downloaded"; the downloaded detect model is preselected.
- Upload an image, run in server mode (torch / cpu), see boxes on the canvas and the table. What the UI shows
  is compared with an **independent** run submitted straight to the API: row count at 25 % and 50 %, and the
  first row's class, confidence and rounded box.
- The confidence slider and the empty state; a second run with the segment model; the Mask column.
- The queue: three runs submitted at once are `running`, `pending`, `pending`, finish in submission order,
  and a UI run submitted behind them shows "Queued on the server..." and then the results.
- A reload on `/run/local` keeps working (token mirrored in `sessionStorage`).
- Error state with a wrong token and with no token: `/health` is public so local mode is still detected;
  the panel shows "Could not talk to the local server." with the unauthorized hint and "Try again".
- No console errors, page errors or failed requests during the main flow.

### Security (`security.mjs`, real Chrome)

- **Host guard / DNS rebinding**: Chrome is started with `--host-resolver-rules="MAP evil.test 127.0.0.1"`;
  `http://evil.test:PORT/` and its `/api/v1/health` answer 403 `forbidden_host`; `localhost:PORT` is allowed.
- **Origin guard**: a hostile page on another origin submits a form (`POST /api/v1/runs`, top-level
  navigation, `Origin: http://evil.test:EPORT`): the response is 403 `forbidden_origin`, checked before the
  token. The same page cannot read `/health` (no CORS headers), cannot embed an API response as an image
  (CORP), cannot read the response of a `no-cors` POST, and a cross-origin POST with a **valid token and
  upload** is stopped at the preflight: no run is created.
- **Token required**: navigations to `/models`, `/devices`, `/runs`, `/onnx/...` and an unknown API path
  return 401 `unauthorized` with `WWW-Authenticate: Bearer`; so do same-origin fetches with no, wrong or
  non-Bearer credentials. Only `/health` is public. The token never appears in a request URL; no `Referer`
  is sent.
- **Headers**: every response of the page, its assets, the API and the 401/403 rejections carries
  `nosniff`, COOP `same-origin`, COEP `require-corp`, CORP `same-origin`, `no-referrer` and the CSP.
- **CSP** inside the real UI page: an inline `<script>`, `eval`, `new Function`, `setTimeout(string)`, an inline
  event handler, a remote `<script>`, `fetch`, image and `<iframe>` and a `data:` worker are all blocked, and
  no request reached the hostile server. `blob:` workers and `WebAssembly.compile` still work (ONNX Runtime
  Web needs them). The page is `crossOriginIsolated` and `SharedArrayBuffer` exists.

### Parity, server vs browser (`parity.mjs`, `test_studio_parity.py`)

The harness (`harness/`, built with Vite using the product's ORT-web setup) is served by the real app, so
it runs under the exact CSP/COOP/COEP headers. In Chrome it downloads the ONNX from the real
`GET /api/v1/onnx/{model}?task=...`, verifies the ETag, and runs the engine (worker, wasm from `/ort/`,
2 threads) on the same image the server processes with `POST /runs` (PyTorch, CPU, `imgsz=640`). Both
tasks, at `conf` 0.5, 0.1 and 0.9.

Measured on the tiny models (nitid 0.9.6, torch 2.13.0, onnxruntime-web 1.31.1, Chrome 153), worst value over
all cases:

| Quantity | Measured | Tolerance in the test | Why this tolerance |
| --- | --- | --- | --- |
| Detections above `conf` | equal (10 / 20 / 0) | equal | Same strict `>` rule; the scores are far from the thresholds. |
| Classes and names | equal | equal | |
| Box coordinates | 0.0017 px | 0.02 px | The server rounds to 0.01 px (up to 0.005 px of difference by itself); the rest is float noise. |
| Box IoU | 0.99997 | >= 0.999 | |
| Score | 5.0e-5 | 1e-4 | The server rounds to 1e-4 (up to 5e-5 by itself). |
| Mask IoU (segment) | 1.0 (30 masks) | >= 0.99 | The server only serves an outline; it is filled with `cv2.fillPoly` and compared with the engine's mask. A boundary pixel flipping at the 0.5 threshold would show here. |

The detections are matched per class by best IoU, not by row index, because the untrained model gives
all queries of a class the same score.

### Browser observations under the Studio headers (real Chrome 153, headless, Linux)

- `crossOriginIsolated` is **true** and `SharedArrayBuffer` is available on the harness page, so the
  engine picks **2 threads** (`numThreads: 2` on a 2-core machine) and blob workers start. COOP
  `same-origin` + COEP `require-corp` + same-origin assets work together.
- The CSP `script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:` is enough for ORT-web: wasm
  compiles, the worker and its thread workers run, no violation is reported.
- Real header (`nitid/studio/security.py`) also has `style-src 'self' 'unsafe-inline'`,
  `img-src 'self' data: blob:` and `connect-src 'self'`. The target CSP written in
  `docs/studio-development.md` does not list them; an inline `style` attribute works (observed).
- **WebGPU**: plain headless Chrome exposes `navigator.gpu` but `requestAdapter()` returns `null`, so the
  engine runs on wasm. With `--enable-unsafe-webgpu` (also with `--use-webgpu-adapter=swiftshader`) an
  adapter is returned (software SwiftShader). Run by hand with those flags and the WebGPU runtime files:
  see "Product findings" 2; the detect model finished in 143 s (first run, software rendering) and the
  segment model had not finished after about 6 minutes more, so it is **not** part of the suite. No real GPU was
  available, so real WebGPU speed and precision are `[NV]`.

## Product findings

1. **Favicon 404 (fixed, one line).** The UI shipped no favicon, so Chrome's `/favicon.ico` request got the
   SPA fallback's 404 and a console error appeared on every page load. `studio/index.html` now has
   `<link rel="icon" href="data:,">`; verified in Chrome (error gone) and covered by the "no console errors"
   checks.
2. **WebGPU can never start with onnxruntime-web 1.31.1 (not fixed).** `scripts/copy-ort.mjs` (and
   `docs/studio-engine.md`) copy `ort-wasm-simd-threaded.jsep.{mjs,wasm}`, but the `onnxruntime-web/webgpu`
   build of 1.31.1 requests `ort-wasm-simd-threaded.asyncify.mjs`. Reproduction: build the harness with
   `ORT_COPY_WEBGPU=1`, start Chrome with `--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader`, run the
   engine with `backend: "webgpu"`: `info.backend` is `wasm` and `fallbackReason` says
   `Failed to fetch dynamically imported module: /ort/ort-wasm-simd-threaded.asyncify.mjs` (404). Copying the
   `asyncify.{mjs,wasm}` files (28.7 MB) instead makes `info.backend` be `webgpu`. The fix is the file list in
   `copy-ort.mjs` plus the docs; not done here because the choice between `asyncify` and `jspi` is a design
   decision.
3. **The worker cannot download the ONNX from the authenticated endpoint (not fixed).** `init({ modelUrl })`
   makes the worker `fetch` the URL without an `Authorization` header (`engine/model-cache.ts`), but
   `/api/v1/onnx/*` requires the token: `EngineError: Could not download the model: HTTP 401` (recorded by
   `parity.mjs` as `workerDownloadWithoutToken`). The harness works around it the way the browser workbench
   will have to: the page fetches with the token and passes `modelBytes`, which gives up the Cache API
   (`modelCached: false`, the file is downloaded on every start). A fix needs a header or token option in
   the worker protocol, or a cache keyed by the ETag filled by the page.
4. **Studio can be framed by any site (not fixed).** The CSP has no `frame-ancestors` and there is no
   `X-Frame-Options`: a page on another origin can embed the UI in an `<iframe>` (observed, status 200).
   Whether a framed copy could reach the token was not checked `[NV]` (it is not in the URL and
   `sessionStorage` is per origin); the clear risk is clickjacking. Adding `frame-ancestors 'none'` would close it.
5. **Stale copy (fixed)**: the UI used to say "Mask outlines are not provided by the server API yet" after a
   segment run, although `/runs` returns `segments`. The note is removed and the outlines are drawn; the UI
   test data key `staleMaskNote` should now be `false`.
6. **No `imgsz` control**: the UI sends no `imgsz`, so the server uses 640. A checkpoint trained at another
   size (the unit-test models are 64 px) must be run through the API with `imgsz`. Not a bug for the
   official 640 px models; noted because it is why the e2e models are anchored at 640.

## Known gaps

- **Browser mode has no UI yet** (`/run/browser` is a placeholder), so the browser half of the parity
  test drives the engine through a harness page instead of the product UI. When the browser workbench
  lands, add a UI scenario for it and keep the parity test as the numerical check.
- The detect fixture is degenerate: all queries of the tiny detect model give the same box
  (a 160x120 px box clipped by the image), so the detect box comparison is weaker than the segment one,
  whose boxes vary. A trained model would be a better oracle; none can be downloaded offline.
- Real pretrained weights, the download flow (`POST /models/{id}/download`) and OpenVINO are not exercised
  (no network, no Intel runtime in the test environment).
- Real WebGPU hardware, and Firefox/Safari, are not covered; only headless Chrome on Linux.
- Mobile layouts and accessibility audits are not part of this suite.
- Exports (PNG/CSV/ZIP/PDF) are covered by the unit tests, not here.
- The suite has not run in GitHub Actions yet `[NV]`.
