# Nitid Studio: frontend development guide

This guide covers the `studio/` folder: the single-page web app (Vite + React +
TypeScript) that becomes the Nitid Studio UI. It is written for people who
change the frontend. Facts that were not verified are marked `[NV]`.

## What the app does (in one paragraph)

Nitid Studio has two ways of running a model: **Local**, where the Python server
on `127.0.0.1` does the work with PyTorch or OpenVINO, and **Browser**, where
ONNX Runtime Web runs the model in a Web Worker so the image and the computation
never leave the page. This scaffold contains only the shell: start page with the
two mode cards, an About page, the API client, backend detection and token
handling. The workbench itself (and ONNX Runtime Web) arrive in later work
packages; `/run/local` and `/run/browser` currently show a placeholder.

## Requirements

- Node.js >= 22.13 (developed with 22.23).
- pnpm, at the exact version in the `packageManager` field of
  `studio/package.json` (currently 10.28.0). With Corepack:
  `corepack enable` and pnpm picks the right version by itself.
- Only the standard npm registry is used. `pnpm-lock.yaml` is committed and
  `pnpm install --frozen-lockfile` must succeed (CI should use that flag).

## Commands

Run everything from `studio/`.

| Command | What it does |
| --- | --- |
| `pnpm install --frozen-lockfile` | Install exactly what the lockfile says. |
| `pnpm run dev` | Vite dev server on `http://localhost:5173`; `/api` is proxied to `http://127.0.0.1:8000`. |
| `pnpm run build` | Production build. Output dir: `$STUDIO_OUT_DIR`, default `../nitid/studio/static` (relative to `studio/`). |
| `pnpm run build:hosted` | Build in `hosted` mode (no backend probing). Output: `dist-hosted/` (or `$STUDIO_OUT_DIR`). |
| `pnpm run lint` | ESLint (typescript-eslint, react-hooks). |
| `pnpm run typecheck` | `tsc --noEmit`. |
| `pnpm run test` | Vitest, run once. |
| `pnpm run check` | lint, typecheck, test and build, in that order. |
| `pnpm run check:size` | Fails if a tracked file under `studio/` is a model/binary type or larger than 1 MB (see "Size policy"). |

`STUDIO_OUT_DIR` is emptied before each build (Vite `emptyOutDir`). Point it at a
dedicated folder, never at a directory that holds other files.

## Folder layout

```
studio/
  index.html            entry document (no inline scripts, lang="en", <title>)
  vite.config.ts        build modes, dev proxy, Vitest config
  eslint.config.js      flat ESLint config
  tsconfig.json         strict TypeScript
  package.json          scripts, pinned packageManager
  pnpm-lock.yaml        committed lockfile
  scripts/check-size.mjs   size / file-type guard
  src/
    main.tsx            entry: consumes the token fragment, mounts <App/>
    App.tsx             backend detection on start-up, router, layout
    backend-context.tsx React context carrying the detection result
    token.ts            #t=... extraction and in-memory token
    use-page-title.ts   per-route document.title
    env.d.ts            declares the build-time __STUDIO_MODE__
    styles.css          plain CSS, light/dark, AA contrast
    api/
      client.ts         same-origin fetch client: bearer token, timeout, typed errors
      backend.ts        detectBackend(), parseHealth()
      types.ts          hand-written API types (to be generated, see below)
      local.ts          typed wrappers for /devices, /models, /jobs, /runs
    pages/              Home, About, RunLocal (/run/local), RunPlaceholder
    local/              Local panel: components, hooks and pure logic (see "Local panel")
    test/setup.ts       Testing Library + jest-dom setup
    **/*.test.ts(x)     tests live next to the code they cover
  api-contract/         NOT part of this scaffold: owned by another work package
```

## How it fits with the backend

During development you run two processes:

1. The Python server: `nitid studio` `[NV: the command and its flags are
   implemented in another work package; the default port 8000 used by the
   proxy is an assumption]`.
2. The frontend: `pnpm run dev`. Vite serves the UI on port 5173 and forwards
   every `/api/*` request to `http://127.0.0.1:8000`, so the browser sees a
   single origin and no CORS configuration is needed.

In production the Python package serves the built files itself, from
`nitid/studio/static/` (git-ignored, produced by `pnpm run build`).

### Backend detection (local vs hosted)

The same frontend ships in two deployments:

- **Local**: served by the Python server. On start-up the SPA calls
  `GET /api/v1/health` (the probe is aborted after 2 s).
  If the answer is HTTP 200 with a JSON body
  `{"status":"ok","nitid_version":"...","api_version":"..."}`, the mode is
  `local` and both cards are enabled.
- **Hosted**: a static site with no backend. Any other outcome (404, an HTML
  fallback page, invalid JSON, wrong shape, network error, timeout) means
  `hosted`: only the Browser card is enabled and nothing is logged to the
  console.

`vite build --mode hosted` (`pnpm run build:hosted`) goes one step further:
`__STUDIO_MODE__` is `"hosted"` and `detectBackend()` returns immediately
without any request, so a static host never sees `/api/v1/health` probes. The
default build uses `"auto"`. (The bundle still contains the probe code; it is
simply never reached. Tree-shaking it away was not attempted.)

If the server reports an `api_version` different from `EXPECTED_API_VERSION` in
`src/api/types.ts`, the mode stays `local` and the home page shows a warning.
The expected value `"1"` is a placeholder `[NV]` until the API contract exists.

The health probe is sent **without** the bearer token on purpose: it is only a
liveness check.

### Token

The server opens the browser at `127.0.0.1:PORT/#t=<token>`. `initToken()`
(called in `main.tsx` before the router mounts) reads the fragment, stores the
token in memory and in `sessionStorage` (so a reload of the same tab keeps
working), and removes the fragment with `history.replaceState`. `localStorage`
is never used; a test fails if application code mentions it. Every API call
sends `Authorization: Bearer <token>` through `src/api/client.ts`.

Reversible choice: persisting to `sessionStorage` is one constant and three small
functions in `token.ts`; delete the mirroring for memory-only tokens (a reload
would then require opening the URL with the fragment again).

### API types

`src/api/types.ts` is hand-written and minimal. When `studio/api-contract/openapi.json`
exists it should be replaced by generated types; the comment at the top of the
file says so. Do not add generator dependencies before that contract lands.

## Local panel (`/run/local`, issue #43)

The workbench for the **local** deployment. `App.tsx` lazy-loads `pages/RunLocal.tsx`
(separate ~24 kB chunk), which shows `LocalPanel` only when `detectBackend()` said
`local`; otherwise it explains how to start `nitid studio`.

### Component map

| File (`src/local/`) | Role |
| --- | --- |
| `LocalPanel.tsx` | Page-level container. Loads data, owns selections, runs. Props: `api?` (tests), `ExportMenu?` (slot, below). |
| `ModelPicker.tsx` | `GET /models` as one `<select>` grouped by task (items are `(id, task)` pairs). Download button, progress bar, failure message. |
| `DevicePicker.tsx` + `devices.ts` | Backend (`torch`/`openvino`) and device from `GET /devices`. Devices the machine lacks are listed disabled ("not detected"); only reported ones are enabled. |
| `ImageDropzone.tsx` + `files.ts` | Drag and drop plus a real file input (keyboard, screen reader); several images; pre-flight checks (image type, 25 MB, duplicates). |
| `ResultsView.tsx` | Canvas, confidence slider (native range input), class checkboxes, labels/masks toggles, image selector, detection table, per-stage timing. |
| `DetectionCanvas.tsx` + `draw.ts` + `geometry.ts` | Canvas drawing; geometry is pure and unit-tested; `draw.ts` is tested with a recording fake context. |
| `filter.ts` | Confidence and class filtering, class summary. |
| `polling.ts` | `pollUntil`/`pollRun`/`pollJob` (backoff 400 ms to 2 s, 10 min timeout, abort, 3 tolerated network blips) and `runReducer`, the idle/uploading/processing/done/failed state machine. |
| `use-local-data.ts`, `use-downloads.ts`, `use-run.ts` | Hooks that connect the above to `src/api/local.ts`. |
| `errors.ts` + `ErrorNotice.tsx` | Contract error envelope (`{"error":{"code","message"}}`) to message plus a hint per code. Rendered with `role="alert"`. |
| `results-model.ts` | UI-facing results types (below). |

### Extension points for the unified UI (issue #46)

- **Results model**: `RunResults` / `ResultItem` / `ResultDetection` in `results-model.ts`
  (camelCase, independent of the API). `fromRun(run, wallClockMs)` builds it from
  `GET /runs/{id}`; the browser engine can build the same shape (`source: "browser"`) and reuse
  `ResultsView`, whose only inputs are `results` and `loadImage(item)`.
- **Exports**: `LocalPanel` and `ResultsView` accept `ExportMenu?: ComponentType<ExportMenuProps>`
  and render it in the controls column. It receives `{ results, filter, selectedItemId }`, where
  `filter` is the user's current threshold and hidden classes, so exports can honour "what I see".
  Nothing is exported by this panel; `RunLocal` also takes the prop so a parent can wire it.

### Behaviour worth knowing

- The run is sent with `conf: 0.1` (`RUN_CONFIDENCE_FLOOR`) and the slider starts at 25 %, so
  raising or lowering the threshold above 10 % is instant and client-side. Detections the server
  dropped below 10 % can never be shown. The `iou`, `imgsz` and `classes` params are not exposed yet.
- Original images are fetched with the bearer token as a `Blob` and decoded with
  `createImageBitmap`; no `blob:` URL is created, so the CSP in "Security and CSP" is unaffected.
- `POST /runs` sends `params` as a plain string form field (not a file part).
- Only the pure `done` payload is shown: there is no cancel endpoint, so "Stop waiting" only stops
  polling; the server may finish the run anyway. Old runs (`GET /runs`, `DELETE /runs/{id}`) are
  not used yet.
- **Masks**: contract v1 only reports `has_mask`; no endpoint serves mask pixels or polygons, so
  segmentation results show boxes with a tinted fill and a note. `ResultDetection.mask` (polygons)
  and the drawing code exist, ready for when the API provides them. [NV: contract gap]

### Running against a real `nitid studio`

```bash
# terminal 1: backend (prints the URL with #t=<token>)
nitid studio                      # [NV] default port 8000; the Vite proxy assumes it
# terminal 2: frontend with hot reload
cd studio && pnpm run dev         # http://localhost:5173, /api proxied to 127.0.0.1:8000
```

Open `http://localhost:5173/#t=<token>` using the token from terminal 1 (the fragment is consumed
and removed). Then Home -> "Use local mode". Without the token every call answers 401 and the
panel shows the hint to open the link printed by `nitid studio`. To test the built UI instead, build
(`pnpm run build`, output in `nitid/studio/static/`) and let `nitid studio` serve it.

### Browser smoke test (no backend)

`pnpm run build && pnpm run smoke:local` starts a throw-away Node server that serves the built UI
plus a mocked `/api/v1`, drives Chrome through select, upload, run, polling, overlay pixels, slider
and class filter, and exits non-zero on failure. It uses `playwright-core` found in
`PLAYWRIGHT_CORE_PATH` or the usual global locations and Chrome at `CHROME_PATH`
(default `/usr/bin/google-chrome`). It is **not** part of `pnpm run check` (no new dependency).

## Security and CSP

The app must work under this Content-Security-Policy:

```
default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:
```

Rules that follow from it (enforced where possible by `src/policy.test.ts`):

- No inline `<script>`, `<style>` or `on*=` attributes; no `eval`.
- No CDN, web fonts or any `http(s)://` URL in code or CSS: everything is
  same-origin. `assetsInlineLimit` is 0, so no `data:` assets are produced.
- Source maps are off in production builds.
- The token never goes to `localStorage` and is never put in a URL after start-up.

Not covered by tests: that the built output actually runs under that CSP in a
browser `[NV]` (needs a manual or end-to-end check once a server sets the header).

## Size policy

The repository must stay small and reviewable:

- Do not commit binaries, `.wasm`, `.onnx` or other model files, archives, or any
  file larger than 1 MB. `pnpm run check:size` enforces this for `studio/`.
- `node_modules/`, `dist/`, `dist-hosted/`, `.vite/`, `coverage/` and
  `nitid/studio/static/` are git-ignored (root `.gitignore`, block "Nitid Studio",
  plus `studio/.gitignore`).
- The one file allowed to exceed 1 MB is `pnpm-lock.yaml`. Today it is about
  80 KB, so the exception is not in use; it is documented because the lockfile
  grows with every dependency. Large runtime assets (ONNX Runtime Web `.wasm`
  files, models) must be fetched or generated at build/runtime, not committed.

Measured on this scaffold (Node 22.23, Vite 8.3.8):

| Build | Output size |
| --- | --- |
| `pnpm run build` (to `nitid/studio/static`) | 266 127 bytes total (`index.html` 0.54 kB, CSS 2.31 kB, JS 263.26 kB, 83.80 kB gzip) |
| `pnpm run build:hosted` (to `dist-hosted`) | 266 129 bytes total (same content) |

## Packaging

`pnpm run build` writes the static site into `nitid/studio/static/`. The Python
packaging work (separate work package) is responsible for including that
directory in the wheel and for CI building the frontend before `uv build`. This
scaffold does not touch `pyproject.toml`, CI or Dependabot; adding an `npm`
ecosystem for `/studio` to `.github/dependabot.yml` is a suggested follow-up.

## Dependencies and licences

Runtime (`pnpm licenses list --prod`, 9 packages, all MIT): react, react-dom,
react-router, scheduler, cookie-es, @remix-run/route-pattern, csstype and the
type packages `@types/react`, `@types/react-dom`.

All installed packages including development tooling (`pnpm licenses list`,
Nov 2026): MIT 178, Apache-2.0 17, ISC 10, BSD-2-Clause 8, BSD-3-Clause 3,
BlueOak-1.0.0 3, MPL-2.0 3, CC-BY-4.0 1.

Non-MIT notes (development-time only, none is bundled into the shipped JS):

- MPL-2.0: `lightningcss` (Vite CSS tooling). File-level copyleft, used unmodified
  as a build tool.
- CC-BY-4.0: `caniuse-lite` (browser data used by build tooling; attribution only).
- Apache-2.0 includes `typescript` and the ESLint core packages.

Whether the shipped bundle needs entries in `THIRD_PARTY_NOTICES.md` (React and
React Router are MIT and require the copyright notice to accompany copies) is an
open question for the maintainers `[NV]`.

TypeScript is pinned to 6.x (`^6`) because `typescript-eslint` 8.73 refuses to run
with TypeScript 7 (it prints "typescript-eslint does not support TS 7.0"). Move to
7.x once typescript-eslint supports it.

## Testing

`pnpm run test` runs Vitest with jsdom. Covered: backend detection (valid 200,
200 with invalid JSON or wrong shape, 404, network failure, timeout, API version
mismatch, hosted build skipping the probe), token extraction and cleanup, the
API client (bearer header, error mapping), rendering of Home in local, hosted and
detecting states, the full App with mocked `fetch`, the Local panel (pure logic: filtering,
drawing geometry, polling state machine, device options; API client against mocked `fetch`;
component tests of the whole panel with a mocked `LocalApi`), and the static policy checks
(no `localStorage`, no external URLs, no inline scripts).

Run a single file: `pnpm exec vitest run src/token.test.ts`.
