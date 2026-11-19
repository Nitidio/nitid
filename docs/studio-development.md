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
    exports/            result exports: PNG, JSON, CSV, YOLO TXT, crops, ZIP, PDF, <ExportMenu/> (see "Exports")
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

Runtime (`pnpm licenses list --prod`, all MIT): react, react-dom,
react-router, scheduler, cookie-es, @remix-run/route-pattern, csstype, the
type packages `@types/react`, `@types/react-dom`, and `fflate` (ZIP, see
"Exports"). The list above predates later packages such as `onnxruntime-web`;
run `pnpm licenses list --prod` for the current set.

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

## Exports (`src/exports/`)

Issue #45. A framework-light module that turns one inference result into files.
The unified UI (issue #46) mounts `<ExportMenu/>`; everything else is plain
functions. Facts verified by running the tests and tools named below are
unmarked; open points are marked `[NV]`.

### One input type: `ExportResults`

Both producers map into the same type (`src/exports/model.ts`), so no exporter
knows where the numbers came from.

| Producer | Mapper | Notes |
| --- | --- | --- |
| Browser engine (`RunResult` of `src/engine`) | `fromEngineResult(result, { imageName, model, task, backend })` | Masks are present (`DetectionMask`, cropped to the box). `device` is `null`. |
| REST server (`GET /runs/{id}`, one `RunItem`) | `fromServerRun(run, itemId)` | The contract reports `has_mask` but **no mask pixels** (no mask endpoint in `api-contract/openapi.json`), so server results carry `hasMask: true` and no `mask`: PNG draws boxes only and YOLO TXT falls back to box lines. `device` is the server's device name. |

```ts
interface ExportResults {
  source: "browser" | "server";
  image: { name: string; width: number; height: number };   // original image
  model: string;  task: "detect" | "segment";
  backend: string;            // browser: wasm | webgpu; server: torch | openvino
  device: string | null;
  timings: { preprocessMs; inferenceMs; postprocessMs } | null;
  createdAt: string;          // ISO 8601, injected so outputs are reproducible
  detections: Array<{
    box: { x1; y1; x2; y2 };  // xyxy, pixels of the ORIGINAL image
    confidence: number;       // 0..1
    classId: number;  name: string;   // any Unicode
    hasMask: boolean;
    mask?: { x; y; width; height; data: Uint8Array };  // 0/1, row-major, top-left at (x, y)
  }>;
}
```

The server image comes from `GET /runs/{id}/items/{n}/image`; the caller turns
it into an `ImageBitmap` and passes it as `image`.

### Formats

| Format | File name | Function | Needs image |
| --- | --- | --- | --- |
| Annotated image | `<base>_annotated.png` | `renderAnnotated`, `annotatedBlob` | yes |
| JSON | `<base>.json` | `formatJson`, `toJsonDocument` | no |
| CSV | `<base>.csv` | `formatCsv` | no |
| YOLO TXT | `<base>.txt` + `classes.txt` | `formatYoloTxt`, `formatClassesTxt`, `parseYoloTxt` | no |
| Crops | `crops/<base>_<NNNN>_<class>.png` | `renderCrops` | yes |
| PDF report | `<base>_report.pdf` | `buildPdfReport` (`pdf.ts`) | yes |
| ZIP bundle | `<base>_export.zip` | `buildZip` (`bundle.ts`) | if a member needs it |

`<base>` is the image file name without extension, with every run of characters
that are not Unicode letters, digits, `_` or `-` replaced by `_`
(`street view.JPG` becomes `street_view`; empty becomes `image`).
`buildFiles(format, results, ctx)` and `buildZip(formats, results, ctx)` in
`bundle.ts` are the entry points when you do not use the component. Multi-file
results (YOLO TXT + `classes.txt`, crops) are delivered by `ExportMenu` as a ZIP.

### JSON schema (`nitid-studio-results`, version 1)

```json
{
  "schema": "nitid-studio-results",
  "version": 1,
  "created_at": "2026-11-19T10:00:00.000Z",
  "source": "browser",
  "image": { "name": "street.jpg", "width": 1280, "height": 720 },
  "model": { "id": "nitid1s", "task": "segment" },
  "runtime": { "backend": "wasm", "device": null },
  "timings_ms": { "preprocess": 3.1, "inference": 28.4, "postprocess": 1.2 },
  "detection_count": 1,
  "detections": [
    {
      "index": 0, "class_id": 0, "name": "person", "confidence": 0.91,
      "box": { "x1": 12.5, "y1": 40, "x2": 210, "y2": 388.25 },
      "has_mask": true,
      "mask": { "polygon": [[12.5, 40], [210, 40]], "pixels": 5123 }
    }
  ]
}
```

- Keys are snake_case like the REST API. `timings_ms` is `null` when unknown.
- Pixels are rounded to 2 decimals, confidence to 6, timings to 3.
- `mask` exists only when mask pixels are available. `polygon` is the contour
  below, in ORIGINAL image pixels (`[]` for an empty mask); `pixels` is the
  number of set mask pixels.
- Adding a key is not a breaking change; removing or renaming one bumps `version`.

### CSV

RFC 4180: header row always present, CRLF line ends, fields containing the
delimiter, `"`, CR or LF are quoted with doubled quotes. Columns, one row per
detection:

`image,index,class_id,class_name,confidence,x1,y1,x2,y2,width,height,has_mask,mask_pixels`

`mask_pixels` is empty when the mask pixels are not available. Options:
`delimiter` (`,` default, `;` or TAB for Excel in European locales) and `bom`
(UTF-8 byte-order mark so Excel detects Unicode; off by default). Text cells that
start with `=`, `+`, `-`, `@`, TAB or CR get a leading `'` so spreadsheets do not
evaluate them (class names come from user-supplied models); numeric cells are
never touched.

### YOLO TXT

One line per detection, class id first, coordinates normalised by the ORIGINAL
image size, clamped to 0..1, six decimals:

- box: `<class> <cx> <cy> <w> <h>`
- polygon (detections with mask pixels): `<class> <x1> <y1> ... <xn> <yn>`,
  `n >= 3` (the Ultralytics segmentation format)

No detections give an empty file, the YOLO convention for a negative sample.
`classes.txt` lists names by class id (ids that never occur are `class_<id>`).
`segmentation: "box"` forces box lines. `parseYoloTxt` reads both line kinds and
throws a `SyntaxError` naming the line; the tests round-trip through it.

### Mask contour extraction (`contour.ts`)

1. Label the 8-connected components of the binary mask and keep the largest. A
   YOLO line holds one polygon, so smaller islands are dropped; holes are
   ignored (outer boundary only).
2. Trace that component's boundary by crack following on pixel corners, with the
   foreground on the right-hand side. At each corner look at the two pixels
   ahead: ahead-left set, turn left; else ahead-right set, go straight; else
   turn right. This joins diagonal neighbours, consistent with step 1. Stop when
   the start corner is reached again with the start direction. The ring encloses
   every pixel of the component, so with tolerance 0 its area equals the pixel
   count (tested on L shapes, discs, strips).
3. Remove collinear vertices, then simplify the closed ring with
   Ramer-Douglas-Peucker. Default tolerance 1 pixel, capped at a quarter of the
   mask's smaller side so thin masks do not collapse. A disc of radius 40 goes
   from 196 vertices (tolerance 0, exact area) to 20 with +1.1 % area (measured; the test asserts under half the vertices and under 3 %).
4. Add the mask origin, divide by image width/height.

Cost is linear in the mask size. Limitation: instances with several separate
parts lose all but the largest part in YOLO TXT and in JSON `mask.polygon`
(the PNG overlay draws every mask pixel).

### Annotated PNG and crops

`renderAnnotated` draws the image and then, on a canvas of the original size:
mask overlays (class colour, 40 % opacity) first, then boxes and labels
(`name NN%`). Colours are deterministic per class id (golden-angle hue). Line
width and font scale with the image. Crops are the box region of the original
image, rounded outwards and clamped (at least 1x1), without overlays.

Canvas access goes through `RasterBackend` (`createCanvas`, `toBlob`,
`makeImageData`). The default uses `OffscreenCanvas` when present, otherwise a
detached `<canvas>`. jsdom has no canvas, so tests use a recording fake
(`fakeRaster()` in `test-fixtures.ts`): drawing order and arguments are tested,
**pixel output in a real browser is not** `[NV]`. Check it by hand once the UI
mounts the menu (a tainted canvas from a cross-origin image would make
`toBlob` fail; the menu shows the error).

### PDF report (`pdf.ts`)

A4, 40 pt margins. Page 1: title, metadata (image, model and task, runtime and
device, timings, creation time, detection count), the annotated image (JPEG,
scaled to fit, at most 300 pt tall), then the detections table: `# | Class |
Conf. | x1 | y1 | x2 | y2 | Mask`. The table breaks across pages with the
header repeated; every page has a `Page n of N` footer. Metadata goes into the
Info dictionary (title, creation date).

Verified on a 1000-detection report (22 pages) with `qpdf --check` (no errors),
`pypdf` (22 pages, text and one image found) and `pdftoppm` (rendered and
inspected). The test suite asserts the header, `%%EOF`, xref offsets, page
count (`/Count` equals the number of `/Type /Page` objects), the text strings, the
image object, and that the file body is ASCII.

Limits: only Helvetica with WinAnsi encoding, so Latin-1 and a few cp1252
marks (euro, curly quotes, dashes, ellipsis, bullet) are shown; any other
character (CJK, Cyrillic, Arabic) prints as `?`. JSON, CSV and YOLO TXT keep
exact Unicode. Embedding a Unicode font would add hundreds of kB; revisit if
needed. Content streams are uncompressed; the image is the annotated picture as
JPEG (quality 0.9) at original size, so the report of a 20-megapixel image is
large `[NV: not measured]`.

### `ExportMenu`

```tsx
<ExportMenu
  results={exportResults}     // ExportResults
  image={imageBitmap}         // optional; without it png/crops/pdf/zip are disabled
  formats={["png", "json", "csv", "yolo", "crops", "pdf", "zip"]}   // default: all, in this order
  context={{ csv: { delimiter: ";", bom: true }, yolo: { tolerance: 1 } }}  // optional
  onDownload={(file) => ...}  // optional; default saves via an <a download> object URL
  onError={(error) => ...}
/>
```

An "Export" disclosure button (`aria-expanded`) opens a group with one
"Download" button per format and a checkbox per bundle member; "Download ZIP of
checked formats" builds the bundle. Crops are disabled without detections;
image formats are disabled without `image`; only one export runs at a time; the
`role="status"` line reports progress and errors. Styles are the `.export-*`
rules in `styles.css`. The component does not know about routes or API calls.

### Dependencies

| Package | Why | Cost |
| --- | --- | --- |
| `fflate` 0.8.3 (MIT, no dependencies) | ZIP. `zipSync` with per-file level (stored for PNG/PDF, deflate for text) and fixed timestamps so the same input gives the same bytes. Chosen over JSZip (larger, with Promise/stream layers we do not use `[NV: size not measured]`) and over writing a ZIP writer by hand (CRC32, deflate, ZIP64 edge cases). | 8.8 kB minified (4.4 kB gzip) lazy chunk |
| none for PDF | A hand-written PDF 1.4 writer (about 250 lines) covers one JPEG, Helvetica text and a table. pdf-lib and jsPDF would add several hundred kB to the browser for the same output. The cost is the limits above (Latin-1 text) and owning the code; the xref/validity risk is covered by tests plus the `qpdf`/`pypdf` check. | 5.7 kB minified (2.7 kB gzip) lazy chunk |

Both load by dynamic `import()` the first time a ZIP or PDF is built
(`bundle.ts`), so the main bundle does not carry them. Measured with
`vite build` (Vite 8.3.8): the baseline main chunk is 263.26 kB (83.80 kB gzip).
Forcing the whole `src/exports` index into the entry (worst case: the menu,
formatters, contour, raster and mapper code all in main) gives 278.10 kB
(89.62 kB gzip), i.e. +14.8 kB (+5.8 kB gzip), plus lazy chunks `browser`
(fflate) 8.76 kB, `pdf` 5.72 kB and `bundle` 1.81 kB. Until the UI imports the
menu, the shipped bundle is unchanged (tree-shaken).

### Tests

`src/exports/*.test.ts(x)`: golden outputs for JSON, CSV and YOLO TXT; YOLO
round trip (box and polygon); CSV escaping, formula neutralisation, Unicode
names, BOM and delimiter; empty results everywhere; contour cases (rectangle, L
shape, hole, islands, diagonal pixels, disc, thin strips); mappers; raster
drawing order with the fake backend; PDF structure (header, xref, page count,
text, multi-page with repeated header, Latin-1 escaping); ZIP round trip and
determinism; a 1000-detection sanity test (all text formats plus ZIP, bound 5 s,
about 0.4 s measured in jsdom with 1000 masks of 40x30 px); `ExportMenu` component tests (expand, download, ZIP of
checked formats, disabled states, error status).

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
