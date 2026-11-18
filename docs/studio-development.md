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
    pages/              Home, About, RunPlaceholder
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
detecting states, the full App with mocked `fetch`, and the static policy checks
(no `localStorage`, no external URLs, no inline scripts).

Run a single file: `pnpm exec vitest run src/token.test.ts`.
