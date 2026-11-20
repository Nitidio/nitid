# Nitid Studio: the hosted site

The hosted site is Nitid Studio **without a server**: a static website where the model runs inside the
visitor's browser. Nothing is uploaded; there is no backend to attack, scale or pay for. This page
explains how it is put together, how to build and deploy it, and what the owner has to do by hand.
Facts that were not verified are marked `[NV]`; verified ones say where.

## The idea in one picture: one counter, two shops

Think of a market stall with **one counter** (one web address, one origin) and **two shops** behind it:

| Shop | What it sells | Where it comes from |
| --- | --- | --- |
| The app | `index.html`, `assets/`, the ONNX Runtime engine in `ort/` | `pnpm run build:hosted` (`studio/dist-hosted`) |
| The models | `models/models.json`, `*.onnx`, `SHA256SUMS`, licence files | the model bundle (`tools/publish_models.py`) |

`scripts/assemble_hosted.py` puts both shops in one folder, which Cloudflare serves as static files:

```text
studio/dist-hosted-site/
  index.html  assets/  ort/        <- the app
  models/  models.json  *.onnx     <- the models (same origin, so STUDIO_MODELS_BASE_URL=/models/)
  _headers                         <- the response headers (see below)
```

Why the same counter? Because the browser only allows the powerful, multi-threaded engine on pages that
are **cross-origin isolated** (next section), and an isolated page may only load things from other origins
if those origins opt in. Serving the models from the same origin avoids asking anybody for anything.

The local app (`nitid studio`) and the hosted site are the **same UI**. The hosted build
(`vite build --mode hosted`) just never looks for a backend: no `/api/v1/health` probe, no "Local" or
"Compare" entry, and the "Browser" page reads `models/models.json`. With no catalogue it shows
"No models are configured for this site".

## Build it

```bash
cd studio && pnpm install --frozen-lockfile
STUDIO_MODELS_BASE_URL=/models/ pnpm run build:hosted      # -> studio/dist-hosted (ORT wasm included)

# a bundle folder from tools/publish_models.py (models.json, *.onnx, SHA256SUMS, licences)
python scripts/assemble_hosted.py --bundle path/to/bundle --clean   # -> studio/dist-hosted-site
```

The assemble script is the last gate before a deploy. It exits with status 1, and writes nothing, when:

- `models.json` is missing, is not `{"models": [...]}` or lists no model; an entry has a bad `task`, a
  duplicate id, or a `url` that is absolute, uses `..` or does not exist;
- a `sha256` in `models.json` or a line in `SHA256SUMS` does not match the file;
- the bundle contains a training checkpoint or pickle (`.pth`, `.pt`, `.ckpt`, `.safetensors`, `.pkl`,
  `.pickle`) or a TensorRT/TorchScript file, or any symlink;
- a file is larger than **25 MiB** or there are more than **20,000** files (Cloudflare limits, below);
- the UI is not built (no `index.html`, no `ort/*.wasm`).

A missing `sha256` or `SHA256SUMS` is only a warning (the browser then cannot verify or cache by hash).

## Test it without any backend

```bash
cd studio && pnpm run smoke:hosted            # or: pytest studio/e2e/test_studio_hosted.py
```

It builds the hosted UI, writes a fixture bundle (a tiny ONNX made in the script), runs the real assemble
script, serves the result with the **production `_headers`** and the single-page-app fallback, and drives
real Chrome. It fails on: loss of cross-origin isolation or `SharedArrayBuffer`, any Content-Security-Policy
violation, any console error (the favicon included), any request to `/api/*` or to another origin, a
missing `application/wasm`, no detection from the catalogue model, no export, or a second model download
after a reload. No Python backend runs. `SMOKE_HEADERS=file` tries another headers file: with `'wasm-unsafe-eval'`
removed the run fails (the engine cannot compile its WebAssembly), and without COEP the page is no longer isolated.

## The headers, and why

`studio/hosting/cloudflare/_headers` carries the same security policy as the local app
(`nitid/studio/security.py`). The unit test `tests/unit/studio/test_hosted_headers.py` compares them
directive by directive, so they cannot drift apart silently. Only the server-side guards (Host, Origin,
bearer token) are absent: there is no server.

| Header | Value | Why |
| --- | --- | --- |
| `Cross-Origin-Opener-Policy` | `same-origin` | Part of isolation: no other window can hold a reference to this page. |
| `Cross-Origin-Embedder-Policy` | `require-corp` | Part of isolation: the page may only load resources that agree to be loaded. |
| `Cross-Origin-Resource-Policy` | `same-origin` | Our files agree to be loaded only by our own pages. |
| `Content-Security-Policy` | `default-src 'self'; worker-src 'self' blob:; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'` | Everything from our own origin only; no CDN, no analytics, no third party can run or receive anything. |
| `X-Frame-Options` | `DENY` | Nobody can frame the site (clickjacking). |
| `X-Content-Type-Options` | `nosniff` | The browser trusts the declared type of a file. |
| `Referrer-Policy` | `no-referrer` | No address is leaked when following a link. |
| `Permissions-Policy` | camera, microphone, geolocation, payment, USB, serial, Bluetooth off | The app uses none of them. |
| `Cache-Control` | `no-cache` everywhere; `public, max-age=31536000, immutable` under `/assets/` | See "Caching". |

**COOP and COEP in plain words.** The fast version of the engine runs on several threads that share memory
(`SharedArrayBuffer`). Browsers switched that off for ordinary pages years ago, because a page that shares
memory can be used to read secrets from other sites (the Spectre family of attacks). They switch it back on
only for a page that proves it is **isolated**: it sets COOP (`same-origin`: "I am alone in my window group")
and COEP (`require-corp`: "I load only what agrees to it"). If either header is missing, `crossOriginIsolated`
is false, `SharedArrayBuffer` disappears, and the engine falls back to one thread or fails. That is why these
headers are not optional, and why the smoke test checks `crossOriginIsolated` first.

What the CSP needs, proven by the smoke test:

- `'wasm-unsafe-eval'` is required: without it Chrome refuses to compile the ONNX Runtime WebAssembly
  (the test fails with "violates ... Content Security policy directive").
- `style-src 'unsafe-inline'` is there because the UI sets inline style attributes; it mirrors the local policy.
- `worker-src blob:` was **not** needed by anything the smoke test exercised (the engine worker and the
  ORT files load from `'self'`). It is kept only so that the hosted and local policies stay identical.
  `[NV]` whether other browsers (Firefox, Safari) need it.

**Content type.** `_headers` does not set `Content-Type`. Cloudflare may join a header set twice with a comma
(see below), which for a content type would break the file, so `.wasm` -> `application/wasm` is left to the
file extension. `[NV]` that Cloudflare infers `application/wasm`: check after the first deploy (below).

### Caching

- `/assets/*` are named after their content (`index-AbC123.js`), so they never change under one address:
  one year, `immutable`.
- `index.html`, every client route and `models/models.json` are `no-cache` (the browser asks every time,
  answered with a tiny "not modified"), so a new release is picked up at once.
- `/ort/*` is **not** immutable, although it is large (the wasm is about 14.8 MB). Its file names do not
  contain a version; if `onnxruntime-web` is upgraded, a browser holding a year-old `.wasm` with a new `.mjs`
  would break. `no-cache` still avoids re-downloading: the browser sends the stored ETag and receives "not modified".
  `[NV]` that Cloudflare sends an `ETag` for static assets; make the folder versioned (`/ort/<version>/`) if you
  want true immutability.
- The model files are also `no-cache`; the app itself stores each model in the browser's Cache API under its
  SHA-256, so it is downloaded once per browser (the smoke test checks this).

## What was verified in Cloudflare's documentation (2026-11-20)

| Fact | Source | |
| --- | --- | --- |
| `_headers` lives in the assets directory, is not served itself; at most **100 rules**, **2,000 characters** per line | [Headers](https://developers.cloudflare.com/workers/static-assets/headers/) | [V] |
| A request inherits every matching rule, in file order; repeated headers are **joined with commas**; `! Name` detaches a header, including defaults | same | [V] |
| `_headers` is **not applied to responses generated by Worker code**, hence an assets-only Worker (no `main`) | same | [V] |
| One splat (`*`) per pattern, greedy; `:name` placeholders | same | [V] |
| Limits: **25 MiB per asset**, **20,000 files** (Free) / 100,000 (Paid) per Worker version | [Limits](https://developers.cloudflare.com/workers/platform/limits/) | [V] |
| `not_found_handling = "single-page-application"` serves `/index.html` with 200 for unknown paths; `main` can be omitted for an assets-only Worker | [SPA routing](https://developers.cloudflare.com/workers/static-assets/routing/single-page-application/) | [V] |
| Custom Domain: `routes` with `custom_domain: true`; needs an active zone on Cloudflare; the hostname must have no CNAME | [Custom domains](https://developers.cloudflare.com/workers/configuration/routing/custom-domains/) | [V] |
| `_headers` rules also apply to the SPA-fallback and 404 responses | not stated in the page | [NV] |
| `.wasm` is served as `application/wasm`; static assets get an `ETag` | not stated in the pages read | [NV] |
| Inputs of `cloudflare/wrangler-action@v3` (`apiToken`, `accountId`, `command`) | not fetched (from memory) | [NV] |
| The workflow `studio-hosted.yml` has run on GitHub Actions | it has not | [NV] |

`ort-wasm-simd-threaded.wasm` is 14,848,852 bytes (14.2 MiB), under the 25 MiB limit; a single ONNX model
must be below 25 MiB too, which is a real constraint for the larger model sizes.

## Deploy to Cloudflare

The workflow `.github/workflows/studio-hosted.yml` is **manual only** (`workflow_dispatch`). It builds the UI,
publishes the bundle, assembles, runs the smoke test and, only if you tick **deploy**, deploys. Nothing
runs on push or pull request. (`tools/publish_models.py` arrives in a sibling pull request; until it is on
the branch the bundle job stops with a clear message. `[NV]` its command line.)

### What the owner does by hand

1. **Cloudflare account.** Free plan is enough. Note the **Account ID** (dashboard, Workers & Pages).
2. **API token.** My Profile > API Tokens > Create Token, template "Edit Cloudflare Workers", or a custom
   token with *Account > Workers Scripts > Edit*. For the custom domain also add *Zone > Workers Routes > Edit*
   and *Zone > Zone > Read* on `nitid.io`. Keep it to this account.
3. **GitHub secrets** (repository Settings > Secrets and variables > Actions): `CLOUDFLARE_API_TOKEN` and
   `CLOUDFLARE_ACCOUNT_ID`. They are never written to the repository; the deploy job is the only one that reads them.
4. **GitHub environment** `cloudflare-production` (Settings > Environments). Optional: add yourself as a required
   reviewer, so a deploy waits for an approval.
5. **Run it.** Actions > "Studio hosted (Cloudflare)" > Run workflow. First without `deploy` (a dry run: build,
   assemble, smoke test; the assembled site is kept as an artifact for 30 days). Then with `deploy` ticked.

### Where it is served: two paths

**Path B, the default: the free `workers.dev` address.** With no custom domain the Worker `nitid-app` answers at
`https://nitid-app.<your-subdomain>.workers.dev`. It needs no DNS and works from the first deploy. The
default `wrangler.jsonc` contains no route, so a deploy never fails for lack of a domain.

**Path A: `app.nitid.io`.** A Workers Custom Domain needs an **active Cloudflare zone for `nitid.io`**; Cloudflare then
creates the DNS record and the certificate itself. Do **not** create a DNS record for `app.nitid.io` by hand,
and the name must not already have a CNAME. `nitid.io` is currently managed at SiteGround, so its DNS is probably
not on Cloudflare. On the Free and Pro plans the only way to add a zone is a *full setup*, which means changing the
domain's nameservers at the registrar (partial/CNAME setup needs Business or Enterprise; see
[Update nameservers](https://developers.cloudflare.com/dns/nameservers/update-nameservers)). So, in this order:

1. Add `nitid.io` to Cloudflare (Free plan). Cloudflare scans existing records; **check the list against the
   current DNS at SiteGround and re-create every record that is missing**: the website's `A`/`CNAME`, **`MX`**
   (mail), **`TXT`** (SPF, DKIM, DMARC, verification codes) and any other. A missing `MX` or `TXT` record
   breaks e-mail the moment the nameservers change.
2. Change the nameservers at the registrar to the two Cloudflare gives you; wait until the zone shows *Active*.
3. Add the custom domain: dashboard (Workers & Pages > `nitid-app` > Settings > Domains & Routes > Add > Custom
   Domain > `app.nitid.io`) **or** uncomment the `routes` line in `studio/hosting/cloudflare/wrangler.jsonc` and deploy again.

Until step 3 is done use Path B. `[NV]` the exact dashboard labels (they change); the config key names are verified.

### Check the result

```bash
curl -sI https://nitid-app.<your-subdomain>.workers.dev/ | grep -iE "cross-origin|content-security|cache-control"
curl -sI https://nitid-app.<your-subdomain>.workers.dev/ort/ort-wasm-simd-threaded.wasm | grep -iE "content-type|etag"
```

Expect COOP `same-origin`, COEP `require-corp`, the CSP above, `application/wasm` and an `etag`. Then open the site
and the browser console: `crossOriginIsolated` must be `true`. If `Content-Type` is not `application/wasm`
or the headers are missing on client routes, that is the `[NV]` points above: report it and adjust `_headers`.

## Privacy

- **The images never leave your browser.** Detection runs in WebAssembly on your device. No image, result or
  file name is sent to any server, ours or a third party's.
- **No telemetry.** No analytics, no cookies, no tracking script, no external font or CDN; the policy
  `connect-src 'self'` would block them anyway, and `Referrer-Policy: no-referrer` hides the page address.
- The only requests are for the site's own files: the app, the ONNX Runtime files and the model you pick.
  The model is kept in your browser's cache; clearing site data removes it.
- Cloudflare, as the host, sees the usual web-server data (IP address, requested file) like any website.

## Limits

- **Speed and size.** It runs on the visitor's CPU through WebAssembly, so large models are slow; each model file
  must be under 25 MiB (Cloudflare) and is downloaded once per browser.
- **No server features.** No Local mode, no comparison with a server, no training, no PyTorch or OpenVINO.
- **Cross-origin isolation** is required for the threaded engine; embedding the app in another site is blocked on purpose.
- **WebGPU** is not enabled in the hosted build `[NV]` (it is not verified in a real browser).
- The smoke test uses a fixture model and Chrome. `[NV]` Firefox and Safari were not tested.

## Rollback

- **A bad release:** run the workflow again from the last good commit or tag with **deploy** ticked; the previous
  assembled site is also kept for 30 days as the `hosted-site` artifact of an earlier run.
- **Quick undo in Cloudflare:** the Worker's *Deployments* tab lists earlier versions and can roll back to one,
  or `npx wrangler rollback --config studio/hosting/cloudflare/wrangler.jsonc`. `[NV]` not exercised here.
- **Stop serving entirely:** delete the Custom Domain (or the Worker) in the dashboard.
- **A wrong model:** fix the bundle and redeploy; browsers re-download a model whose SHA-256 in `models.json` changed.

## Files

| File | Role |
| --- | --- |
| [`studio/hosting/cloudflare/_headers`](https://github.com/Nitidio/nitid/blob/main/studio/hosting/cloudflare/_headers) | Response headers |
| [`studio/hosting/cloudflare/wrangler.jsonc`](https://github.com/Nitidio/nitid/blob/main/studio/hosting/cloudflare/wrangler.jsonc) | Assets-only Worker `nitid-app` |
| [`scripts/assemble_hosted.py`](https://github.com/Nitidio/nitid/blob/main/scripts/assemble_hosted.py) | UI + bundle -> one folder, with validation |
| [`studio/scripts/smoke-hosted.mjs`](https://github.com/Nitidio/nitid/blob/main/studio/scripts/smoke-hosted.mjs) | Real-Chrome smoke test, no backend |
| [`.github/workflows/studio-hosted.yml`](https://github.com/Nitidio/nitid/blob/main/.github/workflows/studio-hosted.yml) | Manual build, smoke test and gated deploy |
| `tests/unit/studio/test_hosted_headers.py` | `_headers` and `wrangler.jsonc` against `security.py` |
