# Nitid Studio: the hosted site

The hosted site is Nitid Studio **without a server**: a static website where the model runs inside the
visitor's browser. Nothing is uploaded; there is no backend to attack, scale or pay for. This page
explains how it is put together, how to build and deploy it, and what the owner has to do by hand.
Facts that were not verified are marked `[NV]`; verified ones say where.

## The idea in one picture: a shop and a warehouse

Think of a market stall (the app) and a **warehouse** (the models) that is somewhere else:

| Part | What it is | Where it lives |
| --- | --- | --- |
| The app | `index.html`, `assets/`, the ONNX Runtime engine in `ort/` | Cloudflare, `app.nitid.io` (`pnpm run build:hosted`, `studio/dist-hosted`) |
| The models | `models.json`, `*.onnx`, `SHA256SUMS`, licence files | **a server of the owner**, for example `https://models.nitid.io/` (the model bundle, `tools/publish_models.py`) |

The reason is size: Cloudflare static assets accept **25 MiB per file** at most and `nitid1s` is 39.7 MiB, so the
models cannot be part of the site. The app learns where the warehouse is from **one build-time setting**,
`STUDIO_MODELS_BASE_URL` (never written in the source code). `scripts/assemble_hosted.py` then writes the site
folder, which Cloudflare serves as static files, and gives its security headers exactly one extra permission: to
talk to that one origin.

```text
studio/dist-hosted-site/           (what Cloudflare serves)
  index.html  assets/  ort/        <- the app
  _headers                         <- the response headers, with the models origin in connect-src
                                      (NO models/ folder, no .onnx: they are on the models server)

https://models.nitid.io/           (what the owner uploads to his own server; docs/studio-hosted-models.md)
  models.json  nitid1n-detect.onnx  nitid1s-detect.onnx  ...  SHA256SUMS  NOTICE.txt  notices/
```

Why does this still work with the isolated page of the next section? A page that is **cross-origin isolated**
may only load things from other origins if those origins agree to it. The app fetches the models in **CORS mode**
(an explicit `mode: "cors"`), and the models server agrees by answering with `Access-Control-Allow-Origin:
https://app.nitid.io`. The request carries **no credentials** (`credentials: "omit"`: no cookie, no
`Authorization`), and no custom header, so the browser needs no preflight. The server should also send
`Cross-Origin-Resource-Policy: cross-origin` (the app's own files keep `same-origin`). The exact server
requirements, with nginx and Apache snippets, are in [the models server guide](studio-hosted-models.md#hosting-the-bundle-on-your-own-server).

There is still a second, simpler topology for **small bundles**: `--bundle DIR` copies the bundle into the site
as `models/` (same origin, `STUDIO_MODELS_BASE_URL=/models/`, every file under 25 MiB). It is what the
smoke test runs first, and what `nitid1n` alone would allow; it is not what production uses.

The local app (`nitid studio`) and the hosted site are the **same UI**. The hosted build
(`vite build --mode hosted`) just never looks for a backend: no `/api/v1/health` probe, no "Local" or
"Compare" entry, and the "Browser" page reads `models.json` under `STUDIO_MODELS_BASE_URL`. With no base URL it shows
"No models are configured for this site".

## Build it

```bash
cd studio && pnpm install --frozen-lockfile
STUDIO_MODELS_BASE_URL=https://models.nitid.io/ pnpm run build:hosted     # -> studio/dist-hosted (ORT wasm included)

# the site: the UI + headers for that models origin. Nothing of the bundle is copied.
python scripts/assemble_hosted.py --models-base-url https://models.nitid.io/ --clean   # -> studio/dist-hosted-site

# the same, and also check a local copy of the bundle (not copied; no 25 MiB limit applies to it)
python scripts/assemble_hosted.py --models-base-url https://models.nitid.io/ \
    --validate-bundle path/to/bundle --clean
```

`STUDIO_MODELS_BASE_URL` is either `/models/` (same origin) or an absolute `https://` URL; `vite build` refuses
anything else: `http://` is accepted only for `localhost`, and a query string, a fragment or credentials in the URL never are.
The base may have a path (`https://example.org/nitid/models/`); only its **origin** goes into the policy. Model
`url` values in `models.json` are resolved against the base (relative file names) and must stay on that origin.

### The headers are generated, not committed

`studio/hosting/cloudflare/_headers.template` is the policy with one placeholder in `connect-src`. The assemble
script renders it into `<site>/_headers`:

- **cross-origin** (`--models-base-url` or `--models-origin`): `connect-src 'self' https://models.nitid.io`.
  The origin is validated first and the script refuses: anything but `https` (a test-only flag,
  `--allow-http-loopback`, admits `http://localhost` and `http://127.0.0.1`; the workflow never uses it),
  wildcards (`*`), a path, a query string, a fragment, credentials, spaces or `;` `,` quotes (CSP injection),
  and malformed host names or ports;
- **same origin** (`--bundle`): `connect-src 'self'`.

It also checks that the UI was **built for the same URL** (the base URL is baked into the JavaScript): a UI built
for another server would request models from an origin the policy does not allow, and the page would only say
"the catalogue could not be reached".

The assemble script is the last gate before a deploy. It exits with status 1, and writes nothing, when:

- the models location is missing or ambiguous (give exactly one of `--bundle`, `--models-base-url`,
  `--models-origin`), or the origin is not acceptable (above), or the UI was built for another URL;
- the bundle (`--bundle`, or `--validate-bundle`) has no `models.json`, it is not `{"models": [...]}` or lists no
  model; an entry has a bad `task`, a duplicate id, or a `url` that is absolute, uses `..` or does not exist;
- a `sha256` in `models.json` or a line in `SHA256SUMS` does not match the file;
- the bundle contains a training checkpoint or pickle (`.pth`, `.pt`, `.ckpt`, `.safetensors`, `.pkl`,
  `.pickle`) or a TensorRT/TorchScript file, or any symlink;
- `--validate-bundle` additionally runs `tools/check_hosted_bundle.py` (licences, `nitid_meta`, lock; needs
  `onnx` and `jsonschema`; `--bundle-check basic` skips it);
- **a file that ends up in the site** is larger than **25 MiB** or there are more than **20,000** (Cloudflare
  limits, below). A bundle that is only validated and hosted elsewhere is not subject to that limit;
- the UI is not built (no `index.html`, no `ort/*.wasm`).

A missing `sha256` or `SHA256SUMS` is only a warning (the browser then cannot verify or cache by hash).

## Test it without any backend

```bash
cd studio && pnpm run smoke:hosted            # or: pytest studio/e2e/test_studio_hosted.py
SMOKE_MODE=cross-origin pnpm run smoke:hosted # only the production topology (also: same-origin)
```

It builds the hosted UI, writes a fixture bundle (a tiny ONNX made in the script), runs the real assemble
script, serves the result with the **production headers** (rendered from the template) and the single-page-app
fallback, and drives real Chrome. No Python backend runs. It runs, in order:

1. **same-origin**: the bundle inside the site (`--bundle`).
2. **cross-origin**: the site on `127.0.0.1:PORT1`, the bundle on **another origin** `127.0.0.1:PORT2`, whose
   server sends **only** `Access-Control-Allow-Origin: <app origin>`, `Cross-Origin-Resource-Policy: cross-origin`
   and the right `Content-Type` (what the owner configures; no cache headers, no `Expose-Headers`). It asserts
   `crossOriginIsolated` and `SharedArrayBuffer`, zero console errors, zero CSP violations, a detection, the
   JSON export, a reload served from the Cache API (one download only), that the model requests went to the
   models origin as CORS requests (`Origin` header, `Sec-Fetch-Mode: cors`), only `GET` (no preflight, no
   `HEAD`), **no cookie** (a probe cookie that the browser would send if credentials were included is seen by the
   site's server and never by the models server), and that nothing of the bundle was requested from the site.
3. **variant, ACAO only**: the same without `Cross-Origin-Resource-Policy`. It passes: in CORS mode COEP
   accepts the response without CORP (measured in Chrome here). The guide still asks for CORP.
4. **negative, no `Access-Control-Allow-Origin`**: the app must not load the models, must say so, and Chrome must
   report `blocked by CORS policy: No 'Access-Control-Allow-Origin' header`; the test prints
   `DIAGNOSIS: the models server does not send Access-Control-Allow-Origin ...`.
5. **negative, origin missing from `connect-src`**: the site's headers without the models origin. Chrome must report
   a `connect-src` violation for the models URL and the request must never reach the models server; the test
   prints `DIAGNOSIS: ... the models origin is missing from connect-src`.

The same checks run on the same-origin site (no request to `/api/*` or to another origin, `application/wasm`,
and so on). `SMOKE_HEADERS=file` tries another headers template: with `'wasm-unsafe-eval'` removed the run fails
(the engine cannot compile its WebAssembly), and without COEP the page is no longer isolated. Only the loopback
`http` origin is used here; real `https` with real certificates is `[NV]` until the first deploy.

## The headers, and why

`studio/hosting/cloudflare/_headers.template` (rendered into `_headers` by the assemble script) carries the same
security policy as the local app (`nitid/studio/security.py`). The unit test `tests/unit/studio/test_hosted_headers.py`
compares them directive by directive, for the same-origin and the cross-origin rendering, so they cannot drift
apart silently; the **only** allowed difference is the models origin in `connect-src`. Only the server-side guards (Host, Origin,
bearer token) are absent: there is no server.

| Header | Value | Why |
| --- | --- | --- |
| `Cross-Origin-Opener-Policy` | `same-origin` | Part of isolation: no other window can hold a reference to this page. |
| `Cross-Origin-Embedder-Policy` | `require-corp` | Part of isolation: the page may only load resources that agree to be loaded. |
| `Cross-Origin-Resource-Policy` | `same-origin` | Our files agree to be loaded only by our own pages. (The models server sends `cross-origin` for its own files: it is a different server.) |
| `Content-Security-Policy` | `default-src 'self'; worker-src 'self' blob:; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' https://models.nitid.io; frame-ancestors 'none'` | Everything from our own origin only, plus **fetch** (not scripts, workers, images or frames) to the one models origin. No CDN, no analytics, no other third party can run or receive anything. |
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
- The model files (on the models server) are `no-cache` with an `ETag`; the app itself stores each model in the
  browser's Cache API under its SHA-256, so it is downloaded once per browser, also when the models come from
  another origin (the smoke test checks this). The cache key uses the app's own origin, never the models server's.

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

`ort-wasm-simd-threaded.wasm` is 14,848,852 bytes (14.2 MiB), under the 25 MiB limit. The ONNX models are what
does not fit (`nitid1s` is 39.7 MiB), hence the separate models server.

## Deploy to Cloudflare

The workflow `.github/workflows/studio-hosted.yml` is **manual only** (`workflow_dispatch`). It builds the UI
for the models server, publishes the model bundle (`tools/publish_models.py`, default sizes `n,s`), runs the
smoke test, assembles the site and, only if you tick **deploy**, deploys it. Nothing runs on push or pull request.

The bundle is **not deployed to Cloudflare**: the `hosted-bundle` artifact (kept 7 days) is what the owner
downloads and uploads to his models server ([how](studio-hosted-models.md#upload-it)). The UI is built with the
**repository variable** `STUDIO_MODELS_BASE_URL` (a variable, not a secret: the address is public by design).
**A deploy with the variable empty fails at once** with a message naming it, before anything is built; a dry run
without it still builds and smoke-tests (the smoke test uses its own local servers) but assembles no site.

### What the owner does by hand

1. **Cloudflare account.** Free plan is enough. Note the **Account ID** (dashboard, Workers & Pages).
2. **API token.** My Profile > API Tokens > Create Token, template "Edit Cloudflare Workers", or a custom
   token with *Account > Workers Scripts > Edit*. For the custom domain also add *Zone > Workers Routes > Edit*
   and *Zone > Zone > Read* on `nitid.io`. Keep it to this account.
3. **GitHub secrets** (repository Settings > Secrets and variables > Actions > Secrets): `CLOUDFLARE_API_TOKEN` and
   `CLOUDFLARE_ACCOUNT_ID`. They are never written to the repository; the deploy job is the only one that reads them.
   **GitHub variable** (same page, tab Variables): `STUDIO_MODELS_BASE_URL`, for example `https://models.nitid.io/`.
   Only the origin matters for security; the path, if any, is where `models.json` lives.
4. **GitHub environment** `cloudflare-production` (Settings > Environments). Optional: add yourself as a required
   reviewer, so a deploy waits for an approval.
5. **Prepare the models server** and upload the bundle ([guide](studio-hosted-models.md#hosting-the-bundle-on-your-own-server)),
   then run the `curl -I` checks of that guide. The app cannot work before this.
6. **Run it.** Actions > "Studio hosted (Cloudflare)" > Run workflow. First without `deploy` (a dry run: build,
   smoke test, and the bundle artifact). Then with `deploy` ticked (the assembled site is kept as an artifact for 30 days).

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

Expect COOP `same-origin`, COEP `require-corp`, the CSP above **with the models origin in `connect-src`**,
`application/wasm` and an `etag`. Then open the site and the browser console: `crossOriginIsolated` must be
`true`, and the Browser page must list the models. If it says "The model catalogue could not be reached", the
console tells which side is wrong: `blocked by CORS policy` means the models server (CORS header), `Refused to
connect ... Content Security Policy` means the site's headers (a UI built for another URL than the one deployed). If `Content-Type` is not `application/wasm`
or the headers are missing on client routes, that is the `[NV]` points above: report it and adjust `_headers`.

## Privacy

- **The images never leave your browser.** Detection runs in WebAssembly on your device. No image, result or
  file name is sent to any server, ours or a third party's.
- **No telemetry.** No analytics, no cookies, no tracking script, no external font or CDN; the policy
  `connect-src` (the site and the one models origin) would block them anyway, and `Referrer-Policy: no-referrer` hides the page address.
- The only requests are for the site's own files (the app, the ONNX Runtime files) and for the model catalogue and
  the model you pick, which come from the models server. Those requests carry **no cookies or credentials**.
  The model is kept in your browser's cache; clearing site data removes it.
- Cloudflare, as the host of the app, and the owner's models server, as the host of the models, each see the
  usual web-server data (IP address, requested file, the page address is not sent: `Referrer-Policy: no-referrer`
  applies to those requests too `[NV]`) like any website.

## Limits

- **Speed and size.** It runs on the visitor's CPU through WebAssembly, so large models are slow; each model is downloaded
  once per browser (about 15 MiB for `nitid1n`, 40 MiB for `nitid1s`).
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
- **A wrong model:** fix the bundle and upload it again to the models server (no Cloudflare deploy needed);
  browsers re-download a model whose SHA-256 in `models.json` changed. Rolling the models back is described in
  [the models server guide](studio-hosted-models.md#roll-back).
- **The models server is down or moved:** the app shows "The model catalogue could not be reached". Moving it
  to a new address needs a new UI build with the new `STUDIO_MODELS_BASE_URL` and a deploy (the address is
  part of the build and of the headers).

## Files

| File | Role |
| --- | --- |
| [`studio/hosting/cloudflare/_headers.template`](https://github.com/Nitidio/nitid/blob/main/studio/hosting/cloudflare/_headers.template) | Response headers, with the models origin as the only variable part |
| [`studio/hosting/cloudflare/wrangler.jsonc`](https://github.com/Nitidio/nitid/blob/main/studio/hosting/cloudflare/wrangler.jsonc) | Assets-only Worker `nitid-app` |
| [`scripts/assemble_hosted.py`](https://github.com/Nitidio/nitid/blob/main/scripts/assemble_hosted.py) | UI + generated headers (+ optionally a same-origin bundle) -> one folder, with validation |
| [`studio/scripts/smoke-hosted.mjs`](https://github.com/Nitidio/nitid/blob/main/studio/scripts/smoke-hosted.mjs) | Real-Chrome smoke test, no backend |
| [`.github/workflows/studio-hosted.yml`](https://github.com/Nitidio/nitid/blob/main/.github/workflows/studio-hosted.yml) | Manual build, smoke test and gated deploy |
| `tests/unit/studio/test_hosted_headers.py` | The rendered headers and `wrangler.jsonc` against `security.py` |
| `tests/unit/test_assemble_hosted.py` | The assemble gate and the origin/headers generator (valid origin, http, wildcard, path, missing) |
