# Nitid Studio: hosted model bundle

Work package WP-6.2. The hosted Studio (`vite build --mode hosted`) is a static site with no
backend: the in-browser engine downloads ONNX files from a folder that the owner hosts **on a server of
his own, on another origin than the UI** (the UI is on Cloudflare at `app.nitid.io`; the models on, for
example, `https://models.nitid.io/`; decision of 2026-11-23, because Cloudflare static assets accept only
25 MiB per file and `nitid1s` is 39.7 MiB). `url` values in the catalogue are relative file names, resolved
against the base URL that the UI was built with. This page explains how that folder is produced, what is in it,
how the licence gate works, how big the files really are, and **what the models server must do**
([jump to it](#hosting-the-bundle-on-your-own-server)).

Marks: **[V]** measured or read in the session that wrote this page (2026-11-20); **[NV]** not
verified, with what is missing.

## In one paragraph

The pipeline reads [`studio/models.lock.json`](https://github.com/Nitidio/nitid/blob/main/studio/models.lock.json)
(the licence-audited list of weights), keeps the entries with `hostable: true`, downloads each
pinned checkpoint, **checks its SHA-256 against the lock before loading it**, exports it with the
same exporter the local Studio server uses (`nitid.studio.onnx_service.export_checkpoint`: static
`[1, 3, S, S]`, post-processed, simplified, `nitid_meta` embedded) and writes the folder below.
A second tool re-reads the folder and checks it. Nothing is uploaded anywhere by these tools.

## Layout

```text
models.json                       catalogue read by studio/src/browser/catalog.ts (version 1)
nitid1s-detect.onnx               one ONNX per (model, task); name = <id>-<task>.onnx
nitid1s-detect.onnx.meta.json     sidecar: sha256, size, source checkpoint, copy of nitid_meta
notices/Peterande-D-FINE/LICENSE  upstream licence files named by the lock's notice_files
NOTICE.txt                        attribution + "modified by exporting to ONNX" notice
SHA256SUMS                        sha256sum -c compatible, every file except itself
```

`models.json` has exactly the keys the browser parser accepts:

```json
{ "models": [ { "id": "nitid1s", "task": "detect", "url": "nitid1s-detect.onnx",
                "sha256": "<64 hex>", "label": "nitid1s (detect, coco)" } ] }
```

`sha256` is the browser's cache key and integrity check, so the pipeline always writes it. The
sidecar is for humans and tools (it says which lock entry, URL and revision the file came from);
the browser does not read it. No `.pth`/`.pt` file is ever written to the bundle.

## The licence gate

- Only `hostable: true` entries are published. Asking for a `hostable: false` entry by lock id
  (every `obj2coco` entry, for example `--models dfine-s-obj2coco`) is an **error**, not a silent
  skip, and nothing is written. Asking for a public name (`--models nitid1s`) selects only its
  hostable entries and fails when it has none.
- The entry needs `sha256_status: "verified"`, no `open_caveats` and its `notice_files`.
- The licence texts come from [`studio/notices/`](https://github.com/Nitidio/nitid/tree/main/studio/notices)
  (`<owner>-<repo>/<file>`), where the upstream `LICENSE` files read during the audit are
  committed. A missing or empty file stops the run: no licence file, no publication.
- `hostable` is the result of a mechanical gate, not a legal opinion; the decision to host stays
  with the owner (see [Licence audit](licensing/studio-weights.md)).

## Producing the bundle

```bash
uv sync --extra dev                                   # onnx, onnxsim, jsonschema
uv run python -m tools.publish_models --out dist-models                 # n and s (default)
uv run python -m tools.check_hosted_bundle dist-models                  # validate
```

| Option | Meaning |
| --- | --- |
| `--out DIR` | Bundle folder (required). |
| `--sizes n,s,m,l,x` | Size allow-list. Default `n,s` (small sizes until sizes are measured) unless `--models` is given. |
| `--models a,b` | Lock ids (`dfine-s-coco`) or public names (`nitid1s`). |
| `--tasks detect,segment` | Default: both. |
| `--offline-from DIR` | Use weights already on disk, named like the release assets (`dfine_s_coco.pth`). They are still checked against the lock. |
| `--cache-dir DIR` | Keep downloaded weights between runs. |
| `--notices-dir DIR` | Where licence texts are read from (default `studio/notices`). |
| `--force` | Rebuild even if up to date. |
| `--dry-run` | Print the selection and stop. |

Properties worth knowing:

- **Idempotent.** If `<file>.meta.json` says the ONNX was built from the same source SHA-256 with
  the same nitid version and the file still has the recorded hash, it is reused without
  downloading or exporting. A second run changes nothing, byte for byte.
- **Deterministic.** Order is by size (n, s, m, l, x), then task, then lock id; JSON is written
  with fixed formatting. A forced rebuild of the tiny test checkpoints gave identical ONNX bytes
  **[V]**; on the same machine the real models were not rebuilt twice, so byte-for-byte
  reproducibility across torch/onnx versions is **[NV]**. Treat the ONNX SHA-256 as belonging to
  the toolchain that built it (see the table).
- **Stale files are pruned.** ONNX files and sidecars of earlier runs that are not in the current
  selection are removed from `--out`, so `SHA256SUMS` and the catalogue always describe the folder.
- `nitid_meta.model` is set from the lock when the exporter could not infer the name, and the run
  fails if an ONNX names a different model or task than the lock.
- Segmentation weights come from Hugging Face; they are built the same way but have **not** been
  built in the sandbox of this work package (Hugging Face is blocked there) **[NV]**.

Then upload the folder to the models server and point the hosted build at it:
`STUDIO_MODELS_BASE_URL=https://models.nitid.io/ pnpm run build:hosted` (see
[the hosted site](studio-hosted.md) and the [frontend guide](studio-development.md)). For a small bundle on the
same origin as the UI, `scripts/assemble_hosted.py --bundle` copies it into the site instead (25 MiB per file).

## Hosting the bundle on your own server

The models server is a plain static file server. The UI at `https://app.nitid.io` reads it from the browser, so
it has to say "yes, this page may read me". Everything the app needs is below; nothing else is required.

### What the server must do

| Requirement | Why |
| --- | --- |
| **HTTPS** with a valid certificate, on a fixed origin (`https://models.nitid.io`, optional port). Serve the files directly at their final URL: no redirect to another origin. | The app is an https page (no mixed content), and the site's policy allows exactly this origin; a redirect to another origin would be blocked by `connect-src` `[NV]`. |
| `Access-Control-Allow-Origin: https://app.nitid.io` (exactly the app's origin; `*` also works because no credentials are sent `[NV]: not exercised, only the exact origin was tested`) on **every** response, errors included | The page is cross-origin isolated (COEP `require-corp`); the app fetches the models in CORS mode, and without this header Chrome refuses the response. **Tested: without it the app shows "The model catalogue could not be reached".** |
| `Cross-Origin-Resource-Policy: cross-origin` | Belt and braces. Measured: the app also works without it (CORS mode is enough), but any future non-CORS load (an image, a script, a worker) would be blocked without it. Send it. |
| `Content-Type: application/json` for `models.json`; `application/octet-stream` (or `application/vnd.onnx`) for `*.onnx` | The catalogue is parsed as JSON; the app reads the ONNX as bytes. Text types for `.onnx` also work, but are wrong. |
| Answer `GET` and `HEAD`; answer `OPTIONS` with a 2xx (and the CORS headers) if you can | The app sends plain `GET` requests (verified: no preflight, no `HEAD` when the catalogue has `sha256`, which `tools/publish_models.py` always writes). `HEAD` is used only when an entry has no `sha256`; `OPTIONS` only if a browser decides to preflight. |
| `Cache-Control: no-cache` with an `ETag` (the default of nginx and Apache) | File names are fixed (`nitid1s-detect.onnx`), so a file can change under the same URL: `public, max-age=31536000, immutable` is **not safe** here. `no-cache` still avoids re-downloading: the browser revalidates, and the app keeps each model in its own Cache API by SHA-256, so a model is downloaded once per browser. |
| `Access-Control-Expose-Headers: ETag, Content-Length` (recommended) | Without it the browser hides these two headers from the app: the download progress bar has no total, and an entry **without** `sha256` cannot be cached by `ETag`. Not needed otherwise (tested without). |
| No `Content-Encoding` on `*.onnx` unless it is real (gzip/brotli are optional: ONNX weights barely compress). Never a wrong `Content-Length`. | The app checks the SHA-256 of the bytes it receives; a double-compressed or truncated file fails the check. |
| No authentication, no cookies, no `Set-Cookie`, no `Access-Control-Allow-Credentials` | The app sends no credentials (`credentials: "omit"`; tested: the server never sees a cookie). |
| No directory listing | Nothing to browse. |

What the server does **not** need: any TLS-client configuration, CORS preflight caching, a special method, or
anything for the UI's own files (those stay on Cloudflare).

### nginx `[NV]` (not tested here)

```nginx
server {
    listen 443 ssl;
    http2 on;
    server_name models.nitid.io;
    # ssl_certificate / ssl_certificate_key: your certificate (for example from certbot)

    root /var/www/models;          # the bundle: /var/www/models/models.json, ...
    autoindex off;
    default_type application/octet-stream;   # .onnx has no entry in mime.types

    # Only reads. (Return 405 for everything else.)
    if ($request_method !~ ^(GET|HEAD|OPTIONS)$) { return 405; }

    location / {
        # add_header is NOT inherited into a block that has its own add_header: keep them all here.
        add_header Access-Control-Allow-Origin "https://app.nitid.io" always;
        add_header Access-Control-Expose-Headers "ETag, Content-Length" always;
        add_header Cross-Origin-Resource-Policy "cross-origin" always;
        add_header Vary "Origin" always;
        add_header Cache-Control "no-cache" always;
        add_header X-Content-Type-Options "nosniff" always;

        # Optional: answer a preflight (the app does not trigger one today).
        if ($request_method = OPTIONS) {
            add_header Access-Control-Allow-Origin "https://app.nitid.io" always;
            add_header Access-Control-Allow-Methods "GET, HEAD, OPTIONS" always;
            add_header Access-Control-Max-Age "86400" always;
            add_header Cross-Origin-Resource-Policy "cross-origin" always;
            add_header Content-Length 0;
            return 204;
        }

        etag on;                    # the default
        gzip off;                   # optional: ONNX weights barely compress
    }
}
```

Check it with `sudo nginx -t` before reloading. `types { }` is deliberately not used: it would replace the
default MIME table, and `.json` would lose its type.

### Apache / `.htaccess` `[NV]` (not tested here)

Needs `mod_headers` (and `mod_mime`; `mod_rewrite` is not used). Put it in the web root of the models host:

```apache
Options -Indexes

<IfModule mod_mime.c>
  AddType application/json .json
  AddType application/octet-stream .onnx
</IfModule>

<IfModule mod_headers.c>
  Header always set Access-Control-Allow-Origin "https://app.nitid.io"
  Header always set Access-Control-Expose-Headers "ETag, Content-Length"
  Header always set Cross-Origin-Resource-Policy "cross-origin"
  Header always merge Vary "Origin"
  Header always set Cache-Control "no-cache"
  Header always set X-Content-Type-Options "nosniff"
</IfModule>

# Reads only; OPTIONS stays allowed for a preflight.
<LimitExcept GET HEAD OPTIONS>
  Require all denied
</LimitExcept>

# Optional: never compress the models (a double-encoded file fails the SHA-256 check).
<IfModule mod_deflate.c>
  SetEnvIfNoCase Request_URI \.onnx$ no-gzip
</IfModule>
```

On a shared host, `AllowOverride` must permit `Options`, `FileInfo` and `Headers`; if the host ignores
`.htaccess`, use the host's panel for custom headers instead. If `Content-Type` stays `text/plain` for `.onnx`,
the `AddType` line was not applied.

### Check it with `curl`

Replace the host if yours differs. These need no browser:

```bash
BASE=https://models.nitid.io
APP=https://app.nitid.io

# 1. the catalogue: 200, JSON, CORS for the app's origin, CORP cross-origin, no-cache
curl -sI -H "Origin: $APP" "$BASE/models.json" \
  | grep -iE "^HTTP|content-type|access-control-allow-origin|cross-origin-resource-policy|cache-control|etag|set-cookie"

# 2. a model: same headers, a Content-Length, no Content-Encoding
curl -sI -H "Origin: $APP" "$BASE/nitid1n-detect.onnx" \
  | grep -iE "^HTTP|content-type|content-length|content-encoding|access-control-allow-origin|cross-origin-resource-policy|etag"

# 3. the bytes are the ones in the catalogue (compare with the sha256 of that entry in models.json)
curl -s "$BASE/nitid1n-detect.onnx" | sha256sum
curl -s "$BASE/models.json"

# 4. an error is also CORS-enabled (a missing file must not hide the real error behind a CORS failure)
curl -sI -H "Origin: $APP" "$BASE/does-not-exist.onnx" | grep -iE "^HTTP|access-control-allow-origin"

# 5. optional: the preflight
curl -si -X OPTIONS -H "Origin: $APP" -H "Access-Control-Request-Method: GET" "$BASE/models.json" | head -n 12
```

Expected: `HTTP/2 200`; `content-type: application/json` (1) and `application/octet-stream` or
`application/vnd.onnx` (2); `access-control-allow-origin: https://app.nitid.io`;
`cross-origin-resource-policy: cross-origin`; `cache-control: no-cache`; an `etag`; **no** `set-cookie`, **no**
`content-encoding`; the same SHA-256 as the catalogue (3); `404` with the CORS header (4); `204` or `200` (5).
The strongest test is the app itself: open `https://app.nitid.io/run/browser`; the model list must appear and the
console must show no `blocked by CORS policy`. A header listed here can still be removed by a CDN or proxy in
front of the server, which is why the check is run against the public URL.

### Upload it

The bundle folder is uploaded **as it is, to the web root of the models host**, so that
`https://models.nitid.io/models.json` exists (if you serve it under a path, `STUDIO_MODELS_BASE_URL` must contain
that path, e.g. `https://example.org/nitid/`):

```text
<web root>/
  models.json                        <- the catalogue, uploaded LAST
  nitid1n-detect.onnx                <- one file per model and task
  nitid1n-detect.onnx.meta.json
  nitid1s-detect.onnx   ...          (and the segmentation files, if published)
  NOTICE.txt
  notices/Peterande-D-FINE/LICENSE   <- licence texts: they must travel with the models
  SHA256SUMS
```

1. Get the bundle: the `hosted-bundle` artifact of the workflow run (kept 7 days), or
   `uv run python -m tools.publish_models --out dist-models` locally. Run
   `uv run python -m tools.check_hosted_bundle dist-models` (the workflow already did).
2. **Keep a copy of every bundle you upload** (a dated folder). It is your rollback.
3. Upload everything except `models.json` (SFTP, `rsync -av --exclude models.json dist-models/ user@host:/var/www/models/`, or the panel).
4. Upload `models.json` last, so no catalogue ever points at a file that is not there yet.
5. Run the `curl` checks above, then `sha256sum -c SHA256SUMS` in the web root on the server.
6. Make sure the UI deployed on Cloudflare was built with the same base URL (`STUDIO_MODELS_BASE_URL`).

The UI and the bundle are independent releases: new models need an upload only, no Cloudflare deploy, as long as
the catalogue format (version 1) and the origin do not change.

### Roll back

- **A bad model or catalogue:** re-upload the previous bundle copy (step 2) in the same order (models first,
  `models.json` last). Names are fixed, so the old file replaces the new one. Browsers that already fetched the
  bad file hold it under its SHA-256; the old `models.json` points to the old SHA-256 and the old bytes are
  downloaded (or served from that earlier cache entry), so nobody keeps using the bad one.
- **The models server is down or broken:** the app shows "The model catalogue could not be reached" and nothing
  else breaks. Restore the server; no Cloudflare action is needed.
- **A wrong origin or CORS value:** fix the server headers (no UI deploy needed). A *new origin* needs a UI
  rebuild and deploy, because the origin is part of the build and of the site's headers.

## What the validator checks

`python -m tools.check_hosted_bundle DIR [--lock FILE] [--max-total-bytes N]` re-reads the folder and
trusts nothing the publisher wrote. It prints a size report and exits 1 on any problem:

- `models.json` has the shape `parseHostedCatalog` accepts, non-empty, with a `sha256` and a
  plain relative `.onnx` file name per entry (no scheme, no path);
- every `sha256` matches its file and every `.onnx` is in the catalogue;
- every ONNX has a `nitid_meta` valid against
  [`onnx-meta.schema.json`](https://github.com/Nitidio/nitid/blob/main/studio/api-contract/onnx-meta.schema.json),
  consistent with the catalogue (task, model, static square input) and with the sidecar copy;
- every sidecar names a lock entry that is `hostable: true` with a verified sha256 and the same
  source hash as the lock;
- the licence files each entry requires, and `NOTICE.txt`, are present;
- no `.pth`, `.pt`, `.ckpt`, `.safetensors`, `.torchscript`, `.engine` or `.bin` file;
- `SHA256SUMS` lists exactly the files present and every hash matches.

The contract with the browser is tested on both sides: `tests/unit/test_publish_models.py` proves
the catalogue the tool writes is byte-identical to
`studio/src/browser/__fixtures__/hosted-models.json`, and `studio/src/browser/hosted-bundle.test.ts`
parses that fixture with the real `parseHostedCatalog`. To check a real bundle:

```bash
cd studio && NITID_HOSTED_CATALOG=../dist-models/models.json pnpm exec vitest run hosted-bundle
```

## Measured sizes (risk R13)

Detection, COCO-only checkpoints from the `dfinev1.0` release (the audited, pinned URLs), exported
at the default size (640 x 640), batch 1, static, post-processed, simplified. Built with nitid
0.9.6, torch 2.13.0, onnx 1.20.1, onnxruntime 1.28.0 on 2 CPUs and 7 GB of RAM, 2026-11-20.

| Model | Checkpoint `.pth` | ONNX bytes | ONNX MiB | ONNX SHA-256 | Build time / peak RSS |
| --- | ---: | ---: | ---: | --- | --- |
| `nitid1n` | 15,489,558 | 15,383,183 | 14.67 [V] | `ef561cd56d4ad0164a6ea326b3f37e4ef6d424f83b75271dfbea7ba79c0907ba` | about 12 s with download [V] / not recorded |
| `nitid1s` | 41,841,422 | 41,638,053 | 39.71 [V] | `a0dd8aee495f81c7ce6e745b00ff2071887a5109216dced6fc9377e0ea920c7f` | 17 s / 1.6 GB [V] |
| `nitid1m` | 79,108,938 | 77,708,250 | 74.11 [V] | `8c67d2d61cb8c657ce14d8ece3fe8bbf3ac6e6005e29739615779571d5ac66fd` | 26 s / 2.1 GB [V] |
| `nitid1l` | 126,059,754 | 123,804,142 | 118.07 [V] | `29a489d3488174c4b06328185b18f9a9616a051a19458f323bcf0d8074ddaaf0` | 43 s / 2.3 GB [V] |
| `nitid1x` | 252,128,986 | 247,641,727 | 236.17 [V] | `9bc94f3e02a4ed8fc91cc0e4f20dd8e31f6b77d3d7aced914201be5fac84d88b` | under 3 min (whole 5-model run took about 3.5 min) / not recorded |

The five ONNX files together are 506,175,355 bytes; the complete bundle with sidecars, notices and
checksums is 506,206,348 bytes (482.76 MiB) **[V]**. The validator accepted this bundle
(`hosted bundle OK`), `sha256sum -c SHA256SUMS` was clean, and the TypeScript parser accepted its
`models.json` **[V]**. Sizes in bytes are the ones the user's browser downloads if nothing
compresses them; ONNX weights compress poorly and compression was **not measured** **[NV]**.
The ONNX is slightly smaller than the checkpoint because the checkpoint stores EMA weights in
addition to the model **[NV: reason not checked]**.

Real inference check **[V]**: each of the five ONNX files loaded in ONNX Runtime 1.28.0 (CPU
provider), read its own `nitid_meta`, and on `docs/assets/demo/street.jpg` (squash-resized to
640 x 640, RGB, 0-1) returned `labels`, `boxes`, `scores` (the meta says `topk` 300); with score above 0.5 the
models returned 18 (n), 24 (s), 23 (m), 29 (l) and 23 (x) detections, mostly `person` and
`traffic light` (the image was not inspected by eye). This confirms the files load and detect; it is
**not** an accuracy evaluation, and not a check of the browser engine (ONNX Runtime Web / WASM).

Consequences to decide, not decided here:

- **Per-file limit [V, Cloudflare changelog of 2025-09-02, may have changed since]:** Workers static
  assets allow 25 MiB per file. Only `nitid1n` (14.67 MiB) fits; `nitid1s` (39.71 MiB) and larger
  do not. That is why the bundle is **not** deployed to Cloudflare: the owner hosts it on his own server
  (below), where no such limit applies, and the default sizes `n,s` can be published. The 25 MiB check of
  `scripts/assemble_hosted.py` applies only to files that are copied into the site.
- **First-use download:** s is about 40 MiB and x about 236 MiB per visitor on first use (then
  cached by the browser by SHA-256). Whether that is acceptable on mobile is a product decision.
- **Not measured:** WASM inference time and memory in a browser for any size; segmentation
  exports (n, s, m, l, x) of any size.

## Tests

| File | What it covers |
| --- | --- |
| `tests/unit/test_publish_models.py` | Selection, size/task/model filters, refusal of non-hostable entries, lock gate, notices, catalogue shape and the TypeScript fixture, checksums, idempotence, pruning, offline and cache behaviour, CLI. |
| `tests/unit/test_check_hosted_bundle.py` | Every validator rule, passing and failing. |
| `tests/integration/test_publish_models.py` | Tiny random checkpoints (detect and segment) through the real exporter: bundle, validator, ONNX Runtime run, byte-identical rebuild, CLI, and the vitest parser check (skipped without `studio/node_modules`). |
| `studio/src/browser/hosted-bundle.test.ts` | `parseHostedCatalog` on the Python-written catalogue, against a relative and an absolute models base URL. |
| `studio/src/browser/catalog.test.ts`, `studio/src/engine/model-cache.test.ts` | Absolute base URL resolution, same-origin guard for `url`, build-time URL validation, CORS-mode and no-credentials requests. |
| `tests/unit/test_assemble_hosted.py` | The cross-origin assemble mode: only the models origin in the CSP, nothing copied, validation without the 25 MiB limit. |
| `studio/scripts/smoke-hosted.mjs` | Real Chrome, the site and the models on two origins, with positive and negative cases ([the hosted site](studio-hosted.md#test-it-without-any-backend)). |

Related: [Licence audit](licensing/studio-weights.md), [Studio frontend guide](studio-development.md),
[Export](export.md) (the `nitid_meta` format).
