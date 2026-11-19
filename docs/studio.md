# Nitid Studio

Nitid Studio is a local web app for working with nitid models from your
browser. It runs on your own machine, in a single process started by
`nitid studio`, and sends no telemetry.

!!! note "Status: backend in progress"
    This release contains the server foundation (settings, security guards,
    static serving, `GET /api/v1/health`) plus device discovery and model
    listing/downloading, and the ONNX endpoints for the browser engine. Prediction, training and export jobs and the web UI
    itself arrive in later changes. Until the UI is built into your installation, opening the URL
    shows a short page that says so (HTTP 503); the API still works.

## Install

Studio is an optional extra, so the base `nitid` install does not pull in a web
framework:

```bash
pip install "nitid[studio]"
```

The extra installs `fastapi`, `uvicorn` and `python-multipart`. If you run
`nitid studio` without it, the command prints the line above and exits with
code 3.

## Run

```bash
nitid studio
# or
python -m nitid.studio
```

Studio prints a URL such as `http://127.0.0.1:8000/#t=<token>` and opens it in
your browser. The token is in the URL *fragment* (after `#`), which browsers
never send to a server, so it does not appear in server logs.

| Option | Default | Meaning |
|---|---|---|
| `host=HOST` | `127.0.0.1` | Interface to listen on. |
| `port=INT` | `8000` | TCP port, `0` to `65535`. `0` picks a free port and prints it. |
| `models=PATH` | `./models` | Directory for models. It is not created at start. |
| `open=true\|false` | `true` | Open the browser when ready. |
| `--no-browser` | | Same as `open=false`. |
| `token=TOKEN` | random | Access token (at least 8 characters, no whitespace). |

Exit codes: `0` normal stop, `1` the address could not be bound, `2` invalid
arguments, `3` the `studio` extra is missing.

## Security model

Studio assumes one trusted user on a trusted machine. The defaults protect
that user from *other web pages* in their browser and from the local network.

- **Loopback by default.** The server listens on `127.0.0.1` only.
- **Token.** A random token (`secrets.token_urlsafe`) is generated at every
  start. Every `/api/v1/*` request except `GET /api/v1/health` must send
  `Authorization: Bearer <token>`. Comparison is constant-time. The token is
  never accepted in a query string.
- **Host check (DNS rebinding).** The `Host` header must be `127.0.0.1`,
  `localhost` or `[::1]` with the server's own port. Anything else gets 403, so
  a malicious site that re-points its DNS name at `127.0.0.1` cannot talk to
  Studio.
- **Origin check (CSRF).** On `POST`, `PUT`, `PATCH` and `DELETE`, a present
  `Origin` header must be Studio's own origin; otherwise 403. Requests without
  `Origin` (for example `curl`) are allowed, but still need the token.
- **No CORS.** No `Access-Control-*` headers are ever sent; the UI is served
  from the same origin as the API.
- **Browser hardening headers on every response:** `X-Content-Type-Options:
  nosniff`, `Cross-Origin-Opener-Policy: same-origin`,
  `Cross-Origin-Embedder-Policy: require-corp`,
  `Cross-Origin-Resource-Policy: same-origin`, `Referrer-Policy: no-referrer`,
  and a Content-Security-Policy that only allows same-origin resources plus
  `blob:` workers and `'wasm-unsafe-eval'` for WebAssembly.
- **No API docs page.** `/docs` and `/redoc` are disabled; the OpenAPI schema is
  at `/api/v1/openapi.json` and needs the token.
- **One process, one worker, no telemetry.**

### Listening beyond this machine

`host=0.0.0.0` (or any non-loopback address) is refused unless you also pass
`token=`, and Studio prints a warning at start. In this mode `Host` values that
are IP addresses are accepted too (an IP address cannot be a DNS-rebinding
vector), but DNS names are still refused. Traffic is **plain HTTP**: the token
and your images are visible to anyone who can observe the network. Use it only
on a network you trust, or put your own TLS-terminating reverse proxy in front
(not covered here).

## Limits

These are constants in `nitid/studio/settings.py`, enforced by the endpoints
that accept images (added later):

| Limit | Value |
|---|---|
| Image upload size | 25 MB (`MAX_IMAGE_BYTES`) |
| Image size in pixels | 20 megapixels (`MAX_IMAGE_PIXELS`) |

## Serving the web UI

The compiled frontend lives in `nitid/studio/static/` and is shipped inside the
wheel (`package-data`). Rules:

- `/` and any unknown path that is not under `/api` and has no file extension
  return `index.html`, so client-side routes work on reload.
- Files under `assets/` (content-hashed by the bundler) are sent with
  `Cache-Control: public, max-age=31536000, immutable`. `index.html` and other
  files use `no-cache`.
- A missing file with an extension (for example `/assets/old.js`) is a plain
  404, never `index.html`. Paths that escape the static directory, including
  through symlinks, are not served.
- If `static/index.html` does not exist, `/` returns a plain-text 503 explaining
  that the UI is not built.

## API (v1)

Only `GET /api/v1/health` is public. Every other endpoint needs the token. The
examples below use `TOKEN` for the token printed at start (the part after `#t=`).

```bash
curl http://127.0.0.1:8000/api/v1/health
# {"status":"ok","nitid_version":"0.9.6","api_version":"1"}

curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/openapi.json
```

Errors always use the contract envelope
`{"error": {"code": "snake_case", "message": "..."}}`. The shape of every response is
specified in `studio/api-contract/openapi.json`.

### Devices and models

#### `GET /api/v1/devices`

Which hardware can run inference. PyTorch CPU is always listed. CUDA devices
appear only if `torch.cuda.is_available()`. OpenVINO is looked up lazily: if it
is not installed, or reports no device, you get `"available": false` and an empty
list, never an error.

```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/devices
# {"torch":{"cpu":true,"cuda":[{"index":0,"name":"NVIDIA GeForce RTX 4070"}]},
#  "openvino":{"available":true,"devices":["CPU","GPU","NPU"]}}
```

#### `GET /api/v1/models`

A flat list of `(id, task)` pairs from two sources:

- **`registry`**: official pretrained weights, from `dfine/utils/downloads.py`.
  Detection: `nitid1s`, `nitid1m`, `nitid1l`, `nitid1x`. Instance segmentation:
  `nitid1n`, `nitid1s`, `nitid1m`, `nitid1l`, `nitid1x`. `downloaded` is `true`
  when the checkpoint file already exists in the models directory. Listing never
  downloads anything.
- **`local`**: your own `*.pth` checkpoints, found directly inside the models
  directory (`models=PATH`, not recursive). Each file is opened with
  `torch.load(weights_only=True, mmap=True)`, which cannot run code from the file
  and does not read the weights into memory; only the embedded `config`,
  `names` and task are used. `size` is inferred from the architecture (`null` if
  it matches no known size). Unreadable, corrupt, non-wrapped or semantic-task
  files are skipped (and logged), as are files over 2 GiB
  (`MAX_LOCAL_CHECKPOINT_BYTES`), more than 500 files
  (`MAX_LOCAL_SCAN_FILES`), the official downloads themselves (already listed as
  `registry`) and files named like an official id (for example `nitid1s.pth`),
  because that id would be ambiguous.

```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/models
# [{"id":"nitid1s","task":"detect","size":"s","downloaded":true,"source":"registry"},
#  ...,
#  {"id":"my-detector","task":"detect","size":"s","downloaded":true,
#   "source":"local","path":"/home/user/models/my-detector.pth"}]
```

!!! note "`nitid1n` for detection is not listed (decision D8)"
    There are no official nano detection weights in the registry, and
    `NITID("nitid1n", task="detect")` is refused by the library. The pair is hidden
    by the constant `EXPOSE_NANO_DETECT = False` in `nitid/studio/registry.py`.
    `nitid1n` for `segment` is available.

#### `POST /api/v1/models/{id}/download?task=detect|segment`

Starts downloading the official checkpoint in the background and returns
`202` with a job id. Only canonical registry ids are accepted: no aliases, no
URLs, no paths. The source URL and the file name come from the registry, never
from the request. The file is written into the models directory through a staging
folder and moved into place when complete, so a half-written file never shows as
downloaded.

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/api/v1/models/nitid1s/download?task=detect"
# {"job_id":"j_3f9c1a2b4d5e"}
```

| Status | `code` | When |
|---|---|---|
| 202 | | Download started. |
| 404 | `not_found` | Unknown id, or no official weights for that task. |
| 409 | `conflict` | The id is a local model, or this `(id, task)` is already downloading. |
| 422 | `invalid_params` | `task` missing or not `detect`/`segment`. |

#### `GET /api/v1/jobs/{job_id}`

```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/jobs/j_3f9c1a2b4d5e
# {"status":"running","progress":0.0}
# {"status":"done","progress":1.0}
# {"status":"failed","progress":0.0,"error":"Network error while downloading the checkpoint."}
```

`status` is `pending`, `running`, `done` or `failed`. The downloader has no
progress hook, so `progress` jumps from `0` to `1` when the file is in place. The
`error` text is deliberately generic (network, disk, checksum, ...): details
such as paths or URLs are logged by the server, never sent to the client. Jobs
live in memory, at most two run at once, and a finished job is forgotten one hour
later (`JOB_TTL_SECONDS` in `nitid/studio/jobs.py`); asking for it afterwards
returns 404. Jobs do not survive a restart. A downloaded checkpoint does.

### ONNX for the browser engine

The browser engine (`studio/src/engine`, see [Studio engine](studio-engine.md)) runs the model
itself with ONNX Runtime Web. These two endpoints give it the file and the facts needed to use it.
Both follow `studio/api-contract/openapi.json` (`/onnx/{model}` and `/onnx/{model}/meta`).

| Query parameter | Required | Meaning |
|---|---|---|
| `task` | yes | `detect` or `segment`. The same id can exist for both. |
| `imgsz` | no | Square input size: a multiple of 32 from 32 to 4096. Default: the checkpoint's own `eval_spatial_size` (640 for the official models). |

The model must be known and **already downloaded** (see `POST /models/{id}/download`): these
endpoints never touch the network. Local `*.pth` checkpoints work too, by file stem.

#### `GET /api/v1/onnx/{model}?task=detect|segment[&imgsz=N]`

Streams the ONNX file (`application/octet-stream`, with `Content-Length`). The first request for a
given model exports it from the checkpoint, which takes from a few seconds (nano/small on a
laptop) to minutes (`nitid1x` on a slow CPU); the HTTP request simply stays open meanwhile, so use
a generous timeout. Later requests are served from the cache.

What you get is always the same shape of file: batch fixed to 1, input `images`
`float32 [1, 3, S, S]` (RGB, 0-1, plain resize without letterbox), post-processed outputs
(`labels`, `boxes`, `scores` and, for `segment`, `masks`) and the descriptor (`nitid_meta`)
embedded in the file's `metadata_props`. Boxes are pixels of the *model input*, so rescale them to
the original image.

```bash
# Download the weights once (see "Devices and models" above), then:
curl -H "Authorization: Bearer $TOKEN" -o nitid1s-detect.onnx -D - \
  "http://127.0.0.1:8000/api/v1/onnx/nitid1s?task=detect"
# HTTP/1.1 200 OK
# content-type: application/octet-stream
# content-length: 41683291
# etag: "9f2c...e1"          <- sha256 of the body, quoted
# cache-control: private, no-cache

sha256sum nitid1s-detect.onnx      # 9f2c...e1, the same value as the ETag
```

**ETag and revalidation.** The `ETag` is a strong validator: the lowercase sha256 of the body,
inside double quotes. Send it back in `If-None-Match` to avoid downloading again; a match returns
`304 Not Modified` with no body (`*` and weak `W/"..."` forms are accepted as well). The response
has `Cache-Control: private, no-cache`: the browser may keep the file but must ask first, which
is cheap once the export exists.

```bash
curl -i -H "Authorization: Bearer $TOKEN" \
  -H 'If-None-Match: "9f2c...e1"' \
  "http://127.0.0.1:8000/api/v1/onnx/nitid1s?task=detect"
# HTTP/1.1 304 Not Modified
```

**Range.** The contract does not define byte ranges. Because the file is served by Starlette's
`FileResponse`, `Range: bytes=...` requests do work (`206 Partial Content`), but treat that as a
transport nicety, not as part of the API.

#### `GET /api/v1/onnx/{model}/meta?task=detect|segment[&imgsz=N]`

Returns the descriptor as JSON: exactly the `nitid_meta` embedded in the file (schema
`studio/api-contract/onnx-meta.schema.json`) plus `sha256`, the same value as the ETag of the
binary. `sha256` is never embedded in the ONNX, since a file cannot contain its own hash. The
response carries the same `ETag` header. The contract has no `size` field in the meta; take the
size from the `Content-Length` of the binary.

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/api/v1/onnx/nitid1s/meta?task=detect"
# {"schema_version":1,"task":"detect","model":"nitid1s","imgsz":640,
#  "input":{"name":"images","shape":[1,3,640,640],"dtype":"float32","range":"0-1",
#           "color":"RGB","resize":"squash"},
#  "outputs":[{"name":"labels","dtype":"int64"},
#             {"name":"boxes","dtype":"float32","format":"xyxy_input_px"},
#             {"name":"scores","dtype":"float32"}],
#  "names":{"0":"person","1":"bicycle", ...},"topk":300,"default_iou":0.85,
#  "nitid_version":"0.9.6","sha256":"9f2c...e1"}
```

Asking for the meta also triggers the export when the file is not cached yet.

#### Errors

| Status | `code` | When |
|---|---|---|
| 401 | `unauthorized` | Missing or wrong token (central middleware). |
| 403 | `forbidden_origin` | Host or Origin not allowed. |
| 404 | `not_found` | Unknown model, no weights for that `task`, or the checkpoint is not downloaded (the message says which and how to fix it). |
| 422 | `invalid_params` | `task` missing or not `detect`/`segment`; `imgsz` not an integer, not a multiple of 32, or outside 32-4096. |
| 500 | `internal_error` | The export failed. The message names only the exception type; details are in the server log. |

All errors use the envelope `{"error": {"code": "...", "message": "..."}}`.

#### How the export is cached and scheduled

- Exports are written to `<models>/.cache/onnx/` (the catalogue ignores that directory). Delete it
  at any time; the next request exports again. Nothing is evicted automatically.
- The cache key is the model id, task, `imgsz`, the nitid version **and** the checkpoint's size and
  modification time. Upgrading nitid, or replacing a local `.pth`, therefore produces a new file
  instead of serving a stale one. File names are hashes: nothing from the URL becomes a path.
- **One export at a time.** A lock serialises them, because an export is CPU- and memory-heavy.
  Requests for the same file that arrive while it is being exported wait and then reuse the result,
  so ten browser tabs cause one export. Requests for other files queue behind it.
- Exports run on CPU in the server's worker thread pool, so the rest of the API (and the web UI)
  keeps answering while one is in progress.
- The sha256 is computed once per file and remembered in memory (recomputed if the file changes).
- Not provided: uploading your own ONNX. The contract defines no endpoint for it.

## What is not here yet

- The web UI (the `static/` directory is empty in a source checkout).
- Model deletion and management beyond listing and downloading.
- Jobs: prediction, training, and a progress-reporting export job. (ONNX export for the browser
  engine exists, but synchronously: see above.)
- Image upload endpoints and the limits above being enforced.

## For developers

`nitid.studio` is organised so each concern is testable on its own:

| Module | Role |
|---|---|
| `settings.py` | Immutable, validated `StudioSettings`; `key=value` parsing. No FastAPI import. |
| `security.py` | Pure ASGI middleware: Host, Origin, token, headers. No FastAPI import. |
| `static.py` | SPA serving, cache headers, 503 page. |
| `api.py` | `build_v1_router()`; add new `/api/v1` routes here. |
| `devices.py` | Tolerant hardware discovery (torch CPU/CUDA, OpenVINO). |
| `registry.py` | Model catalogue: registry pairs, local `*.pth` scan, staged download, safe error text. |
| `jobs.py` | Thread-safe in-memory job store with TTL and one active job per key. |
| `routes_models.py` | `/devices`, `/models`, `/models/{id}/download`, `/jobs/{job_id}`. |
| `onnx_service.py` | On-demand ONNX export, disk cache, serialisation, sha256 and meta of each file. No FastAPI import. |
| `routes_onnx.py` | `/onnx/{model}` and `/onnx/{model}/meta`: ETag, 304, error envelope. |
| `app.py` | `create_app(settings, static_dir=..., extra_routers=...)`. |
| `launcher.py` | Extra check, socket binding, uvicorn, `nitid studio` entry point. |

Routers added through `extra_routers` are registered before the SPA catch-all
(routes match in registration order). Authentication is applied centrally by the
middleware; routers must not re-implement it.

Run the tests with the extra installed:

```bash
uv run --extra studio pytest tests/unit/studio tests/integration/studio
```

Without FastAPI the app-level tests are skipped; the settings tests still run.
