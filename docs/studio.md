# Nitid Studio

Nitid Studio is a local web app for working with nitid models from your
browser. It runs on your own machine, in a single process started by
`nitid studio`, and sends no telemetry.

!!! note "Status: backend skeleton"
    This release contains the server foundation only: settings, security
    guards, static serving and `GET /api/v1/health`. Model management, jobs
    (prediction, training, export) and the web UI itself arrive in later
    changes. Until the UI is built into your installation, opening the URL
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

Only one endpoint exists today.

```bash
curl http://127.0.0.1:8000/api/v1/health
# {"status":"ok","nitid_version":"0.9.6","api_version":"1"}
```

Authenticated calls look like:

```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/openapi.json
```

## What is not here yet

- The web UI (the `static/` directory is empty in a source checkout).
- Model listing, download and management.
- Jobs: prediction, training, export and their progress reporting.
- Image upload endpoints and the limits above being enforced.

## For developers

`nitid.studio` is organised so each concern is testable on its own:

| Module | Role |
|---|---|
| `settings.py` | Immutable, validated `StudioSettings`; `key=value` parsing. No FastAPI import. |
| `security.py` | Pure ASGI middleware: Host, Origin, token, headers. No FastAPI import. |
| `static.py` | SPA serving, cache headers, 503 page. |
| `api.py` | `build_v1_router()`; add new `/api/v1` routes here. |
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
