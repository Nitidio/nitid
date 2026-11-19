"""Serving the compiled single-page app from ``nitid/studio/static/``.

The frontend is compiled into that directory by the release build; a source
checkout has no ``static/index.html`` until the UI is built, and Studio then
answers with a clear 503 page instead of failing obscurely.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response

STATIC_DIR = Path(__file__).resolve().parent / "static"

IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
NO_CACHE = "no-cache"

# Bundlers emit content-hashed files under ``assets/``; those can be cached forever.
HASHED_ASSETS_DIR = "assets"

_MEDIA_TYPES = {
    ".wasm": "application/wasm",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".json": "application/json",
    ".webmanifest": "application/manifest+json",
}

UI_NOT_BUILT_MESSAGE = """\
Nitid app is running, but its web UI is not built into this installation.

The API is available (try /api/v1/health), but there is no static/index.html to serve.

- If you installed a released wheel, reinstall it: pip install -U "nitid[studio]"
- If you work from a source checkout, build the frontend into nitid/studio/static/
  (see docs/studio.md).
"""


def media_type_for(path: Path) -> str | None:
    """Content type for ``path``, with explicit overrides for web assets."""
    override = _MEDIA_TYPES.get(path.suffix.lower())
    return override or mimetypes.guess_type(path.name)[0]


def _resolve_inside(root: Path, rel_path: str) -> Path | None:
    """Resolve ``rel_path`` under ``root``; ``None`` if it escapes or is not a file."""
    try:
        candidate = (root / rel_path).resolve()
        if not candidate.is_relative_to(root):
            return None
        return candidate if candidate.is_file() else None
    except (ValueError, OSError):
        return None


def spa_response(static_dir: Path, rel_path: str) -> Response:
    """Answer a GET for ``rel_path`` (no leading slash) outside ``/api/v1``."""
    if rel_path == "api" or rel_path.startswith("api/"):
        return JSONResponse(
            {"error": {"code": "not_found", "message": "Not found."}},
            status_code=404,
            headers={"Cache-Control": "no-store"},
        )

    root = static_dir.resolve()
    if rel_path:
        file_path = _resolve_inside(root, rel_path)
        if file_path is not None:
            relative = file_path.relative_to(root)
            hashed = relative.parts[0] == HASHED_ASSETS_DIR and len(relative.parts) > 1
            return FileResponse(
                file_path,
                media_type=media_type_for(file_path),
                headers={"Cache-Control": IMMUTABLE_CACHE if hashed else NO_CACHE},
            )
        if "." in rel_path.rsplit("/", 1)[-1]:
            # A missing file with an extension is a broken asset link, not an SPA route.
            return PlainTextResponse("Not Found", status_code=404)

    index = _resolve_inside(root, "index.html")
    if index is None:
        return PlainTextResponse(
            UI_NOT_BUILT_MESSAGE, status_code=503, headers={"Cache-Control": "no-store"}
        )
    return FileResponse(
        index, media_type="text/html; charset=utf-8", headers={"Cache-Control": NO_CACHE}
    )
