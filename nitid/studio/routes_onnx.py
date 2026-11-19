"""``GET /onnx/{model}`` and ``GET /onnx/{model}/meta`` (included by ``build_v1_router``).

Authentication, Host and Origin checks are applied centrally by
:class:`nitid.studio.security.SecurityMiddleware`. Errors use the contract envelope
``{"error": {"code", "message"}}``. The handlers are plain ``def`` so FastAPI runs them in
its thread pool: an export can take minutes and must not block the event loop. The
binary itself is streamed by Starlette's ``FileResponse`` (asynchronous file reads).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from starlette.responses import FileResponse

from nitid.studio import registry
from nitid.studio.onnx_service import (
    ExportFailed,
    InvalidParams,
    OnnxArtifact,
    OnnxService,
    parse_imgsz,
    parse_task,
)
from nitid.studio.routes_models import error_response

#: Revalidate on every use (the body is addressed by its ETag, which is cheap to check).
CACHE_CONTROL = "private, no-cache"


def etag_matches(header_value: str | None, etag: str) -> bool:
    """RFC 9110 ``If-None-Match`` using weak comparison (``W/`` prefixes are ignored)."""
    if header_value is None:
        return False
    if header_value.strip() == "*":
        return True
    wanted = etag.removeprefix("W/")
    for candidate in header_value.split(","):
        if candidate.strip().removeprefix("W/") == wanted:
            return True
    return False


def build_onnx_router(*, service: OnnxService | None = None) -> APIRouter:
    """Create the router. ``service`` can be injected for tests (fake exporter)."""
    onnx_service = service if service is not None else OnnxService()
    router = APIRouter()

    def resolve(
        request: Request, model: str, task: str | None, imgsz: str | None
    ) -> OnnxArtifact | JSONResponse:
        try:
            parsed_task = parse_task(task)
            parsed_imgsz = parse_imgsz(imgsz)
            return onnx_service.get(
                request.app.state.settings.models_dir, model, parsed_task, parsed_imgsz
            )
        except InvalidParams as exc:
            return error_response(422, "invalid_params", str(exc))
        except registry.ModelNotAvailable as exc:
            return error_response(404, "not_found", str(exc))
        except ExportFailed as exc:
            return error_response(500, "internal_error", str(exc))

    @router.get("/onnx/{model}/meta", response_model=None)
    def get_onnx_meta(
        model: str, request: Request, task: str | None = None, imgsz: str | None = None
    ) -> JSONResponse | dict[str, Any]:
        result = resolve(request, model, task, imgsz)
        if isinstance(result, JSONResponse):
            return result
        return JSONResponse(
            result.meta_with_sha256(),
            headers={"ETag": result.etag, "Cache-Control": CACHE_CONTROL},
        )

    @router.get("/onnx/{model}", response_model=None)
    def get_onnx(
        model: str, request: Request, task: str | None = None, imgsz: str | None = None
    ) -> Response:
        result = resolve(request, model, task, imgsz)
        if isinstance(result, JSONResponse):
            return result
        headers = {"ETag": result.etag, "Cache-Control": CACHE_CONTROL}
        if etag_matches(request.headers.get("if-none-match"), result.etag):
            return Response(status_code=304, headers=headers)
        return FileResponse(result.path, media_type="application/octet-stream", headers=headers)

    return router
