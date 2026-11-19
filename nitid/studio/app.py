"""FastAPI application factory for Nitid Studio.

Requires the optional ``studio`` extra. Import this module lazily.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import PlainTextResponse, Response

from nitid.studio.api import build_v1_router
from nitid.studio.routes_models import error_response
from nitid.studio.runs import RunService
from nitid.studio.security import SecurityMiddleware
from nitid.studio.settings import API_PREFIX, StudioSettings
from nitid.studio.static import STATIC_DIR, spa_response


class StudioApp(FastAPI):
    """FastAPI app whose outermost layer is the security middleware.

    Wrapping the *whole* stack (rather than using ``add_middleware``) means even
    unhandled-exception 500 responses, produced by Starlette's outer error
    middleware, carry the security headers.
    """

    def __init__(self, settings: StudioSettings, **kwargs: object) -> None:
        self.studio_settings = settings
        super().__init__(**kwargs)  # type: ignore[arg-type]

    def build_middleware_stack(self):  # type: ignore[no-untyped-def]
        settings = self.studio_settings
        return SecurityMiddleware(
            super().build_middleware_stack(),
            token=settings.api_token,
            port=settings.port,
            allow_ip_hosts=not settings.is_loopback,
        )


def create_app(
    settings: StudioSettings,
    *,
    static_dir: Path | None = None,
    extra_routers: Sequence[APIRouter] = (),
    run_service: RunService | None = None,
) -> FastAPI:
    """Build the Studio app.

    ``extra_routers`` are included before the SPA catch-all route, which would
    otherwise shadow them (routes are matched in registration order).

    ``settings.port`` must be the real listening port (the Host guard compares
    against it); the launcher resolves ``port=0`` before calling this.
    """
    service = run_service if run_service is not None else RunService()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            service.close()  # stop the worker, delete uploaded images

    app = StudioApp(
        settings,
        lifespan=lifespan,
        title="Nitid Studio",
        docs_url=None,
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json",  # under /api/v1, so it needs the token
    )
    app.state.settings = settings
    app.include_router(build_v1_router(run_service=service))
    for router in extra_routers:
        app.include_router(router)

    app.state.run_service = service
    _install_error_handlers(app)

    directory = static_dir if static_dir is not None else STATIC_DIR

    def spa(full_path: str) -> Response:
        return spa_response(directory, full_path)

    app.add_api_route("/{full_path:path}", spa, methods=["GET", "HEAD"], include_in_schema=False)

    return app


_HTTP_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
}


def _is_api(request: Request) -> bool:
    path = request.url.path
    return path == API_PREFIX or path.startswith(API_PREFIX + "/")


def _install_error_handlers(app: FastAPI) -> None:
    """Make framework-generated errors under ``/api/v1`` use the contract envelope.

    Other paths (the SPA) keep Starlette's defaults.
    """

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, StarletteHTTPException)
        if not _is_api(request):
            from fastapi.exception_handlers import http_exception_handler

            return await http_exception_handler(request, exc)
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        response = error_response(exc.status_code, code, str(exc.detail))
        for key, value in (exc.headers or {}).items():
            response.headers[key] = value
        return response

    async def validation_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, RequestValidationError)
        if not _is_api(request):
            from fastapi.exception_handlers import request_validation_exception_handler

            return await request_validation_exception_handler(request, exc)
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(part) for part in first.get("loc", ())[1:])
        message = f"Invalid request: {where or 'body'} ({first.get('type', 'invalid')})."
        return error_response(422, "invalid_params", message)

    async def server_error(request: Request, exc: Exception) -> Response:
        if not _is_api(request):
            return PlainTextResponse("Internal Server Error", status_code=500)
        return error_response(500, "internal_error", "Internal server error.")

    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(Exception, server_error)
