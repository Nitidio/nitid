"""FastAPI application factory for Nitid Studio.

Requires the optional ``studio`` extra. Import this module lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from fastapi import APIRouter, FastAPI
from starlette.responses import Response

from nitid.studio.api import build_v1_router
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
) -> FastAPI:
    """Build the Studio app.

    ``extra_routers`` are included before the SPA catch-all route, which would
    otherwise shadow them (routes are matched in registration order).

    ``settings.port`` must be the real listening port (the Host guard compares
    against it); the launcher resolves ``port=0`` before calling this.
    """
    app = StudioApp(
        settings,
        title="Nitid Studio",
        docs_url=None,
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json",  # under /api/v1, so it needs the token
    )
    app.state.settings = settings
    app.include_router(build_v1_router())
    for router in extra_routers:
        app.include_router(router)

    directory = static_dir if static_dir is not None else STATIC_DIR

    def spa(full_path: str) -> Response:
        return spa_response(directory, full_path)

    app.add_api_route("/{full_path:path}", spa, methods=["GET", "HEAD"], include_in_schema=False)

    return app
