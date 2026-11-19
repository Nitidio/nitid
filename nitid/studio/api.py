"""Versioned HTTP API (``/api/v1``).

The routers of each work package (health, models, ONNX) are assembled here. Later ones
add theirs to :func:`build_v1_router` (or call ``router.include_router`` on the returned
router); authentication and Host/Origin checks are applied centrally by
:class:`nitid.studio.security.SecurityMiddleware`, so routers must not re-implement them.
"""

from __future__ import annotations

from fastapi import APIRouter

from nitid.studio.onnx_service import OnnxService
from nitid.studio.routes_models import build_models_router
from nitid.studio.routes_onnx import build_onnx_router
from nitid.studio.settings import API_PREFIX, API_VERSION


def build_v1_router(*, onnx_service: OnnxService | None = None) -> APIRouter:
    """Create the ``/api/v1`` router with the endpoints available today.

    ``onnx_service`` lets tests inject an :class:`OnnxService` with a fake exporter.
    """
    from nitid import __version__

    router = APIRouter(prefix=API_PREFIX)
    router.include_router(build_models_router())  # /devices, /models, /jobs (WP-1.2)
    router.include_router(build_onnx_router(service=onnx_service))  # /onnx/{model}[/meta]

    @router.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe. The only unauthenticated API endpoint."""
        return {"status": "ok", "nitid_version": __version__, "api_version": API_VERSION}

    return router
