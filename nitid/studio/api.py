"""Versioned HTTP API (``/api/v1``).

Only ``GET /health`` exists in the skeleton. Later work packages add their
routers to :func:`build_v1_router` (or call ``router.include_router`` on the
returned router); authentication and Host/Origin checks are applied centrally by
:class:`nitid.studio.security.SecurityMiddleware`, so routers must not re-implement them.
"""

from __future__ import annotations

from fastapi import APIRouter

from nitid.studio.settings import API_PREFIX, API_VERSION


def build_v1_router() -> APIRouter:
    """Create the ``/api/v1`` router with the endpoints available today."""
    from nitid import __version__

    router = APIRouter(prefix=API_PREFIX)

    @router.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe. The only unauthenticated API endpoint."""
        return {"status": "ok", "nitid_version": __version__, "api_version": API_VERSION}

    return router
