"""``/devices``, ``/models`` and ``/jobs`` routes (included by ``build_v1_router``).

Authentication, Host and Origin checks are applied centrally by
:class:`nitid.studio.security.SecurityMiddleware`; nothing here re-implements them.
Errors use the contract envelope ``{"error": {"code", "message"}}``.

The handlers are plain ``def`` so FastAPI runs them in its thread pool: hardware
probing and checkpoint inspection may block.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from nitid.studio import devices as devices_module
from nitid.studio import registry
from nitid.studio.jobs import JobConflict, JobStore, ProgressCallback

#: ``(entry, models_dir, progress) -> None`` -- replaceable in tests.
Downloader = Callable[[registry.RegistryEntry, Path, ProgressCallback], object]

_MAX_ECHO = 64


def error_response(status: int, code: str, message: str) -> JSONResponse:
    """Contract error envelope (``ErrorResponse``)."""
    return JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status,
        headers={"Cache-Control": "no-store"},
    )


def _echo(value: str) -> str:
    """Quote client input for an error message, bounded in length."""
    shown = value if len(value) <= _MAX_ECHO else value[:_MAX_ECHO] + "..."
    return repr(shown)


def build_models_router(
    *, store: JobStore | None = None, downloader: Downloader | None = None
) -> APIRouter:
    """Create the router. ``store`` and ``downloader`` can be injected for tests.

    Each call gets its own :class:`JobStore` unless one is passed, so two apps never
    share jobs. When ``downloader`` is ``None`` the function
    :func:`nitid.studio.registry.download_registry_model` is looked up at call time.
    """
    jobs = store if store is not None else JobStore()
    router = APIRouter()

    @router.get("/devices")
    def list_devices() -> dict[str, Any]:
        return devices_module.collect_devices()

    @router.get("/models")
    def list_models(request: Request) -> list[dict[str, Any]]:
        return registry.list_models(request.app.state.settings.models_dir)

    @router.post("/models/{model_id}/download", status_code=202, response_model=None)
    def download_model(
        model_id: str, request: Request, task: str | None = None
    ) -> JSONResponse | dict[str, str]:
        if task is None or task not in registry.STUDIO_TASKS:
            allowed = ", ".join(registry.STUDIO_TASKS)
            return error_response(
                422, "invalid_params", f"Query parameter task is required: {allowed}."
            )

        models_dir: Path = request.app.state.settings.models_dir
        entry = registry.find_registry_entry(model_id, task)
        if entry is None:
            if model_id in registry.registry_ids():
                return error_response(
                    404,
                    "not_found",
                    f"No official weights for {_echo(model_id)} with task {task!r}.",
                )
            if model_id in registry.local_model_ids(models_dir):
                return error_response(409, "conflict", "Model is local; nothing to download.")
            return error_response(404, "not_found", f"Model {_echo(model_id)} does not exist.")

        def run(progress: ProgressCallback) -> None:
            fn = downloader if downloader is not None else registry.download_registry_model
            fn(entry, models_dir, progress)

        try:
            job_id = jobs.submit(
                (entry.id, entry.task), run, describe_error=registry.describe_download_error
            )
        except JobConflict:
            return error_response(
                409, "conflict", f"A download of {entry.id} ({entry.task}) is already running."
            )
        return {"job_id": job_id}

    @router.get("/jobs/{job_id}", response_model=None)
    def get_job(job_id: str) -> JSONResponse | dict[str, object]:
        snapshot = jobs.get(job_id)
        if snapshot is None:
            return error_response(404, "not_found", f"Job {_echo(job_id)} does not exist.")
        return snapshot.to_payload()

    return router
