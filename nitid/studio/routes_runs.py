"""``/runs`` routes (included by ``build_v1_router``).

Authentication, Host and Origin checks are applied centrally by
:class:`nitid.studio.security.SecurityMiddleware`. Errors use the contract envelope
``{"error": {"code", "message"}}``. Handlers are plain ``def`` so FastAPI runs them in
its thread pool (image inspection and file reads block).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from nitid.studio import devices as devices_module
from nitid.studio import runs
from nitid.studio.routes_models import error_response
from nitid.studio.settings import MAX_IMAGE_BYTES


def _run_not_found(run_id: str) -> JSONResponse:
    return error_response(404, "not_found", f"Run {runs.echo(run_id)} does not exist.")


def _read_upload(upload: UploadFile) -> bytes:
    """Read at most the limit plus one byte, so a huge body is never fully loaded."""
    return upload.file.read(MAX_IMAGE_BYTES + 1)


def build_runs_router(*, service: runs.RunService | None = None) -> APIRouter:
    """Create the router. ``service`` can be injected for tests."""
    svc = service if service is not None else runs.RunService()
    router = APIRouter()
    router.run_service = svc  # type: ignore[attr-defined]

    @router.post("/runs", status_code=202, response_model=None)
    def create_run(
        request: Request,
        files: list[UploadFile] = File(...),
        params: str = Form(...),
    ) -> JSONResponse | dict[str, str]:
        try:
            parsed = runs.parse_run_params(params)
            if not files:
                raise runs.ParamsError("At least one file is required.")
            if len(files) > runs.MAX_FILES_PER_RUN:
                raise runs.ParamsError(f"At most {runs.MAX_FILES_PER_RUN} files per run.")
            runs.validate_device(parsed, devices_module.collect_devices())
            model = runs.resolve_model(
                request.app.state.settings.models_dir, parsed.model, parsed.task
            )
        except runs.ParamsError as error:
            return error_response(422, "invalid_params", str(error))
        except runs.ModelNotDownloaded as error:
            return error_response(409, "conflict", str(error))

        images: list[tuple[str, bytes, runs.ImageInfo]] = []
        for index, upload in enumerate(files):
            try:
                data = _read_upload(upload)
                info = runs.inspect_image(data)
            except runs.PayloadTooLarge as error:
                return error_response(413, "payload_too_large", str(error))
            except runs.UnsupportedImage as error:
                return error_response(415, "unsupported_media_type", str(error))
            images.append((runs.safe_name(upload.filename, index), data, info))

        return {"id": svc.submit(parsed, model, images)}

    @router.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [snapshot.summary() for snapshot in svc.store.list()]

    @router.get("/runs/{run_id}", response_model=None)
    def get_run(run_id: str) -> JSONResponse | dict[str, Any]:
        snapshot = svc.store.get(run_id)
        if snapshot is None:
            return _run_not_found(run_id)
        return snapshot.payload()

    @router.delete("/runs/{run_id}", status_code=204, response_model=None)
    def delete_run(run_id: str) -> Response:
        if not svc.store.delete(run_id):
            return _run_not_found(run_id)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    def _item_index(raw: str) -> int | None:
        return int(raw) if raw.isascii() and raw.isdigit() else None

    @router.get("/runs/{run_id}/items/{n}/image", response_model=None)
    def get_item_image(run_id: str, n: str) -> Response:
        snapshot = svc.store.get(run_id)
        index = _item_index(n)
        if snapshot is None:
            return _run_not_found(run_id)
        if index is None or index >= len(snapshot.uploads):
            return error_response(404, "not_found", f"Item {runs.echo(n)} does not exist.")
        upload = snapshot.uploads[index]
        try:
            data = upload.path.read_bytes()
        except OSError:
            return _run_not_found(run_id)  # deleted while we were reading
        return Response(
            data, media_type=upload.info.media_type, headers={"Cache-Control": "no-store"}
        )

    @router.get("/runs/{run_id}/items/{n}.json", response_model=None)
    def get_item(run_id: str, n: str) -> JSONResponse | dict[str, Any]:
        snapshot = svc.store.get(run_id)
        if snapshot is None:
            return _run_not_found(run_id)
        index = _item_index(n)
        if index is None or index >= len(snapshot.items):
            return error_response(404, "not_found", f"Item {runs.echo(n)} does not exist.")
        return snapshot.items[index]

    @router.get("/runs/{run_id}/export.csv", response_model=None)
    def export_csv(run_id: str) -> Response:
        snapshot = svc.store.get(run_id)
        if snapshot is None:
            return _run_not_found(run_id)
        return Response(
            runs.items_to_csv(snapshot.items),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{run_id}.csv"',
                "Cache-Control": "no-store",
            },
        )

    @router.get("/runs/{run_id}/export.zip", response_model=None)
    def export_zip(run_id: str) -> Response:
        snapshot = svc.store.get(run_id)
        if snapshot is None:
            return _run_not_found(run_id)
        try:
            body = runs.build_zip(snapshot)
        except OSError:
            return _run_not_found(run_id)
        return Response(
            body,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{run_id}.zip"',
                "Cache-Control": "no-store",
            },
        )

    return router
