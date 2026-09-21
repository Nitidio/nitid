from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import cast

import yaml
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from dfine.utils.dataset_converter import TargetFormat, convert_dataset
from web.api.deps import get_current_user
from web.api.models.user import User

router = APIRouter()


def _extract_archive(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def _find_config(root: Path, requested: str | None) -> Path:
    if requested:
        relative = Path(requested)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("config must be a relative path inside the archive")
        config = root / relative
        if not config.is_file():
            raise ValueError(f"Config not found in archive: {requested}")
        return config

    configs = sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")])
    if len(configs) != 1:
        raise ValueError("Archive must contain one YAML config, or specify config=path/to/data.yml")
    return configs[0]


@router.post("/datasets/convert")
def convert_uploaded_dataset(
    background_tasks: BackgroundTasks,
    archive: UploadFile = File(...),
    target: str = Form(...),
    config: str | None = Form(default=None),
    _current_user: User = Depends(get_current_user),
):
    if target.lower() not in {"coco", "yolo"}:
        raise HTTPException(status_code=422, detail="target must be coco or yolo")
    if Path(archive.filename or "").suffix.lower() != ".zip":
        raise HTTPException(status_code=422, detail="Upload a .zip dataset archive")

    temporary = Path(tempfile.mkdtemp(prefix="nitid-convert-"))
    background_tasks.add_task(shutil.rmtree, temporary, True)
    try:
        archive_path = temporary / "source.zip"
        with archive_path.open("wb") as stream:
            shutil.copyfileobj(archive.file, stream)
        extracted = temporary / "source"
        extracted.mkdir()
        _extract_archive(archive_path, extracted)
        config_path = _find_config(extracted, config)
        config_payload = yaml.safe_load(config_path.read_text())
        if not isinstance(config_payload, dict) or not isinstance(config_payload.get("path"), str):
            raise ValueError("Data YAML must define a dataset 'path'")
        dataset_root = Path(config_payload["path"]).expanduser()
        dataset_root = (
            dataset_root.resolve()
            if dataset_root.is_absolute()
            else (config_path.parent / dataset_root).resolve()
        )
        extracted_root = extracted.resolve()
        if dataset_root != extracted_root and extracted_root not in dataset_root.parents:
            raise ValueError("Dataset path must stay inside the uploaded archive")
        converted = temporary / "converted"
        convert_dataset(config_path, converted, cast(TargetFormat, target.lower()))
        zip_path = Path(
            shutil.make_archive(str(temporary / f"dataset-{target.lower()}"), "zip", converted)
        )
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(error)) from None

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_path.name,
        background=background_tasks,
    )
