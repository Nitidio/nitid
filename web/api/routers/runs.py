from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from web.api.config import settings
from web.api.database import SessionLocal
from web.api.deps import get_current_user, get_db
from web.api.models.run import Run, RunStatus
from web.api.models.user import User
from web.api.schemas.run import RunCreate, RunDetail, RunOut
from web.api.services.inference import run_inference

router = APIRouter()

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v"}


def _detect_input_type(filenames: list[str]) -> str:
    for name in filenames:
        if Path(name).suffix.lower() in _VIDEO_SUFFIXES:
            return "video"
    return "images"


def _save_uploads(files: list[UploadFile], run_id: int) -> list[Path]:
    upload_dir = Path(settings.uploads_dir) / str(run_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, f in enumerate(files):
        suffix = Path(f.filename or "file").suffix
        dest = upload_dir / f"{i:05d}{suffix}"
        dest.write_bytes(f.file.read())
        saved.append(dest)
    return saved


@router.post("/runs", response_model=RunOut, status_code=202)
def create_run(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    params: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run_params = RunCreate.model_validate_json(params)
    input_type = _detect_input_type([f.filename or "" for f in files])

    run = Run(
        user_id=current_user.id,
        model_name=run_params.model_name,
        conf=run_params.conf,
        imgsz=run_params.imgsz,
        classes=json.dumps(run_params.classes) if run_params.classes else None,
        frame_step=run_params.frame_step,
        status=RunStatus.pending,
        input_type=input_type,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    saved_paths = _save_uploads(files, run.id)

    background_tasks.add_task(
        run_inference, run.id, saved_paths, run_params, input_type, SessionLocal
    )
    return RunOut.model_validate(run)


@router.get("/runs", response_model=list[RunOut])
def list_runs(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    runs = (
        db.query(Run)
        .filter(Run.user_id == current_user.id)
        .order_by(Run.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [RunOut.model_validate(r) for r in runs]


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = db.query(Run).filter(Run.id == run_id, Run.user_id == current_user.id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunDetail.model_validate(run)


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = db.query(Run).filter(Run.id == run_id, Run.user_id == current_user.id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    db.delete(run)
    db.commit()
    for subdir in ("uploads", "results"):
        d = Path(getattr(settings, f"{subdir}_dir")) / str(run_id)
        if d.exists():
            shutil.rmtree(d)
