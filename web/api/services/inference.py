from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2

from web.api.config import settings
from web.api.models.run import Run, RunStatus
from web.api.models.run_item import RunItem
from web.api.schemas.run import RunCreate
from web.api.services.telemetry import sample_cpu_percent, sample_device_memory_kib

# Lazy import of DFINE — loaded on first model access, not at module import time.
_model_cache: dict[tuple[str, str, str | None], object] = {}


def get_model(model_name: str, backend: str = "torch", device: str | None = None):
    from dfine import DFINE

    model_path = str(Path(settings.models_dir) / model_name)
    cache_key = (model_path, backend, device)
    if cache_key not in _model_cache:
        # device=None lets DFINE apply its own default: auto-select cuda when
        # available (backend="torch") or prefer NPU, then GPU, then CPU
        # (backend="openvino") — do not hardcode a device here.
        _model_cache[cache_key] = DFINE(model_path, backend=backend, device=device, verbose=False)
    return _model_cache[cache_key]


def _snapshot_path(run_id: int, item_idx: int) -> Path:
    out_dir = Path(settings.results_dir) / str(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"item_{item_idx:05d}.jpg"


def run_inference(
    run_id: int,
    uploaded_paths: list[Path],
    params: RunCreate,
    input_type: str,
    session_factory,
) -> None:
    db = session_factory()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        run.status = RunStatus.running
        db.commit()

        model = get_model(params.model_name, backend=params.backend, device=params.device)
        classes = params.classes if params.classes else None

        if input_type == "images":
            _process_images(db, run_id, uploaded_paths, model, params, classes)
        else:
            _process_video(db, run_id, uploaded_paths[0], model, params, classes)

        run = db.query(Run).filter(Run.id == run_id).first()
        run.status = RunStatus.done
        run.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception:
        try:
            db.rollback()
            run = db.query(Run).filter(Run.id == run_id).first()
            if run:
                run.status = RunStatus.failed
                run.error_msg = traceback.format_exc(limit=10)
                run.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _process_images(db, run_id, paths, model, params, classes):
    sample_cpu_percent()  # prime the "since last call" window
    results = model.predict(
        source=[str(p) for p in paths],
        conf=params.conf,
        imgsz=params.imgsz,
        classes=classes,
        verbose=False,
    )
    cpu_percent = sample_cpu_percent()
    device_memory_kib = sample_device_memory_kib(params.backend, params.device)
    for idx, result in enumerate(results):
        snap = _snapshot_path(run_id, idx)
        annotated = result.plot()
        cv2.imwrite(str(snap), annotated)
        item = RunItem(
            run_id=run_id,
            source_path=str(paths[idx]),
            result_snapshot_path=snap.name,
            detections_json=json.dumps(result.to_json()),
            speed_json=json.dumps(result.speed),
            cpu_percent=cpu_percent,
            device_memory_kib=device_memory_kib,
        )
        db.add(item)
        db.commit()


def _process_video(db, run_id, video_path, model, params, classes):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    item_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % params.frame_step == 0:
            sample_cpu_percent()  # prime the "since last call" window
            # frame is HWC BGR numpy array — DFINE accepts this directly
            results = model.predict(
                source=frame,
                conf=params.conf,
                imgsz=params.imgsz,
                classes=classes,
                verbose=False,
            )
            cpu_percent = sample_cpu_percent()
            device_memory_kib = sample_device_memory_kib(params.backend, params.device)
            result = results[0]
            snap = _snapshot_path(run_id, item_idx)
            cv2.imwrite(str(snap), result.plot())
            item = RunItem(
                run_id=run_id,
                source_path=str(video_path),
                result_snapshot_path=snap.name,
                detections_json=json.dumps(result.to_json()),
                speed_json=json.dumps(result.speed),
                frame_idx=frame_idx,
                cpu_percent=cpu_percent,
                device_memory_kib=device_memory_kib,
            )
            db.add(item)
            db.commit()
            item_idx += 1
        frame_idx += 1
    cap.release()
