from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from web.api.config import settings
from web.api.database import SessionLocal
from web.api.models.run import Run
from web.api.models.user import User
from web.api.services.auth import decode_token

router = APIRouter()


def _get_user_from_token(token: str, db: Session) -> User:
    username = decode_token(token)
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@router.get("/files/{run_id}/{filename}")
def serve_file(
    run_id: int,
    filename: str,
    token: str = Query(..., description="JWT access token"),
):
    db = SessionLocal()
    try:
        user = _get_user_from_token(token, db)
        run = db.query(Run).filter(Run.id == run_id, Run.user_id == user.id).first()
        if run is None:
            raise HTTPException(status_code=404, detail="Not found")
        path = Path(settings.results_dir) / str(run_id) / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(path), media_type="image/jpeg")
    finally:
        db.close()
