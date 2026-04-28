from __future__ import annotations

from pathlib import Path

import torch
from fastapi import APIRouter, Depends, HTTPException

from web.api.config import settings
from web.api.deps import get_current_user
from web.api.models.user import User

router = APIRouter()


@router.get("/models", response_model=list[str])
def list_models(current_user: User = Depends(get_current_user)):
    models_dir = Path(settings.models_dir)
    if not models_dir.exists():
        return []
    return sorted(p.name for p in models_dir.glob("*.pth") if p.is_file())


@router.get("/models/{model_name}/classes", response_model=dict[str, str])
def get_model_classes(model_name: str, current_user: User = Depends(get_current_user)):
    path = Path(settings.models_dir) / model_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    names: dict = ckpt.get("names", {})
    return {str(k): v for k, v in names.items()}
