from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from web.api.schemas.run_item import DetectionEntry, RunItemOut


class RunCreate(BaseModel):
    model_name: str
    conf: float = Field(default=0.5, ge=0.01, le=1.0)
    imgsz: int = Field(default=640, ge=32, le=1920)
    classes: list[int] | None = None
    frame_step: int = Field(default=30, ge=1)
    backend: Literal["torch", "openvino"] = "torch"
    device: str | None = None


class RunOut(BaseModel):
    id: int
    model_name: str
    conf: float
    imgsz: int
    classes: list[int] | None
    frame_step: int
    backend: str
    device: str | None
    status: str
    input_type: str
    created_at: datetime
    completed_at: datetime | None
    error_msg: str | None
    item_count: int

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def _parse_fields(cls, data: Any) -> Any:
        if hasattr(data, "__dict__"):
            # SQLAlchemy ORM object — convert to dict-like
            d: dict = {
                "id": data.id,
                "model_name": data.model_name,
                "conf": data.conf,
                "imgsz": data.imgsz,
                "classes": json.loads(data.classes) if data.classes else None,
                "frame_step": data.frame_step,
                "backend": data.backend,
                "device": data.device,
                "status": data.status,
                "input_type": data.input_type,
                "created_at": data.created_at,
                "completed_at": data.completed_at,
                "error_msg": data.error_msg,
                "item_count": len(data.items) if hasattr(data, "items") else 0,
            }
            return d
        return data


class RunDetail(RunOut):
    items: list[RunItemOut]

    @model_validator(mode="before")
    @classmethod
    def _parse_fields(cls, data: Any) -> Any:
        if hasattr(data, "__dict__"):
            items_out = []
            for item in data.items:
                raw_detections = json.loads(item.detections_json or "[]")
                detections = [DetectionEntry.model_validate(d) for d in raw_detections]
                items_out.append(
                    RunItemOut(
                        id=item.id,
                        source_path=item.source_path,
                        result_snapshot_path=item.result_snapshot_path,
                        detections=detections,
                        speed=json.loads(item.speed_json) if item.speed_json else None,
                        frame_idx=item.frame_idx,
                        cpu_percent=item.cpu_percent,
                        device_memory_kib=item.device_memory_kib,
                    )
                )
            base = RunOut._parse_fields(data)  # type: ignore[operator]
            base["items"] = items_out
            return base
        return data
