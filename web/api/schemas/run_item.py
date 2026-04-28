from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionEntry(BaseModel):
    box: BoundingBox
    confidence: float
    class_id: int = Field(alias="class")
    name: str

    model_config = ConfigDict(populate_by_name=True)


class RunItemOut(BaseModel):
    id: int
    source_path: str
    result_snapshot_path: str | None
    detections: list[DetectionEntry]
    frame_idx: int | None

    model_config = ConfigDict(from_attributes=True)
