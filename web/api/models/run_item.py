from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from web.api.database import Base

if TYPE_CHECKING:
    from web.api.models.run import Run


class RunItem(Base):
    __tablename__ = "run_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), nullable=False, index=True)
    source_path: Mapped[str] = mapped_column(String, nullable=False)
    result_snapshot_path: Mapped[str | None] = mapped_column(String, nullable=True)
    detections_json: Mapped[str | None] = mapped_column(String, nullable=True)
    speed_json: Mapped[str | None] = mapped_column(String, nullable=True)
    frame_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    device_memory_kib: Mapped[int | None] = mapped_column(Integer, nullable=True)

    run: Mapped["Run"] = relationship("Run", back_populates="items")
