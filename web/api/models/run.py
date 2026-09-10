from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from web.api.database import Base

if TYPE_CHECKING:
    from web.api.models.run_item import RunItem


class RunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    conf: Mapped[float] = mapped_column(Float, nullable=False)
    imgsz: Mapped[int] = mapped_column(Integer, nullable=False)
    classes: Mapped[str | None] = mapped_column(String, nullable=True)
    frame_step: Mapped[int] = mapped_column(Integer, nullable=False)
    backend: Mapped[str] = mapped_column(String, nullable=False, default="torch")
    device: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default=RunStatus.pending.value)
    input_type: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_msg: Mapped[str | None] = mapped_column(String, nullable=True)

    items: Mapped[list["RunItem"]] = relationship(
        "RunItem", back_populates="run", cascade="all, delete-orphan"
    )
