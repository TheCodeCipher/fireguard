from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from fireguard.core.database import Base


class FireObservationRecord(Base):
    __tablename__ = "fire_observations"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    source_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    source: Mapped[str] = mapped_column(String(40), index=True, nullable=False)
    platform: Mapped[str] = mapped_column(String(30), nullable=False)
    product: Mapped[str] = mapped_column(String(10), nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    frp: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    raw_attributes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
