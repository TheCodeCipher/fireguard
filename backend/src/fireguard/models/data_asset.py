from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from fireguard.core.database import Base


class DataAsset(Base):
    __tablename__ = "data_assets"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    product_version: Mapped[str | None] = mapped_column(String(100))
    sensing_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    available_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    storage_uri: Mapped[str] = mapped_column(String(1000), nullable=False)
    quality_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
