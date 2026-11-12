from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fireguard.core.database import Base


class WildfireEvent(Base):
    """Metadata-only WSTS event catalogue entry; never stores target rasters."""

    __tablename__ = "wildfire_events"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    event_key: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    region: Mapped[str | None] = mapped_column(String(100))
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    source_dataset: Mapped[str] = mapped_column(String(100), default="WSTS", nullable=False)
    complex_group_id: Mapped[str | None] = mapped_column(String(100), index=True)
    p1_fold_id: Mapped[str | None] = mapped_column(String(50))
    p2_fold_id: Mapped[str | None] = mapped_column(String(50))
    manifest_sha256: Mapped[str | None] = mapped_column(String(64))

    forecast_runs: Mapped[list["ForecastRun"]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
    )
