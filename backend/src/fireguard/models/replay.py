from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fireguard.core.database import Base

if TYPE_CHECKING:
    from fireguard.models.forecast import ForecastRun


class ReplayMetric(Base):
    __tablename__ = "replay_metrics"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    forecast_run_id: Mapped[int] = mapped_column(
        ForeignKey("forecast_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    metric_name: Mapped[str] = mapped_column(String(60), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float | None] = mapped_column(Float)
    aggregation: Mapped[str] = mapped_column(String(30), default="event", nullable=False)
    calibrator_id: Mapped[str | None] = mapped_column(String(100))
    forecast_run: Mapped[ForecastRun] = relationship(back_populates="metrics")


class ReplaySession(Base):
    __tablename__ = "replay_sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    forecast_run_id: Mapped[int] = mapped_column(
        ForeignKey("forecast_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    forecast_viewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    audit_json: Mapped[list[dict]] = mapped_column(
        JSON, default=list, server_default="[]", nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    forecast_run: Mapped[ForecastRun] = relationship(back_populates="replay_sessions")
