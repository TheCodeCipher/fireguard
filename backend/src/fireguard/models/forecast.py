from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fireguard.core.database import Base

if TYPE_CHECKING:
    from fireguard.models.event import WildfireEvent
    from fireguard.models.model_version import ModelVersion

class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "model_version_id",
            "origin_date",
            "fold",
            name="uq_forecast_event_model_origin_fold",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    event_id: Mapped[int] = mapped_column(
        ForeignKey("wildfire_events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"),
        nullable=False,
        index=True,
    )

    origin_date: Mapped[date] = mapped_column(Date, nullable=False)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)

    horizon_hours: Mapped[int] = mapped_column(
        Integer,
        default=24,
        nullable=False,
    )

    fold: Mapped[str] = mapped_column(String(50), nullable=False)
    split_name: Mapped[str] = mapped_column(String(50), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    event: Mapped["WildfireEvent"] = relationship(
        back_populates="forecast_runs",
    )

    model_version: Mapped["ModelVersion"] = relationship(
        back_populates="forecast_runs",
    )

    artifacts: Mapped[list["ForecastArtifact"]] = relationship(
        back_populates="forecast_run",
        cascade="all, delete-orphan",
    )


class ForecastArtifact(Base):
    __tablename__ = "forecast_artifacts"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    forecast_run_id: Mapped[int] = mapped_column(
        ForeignKey("forecast_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    artifact_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    storage_uri: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    content_type: Mapped[str | None] = mapped_column(String(100))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    forecast_run: Mapped["ForecastRun"] = relationship(
        back_populates="artifacts",
    )
