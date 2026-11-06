from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fireguard.core.database import Base

if TYPE_CHECKING:
    from fireguard.models.forecast import ForecastRun
class WildfireEvent(Base):
    __tablename__ = "wildfire_events"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    event_key: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
    )

    name: Mapped[str | None] = mapped_column(String(255))
    region: Mapped[str | None] = mapped_column(String(100))

    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)

    source_dataset: Mapped[str] = mapped_column(
        String(100),
        default="WSTS",
        nullable=False,
    )

    forecast_runs: Mapped[list["ForecastRun"]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
    )
