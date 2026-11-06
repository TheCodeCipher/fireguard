
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ConfidenceLevel = Literal["low", "nominal", "high"]
SatelliteSource = Literal["VIIRS_SNPP_SP", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]


class FireObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    brightness: float | None = Field(default=None, ge=0)
    frp: float | None = Field(default=None, ge=0)

    confidence: ConfidenceLevel
    satellite: str
    instrument: str = "VIIRS"

    acquisition_date: date
    acquisition_time: str = Field(pattern=r"^\d{4}$")

    daynight: Literal["D", "N"]
    source: SatelliteSource


class FireObservationsResponse(BaseModel):
    observations: list[FireObservation]
    count: int
    source: str
    mode: Literal["mock", "live"]
