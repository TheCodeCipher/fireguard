from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FIRMSSource(str, Enum):
    VIIRS_SNPP_NRT = "VIIRS_SNPP_NRT"
    VIIRS_SNPP_SP = "VIIRS_SNPP_SP"
    VIIRS_NOAA20_NRT = "VIIRS_NOAA20_NRT"
    VIIRS_NOAA20_SP = "VIIRS_NOAA20_SP"
    VIIRS_NOAA21_NRT = "VIIRS_NOAA21_NRT"
    VIIRS_NOAA21_SP = "VIIRS_NOAA21_SP"


class FireConfidence(str, Enum):
    low = "low"
    nominal = "nominal"
    high = "high"
    unknown = "unknown"


class CacheMetadata(BaseModel):
    status: str = Field(pattern=r"^(HIT|MISS|DISABLED)$")
    ttl_seconds: int = Field(ge=0)
    expires_at: datetime | None = None


class FIRMSAvailabilityResponse(BaseModel):
    source: FIRMSSource
    min_date: date
    max_date: date
    retrieved_at: datetime
    cache: CacheMetadata


class FireObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    brightness: float | None = None
    frp: float | None = Field(default=None, ge=0)
    confidence: FireConfidence = FireConfidence.unknown
    confidence_raw: str | None = None
    platform: str
    instrument: str | None = None
    acquisition_date: date
    acquisition_time: str = Field(pattern=r"^\d{4}$")
    daynight: str | None = None
    source: FIRMSSource
    product: str
    source_id: str


class FiresResponse(BaseModel):
    source: FIRMSSource
    platform: str
    product: str
    start_date: date
    end_date: date
    bbox: str
    retrieved_at: datetime
    data_age_seconds: float | None = Field(default=None, ge=0)
    count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    has_more: bool
    cache: CacheMetadata
    observations: list[FireObservation]


class MockFireObservationsResponse(BaseModel):
    observations: list[FireObservation]
    count: int
    source: str
    mode: str = "mock"
