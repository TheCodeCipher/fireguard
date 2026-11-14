"""Response models of the live-forecast routes (retrieve-only; see api/routes/live.py)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LiveArtifactOut(BaseModel):
    url: str = Field(description="Relative API URL of the GeoTIFF; clients must not build it")
    checksum_sha256: str
    file_size_bytes: int
    content_type: str = "image/tiff"


class LiveRegionOut(BaseModel):
    region_id: str
    center_lon: float
    center_lat: float
    bbox_wgs84: list[float] = Field(min_length=4, max_length=4)
    height: int
    width: int
    crs: str
    transform: list[float] = Field(min_length=6, max_length=6)
    n_detections: int
    nan_fraction: dict[str, float]
    reflectance_source: str
    artifacts: dict[str, LiveArtifactOut]


class LiveForecastOut(BaseModel):
    live_run_id: str
    origin_date: str
    target_date: str
    horizon_hours: int
    generated_at: str
    stale: bool = Field(description="True when generated_at is older than stale_after_hours")
    stale_after_hours: int
    model_version: str
    architecture: str
    folds: list[int]
    probability_kind: str
    gate_status: str = Field(description="GATED, or UNGATED when Gate 2 was not passed")
    is_mock: bool
    data_contract_version: str
    firms_source: str
    disclaimer: str
    regions: list[LiveRegionOut]
