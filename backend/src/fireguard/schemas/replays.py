from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class ReplayOriginItem(BaseModel):
    origin_date: date
    target_date: date
    horizon_hours: int
    protocol: str
    fold_id: str
    model_version: str
    split_name: str
    is_oof: bool


class ReplayOriginListResponse(BaseModel):
    event_id: str
    count: int
    mode: str
    origins: list[ReplayOriginItem]


class ReplaySessionResponse(BaseModel):
    session_id: str
    event_id: str
    origin_date: date
    expires_at: datetime


class ReplayInputsResponse(BaseModel):
    event_id: str
    origin_date: date
    mode: str
    input_metadata: dict[str, Any]


class ArtifactDescriptor(BaseModel):
    artifact_id: int
    artifact_type: str
    content_type: str | None = None
    url: str
    checksum_sha256: str | None = None


class ReplayForecastResponse(BaseModel):
    event_id: str
    origin_date: date
    target_date: date
    protocol: str
    fold_id: str
    model_version: str
    config_hash: str | None = None
    channel_manifest_sha256: str | None = None
    time_safety_status: str
    artifacts: list[ArtifactDescriptor]


class ReplayMetricItem(BaseModel):
    metric_name: str
    value: float
    threshold: float | None = None
    aggregation: str
    calibrator_id: str | None = None


class ReplayMetricsResponse(BaseModel):
    event_id: str
    origin_date: date
    metrics: list[ReplayMetricItem]


class ReplayRevealResponse(BaseModel):
    event_id: str
    origin_date: date
    revealed: bool
    revealed_at: datetime


class ReplayActualResponse(BaseModel):
    event_id: str
    origin_date: date
    revealed: bool
    artifacts: list[ArtifactDescriptor]
