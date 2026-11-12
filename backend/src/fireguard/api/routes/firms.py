from datetime import UTC, date, datetime, time
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fireguard.api.dependencies import get_firms_gateway
from fireguard.core.config import get_settings
from fireguard.core.database import get_db
from fireguard.models import FireObservationRecord
from fireguard.schemas.firms import (
    CacheMetadata,
    FireConfidence,
    FireObservation,
    FIRMSAvailabilityResponse,
    FIRMSSource,
    FiresResponse,
    MockFireObservationsResponse,
)
from fireguard.services.firms_service import FIRMSGateway, SOURCE_LABELS, cache_expiry

router = APIRouter(tags=["FIRMS"])


@router.get("/firms/availability", response_model=FIRMSAvailabilityResponse)
async def firms_availability(
    source: FIRMSSource,
    gateway: Annotated[FIRMSGateway, Depends(get_firms_gateway)],
) -> FIRMSAvailabilityResponse:
    result = await gateway.get_availability(source)
    settings = get_settings()
    return FIRMSAvailabilityResponse(
        source=source,
        min_date=result.value[0],
        max_date=result.value[1],
        retrieved_at=datetime.now(UTC),
        cache=CacheMetadata(
            status=(
                "DISABLED"
                if settings.firms_availability_cache_ttl_seconds <= 0
                else "HIT" if result.hit else "MISS"
            ),
            ttl_seconds=settings.firms_availability_cache_ttl_seconds,
            expires_at=cache_expiry(result, settings.firms_availability_cache_ttl_seconds),
        ),
    )


def _parse_bbox(value: str) -> str:
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("bbox must be west,south,east,north numeric coordinates") from exc
    if len(parts) != 4:
        raise ValueError("bbox must contain exactly four coordinates: west,south,east,north")
    west, south, east, north = parts
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must satisfy -180<=west<east<=180 and -90<=south<north<=90")
    return ",".join(f"{part:g}" for part in parts)


def _persist_observations(db: Session, observations: list[FireObservation]) -> None:
    if not observations:
        return
    ids = [item.source_id for item in observations]
    existing = set(
        db.scalars(
            select(FireObservationRecord.source_id).where(
                FireObservationRecord.source_id.in_(ids)
            )
        ).all()
    )
    for item in observations:
        if item.source_id in existing:
            continue
        acquired_at = datetime.combine(
            item.acquisition_date,
            time(int(item.acquisition_time[:2]), int(item.acquisition_time[2:])),
            tzinfo=UTC,
        )
        db.add(
            FireObservationRecord(
                source_id=item.source_id,
                source=item.source.value,
                platform=item.platform,
                product=item.product,
                acquired_at=acquired_at,
                latitude=item.latitude,
                longitude=item.longitude,
                frp=item.frp,
                confidence=item.confidence.value,
                raw_attributes={
                    "brightness": item.brightness,
                    "confidence_raw": item.confidence_raw,
                    "daynight": item.daynight,
                    "instrument": item.instrument,
                },
            )
        )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # Duplicate source IDs can race between two concurrent requests. The API
        # response remains valid; a later request can retry persistence.


@router.get("/fires", response_model=FiresResponse)
async def query_fires(
    source: FIRMSSource,
    start_date: date,
    end_date: date,
    gateway: Annotated[FIRMSGateway, Depends(get_firms_gateway)],
    db: Annotated[Session, Depends(get_db)],
    bbox: Annotated[str, Query(description="west,south,east,north")] = "-180,-90,180,90",
    confidence: Annotated[FireConfidence | None, Query()] = None,
    min_frp: Annotated[float | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> FiresResponse:
    try:
        bbox = _parse_bbox(bbox)
    except ValueError as exc:
        from fireguard.core.errors import APIError

        raise APIError(422, "INVALID_BBOX", str(exc)) from exc

    if end_date < start_date:
        from fireguard.core.errors import APIError

        raise APIError(422, "INVALID_DATE_RANGE", "end_date must be on or after start_date")
    if (end_date - start_date).days > 30:
        from fireguard.core.errors import APIError

        raise APIError(
            422,
            "DATE_RANGE_TOO_LARGE",
            "A single API query may span at most 31 inclusive dates",
        )
    if end_date > datetime.now(UTC).date():
        from fireguard.core.errors import APIError

        raise APIError(422, "FUTURE_DATE", "FIRMS queries cannot end in the future")

    result = await gateway.get_observations(source, bbox, start_date, end_date, confidence, min_frp)
    _persist_observations(db, result.value)
    platform, product = SOURCE_LABELS[source]
    page = result.value[offset : offset + limit]
    latest = max(
        (
            datetime.combine(
                item.acquisition_date,
                time(int(item.acquisition_time[:2]), int(item.acquisition_time[2:])),
                tzinfo=UTC,
            )
            for item in result.value
        ),
        default=None,
    )
    age = max(0.0, (datetime.now(UTC) - latest).total_seconds()) if latest else None
    settings = get_settings()
    return FiresResponse(
        source=source,
        platform=platform,
        product=product,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        retrieved_at=datetime.now(UTC),
        data_age_seconds=age,
        count=len(page),
        limit=limit,
        offset=offset,
        has_more=offset + limit < len(result.value),
        total_count=len(result.value),
        cache=CacheMetadata(
            status=(
                "DISABLED"
                if settings.firms_observation_cache_ttl_seconds <= 0
                else "HIT" if result.hit else "MISS"
            ),
            ttl_seconds=settings.firms_observation_cache_ttl_seconds,
            expires_at=cache_expiry(result, settings.firms_observation_cache_ttl_seconds),
        ),
        observations=page,
    )


# Phase 1 stable mock contract retained for offline UI development and CI.
_MOCK_ROWS = [
    FireObservation(latitude=34.0522, longitude=-116.6810, brightness=326.4, frp=18.7,
        confidence="high", confidence_raw="h", platform="Suomi-NPP", instrument="VIIRS",
        acquisition_date=date(2026, 11, 5), acquisition_time="1830", daynight="D",
        source="VIIRS_SNPP_SP", product="SP", source_id="mock-001"),
    FireObservation(latitude=36.7783, longitude=-119.4179, brightness=312.1, frp=12.3,
        confidence="nominal", confidence_raw="n", platform="Suomi-NPP", instrument="VIIRS",
        acquisition_date=date(2026, 11, 5), acquisition_time="1842", daynight="D",
        source="VIIRS_SNPP_SP", product="SP", source_id="mock-002"),
    FireObservation(latitude=39.5501, longitude=-120.2420, brightness=341.8, frp=25.6,
        confidence="high", confidence_raw="h", platform="Suomi-NPP", instrument="VIIRS",
        acquisition_date=date(2026, 11, 6), acquisition_time="1912", daynight="N",
        source="VIIRS_SNPP_SP", product="SP", source_id="mock-003"),
    FireObservation(latitude=37.7749, longitude=-122.4194, brightness=300.5, frp=8.4,
        confidence="low", confidence_raw="l", platform="Suomi-NPP", instrument="VIIRS",
        acquisition_date=date(2026, 11, 6), acquisition_time="1924", daynight="N",
        source="VIIRS_SNPP_SP", product="SP", source_id="mock-004"),
]


@router.get("/firms/observations", response_model=MockFireObservationsResponse)
async def mock_firms_observations(
    confidence: FireConfidence | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> MockFireObservationsResponse:
    observations = _MOCK_ROWS
    if confidence is not None:
        observations = [item for item in observations if item.confidence == confidence]
    page = observations[:limit]
    return MockFireObservationsResponse(
        observations=page,
        count=len(page),
        source="synthetic development fixture; not NASA observations",
    )
