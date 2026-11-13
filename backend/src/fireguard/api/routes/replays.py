from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Header
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from fireguard.core.config import get_settings
from fireguard.core.database import get_db
from fireguard.core.errors import APIError
from fireguard.models import (
    ForecastArtifact,
    ForecastRun,
    ReplayMetric,
    ReplaySession,
    WildfireEvent,
)
from fireguard.schemas.replays import (
    ArtifactDescriptor,
    ReplayActualResponse,
    ReplayForecastResponse,
    ReplayInputsResponse,
    ReplayMetricItem,
    ReplayMetricsResponse,
    ReplayOriginItem,
    ReplayOriginListResponse,
    ReplayRevealResponse,
    ReplaySessionResponse,
)
from fireguard.services.artifact_storage import LocalArtifactStorage

router = APIRouter(prefix="/replays", tags=["Historical Replay"])
artifacts_router = APIRouter(tags=["Static Artifacts"])

VISIBLE_FORECAST_TYPES = {"probability_map", "uncertainty_map", "explanation_map"}
ACTUAL_TYPES = {"actual_target"}
INPUT_METADATA_ALLOWLIST = {
    "input_window_start", "input_window_end", "channel_manifest",
    "data_provenance", "level_a_status", "level_b_status",
    "input_preview_artifact_id", "missingness_summary",
}


def _find_run(db: Session, event_id: str, origin: date) -> ForecastRun:
    run = db.scalar(
        select(ForecastRun)
        .join(WildfireEvent)
        .options(joinedload(ForecastRun.event), joinedload(ForecastRun.model_version))
        .where(WildfireEvent.event_key == event_id, ForecastRun.origin_date == origin)
    )
    if run is None:
        raise APIError(
            404, "REPLAY_ORIGIN_NOT_FOUND",
            "No replay origin exists for this event and date",
        )
    if not run.is_oof or run.split_name not in {"test", "held_out"}:
        raise APIError(
            409, "REPLAY_NOT_OOF",
            "Only explicitly marked held-out out-of-fold runs can be replayed",
        )
    return run


def _get_session(db: Session, session_id: str, run_id: int) -> ReplaySession:
    session = db.get(ReplaySession, session_id)
    if session is None or session.forecast_run_id != run_id:
        raise APIError(
            404, "REPLAY_SESSION_NOT_FOUND",
            "Replay session is missing or does not match this origin",
        )
    expires_at = session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at <= datetime.now(UTC):
        raise APIError(
            410, "REPLAY_SESSION_EXPIRED",
            "Replay session has expired; start a new session",
        )
    return session


def _artifact_descriptor(artifact: ForecastArtifact) -> ArtifactDescriptor:
    settings = get_settings()
    LocalArtifactStorage(settings.artifact_root).resolve(artifact.storage_uri)
    return ArtifactDescriptor(
        artifact_id=artifact.id,
        artifact_type=artifact.artifact_type,
        content_type=artifact.content_type,
        url=f"{settings.api_v1_prefix}/artifacts/{artifact.id}",
        checksum_sha256=artifact.checksum_sha256,
    )


@router.get("/{event_id}/origins", response_model=ReplayOriginListResponse)
def list_replay_origins(
    event_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> ReplayOriginListResponse:
    event = db.scalar(select(WildfireEvent).where(WildfireEvent.event_key == event_id))
    if event is None:
        raise APIError(404, "EVENT_NOT_FOUND", "WSTS event was not found")
    runs = db.scalars(
        select(ForecastRun)
        .options(joinedload(ForecastRun.model_version))
        .where(
            ForecastRun.event_id == event.id,
            ForecastRun.is_oof.is_(True),
            ForecastRun.split_name.in_(["test", "held_out"]),
        )
        .order_by(ForecastRun.origin_date)
    ).all()
    origins = [
        ReplayOriginItem(
            origin_date=run.origin_date,
            target_date=run.target_date,
            horizon_hours=run.horizon_hours,
            protocol=run.protocol,
            fold_id=run.fold,
            model_version=f"{run.model_version.name}:{run.model_version.version}",
            split_name=run.split_name,
            is_oof=run.is_oof,
        )
        for run in runs
    ]
    return ReplayOriginListResponse(
        event_id=event_id, count=len(origins), mode="catalogue", origins=origins
    )


@router.post("/{event_id}/{origin}/session", response_model=ReplaySessionResponse)
def create_replay_session(
    event_id: str,
    origin: date,
    db: Annotated[Session, Depends(get_db)],
) -> ReplaySessionResponse:
    run = _find_run(db, event_id, origin)
    settings = get_settings()
    now = datetime.now(UTC)
    expires = now + timedelta(seconds=settings.replay_session_ttl_seconds)
    session = ReplaySession(
        session_id=str(uuid4()), forecast_run_id=run.id, expires_at=expires
    )
    db.add(session)
    db.commit()
    return ReplaySessionResponse(
        session_id=session.session_id,
        event_id=event_id,
        origin_date=origin,
        expires_at=expires,
    )


@router.get("/{event_id}/{origin}/inputs", response_model=ReplayInputsResponse)
def get_replay_inputs(
    event_id: str,
    origin: date,
    db: Annotated[Session, Depends(get_db)],
    session_id: Annotated[str, Header(alias="X-Replay-Session-ID")],
) -> ReplayInputsResponse:
    run = _find_run(db, event_id, origin)
    _get_session(db, session_id, run.id)
    metadata = run.metadata_json or {}
    safe_metadata = {key: metadata[key] for key in INPUT_METADATA_ALLOWLIST if key in metadata}
    return ReplayInputsResponse(
        event_id=event_id,
        origin_date=origin,
        mode="metadata",
        input_metadata=safe_metadata,
    )


@router.get("/{event_id}/{origin}/forecast", response_model=ReplayForecastResponse)
def get_replay_forecast(
    event_id: str,
    origin: date,
    db: Annotated[Session, Depends(get_db)],
    session_id: Annotated[str, Header(alias="X-Replay-Session-ID")],
) -> ReplayForecastResponse:
    run = _find_run(db, event_id, origin)
    session = _get_session(db, session_id, run.id)
    artifacts = [
        artifact for artifact in run.artifacts
        if artifact.artifact_type in VISIBLE_FORECAST_TYPES and artifact.status == "available"
    ]
    if not artifacts:
        raise APIError(
            404, "FORECAST_ARTIFACT_UNAVAILABLE",
            "No precomputed OOF forecast artifact is registered",
        )
    # Validate files before marking the forecast as viewed.
    descriptors = [_artifact_descriptor(artifact) for artifact in artifacts]
    session.forecast_viewed_at = datetime.now(UTC)
    db.commit()
    return ReplayForecastResponse(
        event_id=event_id,
        origin_date=origin,
        target_date=run.target_date,
        protocol=run.protocol,
        fold_id=run.fold,
        model_version=f"{run.model_version.name}:{run.model_version.version}",
        config_hash=(run.model_version.model_config_json or {}).get("config_hash"),
        channel_manifest_sha256=run.channel_manifest_sha256,
        time_safety_status=run.time_safety_status,
        artifacts=descriptors,
    )


@router.get("/{event_id}/{origin}/metrics", response_model=ReplayMetricsResponse)
def get_replay_metrics(
    event_id: str,
    origin: date,
    db: Annotated[Session, Depends(get_db)],
    session_id: Annotated[str, Header(alias="X-Replay-Session-ID")],
) -> ReplayMetricsResponse:
    run = _find_run(db, event_id, origin)
    session = _get_session(db, session_id, run.id)
    if session.revealed_at is None:
        raise APIError(
            403, "ACTUAL_NOT_REVEALED",
            "Replay metrics are available only after the actual target is revealed",
        )
    metrics = db.scalars(
        select(ReplayMetric)
        .where(ReplayMetric.forecast_run_id == run.id)
        .order_by(ReplayMetric.metric_name)
    ).all()
    return ReplayMetricsResponse(
        event_id=event_id,
        origin_date=origin,
        metrics=[
            ReplayMetricItem(
                metric_name=item.metric_name,
                value=item.value,
                threshold=item.threshold,
                aggregation=item.aggregation,
                calibrator_id=item.calibrator_id,
            )
            for item in metrics
        ],
    )


@router.post("/{event_id}/{origin}/reveal", response_model=ReplayRevealResponse)
def reveal_actual(
    event_id: str,
    origin: date,
    db: Annotated[Session, Depends(get_db)],
    session_id: Annotated[str, Header(alias="X-Replay-Session-ID")],
) -> ReplayRevealResponse:
    run = _find_run(db, event_id, origin)
    session = _get_session(db, session_id, run.id)
    if session.forecast_viewed_at is None:
        raise APIError(
            409, "FORECAST_NOT_VIEWED",
            "Load the OOF forecast before requesting the reveal",
        )
    if session.revealed_at is None:
        session.revealed_at = datetime.now(UTC)
        db.commit()
    return ReplayRevealResponse(
        event_id=event_id,
        origin_date=origin,
        revealed=True,
        revealed_at=session.revealed_at,
    )


@router.get("/{event_id}/{origin}/actual", response_model=ReplayActualResponse)
def get_actual_target(
    event_id: str,
    origin: date,
    db: Annotated[Session, Depends(get_db)],
    session_id: Annotated[str, Header(alias="X-Replay-Session-ID")],
) -> ReplayActualResponse:
    run = _find_run(db, event_id, origin)
    session = _get_session(db, session_id, run.id)
    if session.revealed_at is None:
        raise APIError(
            403, "ACTUAL_NOT_REVEALED",
            "The next-day target is locked until the reveal action",
        )
    artifacts = [
        item for item in run.artifacts
        if item.artifact_type in ACTUAL_TYPES and item.status == "available"
    ]
    if not artifacts:
        raise APIError(
            404, "ACTUAL_ARTIFACT_UNAVAILABLE",
            "No actual-target artifact is registered",
        )
    return ReplayActualResponse(
        event_id=event_id,
        origin_date=origin,
        revealed=True,
        artifacts=[_artifact_descriptor(item) for item in artifacts],
    )


@artifacts_router.get("/artifacts/{artifact_id}")
def get_static_artifact(
    artifact_id: int,
    db: Annotated[Session, Depends(get_db)],
    session_id: Annotated[str, Header(alias="X-Replay-Session-ID")],
):
    artifact = db.scalar(
        select(ForecastArtifact)
        .options(joinedload(ForecastArtifact.forecast_run))
        .where(ForecastArtifact.id == artifact_id)
    )
    if artifact is None or artifact.status != "available":
        raise APIError(404, "ARTIFACT_NOT_FOUND", "The requested static artifact is not available")
    run = artifact.forecast_run
    if not run.is_oof or run.split_name not in {"test", "held_out"}:
        raise APIError(
            409, "REPLAY_NOT_OOF",
            "Only held-out OOF artifacts can be served through replay",
        )
    session = _get_session(db, session_id, run.id)
    if artifact.artifact_type in ACTUAL_TYPES and session.revealed_at is None:
        raise APIError(
            403, "ACTUAL_NOT_REVEALED",
            "The actual-target artifact is locked until the reveal action",
        )
    path = LocalArtifactStorage(get_settings().artifact_root).resolve(artifact.storage_uri)
    return FileResponse(
        path,
        media_type=artifact.content_type or "application/octet-stream",
        filename=Path(path).name,
        headers={"Cache-Control": "private, max-age=60"},
    )
