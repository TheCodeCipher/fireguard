from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from fireguard.core.database import get_db
from fireguard.models import ForecastRun, WildfireEvent
from fireguard.schemas.events import WSTSEventItem, WSTSEventListResponse

router = APIRouter(prefix="/wsts", tags=["WSTS Event Catalogue"])


@router.get("/events", response_model=WSTSEventListResponse)
def list_wsts_events(
    db: Annotated[Session, Depends(get_db)],
    year: Annotated[int | None, Query(ge=2018, le=2035)] = None,
    region: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> WSTSEventListResponse:
    replay_exists = exists(
        select(ForecastRun.id).where(
            ForecastRun.event_id == WildfireEvent.id,
            ForecastRun.is_oof.is_(True),
            ForecastRun.split_name.in_(["test", "held_out"]),
        )
    )
    statement = select(WildfireEvent, replay_exists.label("has_replay"))
    if year is not None:
        statement = statement.where(WildfireEvent.year == year)
    if region:
        statement = statement.where(WildfireEvent.region == region)

    total_statement = select(func.count()).select_from(WildfireEvent)
    if year is not None:
        total_statement = total_statement.where(WildfireEvent.year == year)
    if region:
        total_statement = total_statement.where(WildfireEvent.region == region)
    total = db.scalar(total_statement) or 0

    rows = db.execute(
        statement.order_by(WildfireEvent.event_key).offset(offset).limit(limit)
    ).all()
    events = [
        WSTSEventItem(
            event_id=event.event_key,
            name=event.name,
            year=event.year,
            region=event.region,
            start_date=event.start_date,
            end_date=event.end_date,
            complex_group_id=event.complex_group_id,
            p1_fold_id=event.p1_fold_id,
            p2_fold_id=event.p2_fold_id,
            source_dataset=event.source_dataset,
            has_replay=bool(has_replay),
        )
        for event, has_replay in rows
    ]
    return WSTSEventListResponse(
        count=len(events), total_count=total, limit=limit, offset=offset,
        mode="catalogue", events=events
    )
