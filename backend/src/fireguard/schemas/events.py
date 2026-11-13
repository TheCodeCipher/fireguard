from datetime import date

from pydantic import BaseModel


class WSTSEventItem(BaseModel):
    event_id: str
    name: str | None = None
    year: int | None = None
    region: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    complex_group_id: str | None = None
    p1_fold_id: str | None = None
    p2_fold_id: str | None = None
    source_dataset: str
    has_replay: bool


class WSTSEventListResponse(BaseModel):
    count: int
    total_count: int
    limit: int
    offset: int
    mode: str
    events: list[WSTSEventItem]
