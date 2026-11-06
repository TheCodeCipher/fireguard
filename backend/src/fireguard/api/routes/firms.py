
from typing import Annotated

from fastapi import APIRouter, Query

from fireguard.schemas.firms import (
    ConfidenceLevel,
    FireObservationsResponse,
)
from fireguard.services.firms_service import get_fire_observations

router = APIRouter(
    prefix="/firms",
    tags=["FIRMS Observations"],
)


@router.get(
    "/observations",
    response_model=FireObservationsResponse,
    summary="Retrieve active-fire observations",
    description=(
        "Retrieve active-fire observations from the FIRMS service. "
        "The current implementation uses synthetic mock data."
    ),
)
async def list_fire_observations(
    confidence: Annotated[
        ConfidenceLevel | None,
        Query(description="Filter by detection confidence"),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=1000, description="Maximum number of observations"),
    ] = 100,
) -> FireObservationsResponse:
    return get_fire_observations(
        confidence=confidence,
        limit=limit,
    )
