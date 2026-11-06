
from datetime import date

from fireguard.schemas.firms import (
    ConfidenceLevel,
    FireObservation,
    FireObservationsResponse,
)

# Synthetic observations for API development and testing.
# These are NOT real NASA FIRMS detections.
MOCK_OBSERVATIONS = [
    FireObservation(
        latitude=34.0522,
        longitude=-116.6810,
        brightness=326.4,
        frp=18.7,
        confidence="high",
        satellite="Suomi-NPP",
        acquisition_date=date(2026, 11, 5),
        acquisition_time="1830",
        daynight="D",
        source="VIIRS_SNPP_SP",
    ),
    FireObservation(
        latitude=36.7783,
        longitude=-119.4179,
        brightness=312.1,
        frp=12.3,
        confidence="nominal",
        satellite="Suomi-NPP",
        acquisition_date=date(2026, 11, 5),
        acquisition_time="1842",
        daynight="D",
        source="VIIRS_SNPP_SP",
    ),
    FireObservation(
        latitude=39.5501,
        longitude=-120.2420,
        brightness=341.8,
        frp=25.6,
        confidence="high",
        satellite="Suomi-NPP",
        acquisition_date=date(2026, 11, 6),
        acquisition_time="1912",
        daynight="N",
        source="VIIRS_SNPP_SP",
    ),
    FireObservation(
        latitude=37.7749,
        longitude=-122.4194,
        brightness=300.5,
        frp=8.4,
        confidence="low",
        satellite="Suomi-NPP",
        acquisition_date=date(2026, 11, 6),
        acquisition_time="1924",
        daynight="N",
        source="VIIRS_SNPP_SP",
    ),
]


def get_fire_observations(
    confidence: ConfidenceLevel | None = None,
    limit: int = 100,
) -> FireObservationsResponse:
    observations = MOCK_OBSERVATIONS.copy()

    if confidence is not None:
        observations = [
            observation
            for observation in observations
            if observation.confidence == confidence
        ]

    observations = observations[:limit]

    return FireObservationsResponse(
        observations=observations,
        count=len(observations),
        source="NASA FIRMS-compatible mock dataset",
        mode="mock",
    )
