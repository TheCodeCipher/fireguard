import hashlib
import json
import math
from datetime import UTC, date, datetime, timedelta
from time import monotonic

import httpx

from fireguard.core.config import Settings
from fireguard.core.errors import APIError
from fireguard.schemas.firms import (
    FireConfidence,
    FireObservation,
    FIRMSSource,
)
from fireguard.services.cache import AsyncTTLCache, CacheResult
from fireguard.services.firms_client import FIRMSClient, FIRMSUpstreamError


OBSERVATION_CACHE: AsyncTTLCache[list[FireObservation]] = AsyncTTLCache()
AVAILABILITY_CACHE: AsyncTTLCache[tuple[date, date]] = AsyncTTLCache()

SOURCE_LABELS = {
    FIRMSSource.VIIRS_SNPP_NRT: ("Suomi-NPP", "NRT"),
    FIRMSSource.VIIRS_SNPP_SP: ("Suomi-NPP", "SP"),
    FIRMSSource.VIIRS_NOAA20_NRT: ("NOAA-20", "NRT"),
    FIRMSSource.VIIRS_NOAA20_SP: ("NOAA-20", "SP"),
    FIRMSSource.VIIRS_NOAA21_NRT: ("NOAA-21", "NRT"),
    FIRMSSource.VIIRS_NOAA21_SP: ("NOAA-21", "SP"),
}


def confidence_from_raw(value: str | None) -> FireConfidence:
    normalized = (value or "").strip().lower()
    return {
        "l": FireConfidence.low,
        "low": FireConfidence.low,
        "n": FireConfidence.nominal,
        "nominal": FireConfidence.nominal,
        "h": FireConfidence.high,
        "high": FireConfidence.high,
    }.get(normalized, FireConfidence.unknown)


def _float_or_none(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def normalize_firms_row(row: dict[str, str], source: FIRMSSource) -> FireObservation | None:
    try:
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        acquired_on = date.fromisoformat(row["acq_date"])
        acquisition_time = str(row["acq_time"]).strip().zfill(4)
    except (KeyError, TypeError, ValueError):
        return None

    if len(acquisition_time) != 4 or not acquisition_time.isdigit():
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    try:
        hour, minute = int(acquisition_time[:2]), int(acquisition_time[2:])
        if hour > 23 or minute > 59:
            return None
    except ValueError:
        return None
    platform, product = SOURCE_LABELS[source]
    confidence_raw = row.get("confidence")
    brightness = _float_or_none(row.get("bright_ti4"))
    if brightness is None:
        brightness = _float_or_none(row.get("bright_ti5"))
    frp = _float_or_none(row.get("frp"))
    if frp is not None and frp < 0:
        frp = None
    identity = json.dumps(
        {"source": source.value, "row": row},
        sort_keys=True,
        separators=(",", ":"),
    )
    source_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    return FireObservation(
        latitude=latitude,
        longitude=longitude,
        brightness=brightness,
        frp=frp,
        confidence=confidence_from_raw(confidence_raw),
        confidence_raw=confidence_raw,
        platform=platform,
        instrument=row.get("instrument") or "VIIRS",
        acquisition_date=acquired_on,
        acquisition_time=acquisition_time,
        daynight=row.get("daynight"),
        source=source,
        product=product,
        source_id=source_id,
    )


def cache_key(payload: dict) -> str:
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def cache_expiry(result: CacheResult, ttl_seconds: int) -> datetime | None:
    if ttl_seconds <= 0:
        return None
    remaining = max(0.0, result.expires_at_epoch - monotonic())
    return datetime.now(UTC) + timedelta(seconds=remaining)


class FIRMSGateway:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _require_key(self) -> str:
        if not self.settings.firms_map_key:
            raise APIError(
                503, "FIRMS_NOT_CONFIGURED",
                "FIRMS MAP_KEY is not configured on the server",
            )
        return self.settings.firms_map_key

    async def _with_client(self, operation):
        key = self._require_key()
        timeout = httpx.Timeout(self.settings.firms_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            client = FIRMSClient(http_client, self.settings.firms_base_url, key)
            try:
                return await operation(client)
            except FIRMSUpstreamError as exc:
                if exc.kind == "timeout":
                    raise APIError(504, "FIRMS_TIMEOUT", exc.message) from exc
                if exc.kind == "rate_limited":
                    raise APIError(503, "FIRMS_RATE_LIMITED", exc.message) from exc
                raise APIError(502, "FIRMS_UPSTREAM_ERROR", exc.message) from exc

    async def get_availability(self, source: FIRMSSource):
        key = cache_key({"kind": "availability", "source": source.value})

        async def load():
            rows = await self._with_client(lambda client: client.availability(source))
            matching = [row for row in rows if row.get("data_id") == source.value]
            if not matching:
                raise APIError(
                    503, "FIRMS_AVAILABILITY_UNKNOWN",
                    "FIRMS did not return availability for the requested source",
                )
            try:
                return (
                    date.fromisoformat(matching[0]["min_date"]),
                    date.fromisoformat(matching[0]["max_date"]),
                )
            except (KeyError, ValueError) as exc:
                raise APIError(
                    502, "FIRMS_AVAILABILITY_INVALID",
                    "FIRMS availability response was not recognized",
                ) from exc

        try:
            return await AVAILABILITY_CACHE.get_or_load(
                key, self.settings.firms_availability_cache_ttl_seconds, load
            )
        except APIError:
            raise

    async def get_observations(
        self,
        source: FIRMSSource,
        bbox: str,
        start_date: date,
        end_date: date,
        confidence: FireConfidence | None,
        min_frp: float | None,
    ) -> CacheResult[list[FireObservation]]:
        query = {
            "source": source.value,
            "bbox": bbox,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "confidence": confidence.value if confidence else None,
            "min_frp": min_frp,
        }
        key = cache_key(query)

        async def load():
            availability = await self.get_availability(source)
            min_available, max_available = availability.value
            if start_date < min_available or end_date > max_available:
                raise APIError(
                    422,
                    "FIRMS_DATE_UNAVAILABLE",
                    (
                        "Requested dates are outside source availability "
                        f"({min_available} to {max_available})"
                    ),
                )
            rows = await self._with_client(
                lambda client: client.area(source, bbox, start_date, end_date)
            )
            observations = [
                item for row in rows if (item := normalize_firms_row(row, source)) is not None
            ]
            if confidence is not None:
                observations = [item for item in observations if item.confidence == confidence]
            if min_frp is not None:
                observations = [
                    item for item in observations
                    if item.frp is not None and item.frp >= min_frp
                ]
            return observations

        return await OBSERVATION_CACHE.get_or_load(
            key, self.settings.firms_observation_cache_ttl_seconds, load
        )
