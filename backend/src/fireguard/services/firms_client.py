import csv
from datetime import date, timedelta
from io import StringIO
from urllib.parse import quote

import httpx

from fireguard.schemas.firms import FIRMSSource


class FIRMSUpstreamError(Exception):
    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.message = message
        super().__init__(message)


def date_chunks(start_date: date, end_date: date, max_days: int = 5):
    if max_days < 1 or max_days > 5:
        raise ValueError("FIRMS area requests must use a day range from 1 to 5")
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=max_days - 1), end_date)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def parse_csv_records(text: str) -> list[dict[str, str]]:
    cleaned = text.lstrip("\ufeff").strip()
    if not cleaned:
        return []
    return [dict(row) for row in csv.DictReader(StringIO(cleaned))]


class FIRMSClient:
    """Small async adapter around NASA FIRMS CSV endpoints."""

    def __init__(self, client: httpx.AsyncClient, base_url: str, map_key: str):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.map_key = map_key

    async def _csv(self, path: str) -> list[dict[str, str]]:
        try:
            response = await self.client.get(f"{self.base_url}{path}")
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise FIRMSUpstreamError("timeout", "NASA FIRMS request timed out") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            kind = "rate_limited" if status == 429 else "http_error"
            raise FIRMSUpstreamError(kind, f"NASA FIRMS returned HTTP {status}") from exc
        except httpx.RequestError as exc:
            raise FIRMSUpstreamError("connection", "NASA FIRMS could not be reached") from exc
        return parse_csv_records(response.text)

    async def availability(self, source: FIRMSSource) -> list[dict[str, str]]:
        key = quote(self.map_key, safe="")
        sensor = quote(source.value, safe="")
        path = f"/api/data_availability/csv/{key}/{sensor}"
        return await self._csv(path)

    async def area(
        self,
        source: FIRMSSource,
        bbox: str,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, str]]:
        key = quote(self.map_key, safe="")
        sensor = quote(source.value, safe="")
        area = quote(bbox, safe=",")
        records: list[dict[str, str]] = []
        for chunk_start, chunk_end in date_chunks(start_date, end_date, max_days=5):
            day_range = (chunk_end - chunk_start).days + 1
            path = (
                f"/api/area/csv/{key}/{sensor}/{area}/"
                f"{day_range}/{chunk_start.isoformat()}"
            )
            records.extend(await self._csv(path))
        return records
