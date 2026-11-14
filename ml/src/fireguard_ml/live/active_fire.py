"""Active-fire band (raw band 22) from VIIRS detections, following the WSTS recipe.

WSTS (``docs/data_sources_and_licenses.md`` 4.7): filter detections by date, drop
``confidence == 'l'``, buffer each point by 187.5 m (half a 375 m pixel), rasterise the acquisition
time (HHMM) with ``Reducer.last``. NaN means no detection. The raster produced here is the *raw*
band; ``process_raw`` then applies ``floor(HHMM / 100)`` and ``> 0`` exactly as for training data
(including the documented quirk that a 00:00-00:59 UTC detection is not in the input mask).

Two points of the original Earth Engine rasterisation are not documented and are therefore
parameters, to be fixed by ``scripts/check_live_input_parity.py`` against real WSTS frames:

* ``rule``: ``"intersects"`` marks every cell the buffered disc touches; ``"containing"`` marks
  only the cell that contains the detection.
* ``day_utc_offset_hours``: which clock the "date" filter used (0 = UTC acquisition date).
"""

from __future__ import annotations

import csv
import math
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import numpy as np

from fireguard_ml.live.grid import LiveRegion, project

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov"
FIRMS_MAX_DAY_RANGE = 5
RASTERIZE_RULES = ("intersects", "containing")
_TOUCH_TOLERANCE = 1e-4  # cells (about 4 cm): absorbs reprojection round-off at tangency


class FirmsError(RuntimeError):
    """FIRMS could not be queried. The message never contains the MAP_KEY."""


@dataclass(frozen=True)
class Detection:
    lon: float
    lat: float
    acq_date: date
    acq_time: int  # HHMM, UTC
    confidence: str  # "l" | "n" | "h" | "" (unknown)

    def acquired_at(self) -> datetime:
        return datetime(self.acq_date.year, self.acq_date.month, self.acq_date.day,
                        self.acq_time // 100, self.acq_time % 100)

    def day(self, utc_offset_hours: float = 0.0) -> date:
        """The calendar day this detection belongs to under the given clock offset."""
        return (self.acquired_at() + timedelta(hours=utc_offset_hours)).date()


def _confidence(raw: str) -> str:
    c = (raw or "").strip().lower()
    if c in ("l", "low"):
        return "l"
    if c in ("n", "nominal"):
        return "n"
    if c in ("h", "high"):
        return "h"
    return ""


def parse_firms_rows(rows: Iterable[dict]) -> list[Detection]:
    """FIRMS VIIRS CSV rows -> detections. Rows with unusable fields raise ``FirmsError``."""
    out = []
    for i, r in enumerate(rows):
        try:
            hhmm = int(str(r["acq_time"]).strip())
            if not (0 <= hhmm // 100 <= 23 and 0 <= hhmm % 100 <= 59):
                raise ValueError(f"acq_time {hhmm}")
            out.append(Detection(
                float(r["longitude"]), float(r["latitude"]),
                date.fromisoformat(str(r["acq_date"]).strip()), hhmm,
                _confidence(str(r.get("confidence", ""))),
            ))
        except (KeyError, ValueError) as exc:
            raise FirmsError(f"FIRMS row {i} is not a VIIRS detection record ({exc})") from exc
    return out


def parse_firms_csv(text: str) -> list[Detection]:
    cleaned = text.lstrip("﻿").strip()
    if not cleaned:
        return []
    first = cleaned.splitlines()[0].lower()
    if "latitude" not in first or "acq_date" not in first:
        # FIRMS answers errors (bad key, rate limit) as plain text with HTTP 200
        raise FirmsError("FIRMS did not return a detection table (check the MAP_KEY and quota)")
    return parse_firms_rows(csv.DictReader(cleaned.splitlines()))


def read_firms_csv(path: Path) -> list[Detection]:
    return parse_firms_csv(Path(path).read_text(encoding="utf-8"))


def date_chunks(start: date, end: date, max_days: int = FIRMS_MAX_DAY_RANGE):
    """Same chunking rule as the backend's ``firms_client.date_chunks`` (1-5 days per request)."""
    if not 1 <= max_days <= FIRMS_MAX_DAY_RANGE:
        raise ValueError("FIRMS area requests must use a day range from 1 to 5")
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=max_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


class DetectionProvider(Protocol):
    source: str

    def detections(self, bbox: Sequence[float], start: date, end: date) -> list[Detection]:
        """Detections with ``start <= acq_date <= end`` in ``bbox`` (west, south, east, north)."""


def _http_get(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        return resp.read().decode("utf-8", errors="replace")


class FirmsAreaProvider:
    """NASA FIRMS Area API (CSV). The MAP_KEY is read by the caller from ``FIRMS_MAP_KEY``."""

    def __init__(self, map_key: str, source: str = "VIIRS_SNPP_NRT",
                 base_url: str = FIRMS_BASE_URL, timeout: float = 60.0,
                 fetch: Callable[[str, float], str] | None = None) -> None:
        if not map_key:
            raise FirmsError("FIRMS_MAP_KEY is not set")
        self._key = map_key
        self.source = source
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._fetch = fetch or _http_get

    def detections(self, bbox: Sequence[float], start: date, end: date) -> list[Detection]:
        west, south, east, north = (float(v) for v in bbox)
        area = f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"
        out: list[Detection] = []
        for c0, c1 in date_chunks(start, end):
            n_days = (c1 - c0).days + 1
            url = (f"{self._base}/api/area/csv/{quote(self._key, safe='')}/"
                   f"{quote(self.source, safe='')}/{area}/{n_days}/{c0.isoformat()}")
            try:
                text = self._fetch(url, self._timeout)
            except urllib.error.HTTPError as exc:
                raise FirmsError(f"NASA FIRMS returned HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                raise FirmsError("NASA FIRMS could not be reached") from None
            out += [d for d in parse_firms_csv(text) if start <= d.acq_date <= end]
        return out


class CsvDetectionProvider:
    """Detections from a saved FIRMS CSV (offline runs and tests)."""

    def __init__(self, path: Path, source: str = "FIRMS_CSV_FILE") -> None:
        self.source = source
        self._all = read_firms_csv(path)

    def detections(self, bbox: Sequence[float], start: date, end: date) -> list[Detection]:
        west, south, east, north = (float(v) for v in bbox)
        return [d for d in self._all
                if start <= d.acq_date <= end and west <= d.lon <= east and south <= d.lat <= north]


def split_time_safe(dets: Iterable[Detection], origin: date,
                    utc_offset_hours: float = 0.0) -> tuple[list[Detection], int]:
    """Keep detections whose day is on or before the forecast origin; count the rest."""
    kept, dropped = [], 0
    for d in dets:
        if d.day(utc_offset_hours) <= origin:
            kept.append(d)
        else:
            dropped += 1
    return kept, dropped


def rasterize_active_fire(
    dets: Iterable[Detection],
    region: LiveRegion,
    day: date,
    *,
    exclude_confidence: Sequence[str] = ("l",),
    buffer_m: float = 187.5,
    rule: str = "intersects",
    utc_offset_hours: float = 0.0,
) -> np.ndarray:
    """Raw active-fire band for one day: ``[H, W]`` float32, HHMM where detected, NaN elsewhere.

    When several detections cover a cell the latest acquisition wins.
    """
    if rule not in RASTERIZE_RULES:
        raise ValueError(f"rule must be one of {RASTERIZE_RULES}")
    h, w = region.hw
    out = np.full((h, w), np.nan, np.float32)
    excluded = set(exclude_confidence)
    todays = sorted((d for d in dets
                     if d.day(utc_offset_hours) == day and d.confidence not in excluded),
                    key=lambda d: d.acquired_at())
    if not todays:
        return out
    xs, ys = project("EPSG:4326", region.crs, [d.lon for d in todays], [d.lat for d in todays])
    px = region.pixel_m
    left, _, _, top = region.bounds
    for d, x, y in zip(todays, xs, ys, strict=True):
        fc, fr = (x - left) / px, (top - y) / px  # fractional column / row
        if rule == "containing":
            cells = [(math.floor(fr), math.floor(fc))]
        else:
            rb = buffer_m / px
            cells = []
            for r in range(math.floor(fr - rb), math.floor(fr + rb) + 1):
                for c in range(math.floor(fc - rb), math.floor(fc + rb) + 1):
                    dr = max(r - fr, 0.0, fr - (r + 1))  # distance from the point to the cell
                    dc = max(c - fc, 0.0, fc - (c + 1))
                    # strictly inside: a disc that only grazes a cell edge does not mark it
                    if math.hypot(dr, dc) < rb - _TOUCH_TOLERANCE:
                        cells.append((r, c))
        for r, c in cells:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = float(d.acq_time)
    return out


class ActiveFireRasterProvider(Protocol):
    source: str

    def fetch(self, region: LiveRegion, day: date) -> np.ndarray:
        """Raw active-fire band ``[H, W]`` (HHMM / NaN) for ``day`` on the region grid."""


class DetectionRasterizer:
    """Adapter: a fixed, already time-filtered detection list -> per-day raw active-fire bands."""

    def __init__(self, dets: Sequence[Detection], source: str, **rasterize_kwargs) -> None:
        self._dets = list(dets)
        self.source = source
        self._kw = rasterize_kwargs

    def fetch(self, region: LiveRegion, day: date) -> np.ndarray:
        return rasterize_active_fire(self._dets, region, day, **self._kw)

    def count(self, region: LiveRegion, day: date) -> int:
        offset = self._kw.get("utc_offset_hours", 0.0)
        excluded = set(self._kw.get("exclude_confidence", ("l",)))
        west, south, east, north = region.bbox_wgs84
        return sum(1 for d in self._dets
                   if d.day(offset) == day and d.confidence not in excluded
                   and west <= d.lon <= east and south <= d.lat <= north)
