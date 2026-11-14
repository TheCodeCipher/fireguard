"""Live next-day forecasts: retrieve-only.

The forecasts are produced offline by ``scripts/run_live_forecast.py`` (ML side), which writes
GeoTIFFs and, last and atomically, ``live/index.json`` under the artifact root. These routes only
read that index and serve its files after a SHA-256 check. Nothing here imports ``fireguard_ml``
or torch, runs a model, or accepts a write. A live forecast has no target yet, so there is no
replay session or reveal step.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Path
from fastapi.responses import FileResponse

from fireguard.core.config import get_settings
from fireguard.core.errors import APIError
from fireguard.schemas.live import LiveForecastOut
from fireguard.services.artifact_storage import LocalArtifactStorage

router = APIRouter(prefix="/live", tags=["Live forecast"])

INDEX_URI = "live/index.json"
LAYERS = ("probability", "fold_spread")
SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
DISCLAIMER = (
    "Research next-day forecast, not an official warning. The model was validated on "
    "2018-2021 western-US fires only. Probabilities are model outputs, not calibrated risks."
)


def _storage() -> LocalArtifactStorage:
    return LocalArtifactStorage(get_settings().artifact_root)


def _not_available(reason: str) -> APIError:
    return APIError(404, "LIVE_FORECAST_NOT_AVAILABLE", reason)


def _load_index() -> dict:
    storage = _storage()
    try:
        path = storage.resolve(INDEX_URI)
    except APIError as exc:
        if exc.code == "ARTIFACT_NOT_FOUND":
            raise _not_available("No live forecast has been published yet") from None
        raise
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise APIError(
            503, "LIVE_INDEX_UNREADABLE", "The live forecast index is unreadable"
        ) from None
    if not isinstance(index, dict) or index.get("schema_version") != 1:
        raise APIError(503, "LIVE_INDEX_UNREADABLE", "Unsupported live forecast index version")
    return index


def _parse_time(value: str) -> datetime:
    t = datetime.fromisoformat(value)
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _artifact_url(live_run_id: str, region_id: str, layer: str) -> str:
    prefix = get_settings().api_v1_prefix
    return f"{prefix}/live/forecasts/{live_run_id}/{region_id}/{layer}"


@router.get("/forecasts", response_model=LiveForecastOut)
def latest_live_forecast() -> dict:
    """The most recently published live forecast run, with one entry per fire region."""
    index = _load_index()
    try:
        generated = _parse_time(index["generated_at"])
        stale_after = int(index["stale_after_hours"])
        regions = []
        for reg in index["regions"]:
            arts = {}
            for layer in LAYERS:
                a = reg["artifacts"][layer]
                arts[layer] = {
                    "url": _artifact_url(index["live_run_id"], reg["region_id"], layer),
                    "checksum_sha256": a["checksum_sha256"],
                    "file_size_bytes": a["file_size_bytes"],
                    "content_type": a.get("content_type", "image/tiff"),
                }
            regions.append({**{k: v for k, v in reg.items() if k != "artifacts"},
                            "artifacts": arts})
        return {
            **{k: v for k, v in index.items() if k not in ("schema_version", "regions")},
            "stale": datetime.now(UTC) - generated > timedelta(hours=stale_after),
            "disclaimer": DISCLAIMER,
            "regions": regions,
        }
    except (KeyError, TypeError, ValueError):
        raise APIError(
            503, "LIVE_INDEX_UNREADABLE", "The live forecast index is incomplete"
        ) from None


@router.get("/forecasts/{live_run_id}/{region_id}/{layer}", response_class=FileResponse)
def live_forecast_artifact(
    live_run_id: str = Path(pattern=SAFE_ID),
    region_id: str = Path(pattern=SAFE_ID),
    layer: str = Path(pattern="^(probability|fold_spread)$"),
):
    """One GeoTIFF of the current live run, after its checksum is verified."""
    index = _load_index()
    if index.get("live_run_id") != live_run_id or not re.match(SAFE_ID, live_run_id):
        raise _not_available("This live run is not the published one")
    region = next((r for r in index.get("regions", []) if r.get("region_id") == region_id), None)
    if region is None:
        raise _not_available("No such region in the published live run")
    art = region.get("artifacts", {}).get(layer)
    if not art:
        raise _not_available("No such layer for this region")
    path = _storage().verify(art.get("storage_uri", ""), art.get("checksum_sha256"),
                             art.get("file_size_bytes"))
    return FileResponse(path, media_type="image/tiff",
                        filename=f"{live_run_id}_{region_id}_{layer}.tif")
