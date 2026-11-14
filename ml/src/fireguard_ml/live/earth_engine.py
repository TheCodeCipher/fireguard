"""Earth Engine providers for terrain and VNP09GA reflectance (optional dependency).

STATUS: written against the documented ``ee.data.computePixels`` API but NOT executed in the
environment this code was built in (no Earth Engine credentials there). The first real run must
be ``scripts/check_live_input_parity.py``, which compares what these providers return with real
WSTS frames; do not trust live forecasts before that check passes.

Why ``computePixels`` on the region's own grid: WSTS was exported from Earth Engine at 375 m in
EPSG:32610, so Earth Engine evaluated ``ee.Terrain.slope`` / ``aspect`` and resampled every source
on that output grid. Requesting the same expressions on the same grid repeats that computation;
no local reprojection or slope formula is involved.

Needs ``pip install -r ml/requirements-live.txt`` and ``earthengine authenticate``; the Cloud
project is read from ``EARTHENGINE_PROJECT``.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np

from fireguard_ml.live.grid import LiveRegion
from fireguard_ml.live.reflectance import (
    V2_TO_V1_SCALE,
    LinearReflectanceMapping,
    ReflectanceUnavailable,
)
from fireguard_ml.live.terrain import TerrainUnavailable

SRTM_ASSET = "USGS/SRTMGL1_003"
VNP09GA_V1 = "NOAA/VIIRS/001/VNP09GA"  # what WSTS used; ends 2024-06-16
VNP09GA_V2 = "NASA/VIIRS/002/VNP09GA"  # current collection, reflectance units
VNP09GA_BANDS = ("M11", "I2", "I1")  # raw bands 0, 1, 2
_FILL = -32768.0
_initialized = False


class EarthEngineUnavailable(RuntimeError):
    """The Earth Engine client is not installed or not authenticated."""


def _ee():
    global _initialized
    try:
        import ee
    except ImportError as exc:
        raise EarthEngineUnavailable(
            "earthengine-api is not installed: pip install -r ml/requirements-live.txt") from exc
    if not _initialized:
        try:
            ee.Initialize(project=os.environ.get("EARTHENGINE_PROJECT") or None)
        except Exception as exc:  # the client raises several types when not authenticated
            raise EarthEngineUnavailable(
                f"Earth Engine is not initialised ({type(exc).__name__}); run "
                "'earthengine authenticate' and set EARTHENGINE_PROJECT") from exc
        _initialized = True
    return ee


def compute_pixels(image, region: LiveRegion, bands: tuple[str, ...]) -> np.ndarray:
    """``[len(bands), H, W]`` float32 of ``image`` on the region grid; masked pixels are NaN."""
    ee = _ee()
    a, b, c, d, e, f = region.transform
    structured = ee.data.computePixels({
        "expression": image.select(list(bands)).toFloat().unmask(_FILL),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": region.width, "height": region.height},
            "affineTransform": {"scaleX": a, "shearX": b, "translateX": c,
                                "shearY": d, "scaleY": e, "translateY": f},
            "crsCode": region.crs,
        },
    })
    arr = np.stack([np.asarray(structured[name], dtype=np.float32) for name in bands])
    if arr.shape != (len(bands), *region.hw):
        expected = (len(bands), *region.hw)
        raise RuntimeError(f"Earth Engine returned {arr.shape}, expected {expected}")
    arr[arr == _FILL] = np.nan
    return arr


class EarthEngineTerrain:
    """Slope (deg), aspect (deg), elevation (m) from SRTM, the WSTS recipe."""

    name = "earth_engine_srtmgl1_003"

    def fetch(self, region: LiveRegion) -> np.ndarray:
        try:
            ee = _ee()
            dem = ee.Image(SRTM_ASSET).select("elevation")
            img = ee.Image.cat([ee.Terrain.slope(dem), ee.Terrain.aspect(dem), dem]).rename(
                ["slope", "aspect", "elevation"])
            return compute_pixels(img, region, ("slope", "aspect", "elevation"))
        except EarthEngineUnavailable as exc:
            raise TerrainUnavailable(str(exc)) from exc


class EarthEngineVNP09GA:
    """Daily VNP09GA reflectance (median of the day's images, as WSTS), on the Collection 1 scale.

    ``collection="v1"`` reads the decommissioned collection WSTS used (historical checks only).
    ``collection="v2"`` reads the current collection, multiplies by 10,000 and applies ``mapping``.
    ``day_offset`` shifts which collection day is taken for a frame day (0 = same UTC date).
    """

    def __init__(self, collection: str = "v2", mapping: LinearReflectanceMapping | None = None,
                 day_offset: int = 0) -> None:
        if collection not in ("v1", "v2"):
            raise ValueError("collection must be 'v1' or 'v2'")
        self.collection = collection
        self.mapping = mapping
        self.day_offset = int(day_offset)
        mapped = "" if mapping is None or collection == "v1" else f"+map[{mapping.fitted_on}]"
        self.name = f"earth_engine_vnp09ga_{collection}{mapped}"

    def fetch(self, region: LiveRegion, day: date) -> np.ndarray:
        try:
            ee = _ee()
        except EarthEngineUnavailable as exc:
            raise ReflectanceUnavailable(str(exc)) from exc
        d0 = day + timedelta(days=self.day_offset)
        cid = VNP09GA_V1 if self.collection == "v1" else VNP09GA_V2
        col = (ee.ImageCollection(cid)
               .filterDate(d0.isoformat(), (d0 + timedelta(days=1)).isoformat())
               .select(list(VNP09GA_BANDS)))
        if int(col.size().getInfo()) == 0:
            raise ReflectanceUnavailable(f"{cid}: no image for {d0}")
        arr = compute_pixels(col.median(), region, VNP09GA_BANDS)
        if self.collection == "v2":
            arr = arr * np.float32(V2_TO_V1_SCALE)
            if self.mapping is not None:
                arr = self.mapping.apply(arr)
        return arr.astype(np.float32)
