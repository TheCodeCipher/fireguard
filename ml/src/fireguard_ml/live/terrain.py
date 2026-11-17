"""Static terrain (raw bands 12-14: slope deg, aspect deg, elevation m) for live frames.

WSTS: ``USGS/SRTMGL1_003`` with ``ee.Terrain.slope`` / ``ee.Terrain.aspect``. Aspect stays in
degrees here; ``process_raw`` applies the sine, once, as for training data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from fireguard_ml.live.grid import LiveRegion

TERRAIN_BANDS = ("slope", "aspect", "elevation")  # raw bands 12, 13, 14


class TerrainUnavailable(RuntimeError):
    """Terrain could not be obtained. A forecast without terrain is refused, never imputed."""


class TerrainProvider(Protocol):
    name: str

    def fetch(self, region: LiveRegion) -> np.ndarray:
        """``[3, H, W]`` float32: slope (degrees), aspect (degrees, 0 = N), elevation (m)."""


def write_terrain_tif(path: Path, arr: np.ndarray, region: LiveRegion) -> None:
    import rasterio
    from rasterio.transform import Affine

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=region.height, width=region.width,
                       count=3, dtype="float32", nodata=float("nan"), compress="deflate",
                       crs=region.crs, transform=Affine(*region.transform)) as dst:
        dst.write(arr.astype(np.float32))
        for i, name in enumerate(TERRAIN_BANDS, start=1):
            dst.set_band_description(i, name)


class TerrainMosaic:
    """Window reads from one GeoTIFF already on the region lattice (same CRS, pixel and origin
    alignment). A region that is not fully inside the mosaic makes the fetch fail."""

    name = "mosaic"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise TerrainUnavailable(f"terrain mosaic not found: {self.path}")

    def fetch(self, region: LiveRegion) -> np.ndarray:
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(self.path) as ds:
            if ds.count != 3:
                raise TerrainUnavailable(f"{self.path}: expected 3 bands, found {ds.count}")
            if (ds.crs is None or ds.crs.to_string() != region.crs
                    or abs(ds.transform.a - region.pixel_m) > 1e-6
                    or abs(ds.transform.e + region.pixel_m) > 1e-6):
                raise TerrainUnavailable(f"{self.path}: not on the {region.crs} 375 m grid")
            left, _, _, top = region.bounds
            col = (left - ds.transform.c) / region.pixel_m
            row = (ds.transform.f - top) / region.pixel_m
            if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
                raise TerrainUnavailable(f"{self.path}: origin is not aligned with the region")
            col, row = round(col), round(row)
            if (col < 0 or row < 0 or col + region.width > ds.width
                    or row + region.height > ds.height):
                raise TerrainUnavailable(f"{self.path}: does not fully cover {region.region_id}")
            arr = ds.read(window=Window(col, row, region.width, region.height))
        return arr.astype(np.float32)


class CachedTerrain:
    """Per-region GeoTIFF cache in front of a slower provider (terrain never changes)."""

    def __init__(self, inner: TerrainProvider, cache_dir: Path) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.name = f"{inner.name}+cache"

    def fetch(self, region: LiveRegion) -> np.ndarray:
        import rasterio

        path = self.cache_dir / f"{region.region_id}.tif"
        if path.is_file():
            with rasterio.open(path) as ds:
                same = (ds.height, ds.width) == region.hw and tuple(ds.transform)[:6] == tuple(
                    region.transform)
                if same:
                    return ds.read().astype(np.float32)
        arr = self.inner.fetch(region)
        if arr.shape != (3, *region.hw) or any(not np.isfinite(b).any() for b in arr):
            raise TerrainUnavailable(
                f"{self.inner.name}: incomplete terrain for {region.region_id}")
        write_terrain_tif(path, arr, region)
        return arr


class ConstantTerrain:
    """Flat synthetic terrain. Only for MOCK runs and tests; never a real forecast."""

    name = "constant_mock"

    def __init__(self, slope: float = 0.0, aspect: float = 0.0, elevation: float = 500.0) -> None:
        self._v = (slope, aspect, elevation)

    def fetch(self, region: LiveRegion) -> np.ndarray:
        return np.stack([np.full(region.hw, v, np.float32) for v in self._v])
