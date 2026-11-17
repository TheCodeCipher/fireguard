"""Live forecast regions on the WSTS grid (EPSG:32610, 375 m, origin on the 375 m lattice).

WSTS exported, per fire event, a longitude/latitude rectangle of +-0.5 degrees around the event
location (``rectangular_size = 0.5`` in the dataset-creation code), at 375 m in EPSG:32610. The
resulting frames are 207-311 px wide and 297-356 px high (``wsts_event_spatial_metadata.csv``).
``region_for_center`` reproduces that framing for an arbitrary centre.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LIVE_CRS = "EPSG:32610"
PIXEL_M = 375.0
HALF_SIZE_DEG = 0.5
_EDGE_POINTS = 21  # densification of the lon/lat rectangle before taking its projected bounds


class OutOfDomainError(ValueError):
    """The requested region lies outside the footprint of the training events."""


def project(src_crs: str, dst_crs: str, xs: Sequence[float], ys: Sequence[float]):
    """Reproject coordinates (rasterio's bundled PROJ; rasterio is already needed for GeoTIFFs)."""
    try:
        from rasterio.warp import transform as warp_transform
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError('live forecasting needs rasterio: pip install -e "ml[scripts]"') from exc
    px, py = warp_transform(src_crs, dst_crs, list(map(float, xs)), list(map(float, ys)))
    return np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64)


def _coord(value: float, pos: str, neg: str) -> str:
    return f"{abs(value):.3f}".replace(".", "p") + (pos if value >= 0 else neg)


def region_id_for(lon: float, lat: float) -> str:
    """URL- and path-safe id, e.g. ``r_38p412n_120p215w``. Never starts with ``fire_``."""
    return f"r_{_coord(lat, 'n', 's')}_{_coord(lon, 'e', 'w')}"


@dataclass(frozen=True)
class LiveRegion:
    region_id: str
    center_lon: float
    center_lat: float
    transform: tuple[float, float, float, float, float, float]  # GDAL order a, b, c, d, e, f
    height: int
    width: int
    crs: str = LIVE_CRS

    @property
    def hw(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def pixel_m(self) -> float:
        return float(self.transform[0])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) in the region CRS."""
        a, _, c, _, e, f = self.transform
        return c, f + e * self.height, c + a * self.width, f

    @property
    def bbox_wgs84(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) enclosing the frame, in degrees."""
        left, bottom, right, top = self.bounds
        xs = np.concatenate([np.linspace(left, right, _EDGE_POINTS)] * 2
                            + [np.full(_EDGE_POINTS, left), np.full(_EDGE_POINTS, right)])
        ys = np.concatenate([np.full(_EDGE_POINTS, bottom), np.full(_EDGE_POINTS, top)]
                            + [np.linspace(bottom, top, _EDGE_POINTS)] * 2)
        lon, lat = project(self.crs, "EPSG:4326", xs, ys)
        return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())

    def contains_xy(self, x: float, y: float) -> bool:
        left, bottom, right, top = self.bounds
        return left <= x < right and bottom < y <= top


def region_from_grid(region_id: str, transform: Sequence[float], height: int, width: int,
                     crs: str = LIVE_CRS) -> LiveRegion:
    """A region on an existing raster grid (e.g. a WSTS event frame, for parity checks)."""
    t = tuple(float(v) for v in transform[:6])
    left, top = t[2], t[5]
    cx = left + t[0] * width / 2.0
    cy = top + t[4] * height / 2.0
    lon, lat = project(crs, "EPSG:4326", [cx], [cy])
    return LiveRegion(region_id, float(lon[0]), float(lat[0]), t, int(height), int(width), crs)  # type: ignore[arg-type]


def region_for_center(lon: float, lat: float, half_size_deg: float = HALF_SIZE_DEG,
                      pixel_m: float = PIXEL_M, crs: str = LIVE_CRS) -> LiveRegion:
    """The WSTS-style frame around ``(lon, lat)``: a +-``half_size_deg`` rectangle, snapped outward
    to the ``pixel_m`` lattice of ``crs``."""
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError(f"not a longitude/latitude: {lon}, {lat}")
    w, e, s, n = lon - half_size_deg, lon + half_size_deg, lat - half_size_deg, lat + half_size_deg
    lons = np.concatenate([np.linspace(w, e, _EDGE_POINTS)] * 2
                          + [np.full(_EDGE_POINTS, w), np.full(_EDGE_POINTS, e)])
    lats = np.concatenate([np.full(_EDGE_POINTS, s), np.full(_EDGE_POINTS, n)]
                          + [np.linspace(s, n, _EDGE_POINTS)] * 2)
    xs, ys = project("EPSG:4326", crs, lons, lats)
    left = math.floor(xs.min() / pixel_m) * pixel_m
    right = math.ceil(xs.max() / pixel_m) * pixel_m
    bottom = math.floor(ys.min() / pixel_m) * pixel_m
    top = math.ceil(ys.max() / pixel_m) * pixel_m
    width = int(round((right - left) / pixel_m))
    height = int(round((top - bottom) / pixel_m))
    return LiveRegion(
        region_id_for(lon, lat), float(lon), float(lat),
        (pixel_m, 0.0, left, 0.0, -pixel_m, top), height, width, crs,
    )


@dataclass(frozen=True)
class TrainingDomain:
    """Bounding box (region CRS) of all training event frames; live regions are centred in it."""

    left: float
    bottom: float
    right: float
    top: float
    crs: str = LIVE_CRS
    n_events: int = 0

    def contains_lonlat(self, lon: float, lat: float) -> bool:
        x, y = project("EPSG:4326", self.crs, [lon], [lat])
        return self.left <= x[0] <= self.right and self.bottom <= y[0] <= self.top

    def require(self, lon: float, lat: float) -> None:
        if not self.contains_lonlat(lon, lat):
            raise OutOfDomainError(
                f"({lon:.3f}, {lat:.3f}) is outside the footprint of the {self.n_events} training "
                "events; the model has no evidence there"
            )

    @property
    def bbox_wgs84(self) -> tuple[float, float, float, float]:
        r = LiveRegion("domain", 0.0, 0.0, (1.0, 0.0, self.left, 0.0, -1.0, self.top),
                       int(self.top - self.bottom), int(self.right - self.left), self.crs)
        return r.bbox_wgs84


def load_training_domain(spatial_metadata_csv: Path) -> TrainingDomain:
    """Domain from ``data/manifests/wsts_event_spatial_metadata.csv`` (left/bottom/right/top)."""
    lefts, bottoms, rights, tops, crss = [], [], [], [], set()
    with Path(spatial_metadata_csv).open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lefts.append(float(r["left"]))
            bottoms.append(float(r["bottom"]))
            rights.append(float(r["right"]))
            tops.append(float(r["top"]))
            crss.add(r["crs"])
    if not lefts:
        raise ValueError(f"{spatial_metadata_csv}: no events")
    if len(crss) != 1:
        raise ValueError(f"{spatial_metadata_csv}: mixed CRS {sorted(crss)}")
    return TrainingDomain(min(lefts), min(bottoms), max(rights), max(tops), crss.pop(), len(lefts))


def cluster_points(xy: np.ndarray, link_m: float) -> np.ndarray:
    """Single-linkage labels: points closer than ``link_m`` share a cluster. ``xy`` is [N, 2]."""
    from scipy.spatial import cKDTree

    n = len(xy)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        for i, j in cKDTree(xy).query_pairs(link_m):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def regions_for_points(
    lons: Sequence[float],
    lats: Sequence[float],
    domain: TrainingDomain | None,
    *,
    link_m: float = 2000.0,
    min_cluster_points: int = 1,
    max_regions: int | None = None,
    half_size_deg: float = HALF_SIZE_DEG,
    pixel_m: float = PIXEL_M,
    crs: str = LIVE_CRS,
) -> tuple[list[LiveRegion], dict]:
    """One WSTS-style region per fire cluster, largest cluster first.

    A cluster whose centroid already lies inside an accepted region is absorbed by it (the fire is
    forecast there), so frames keep the size the model was trained on instead of being merged into
    larger ones. Returns the regions and a small report (clusters seen / skipped and why).
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    report = {"n_points": int(len(lons)), "n_clusters": 0, "too_small": 0, "out_of_domain": 0,
              "absorbed": 0, "over_limit": 0, "cluster_sizes": {}}
    if not len(lons):
        return [], report
    xs, ys = project("EPSG:4326", crs, lons, lats)
    labels = cluster_points(np.column_stack([xs, ys]), link_m)
    sizes = np.bincount(labels)
    report["n_clusters"] = int(len(sizes))
    regions: list[LiveRegion] = []
    for lab in np.argsort(-sizes, kind="stable"):
        if sizes[lab] < min_cluster_points:
            report["too_small"] += 1
            continue
        member = labels == lab
        cx, cy = float(xs[member].mean()), float(ys[member].mean())
        if any(r.contains_xy(cx, cy) for r in regions):
            report["absorbed"] += 1
            continue
        clon, clat = project(crs, "EPSG:4326", [cx], [cy])
        clon, clat = float(clon[0]), float(clat[0])
        if domain is not None and not domain.contains_lonlat(clon, clat):
            report["out_of_domain"] += 1
            continue
        if max_regions is not None and len(regions) >= max_regions:
            report["over_limit"] += 1
            continue
        region = region_for_center(clon, clat, half_size_deg, pixel_m, crs)
        regions.append(region)
        report["cluster_sizes"][region.region_id] = int(sizes[lab])
    return regions, report
