"""Live regions sit on the WSTS lattice and reproduce the WSTS framing (layer 1)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

pytest.importorskip("rasterio")

from live_test_helpers import CENTER, REPO  # noqa: E402

from fireguard_ml.live.grid import (  # noqa: E402
    LIVE_CRS,
    OutOfDomainError,
    cluster_points,
    load_training_domain,
    project,
    region_for_center,
    region_from_grid,
    region_id_for,
    regions_for_points,
)

SPATIAL = REPO / "data" / "manifests" / "wsts_event_spatial_metadata.csv"


def _events():
    with SPATIAL.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_region_is_on_the_375m_lattice_in_epsg_32610():
    r = region_for_center(*CENTER)
    left, bottom, right, top = r.bounds
    assert r.crs == LIVE_CRS == "EPSG:32610"
    assert r.transform[0] == 375.0 and r.transform[4] == -375.0
    assert r.transform[1] == 0.0 and r.transform[3] == 0.0
    for edge in (left, bottom, right, top):
        assert edge % 375.0 == 0.0
    assert (right - left) / 375.0 == r.width and (top - bottom) / 375.0 == r.height


def test_region_size_is_inside_the_range_of_real_wsts_frames():
    ev = _events()
    widths, heights = [int(e["width"]) for e in ev], [int(e["height"]) for e in ev]
    r = region_for_center(*CENTER)
    assert min(widths) <= r.width <= max(widths)
    assert min(heights) <= r.height <= max(heights)


def test_region_reproduces_real_wsts_frames_from_their_centres():
    """Rebuilding a frame from an event's own centre gives that event's size within 2 px."""
    ev = _events()[:: max(1, len(_events()) // 40)]
    cx = [(float(e["left"]) + float(e["right"])) / 2 for e in ev]
    cy = [(float(e["bottom"]) + float(e["top"])) / 2 for e in ev]
    lons, lats = project(LIVE_CRS, "EPSG:4326", cx, cy)
    for e, lon, lat in zip(ev, lons, lats, strict=True):
        r = region_for_center(float(lon), float(lat))
        assert abs(r.width - int(e["width"])) <= 2, e["event_id"]
        assert abs(r.height - int(e["height"])) <= 2, e["event_id"]
        assert abs(r.bounds[0] - float(e["left"])) <= 2 * 375, e["event_id"]
        assert abs(r.bounds[3] - float(e["top"])) <= 2 * 375, e["event_id"]


def test_region_contains_its_centre_and_bbox_encloses_the_box():
    r = region_for_center(*CENTER)
    x, y = project("EPSG:4326", r.crs, [CENTER[0]], [CENTER[1]])
    assert r.contains_xy(float(x[0]), float(y[0]))
    west, south, east, north = r.bbox_wgs84
    assert west <= CENTER[0] - 0.5 + 1e-6 and east >= CENTER[0] + 0.5 - 1e-6
    assert south <= CENTER[1] - 0.5 + 1e-6 and north >= CENTER[1] + 0.5 - 1e-6


def test_region_from_grid_round_trips_a_real_frame():
    e = _events()[0]
    t = (375.0, 0.0, float(e["left"]), 0.0, -375.0, float(e["top"]))
    r = region_from_grid(e["event_id"], t, int(e["height"]), int(e["width"]))
    assert r.bounds == (float(e["left"]), float(e["bottom"]), float(e["right"]), float(e["top"]))


def test_region_id_is_path_and_url_safe_and_never_a_wsts_event_id():
    rid = region_id_for(-120.215, 38.412)
    assert rid == "r_38p412n_120p215w"
    assert not rid.startswith("fire_")
    assert all(ch.isalnum() or ch == "_" for ch in rid)
    assert region_id_for(10.5, -3.25) == "r_3p250s_10p500e"


def test_domain_gate():
    domain = load_training_domain(SPATIAL)
    assert domain.n_events == 607 and domain.crs == LIVE_CRS
    assert domain.contains_lonlat(*CENTER)
    domain.require(*CENTER)
    for lon, lat in ((-80.0, 35.0), (-120.0, 60.0), (140.0, -30.0)):
        assert not domain.contains_lonlat(lon, lat)
        with pytest.raises(OutOfDomainError):
            domain.require(lon, lat)


def test_clustering_is_single_linkage():
    xy = np.array([[0, 0], [1500, 0], [3000, 0], [50_000, 0], [51_000, 0], [200_000, 0]], float)
    labels = cluster_points(xy, 2000.0)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] != labels[0]
    assert len(set(labels)) == 3
    assert len(cluster_points(np.zeros((0, 2)), 2000.0)) == 0


def test_regions_for_points_orders_filters_and_absorbs():
    domain = load_training_domain(SPATIAL)
    rng = np.random.default_rng(0)
    big = np.column_stack([CENTER[0] + rng.uniform(-0.01, 0.01, 40),
                           CENTER[1] + rng.uniform(-0.01, 0.01, 40)])
    near = np.column_stack([CENTER[0] + 0.2 + rng.uniform(-0.01, 0.01, 15),
                            CENTER[1] + rng.uniform(-0.01, 0.01, 15)])  # inside the first box
    far = np.column_stack([-112.0 + rng.uniform(-0.01, 0.01, 20),
                           44.0 + rng.uniform(-0.01, 0.01, 20)])
    tiny = np.array([[-116.0, 40.0]])
    outside = np.column_stack([-80.0 + rng.uniform(-0.01, 0.01, 30),
                               35.0 + rng.uniform(-0.01, 0.01, 30)])
    pts = np.vstack([tiny, far, near, outside, big])
    regions, report = regions_for_points(pts[:, 0], pts[:, 1], domain, min_cluster_points=10)
    assert len(regions) == 2
    assert report["cluster_sizes"][regions[0].region_id] == 40  # largest first
    assert report["cluster_sizes"][regions[1].region_id] == 20
    assert report["absorbed"] == 1 and report["too_small"] == 1 and report["out_of_domain"] == 1
    assert len({r.region_id for r in regions}) == 2
    limited, rep2 = regions_for_points(pts[:, 0], pts[:, 1], domain, min_cluster_points=10,
                                       max_regions=1)
    assert len(limited) == 1 and rep2["over_limit"] == 1
    assert regions_for_points([], [], domain)[0] == []
