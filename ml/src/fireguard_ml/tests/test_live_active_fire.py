"""Active-fire rasterisation follows the WSTS recipe (layer 1)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

pytest.importorskip("rasterio")

from live_test_helpers import small_region  # noqa: E402

from fireguard_ml.data.wsts_dataset import RAW_BANDS, process_raw  # noqa: E402
from fireguard_ml.live.active_fire import (  # noqa: E402
    CsvDetectionProvider,
    Detection,
    FirmsAreaProvider,
    FirmsError,
    date_chunks,
    parse_firms_csv,
    rasterize_active_fire,
    split_time_safe,
)
from fireguard_ml.live.grid import project  # noqa: E402

DAY = date(2026, 11, 1)
HEADER = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,"
          "confidence,version,bright_ti5,frp,daynight")


def at_cell(region, row: float, col: float, hhmm: int = 2030, conf: str = "n", day: date = DAY):
    """A detection at fractional (row, col) of the region grid."""
    left, _, _, top = region.bounds
    x, y = left + col * region.pixel_m, top - row * region.pixel_m
    lon, lat = project(region.crs, "EPSG:4326", [x], [y])
    return Detection(float(lon[0]), float(lat[0]), day, hhmm, conf)


def test_cell_value_is_the_acquisition_time_and_nan_elsewhere():
    r = small_region()
    out = rasterize_active_fire([at_cell(r, 10.5, 20.5, 2030)], r, DAY, rule="containing")
    assert out.dtype == np.float32 and out.shape == r.hw
    assert out[10, 20] == 2030.0
    assert np.isnan(out).sum() == out.size - 1


def test_low_confidence_is_dropped_and_unknown_is_kept():
    r = small_region()
    dets = [at_cell(r, 5.5, 5.5, conf="l"), at_cell(r, 6.5, 6.5, conf="n"),
            at_cell(r, 7.5, 7.5, conf="h"), at_cell(r, 8.5, 8.5, conf="")]
    out = rasterize_active_fire(dets, r, DAY, rule="containing")
    assert np.isnan(out[5, 5])
    assert out[6, 6] == out[7, 7] == out[8, 8] == 2030.0


def test_latest_detection_wins():
    r = small_region()
    dets = [at_cell(r, 3.5, 3.5, 2110), at_cell(r, 3.4, 3.6, 930), at_cell(r, 3.6, 3.4, 1045)]
    assert rasterize_active_fire(dets, r, DAY, rule="containing")[3, 3] == 2110.0


def test_buffer_marks_the_containing_cell_and_touched_neighbours():
    r = small_region()
    centre = rasterize_active_fire([at_cell(r, 10.5, 20.5)], r, DAY, rule="intersects")
    assert np.isfinite(centre).sum() == 1 and centre[10, 20] == 2030.0  # disc stays inside
    corner = rasterize_active_fire([at_cell(r, 10.02, 20.02)], r, DAY, rule="intersects")
    assert {tuple(i) for i in np.argwhere(np.isfinite(corner))} == {
        (9, 19), (9, 20), (10, 19), (10, 20)}
    edge = rasterize_active_fire([at_cell(r, 10.5, 20.02)], r, DAY, rule="intersects")
    assert {tuple(i) for i in np.argwhere(np.isfinite(edge))} == {(10, 19), (10, 20)}
    only = rasterize_active_fire([at_cell(r, 10.02, 20.02)], r, DAY, rule="containing")
    assert {tuple(i) for i in np.argwhere(np.isfinite(only))} == {(10, 20)}


def test_points_outside_the_region_and_other_days_are_ignored():
    r = small_region()
    dets = [at_cell(r, -3.0, 5.0), at_cell(r, 5.0, r.width + 4.0),
            at_cell(r, 5.5, 5.5, day=DAY - timedelta(days=1))]
    assert np.isnan(rasterize_active_fire(dets, r, DAY)).all()


def test_midnight_hour_detection_is_absent_from_the_input_mask():
    """Pinned WSTS quirk (data dictionary, defect 2): 00:00-00:59 UTC gives mask 0 after
    floor(HHMM / 100), although the same detection is a positive target."""
    r = small_region()
    band = rasterize_active_fire([at_cell(r, 4.5, 4.5, 30), at_cell(r, 9.5, 9.5, 130)], r, DAY,
                                 rule="containing")
    assert band[4, 4] == 30.0
    raw = np.full((len(RAW_BANDS), *r.hw), np.nan, np.float32)
    raw[22] = band
    mask = process_raw(raw)[39]
    assert mask[4, 4] == 0.0 and mask[9, 9] == 1.0 and mask.sum() == 1.0


def test_day_offset_moves_detections_between_days():
    r = small_region()
    d = at_cell(r, 5.5, 5.5, 300)  # 03:00 UTC on DAY = 19:00 the previous day at UTC-8
    assert np.isfinite(rasterize_active_fire([d], r, DAY)).sum() == 1
    prev = DAY - timedelta(days=1)
    assert np.isfinite(rasterize_active_fire([d], r, prev, utc_offset_hours=-8)).sum() == 1
    assert np.isnan(rasterize_active_fire([d], r, DAY, utc_offset_hours=-8)).all()


def test_split_time_safe_drops_and_counts_future_detections():
    r = small_region()
    dets = [at_cell(r, 1.5, 1.5), at_cell(r, 2.5, 2.5, day=DAY + timedelta(days=1)),
            at_cell(r, 3.5, 3.5, day=DAY - timedelta(days=2))]
    kept, dropped = split_time_safe(dets, DAY)
    assert dropped == 1 and {d.acq_date for d in kept} == {DAY, DAY - timedelta(days=2)}


def test_parse_firms_csv_and_error_texts():
    text = HEADER + "\n38.4,-120.2,330,0.4,0.4,2026-11-01,0930,N,VIIRS,n,2.0NRT,290,5,D\n"
    (d,) = parse_firms_csv(text)
    assert (d.lon, d.lat, d.acq_date, d.acq_time, d.confidence) == (-120.2, 38.4, DAY, 930, "n")
    assert parse_firms_csv("") == [] and parse_firms_csv(HEADER + "\n") == []
    with pytest.raises(FirmsError):
        parse_firms_csv("Invalid MAP_KEY.")
    with pytest.raises(FirmsError):
        parse_firms_csv(HEADER + "\n38.4,-120.2,330,0.4,0.4,2026-11-01,2575,N,VIIRS,n,2,290,5,D\n")


def test_date_chunks_never_exceed_five_days():
    chunks = list(date_chunks(date(2026, 11, 1), date(2026, 11, 12)))
    assert chunks[0] == (date(2026, 11, 1), date(2026, 11, 5))
    assert chunks[-1][1] == date(2026, 11, 12)
    assert all((b - a).days <= 4 for a, b in chunks)
    with pytest.raises(ValueError):
        list(date_chunks(date(2026, 11, 1), date(2026, 11, 2), max_days=6))


def test_firms_provider_builds_requests_filters_dates_and_never_leaks_the_key():
    urls = []

    def fetch(url, timeout):
        urls.append(url)
        return (HEADER + "\n38.4,-120.2,330,0.4,0.4,2026-11-01,0930,N,VIIRS,n,2.0NRT,290,5,D\n"
                "38.4,-120.2,330,0.4,0.4,2026-11-09,0930,N,VIIRS,n,2.0NRT,290,5,D\n")

    p = FirmsAreaProvider("SECRETKEY", "VIIRS_SNPP_NRT", fetch=fetch)
    dets = p.detections((-125, 32, -102, 49), date(2026, 10, 30), date(2026, 11, 5))
    assert len(urls) == 2  # 7 days -> 5 + 2
    assert urls[0].endswith("/VIIRS_SNPP_NRT/-125.0000,32.0000,-102.0000,49.0000/5/2026-10-30")
    assert urls[1].endswith("/2/2026-11-04")
    assert {d.acq_date for d in dets} == {DAY}  # the 2026-11-09 row is outside the window

    def boom(url, timeout):
        raise OSError("connect to " + url)

    with pytest.raises(FirmsError) as e:
        FirmsAreaProvider("SECRETKEY", fetch=boom).detections((-125, 32, -102, 49), DAY, DAY)
    assert "SECRETKEY" not in str(e.value) and e.value.__cause__ is None
    with pytest.raises(FirmsError):
        FirmsAreaProvider("")


def test_csv_provider_filters_by_bbox_and_date(tmp_path):
    f = tmp_path / "f.csv"
    f.write_text(HEADER + "\n38.4,-120.2,330,0.4,0.4,2026-11-01,0930,N,VIIRS,n,2.0NRT,290,5,D\n"
                 "45.0,-110.0,330,0.4,0.4,2026-11-01,0930,N,VIIRS,n,2.0NRT,290,5,D\n"
                 "38.4,-120.2,330,0.4,0.4,2026-11-02,0930,N,VIIRS,n,2.0NRT,290,5,D\n")
    p = CsvDetectionProvider(f)
    assert len(p.detections((-121, 38, -120, 39), DAY, DAY)) == 1
    assert len(p.detections((-125, 32, -102, 49), DAY, DAY + timedelta(days=1))) == 3
