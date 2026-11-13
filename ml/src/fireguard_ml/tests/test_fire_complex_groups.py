"""Synthetic tests for the Plan v4 related-fire grouping rule. No WSTS data is needed."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from fireguard_ml.data.fire_complex_groups import (
    Event,
    group_events,
    haversine_km,
    interval_gap_days,
    yearly_largest_share,
)

KM_PER_DEG_LAT = 111.195  # mean-earth-radius metres per degree of latitude / 1000
BASE = date(2020, 8, 1)


def ev(eid, km_north=0.0, start_offset=0, length=5, year=2020, cells=None, lon=-120.0):
    start = BASE + timedelta(days=start_offset)
    return Event(
        eid,
        year,
        start,
        start + timedelta(days=length - 1),
        lon,
        40.0 + km_north / KM_PER_DEG_LAT,
        frozenset(cells) if cells is not None else None,
    )


def gap_after(a: Event, days: int, eid: str, km_north: float) -> Event:
    """An event that starts exactly ``days`` days after ``a`` ends."""
    start = a.end + timedelta(days=days)
    return Event(
        eid, a.year, start, start + timedelta(days=4), a.lon, a.lat + km_north / KM_PER_DEG_LAT
    )


def groups_as_sets(result):
    return sorted(sorted(m) for m in result.members.values())


def test_haversine_one_degree_of_latitude():
    assert haversine_km(-120, 40, -120, 41) == pytest.approx(KM_PER_DEG_LAT, rel=1e-3)


def test_interval_gap_is_zero_for_overlapping_intervals_and_counts_days_otherwise():
    a = ev("a", length=5)  # Aug 1-5
    assert interval_gap_days(a, ev("b", start_offset=3)) == 0
    assert interval_gap_days(a, ev("c", start_offset=10)) == 6  # Aug 5 -> Aug 11
    assert interval_gap_days(ev("c", start_offset=10), a) == 6  # symmetric


def test_exact_footprint_overlap_groups_even_when_far_apart_in_space_and_time():
    a = ev("a", cells={1, 2, 3})
    b = ev("b", km_north=200, start_offset=300, cells={3, 4})
    r = group_events([a, b])
    assert groups_as_sets(r) == [["a", "b"]]
    assert r.pair_reasons == {"overlap": 1}


def test_29km_13day_gap_groups():
    a = ev("a")
    assert groups_as_sets(group_events([a, gap_after(a, 13, "b", 29)])) == [["a", "b"]]


def test_31km_13day_gap_does_not_group():
    a = ev("a")
    assert groups_as_sets(group_events([a, gap_after(a, 13, "b", 31)])) == [["a"], ["b"]]


def test_29km_15day_gap_does_not_group():
    a = ev("a")
    assert groups_as_sets(group_events([a, gap_after(a, 15, "b", 29)])) == [["a"], ["b"]]


def test_both_thresholds_are_inclusive():
    a = ev("a")
    assert groups_as_sets(group_events([a, gap_after(a, 14, "b", 29.9)])) == [["a", "b"]]


def test_unknown_footprints_never_count_as_overlap():
    a = ev("a", cells=None)
    b = ev("b", km_north=200, cells={1})
    assert groups_as_sets(group_events([a, b])) == [["a"], ["b"]]


def test_single_linkage_chains_and_complete_linkage_breaks_the_chain():
    # a-b 25 km, b-c 25 km, a-c 50 km, all simultaneous
    events = [ev("a", 0), ev("b", 25), ev("c", 50)]
    single = group_events(events, linkage="single")
    complete = group_events(events, linkage="complete")
    assert groups_as_sets(single) == [["a", "b", "c"]]
    assert len(complete.members) == 2
    for members in complete.members.values():  # every pair inside a complete group is related
        lats = [e.lat for e in events if e.event_id in members]
        assert (max(lats) - min(lats)) * KM_PER_DEG_LAT <= 30


def test_largest_group_share_warning_uses_events_of_that_year():
    # 5 events in 2020; two of them form one group -> 40 % share -> warning
    events = [ev("a"), ev("b", 10)] + [ev(f"x{i}", 500 * (i + 1)) for i in range(3)]
    shares = yearly_largest_share(events, group_events(events))
    assert shares[2020]["largest_group_events_in_year"] == 2
    assert shares[2020]["largest_group_share"] == pytest.approx(0.4)
    assert shares[2020]["warning"] is True


def test_anchor_year_is_manifest_year_of_earliest_event_not_calendar_year_of_its_start():
    # mirrors fire_21458836: manifest year 2018 but first date 2017-12-29
    early = Event("e18", 2018, date(2017, 12, 29), date(2018, 1, 5), -120, 40)
    later = Event("e18b", 2018, date(2018, 1, 10), date(2018, 1, 12), -120, 40.1)
    r = group_events([early, later])
    (gid,) = r.members
    assert r.anchor_year[gid] == 2018
    assert gid.startswith("G2018_")


def test_cross_year_group_is_anchored_to_its_earliest_event():
    a = Event("dec", 2020, date(2020, 12, 20), date(2020, 12, 31), -120, 40, frozenset({9}))
    b = Event("jan", 2021, date(2021, 1, 3), date(2021, 1, 9), -120, 40, frozenset({9}))
    r = group_events([b, a])
    (gid,) = r.members
    assert r.anchor_year[gid] == 2020


def test_group_ids_are_deterministic_and_input_order_independent():
    events = [ev("a"), ev("b", 10), ev("c", 500, start_offset=40)]
    r1 = group_events(events)
    r2 = group_events(list(reversed(events)))
    assert r1.group_of == r2.group_of
    assert r1.group_of["a"] == "G2020_0001"
    assert r1.group_of["c"] == "G2020_0002"


def test_duplicate_event_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        group_events([ev("a"), ev("a", 100)])
