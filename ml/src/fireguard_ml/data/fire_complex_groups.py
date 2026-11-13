"""Related-fire ("fire complex") grouping for WSTS events, following Plan v4 section 6.3.

Two WSTS events are *related* when either

* their active-fire footprints overlap (share at least one 375 m grid cell), or
* their footprint centroids are within ``max_km`` and the gap between their event-date
  intervals is at most ``max_gap_days``.

Plan v4 primary rule: 30 km / 14 days. Sensitivity rule: 50 km / 30 days.

Groups are formed in one of two ways:

* ``single`` linkage: connected components of the "related" graph. Chains of individually
  related events can merge into one large component, which is why diagnostics are required.
* ``complete`` linkage: a partition in which *every pair* inside a group is related. It is built
  greedily and deterministically (events ordered by start date, then event id; each event joins
  the earliest-created group whose every member it is related to, otherwise it starts a new group).

Footprint cleaning (rule v2, 2026-11-13). The raw footprint of an event is every active-fire cell
anywhere in its ~90 x 115 km image, which also contains neighbouring fires and permanent thermal
anomalies. Two evidence-based steps make it the event's own fire:

* ``persistent_cells``: cells detected in events from at least ``min_years`` (3) different years.
  A wildfire does not burn the same 375 m cell in three of four years; these are industrial or
  other static heat sources. On the real archive: 134 cells, nine of them in all four years.
* ``own_fire_cells``: keep the fire clusters (cells bridged across gaps of up to 2 cells, i.e.
  750 m) that come within ``near_km`` (5 km) of the image centre. WSTS images are centred on their
  event: 95.9 % of events have a cluster within 5 km of the centre (median 0.27 km), and the next
  distances jump beyond 40 km. If no cluster is that close, the nearest cluster is kept.

This module is pure Python + NumPy/SciPy and does no file I/O, so it can be unit-tested on
synthetic events. Raw footprints are produced by ``scripts/derive_event_footprints.py``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy import ndimage

PIXEL_M = 375.0
CELL_ROW_FACTOR = 100_000  # cell id = row * 100000 + col (see derive_event_footprints.py)
PERSISTENT_MIN_YEARS = 3
OWN_FIRE_BRIDGE_CELLS = 2
OWN_FIRE_NEAR_KM = 5.0
FOOTPRINT_RULE_VERSION = "v2_own_fire_no_persistent"
EARTH_RADIUS_KM = 6371.0088
PRIMARY_RULE = (30.0, 14)
SENSITIVITY_RULE = (50.0, 30)
LARGEST_GROUP_WARNING_SHARE = 0.20


@dataclass(frozen=True)
class Event:
    """One WSTS event as seen by the grouping rule.

    ``year`` is the WSTS manifest year (the year folder the event lives in). It is kept as given:
    one 2018 event starts on 2017-12-29 and must stay a 2018 event.
    ``cells`` holds global 375 m grid-cell ids of the active-fire footprint; ``None`` means the
    footprint is unknown, in which case only the distance/time arm of the rule can fire.
    """

    event_id: str
    year: int
    start: date
    end: date
    lon: float
    lat: float
    cells: frozenset[int] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"{self.event_id}: end {self.end} is before start {self.start}")


def persistent_cells(
    cells_by_event: dict[str, Iterable[int]],
    year_of: dict[str, int],
    min_years: int = PERSISTENT_MIN_YEARS,
) -> frozenset[int]:
    """Cells that appear in footprints of events from at least ``min_years`` distinct years."""
    years: dict[int, set[int]] = defaultdict(set)
    for ev, cells in cells_by_event.items():
        for c in cells:
            years[int(c)].add(year_of[ev])
    return frozenset(c for c, ys in years.items() if len(ys) >= min_years)


def own_fire_cells(
    cells: Iterable[int],
    frame_center_xy: tuple[float, float],
    excluded: frozenset[int] = frozenset(),
    bridge_cells: int = OWN_FIRE_BRIDGE_CELLS,
    near_km: float = OWN_FIRE_NEAR_KM,
) -> frozenset[int]:
    """The event's own fire: clusters within ``near_km`` of the image centre, minus ``excluded``.

    ``frame_center_xy`` is the image centre in EPSG:32610 metres (Role 1's ``center_x``,
    ``center_y``). Clusters are 8-connected after bridging gaps of up to ``bridge_cells`` cells;
    the bridge only decides membership, it never adds cells.
    """
    c = np.array(sorted(set(int(x) for x in cells) - excluded), dtype=np.int64)
    if c.size == 0:
        return frozenset()
    row, col = c // CELL_ROW_FACTOR, c % CELL_ROW_FACTOR
    r0, c0 = row.min(), col.min()
    grid = np.zeros((row.max() - r0 + 1, col.max() - c0 + 1), dtype=bool)
    grid[row - r0, col - c0] = True
    bridged = ndimage.binary_dilation(grid, iterations=bridge_cells) if bridge_cells else grid
    labels, n = ndimage.label(bridged, structure=np.ones((3, 3)))
    lab = labels[row - r0, col - c0]
    dist_km = (
        np.hypot(
            (col + 0.5) * PIXEL_M - frame_center_xy[0], (row + 0.5) * PIXEL_M - frame_center_xy[1]
        )
        / 1000.0
    )
    nearest = np.full(n + 1, np.inf)
    np.minimum.at(nearest, lab, dist_km)
    keep = np.nonzero(nearest[1:] <= near_km)[0] + 1
    if keep.size == 0:
        keep = np.array([int(np.argmin(nearest[1:])) + 1])
    return frozenset(c[np.isin(lab, keep)].tolist())


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres between two lon/lat points (degrees)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def interval_gap_days(a: Event, b: Event) -> int:
    """Days between two closed date intervals; 0 when they overlap or touch."""
    return max(0, (b.start - a.end).days, (a.start - b.end).days)


def footprints_overlap(a: Event, b: Event) -> bool:
    """True only when both footprints are known and share a grid cell."""
    if a.cells is None or b.cells is None:
        return False
    small, large = (a.cells, b.cells) if len(a.cells) <= len(b.cells) else (b.cells, a.cells)
    return any(c in large for c in small)


def relation(a: Event, b: Event, max_km: float, max_gap_days: int) -> str | None:
    """Return why two events are related ("overlap" or "near"), or ``None`` if they are not."""
    if footprints_overlap(a, b):
        return "overlap"
    if interval_gap_days(a, b) <= max_gap_days and (
        haversine_km(a.lon, a.lat, b.lon, b.lat) <= max_km
    ):
        return "near"
    return None


def related_pairs(
    events: Sequence[Event], max_km: float, max_gap_days: int
) -> dict[tuple[int, int], str]:
    """All related index pairs ``(i, j)`` with ``i < j``, mapped to the reason."""
    n = len(events)
    pairs: dict[tuple[int, int], str] = {}

    # footprint overlap via an inverted index: cell id -> events containing it
    owners: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(events):
        if e.cells:
            for c in e.cells:
                owners[c].append(i)
    for idx in owners.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pairs[(min(idx[a], idx[b]), max(idx[a], idx[b]))] = "overlap"

    # distance/time arm, vectorised
    lon = np.radians([e.lon for e in events])
    lat = np.radians([e.lat for e in events])
    start = np.array([e.start.toordinal() for e in events])
    end = np.array([e.end.toordinal() for e in events])
    for i in range(n - 1):
        dphi = lat[i + 1 :] - lat[i]
        dlmb = lon[i + 1 :] - lon[i]
        h = np.sin(dphi / 2) ** 2 + np.cos(lat[i]) * np.cos(lat[i + 1 :]) * np.sin(dlmb / 2) ** 2
        dist = 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(h)))
        gap = np.maximum(0, np.maximum(start[i + 1 :] - end[i], start[i] - end[i + 1 :]))
        for k in np.nonzero((dist <= max_km) & (gap <= max_gap_days))[0]:
            pairs.setdefault((i, i + 1 + int(k)), "near")
    return pairs


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _order(events: Sequence[Event]) -> list[int]:
    return sorted(range(len(events)), key=lambda i: (events[i].start, events[i].event_id))


def single_linkage(events: Sequence[Event], pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Connected components of the related-events graph (lists of event indices)."""
    uf = _UnionFind(len(events))
    for i, j in pairs:
        uf.union(i, j)
    comps: dict[int, list[int]] = defaultdict(list)
    for i in _order(events):
        comps[uf.find(i)].append(i)
    return list(comps.values())


def complete_linkage(events: Sequence[Event], pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Greedy deterministic partition in which every pair inside a group is related."""
    adj: dict[int, set[int]] = defaultdict(set)
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    groups: list[list[int]] = []
    for i in _order(events):
        for g in groups:
            if all(m in adj[i] for m in g):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


@dataclass
class GroupingResult:
    """Group assignment plus everything the Plan v4 diagnostic asks for."""

    rule_km: float
    rule_days: int
    linkage: str
    group_of: dict[str, str]  # event_id -> group_id
    anchor_year: dict[str, int]  # group_id -> anchor year
    members: dict[str, list[str]]  # group_id -> event ids (start-date order)
    pair_reasons: Counter

    def size_distribution(self) -> dict[int, int]:
        """Number of groups for each group size."""
        return dict(sorted(Counter(len(m) for m in self.members.values()).items()))

    def size_summary(self) -> dict[str, float]:
        sizes = np.array([len(m) for m in self.members.values()])
        return {
            "n_groups": int(sizes.size),
            "n_singletons": int((sizes == 1).sum()),
            "median": float(np.median(sizes)),
            "p90": float(np.percentile(sizes, 90)),
            "p95": float(np.percentile(sizes, 95)),
            "max": int(sizes.max()),
        }


def group_events(
    events: Sequence[Event],
    max_km: float = PRIMARY_RULE[0],
    max_gap_days: int = PRIMARY_RULE[1],
    linkage: str = "single",
) -> GroupingResult:
    """Group events and assign deterministic group ids and P1 anchor years.

    Group id format: ``G<anchor_year>_<nnnn>``, numbered by earliest start date then event id.
    Anchor year: the manifest year of the group's earliest-starting event (Plan v4: "assign the
    entire group to the year of its earliest event start date"; we use the event's manifest year,
    not the calendar year of its first date, so that a 2018 event beginning on 2017-12-29 does not
    create a fold that does not exist).
    """
    ids = [e.event_id for e in events]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate event ids")
    pairs = related_pairs(events, max_km, max_gap_days)
    if linkage == "single":
        comps = single_linkage(events, pairs)
    elif linkage == "complete":
        comps = complete_linkage(events, pairs)
    else:
        raise ValueError(f"unknown linkage {linkage!r}")

    def first(c: list[int]) -> int:
        return min(c, key=lambda i: (events[i].start, events[i].event_id))

    comps.sort(key=lambda c: (events[first(c)].start, events[first(c)].event_id))
    counters: Counter = Counter()
    group_of: dict[str, str] = {}
    anchor: dict[str, int] = {}
    members: dict[str, list[str]] = {}
    for c in comps:
        y = events[first(c)].year
        counters[y] += 1
        gid = f"G{y}_{counters[y]:04d}"
        anchor[gid] = y
        ordered = sorted(c, key=lambda i: (events[i].start, events[i].event_id))
        members[gid] = [events[i].event_id for i in ordered]
        for i in c:
            group_of[events[i].event_id] = gid
    return GroupingResult(
        rule_km=max_km,
        rule_days=max_gap_days,
        linkage=linkage,
        group_of=group_of,
        anchor_year=anchor,
        members=members,
        pair_reasons=Counter(pairs.values()),
    )


def yearly_largest_share(events: Sequence[Event], result: GroupingResult) -> dict[int, dict]:
    """For every manifest year: raw events, groups touching the year, and the largest group share.

    The share is (events of the largest group that belong to year Y) / (all events of year Y),
    which is the quantity Plan v4 compares with the 20 % warning threshold.
    """
    by_year: dict[int, list[Event]] = defaultdict(list)
    for e in events:
        by_year[e.year].append(e)
    out: dict[int, dict] = {}
    for y, evs in sorted(by_year.items()):
        per_group = Counter(result.group_of[e.event_id] for e in evs)
        gid, n = per_group.most_common(1)[0]
        out[y] = {
            "raw_events": len(evs),
            "groups_touching_year": len(per_group),
            "groups_anchored_to_year": sum(1 for a in result.anchor_year.values() if a == y),
            "largest_group_id": gid,
            "largest_group_events_in_year": n,
            "largest_group_share": n / len(evs),
            "warning": n / len(evs) > LARGEST_GROUP_WARNING_SHARE,
        }
    return out


def any_warning(shares: dict[int, dict]) -> bool:
    return any(v["warning"] for v in shares.values())
