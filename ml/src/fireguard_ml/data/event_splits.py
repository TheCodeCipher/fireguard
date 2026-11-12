"""P1 leave-one-year-out (LOYO) split construction and validation (Plan v4 section 6.3).

For test year Y:

* test       = every group whose anchor year is Y (whole groups, all their events);
* validation = whole groups drawn from the other three anchor years, about 15-20 % of those groups;
* train      = the remaining groups.

Validation groups are drawn separately from each remaining anchor year (stratified), with a
fixed seed, so every fold's validation set spans the training years and is reproducible.

``validate_manifest`` is the single source of truth for split integrity; the tests and the
manifest builder both call it, and any experiment can call it before training.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

P1_YEARS = (2018, 2019, 2020, 2021)
VALIDATION_TARGET = 0.175
VALIDATION_RANGE = (0.15, 0.20)
MANIFEST_COLUMNS = (
    "fold_test_year",
    "split",
    "group_id",
    "event_id",
    "event_year",
    "anchor_year",
)
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class ManifestRow:
    fold_test_year: int
    split: str
    group_id: str
    event_id: str
    event_year: int
    anchor_year: int


def choose_validation_groups(
    groups_by_year: Mapping[int, list[str]],
    target: float = VALIDATION_TARGET,
    seed: int = 0,
) -> set[str]:
    """Pick ``round(target * n)`` whole groups from each year's sorted group list (min 1)."""
    chosen: set[str] = set()
    for year in sorted(groups_by_year):
        gids = sorted(groups_by_year[year])
        if not gids:
            continue
        k = max(1, round(target * len(gids)))
        rng = random.Random(f"{seed}-{year}")  # string seeds are stable across Python versions
        chosen.update(rng.sample(gids, k))
    return chosen


def build_p1_manifest(
    group_of: Mapping[str, str],
    anchor_year: Mapping[str, int],
    event_year: Mapping[str, int],
    years: Iterable[int] = P1_YEARS,
    target: float = VALIDATION_TARGET,
    seed: int = 0,
) -> list[ManifestRow]:
    """Build every P1 fold. One row per (fold, event)."""
    years = tuple(years)
    members: dict[str, list[str]] = defaultdict(list)
    for ev, gid in group_of.items():
        members[gid].append(ev)
    unknown = {anchor_year[g] for g in members} - set(years)
    if unknown:
        raise ValueError(f"groups anchored to years outside P1: {sorted(unknown)}")

    rows: list[ManifestRow] = []
    for test_year in years:
        remaining: dict[int, list[str]] = defaultdict(list)
        for gid in members:
            if anchor_year[gid] != test_year:
                remaining[anchor_year[gid]].append(gid)
        val = choose_validation_groups(remaining, target=target, seed=seed + test_year)
        for gid in sorted(members):
            if anchor_year[gid] == test_year:
                split = "test"
            elif gid in val:
                split = "validation"
            else:
                split = "train"
            for ev in sorted(members[gid]):
                rows.append(
                    ManifestRow(test_year, split, gid, ev, event_year[ev], anchor_year[gid])
                )
    return rows


def validate_manifest(rows: list[ManifestRow], years: Iterable[int] = P1_YEARS) -> dict:
    """Raise ``AssertionError`` on any integrity violation; return per-fold counts.

    Checks: every fold has all three splits; every event appears exactly once per fold;
    a group never spans two splits in one fold; test = exactly the groups anchored to the test
    year; validation and train groups come only from other anchor years; validation share of the
    non-test groups is within the 15-20 % target (warning only when the group count makes that
    impossible); every event is tested in exactly one fold.
    """
    years = tuple(years)
    by_fold: dict[int, list[ManifestRow]] = defaultdict(list)
    for r in rows:
        assert r.split in SPLITS, f"bad split {r.split!r}"
        by_fold[r.fold_test_year].append(r)
    assert sorted(by_fold) == sorted(years), f"folds {sorted(by_fold)} != {sorted(years)}"

    all_events = {r.event_id for r in rows}
    tested_in: dict[str, list[int]] = defaultdict(list)
    summary: dict[int, dict] = {}
    for y, fr in sorted(by_fold.items()):
        events = [r.event_id for r in fr]
        assert len(events) == len(set(events)), f"fold {y}: event listed twice"
        assert set(events) == all_events, f"fold {y}: events missing from the fold"
        split_of_group: dict[str, set[str]] = defaultdict(set)
        for r in fr:
            split_of_group[r.group_id].add(r.split)
            if r.split == "test":
                assert r.anchor_year == y, f"fold {y}: test group {r.group_id} anchored elsewhere"
                tested_in[r.event_id].append(y)
            else:
                assert r.anchor_year != y, f"fold {y}: {r.group_id} from the test year in {r.split}"
        leaks = {g: s for g, s in split_of_group.items() if len(s) > 1}
        assert not leaks, f"fold {y}: groups split across splits: {leaks}"
        counts = {s: {"groups": 0, "events": 0} for s in SPLITS}
        for s in split_of_group.values():
            counts[next(iter(s))]["groups"] += 1
        for r in fr:
            counts[r.split]["events"] += 1
        for s in SPLITS:
            assert counts[s]["groups"] > 0, f"fold {y}: empty {s} split"
        non_test = counts["train"]["groups"] + counts["validation"]["groups"]
        share = counts["validation"]["groups"] / non_test
        counts["validation_group_share"] = share
        counts["validation_share_in_range"] = VALIDATION_RANGE[0] <= share <= VALIDATION_RANGE[1]
        summary[y] = counts
    never = all_events - set(tested_in)
    twice = {e: f for e, f in tested_in.items() if len(f) > 1}
    assert not never, f"events never tested: {sorted(never)[:5]}"
    assert not twice, f"events tested in two folds: {twice}"
    return summary


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(MANIFEST_COLUMNS)
        for r in sorted(rows, key=lambda r: (r.fold_test_year, r.split, r.group_id, r.event_id)):
            w.writerow([getattr(r, c) for c in MANIFEST_COLUMNS])


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == MANIFEST_COLUMNS, reader.fieldnames
        return [
            ManifestRow(
                int(r["fold_test_year"]),
                r["split"],
                r["group_id"],
                r["event_id"],
                int(r["event_year"]),
                int(r["anchor_year"]),
            )
            for r in reader
        ]


def sha256_file(path: Path) -> str:
    """SHA-256 of a text file with line endings normalised to LF.

    The repository's ``.gitattributes`` does not pin ``*.csv`` to LF, so a Windows checkout may
    rewrite manifests with CRLF. Normalising first keeps the frozen hash valid on every OS.
    """
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def fold_events(rows: Iterable[ManifestRow], test_year: int, split: str) -> list[str]:
    """Event ids of one split of one fold, sorted. Every experiment should read splits this way."""
    return sorted(r.event_id for r in rows if r.fold_test_year == test_year and r.split == split)
