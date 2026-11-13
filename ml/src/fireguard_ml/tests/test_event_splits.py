"""P1 leave-one-year-out split tests: synthetic integrity checks plus a check of the committed
manifest against its recorded hash (so every experiment can trust it is unchanged)."""

from __future__ import annotations

import dataclasses
import re

import pytest

from fireguard_ml.data.event_splits import (
    build_p1_manifest,
    fold_events,
    read_manifest,
    sha256_file,
    validate_manifest,
    write_manifest,
)


def synthetic(groups_per_year=None, cross_year=True):
    """Groups of 1-3 events in every year, plus one group spanning 2020 -> 2021."""
    groups_per_year = groups_per_year or {2018: 30, 2019: 12, 2020: 34, 2021: 26}
    group_of, anchor, event_year = {}, {}, {}
    for y, n in groups_per_year.items():
        for g in range(n):
            gid = f"G{y}_{g:04d}"
            anchor[gid] = y
            for k in range(1 + g % 3):
                eid = f"e{y}_{g}_{k}"
                group_of[eid], event_year[eid] = gid, y
    if cross_year:
        anchor["G2020_X"] = 2020
        for eid, y in (("dec2020", 2020), ("jan2021", 2021)):
            group_of[eid], event_year[eid] = "G2020_X", y
    return group_of, anchor, event_year


def test_synthetic_manifest_passes_every_integrity_rule():
    rows = build_p1_manifest(*synthetic())
    summary = validate_manifest(rows)
    for counts in summary.values():
        assert counts["validation_share_in_range"]


def test_test_split_is_exactly_the_groups_anchored_to_the_test_year():
    group_of, anchor, event_year = synthetic()
    rows = build_p1_manifest(group_of, anchor, event_year)
    for y in (2018, 2019, 2020, 2021):
        expected = sorted(e for e, g in group_of.items() if anchor[g] == y)
        assert fold_events(rows, y, "test") == expected


def test_cross_year_group_stays_whole_in_its_anchor_fold():
    rows = build_p1_manifest(*synthetic())
    assert {"dec2020", "jan2021"} <= set(fold_events(rows, 2020, "test"))
    assert "jan2021" not in fold_events(rows, 2021, "test")


def test_validation_groups_come_only_from_training_years():
    rows = build_p1_manifest(*synthetic())
    for r in rows:
        if r.split == "validation":
            assert r.anchor_year != r.fold_test_year


def test_a_group_split_across_train_and_validation_is_rejected():
    rows = build_p1_manifest(*synthetic())
    # find a multi-event validation group and move one of its events to train
    idx = next(
        i
        for i, r in enumerate(rows)
        if r.split == "validation"
        and sum(
            1 for s in rows if s.fold_test_year == r.fold_test_year and s.group_id == r.group_id
        )
        > 1
    )
    rows[idx] = dataclasses.replace(rows[idx], split="train")
    with pytest.raises(AssertionError, match="split across"):
        validate_manifest(rows)


def test_a_test_year_group_leaking_into_training_is_rejected():
    rows = build_p1_manifest(*synthetic())
    idx = next(i for i, r in enumerate(rows) if r.split == "test")
    rows[idx] = dataclasses.replace(rows[idx], split="train")
    with pytest.raises(AssertionError):
        validate_manifest(rows)


def test_manifest_is_deterministic_for_a_seed_and_changes_with_the_seed():
    a = build_p1_manifest(*synthetic(), seed=0)
    b = build_p1_manifest(*synthetic(), seed=0)
    c = build_p1_manifest(*synthetic(), seed=1)
    assert a == b
    assert a != c


def test_round_trip_and_hash_ignore_line_endings(tmp_path):
    rows = build_p1_manifest(*synthetic())
    p = tmp_path / "m.csv"
    write_manifest(rows, p)
    assert read_manifest(p) == sorted(
        rows, key=lambda r: (r.fold_test_year, r.split, r.group_id, r.event_id)
    )
    h = sha256_file(p)
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))  # what a Windows checkout may do
    assert sha256_file(p) == h


def test_committed_p1_manifest_is_valid_and_unchanged(repo_root):
    manifest = repo_root / "data" / "manifests" / "p1_loyo_fold_manifest.csv"
    hashes = repo_root / "data" / "manifests" / "p1_split_hashes.txt"
    if not manifest.exists():
        pytest.skip("P1 manifest not generated yet")
    validate_manifest(read_manifest(manifest))
    recorded = dict(
        reversed(line.split("  ", 1))
        for line in hashes.read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert recorded["p1_loyo_fold_manifest.csv"] == sha256_file(manifest), (
        "p1_loyo_fold_manifest.csv changed after it was hashed; experiments must use it unchanged"
    )
    header = hashes.read_text().splitlines()[1]
    assert re.search(r"status=(FROZEN|PROVISIONAL)", header)
