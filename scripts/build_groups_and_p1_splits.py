"""Build related-fire groups, run the Plan v4 diagnostics, and freeze + hash the P1 LOYO manifests.

Stage 1  load events (Role 1 metadata + this role's footprints)
Stage 2  group with the primary rule (30 km / 14 days, single linkage); if any group holds more than
         20 % of a year's events, switch to complete linkage with the same thresholds (Plan v4)
Stage 3  run the 50 km / 30 day sensitivity rule for comparison (reported, never used for splits)
Stage 4  build P1 leave-one-year-out folds with whole-group validation (~17.5 % of non-test groups)
Stage 5  validate every integrity rule, write manifests + reports, and hash them

Usage (from the repository root)::

    python scripts/build_groups_and_p1_splits.py

If ``data/manifests/wsts_event_footprints.csv`` does not exist yet the script stops, unless
``--allow-frame-proxy`` is given, in which case it uses Role 1's frame centres, ignores footprint
overlap, and labels every output PROVISIONAL (never freeze those).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml" / "src"))

from fireguard_ml.data.event_splits import (  # noqa: E402
    VALIDATION_TARGET,
    build_p1_manifest,
    sha256_file,
    validate_manifest,
    write_manifest,
)
from fireguard_ml.data.fire_complex_groups import (  # noqa: E402
    CELL_ROW_FACTOR,
    FOOTPRINT_RULE_VERSION,
    PIXEL_M,
    PRIMARY_RULE,
    SENSITIVITY_RULE,
    Event,
    any_warning,
    group_events,
    own_fire_cells,
    persistent_cells,
    yearly_largest_share,
)

MANIFESTS = REPO / "data" / "manifests"
DOCS = REPO / "docs"
SEED = 0
CLEANING: dict = {}  # filled by load_events; reported in diagnostics


def load_events(allow_proxy: bool) -> tuple[list[Event], str, int]:
    meta_path = MANIFESTS / "wsts_event_metadata.csv"
    with meta_path.open(newline="", encoding="utf-8-sig") as f:
        meta = {r["event_id"]: r for r in csv.DictReader(f)}
    fp_path = MANIFESTS / "wsts_event_footprints.csv"
    cells_path = MANIFESTS / "wsts_event_fire_cells.csv.gz"

    if fp_path.exists() and cells_path.exists():
        with fp_path.open(newline="", encoding="utf-8") as f:
            fps = {r["event_id"]: r for r in csv.DictReader(f)}
        cells: dict[str, frozenset[int]] = {}
        with gzip.open(cells_path, "rt", encoding="utf-8") as f:
            next(f)
            for line in f:
                ev, ids = line.rstrip("\n").split(",", 1)
                cells[ev] = frozenset(int(x) for x in ids.split()) if ids else frozenset()
        missing = set(meta) - set(fps)
        if missing:
            raise SystemExit(
                f"footprints missing for {len(missing)} events, e.g. {sorted(missing)[:3]}"
            )
        unreadable = sum(int(r["tiffs_unreadable"]) for r in fps.values())
        from pyproj import Transformer

        to_lonlat = Transformer.from_crs(32610, 4326, always_xy=True)
        year_of = {ev: int(m["year"]) for ev, m in meta.items()}
        excluded = persistent_cells(cells, year_of)
        events, kept_share = [], []
        for ev, m in sorted(meta.items()):
            fp = fps[ev]
            own = own_fire_cells(
                cells.get(ev, ()), (float(m["center_x"]), float(m["center_y"])), excluded
            )
            if own:
                # Dates: first/last day with any detection in the image (raw footprint run).
                # They can include a neighbouring fire's days, which only widens the interval
                # and therefore only ever adds groupings (conservative).
                start = date.fromisoformat(fp["fire_first_date"])
                end = date.fromisoformat(fp["fire_last_date"])
                ids = np.fromiter(own, dtype=np.int64)
                x = float(((ids % CELL_ROW_FACTOR) + 0.5).mean() * PIXEL_M)
                y = float(((ids // CELL_ROW_FACTOR) + 0.5).mean() * PIXEL_M)
                lon, lat = to_lonlat.transform(x, y)
                kept_share.append(len(own) / max(len(cells.get(ev, ())), 1))
            else:  # no detected fire: fall back to the image sequence and the frame centre
                start, end = date.fromisoformat(m["first_date"]), date.fromisoformat(m["last_date"])
                lon, lat = _frame_centre_lonlat(m)
            events.append(Event(ev, int(m["year"]), start, end, lon, lat, own or None))
        CLEANING.update(
            {
                "footprint_rule": FOOTPRINT_RULE_VERSION,
                "persistent_cells_removed": len(excluded),
                "events_with_own_fire": len(kept_share),
                "median_share_of_raw_cells_kept": float(np.median(kept_share))
                if kept_share
                else None,
            }
        )
        return events, "footprints", unreadable

    if not allow_proxy:
        raise SystemExit(
            "data/manifests/wsts_event_footprints.csv not found. Run "
            "scripts/derive_event_footprints.py first, or pass --allow-frame-proxy for a "
            "PROVISIONAL dry run."
        )
    events = [
        Event(
            ev,
            int(m["year"]),
            date.fromisoformat(m["first_date"]),
            date.fromisoformat(m["last_date"]),
            *_frame_centre_lonlat(m),
        )
        for ev, m in sorted(meta.items())
    ]
    return events, "frame_centre_proxy", 0


def _frame_centre_lonlat(m: dict) -> tuple[float, float]:
    from pyproj import Transformer

    t = Transformer.from_crs(32610, 4326, always_xy=True)
    return t.transform(float(m["center_x"]), float(m["center_y"]))


def grouping_table(events, result) -> list[dict]:
    year = {e.event_id: e.year for e in events}
    return [
        {
            "event_id": ev,
            "event_year": year[ev],
            "group_id": gid,
            "anchor_year": result.anchor_year[gid],
            "group_size": len(result.members[gid]),
        }
        for ev, gid in sorted(result.group_of.items())
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def first_target_index() -> int:
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "evaluation.yaml").read_text(encoding="utf-8"))
    return int(cfg["samples"]["first_target_index"])


def write_eval_samples(path: Path) -> int:
    """One row per scored (event, target day). Every model and baseline must use this list.

    Dates come from Role 1's metadata; Role 4's archive audit found every event's TIFF dates
    consecutive with no gaps or duplicates, so position k is first_date + k days.
    """
    from datetime import timedelta

    k0 = first_target_index()
    with (MANIFESTS / "wsts_event_metadata.csv").open(newline="", encoding="utf-8-sig") as f:
        meta = sorted(csv.DictReader(f), key=lambda r: r["event_id"])
    rows = []
    for m in meta:
        first = date.fromisoformat(m["first_date"])
        n = int(m["tiff_count"])
        if (date.fromisoformat(m["last_date"]) - first).days + 1 != n:
            raise SystemExit(f"{m['event_id']}: dates are not consecutive; fix before sampling")
        for k in range(k0, n):
            rows.append(
                {
                    "event_id": m["event_id"],
                    "event_year": m["year"],
                    "target_index": k,
                    "last_input_date": (first + timedelta(days=k - 1)).isoformat(),
                    "target_date": (first + timedelta(days=k)).isoformat(),
                }
            )
    write_csv(path, rows)
    return len(rows)


def describe(events, result) -> dict:
    shares = yearly_largest_share(events, result)
    cross_year = [
        g
        for g, m in result.members.items()
        if len({e.year for e in events if e.event_id in set(m)}) > 1
    ]
    return {
        "rule": f"{result.rule_km:g} km / {result.rule_days} d",
        "linkage": result.linkage,
        "related_pairs": dict(result.pair_reasons),
        "size_summary": result.size_summary(),
        "size_distribution": result.size_distribution(),
        "per_year": shares,
        "warning": any_warning(shares),
        "cross_year_groups": len(cross_year),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--allow-frame-proxy", action="store_true")
    args = ap.parse_args(argv)

    events, source, unreadable = load_events(args.allow_frame_proxy)
    status = "FROZEN" if source == "footprints" and unreadable == 0 else "PROVISIONAL"

    # Stage 2 - primary rule, with the pre-registered fallback
    primary = group_events(events, *PRIMARY_RULE, linkage="single")
    diag_primary = describe(events, primary)
    chosen, decision = primary, "single linkage kept: no group exceeds 20 % of any year's events"
    diag_complete = None
    if diag_primary["warning"]:
        chosen = group_events(events, *PRIMARY_RULE, linkage="complete")
        diag_complete = describe(events, chosen)
        decision = "20 % warning triggered: switched to complete linkage at 30 km / 14 d"

    # Stage 3 - sensitivity (reported only)
    sens = group_events(events, *SENSITIVITY_RULE, linkage="single")
    diag_sens = describe(events, sens)

    # Stage 4/5 - P1 folds
    event_year = {e.event_id: e.year for e in events}
    rows = build_p1_manifest(chosen.group_of, chosen.anchor_year, event_year, seed=SEED)
    fold_summary = validate_manifest(rows)
    sens_rows = build_p1_manifest(sens.group_of, sens.anchor_year, event_year, seed=SEED)
    sens_summary = validate_manifest(sens_rows)

    groups_path = MANIFESTS / "fire_complex_groups.csv"
    manifest_path = MANIFESTS / "p1_loyo_fold_manifest.csv"
    samples_path = MANIFESTS / "wsts_eval_samples.csv"
    write_csv(groups_path, grouping_table(events, chosen))
    write_manifest(rows, manifest_path)
    n_samples = write_eval_samples(samples_path)

    hashed = [manifest_path, groups_path, samples_path]
    for extra in ("wsts_event_footprints.csv", "wsts_event_metadata.csv"):
        if (MANIFESTS / extra).exists():
            hashed.append(MANIFESTS / extra)
    cells = MANIFESTS / "wsts_event_fire_cells.csv.gz"
    lines = [
        "# FireGuard P1 split hashes. SHA-256 with CRLF normalised to LF (binary files as-is).",
        f"# status={status} grouping_input={source} rule={chosen.rule_km:g}km/{chosen.rule_days}d "
        f"linkage={chosen.linkage} validation_target={VALIDATION_TARGET} seed={SEED} "
        f"footprint_rule={CLEANING.get('footprint_rule', 'none')}",
    ]
    lines += [f"{sha256_file(p)}  {p.name}" for p in hashed]
    if cells.exists():
        import hashlib

        lines.append(f"{hashlib.sha256(cells.read_bytes()).hexdigest()}  {cells.name}")
    (MANIFESTS / "p1_split_hashes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    diagnostics = {
        "status": status,
        "grouping_input": source,
        "footprint_cleaning": CLEANING,
        "tiffs_unreadable": unreadable,
        "events": len(events),
        "eval_samples": n_samples,
        "first_target_index": first_target_index(),
        "decision": decision,
        "primary_30km_14d_single": diag_primary,
        "primary_30km_14d_complete": diag_complete,
        "sensitivity_50km_30d_single": diag_sens,
        "p1_folds": fold_summary,
        "p1_folds_under_sensitivity_rule": sens_summary,
    }
    (DOCS / "evidence" / "role4").mkdir(parents=True, exist_ok=True)
    (DOCS / "evidence" / "role4" / "grouping_and_split_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    (DOCS / "fire_complex_groups_report.md").write_text(
        render_report(diagnostics, events, chosen), encoding="utf-8"
    )
    print(
        f"{status}: {len(chosen.members)} groups, {n_samples} scored samples, decision: {decision}"
    )
    for y, c in fold_summary.items():
        print(
            f"  fold {y}: "
            + ", ".join(
                f"{s} {c[s]['groups']}g/{c[s]['events']}e" for s in ("train", "validation", "test")
            )
            + f", validation share {c['validation_group_share']:.3f}"
        )
    return 0


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def render_report(d: dict, events, chosen) -> str:
    def size_table(diag):
        dist = diag["size_distribution"]
        return "\n".join(
            ["| Group size (events) | Number of groups |", "|---:|---:|"]
            + [f"| {k} | {v} |" for k, v in dist.items()]
        )

    def year_table(diag):
        out = [
            "| Year | Raw events | Groups anchored | Largest group | Its events in year | "
            "Share | > 20 %? |",
            "|---:|---:|---:|---|---:|---:|---|",
        ]
        for y, v in diag["per_year"].items():
            out.append(
                f"| {y} | {v['raw_events']} | {v['groups_anchored_to_year']} | "
                f"{v['largest_group_id']} | {v['largest_group_events_in_year']} | "
                f"{v['largest_group_share']:.1%} | {'YES' if v['warning'] else 'no'} |"
            )
        return "\n".join(out)

    def summary_line(diag):
        s = diag["size_summary"]
        return (
            f"{s['n_groups']} groups ({s['n_singletons']} singletons); size median "
            f"{s['median']:g}, p90 {s['p90']:g}, p95 {s['p95']:g}, max {s['max']}; "
            f"related pairs {diag['related_pairs']}; cross-year groups "
            f"{diag['cross_year_groups']}"
        )

    def fold_table(folds):
        out = [
            "| Test year | Train groups / events | Validation groups / events | "
            "Test groups / events | Validation share of non-test groups |",
            "|---:|---:|---:|---:|---:|",
        ]
        for y, c in folds.items():
            out.append(
                f"| {y} | {c['train']['groups']} / {c['train']['events']} | "
                f"{c['validation']['groups']} / {c['validation']['events']} | "
                f"{c['test']['groups']} / {c['test']['events']} | "
                f"{c['validation_group_share']:.1%} |"
            )
        return "\n".join(out)

    biggest = max(chosen.members.items(), key=lambda kv: len(kv[1]))
    p = d["primary_30km_14d_single"]
    parts = [
        "# Fire-complex grouping and P1 split report (generated)",
        "",
        "> Generated by `scripts/build_groups_and_p1_splits.py`. Do not edit by hand; re-run the "
        "script. Method and rationale: `docs/fire_complex_groups.md`.",
        "",
        f"**Status: {d['status']}** (grouping input: `{d['grouping_input']}`, unreadable TIFFs: "
        f"{d['tiffs_unreadable']}, events: {d['events']}).",
        "",
    ]
    cl = d.get("footprint_cleaning") or {}
    if cl:
        parts += [
            f"**Footprint rule {cl['footprint_rule']}:** {cl['persistent_cells_removed']} "
            "persistent thermal-anomaly cells (seen in events from 3+ different years) removed; "
            "each event keeps only its own fire clusters (within 5 km of the image centre); median "
            f"share of raw cells kept {cl['median_share_of_raw_cells_kept']:.1%}. See "
            "`docs/fire_complex_groups.md` section 3a.",
            "",
        ]
    if d["status"] != "FROZEN":
        parts += [
            "> **PROVISIONAL.** These groups were built without real fire footprints, so the "
            "footprint-overlap arm of the rule could not fire. Do not train on these manifests. "
            "Run `scripts/derive_event_footprints.py`, then re-run this script.",
            "",
        ]
    parts += [
        f"**Decision:** {d['decision']}.",
        "",
        "## Primary rule: 30 km / 14 days, single linkage",
        "",
        summary_line(p),
        "",
        size_table(p),
        "",
        year_table(p),
        "",
    ]
    if d["primary_30km_14d_complete"]:
        c = d["primary_30km_14d_complete"]
        parts += [
            "## Fallback actually used: 30 km / 14 days, complete linkage",
            "",
            summary_line(c),
            "",
            size_table(c),
            "",
            year_table(c),
            "",
        ]
    s = d["sensitivity_50km_30d_single"]
    parts += [
        "## Sensitivity rule: 50 km / 30 days, single linkage (reported, not used for splits)",
        "",
        summary_line(s),
        "",
        size_table(s),
        "",
        year_table(s),
        "",
        f"Largest group under the chosen rule: `{biggest[0]}` with {len(biggest[1])} events: "
        + ", ".join(f"`{e}`" for e in biggest[1])
        + ".",
        "",
        "## P1 leave-one-year-out folds (chosen rule)",
        "",
        fold_table(d["p1_folds"]),
        "",
        "Every fold passed `validate_manifest`: no group spans two splits, test = exactly the "
        "groups anchored to the test year, validation groups come only from the other anchor "
        "years, and every event is tested in exactly one fold.",
        "",
        "**2019 caveat:** 2019 has 74 raw events against 156-201 in the other years, so the 2019 "
        "test fold has the fewest groups and its metrics will have the widest confidence "
        "intervals. "
        "Report it, but do not read a 2019-only difference as strong evidence.",
        "",
        "## P1 folds under the sensitivity rule (comparison only)",
        "",
        fold_table(d["p1_folds_under_sensitivity_rule"]),
        "",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
