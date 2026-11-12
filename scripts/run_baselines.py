"""Run the Phase 2 baselines on every P1 fold and write docs/baseline_results.md.

Stage A (one pass over the data, cached): for every scored sample in
``data/manifests/wsts_eval_samples.csv`` read the active-fire band of the last input day and of the
target day, and accumulate per-event counts of next-day fire by distance to today's fire.

Stage B (per fold, seconds): fit the climatology on training groups, choose the decay scale, the
dilation radius and the F1 thresholds on validation groups, then score the test groups once.

Usage (repo root, scripts extra installed)::

    python scripts/run_baselines.py --data-root <folder with 2018..2021>

Add ``--events fire_a fire_b ...`` for a quick trial; trial results are never written to docs/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml" / "src"))

from fireguard_ml.baselines.distance_decay import (  # noqa: E402
    DistanceCounts,
    decay_scores,
    dilated_scores,
    fit_climatology,
    persistence_scores,
    select_decay_scale,
)
from fireguard_ml.baselines.persistence import fire_mask  # noqa: E402
from fireguard_ml.data.event_splits import (  # noqa: E402
    fold_events,
    read_manifest,
    sha256_file,
    validate_manifest,
)
from fireguard_ml.evaluation.bootstrap import paired_group_bootstrap  # noqa: E402
from fireguard_ml.evaluation.metrics import (  # noqa: E402
    average_precision_from_counts,
    best_f1_threshold,
    brier_from_counts,
    brier_skill_score,
    f1_iou_at_threshold,
)

try:
    import rasterio
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f'Missing dependency ({exc.name}). Run: pip install -e "ml[scripts,dev]"'
    ) from exc

SOFT = ("distance_decay", "climatology")
BINARY = ("persistence", "dilated_persistence")


# ------------------------------------------------------------------------------------- stage A


def read_fire(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return fire_mask(src.read(src.count))


def cached_fire(frames: dict, folder: Path, day: str, problems: dict, ev: str):
    """Read one day's fire mask once per event; record (not raise) unreadable files."""
    if day not in frames:
        try:
            frames[day] = read_fire(folder / f"{day}.tif")
        except Exception as exc:  # noqa: BLE001
            frames[day] = None
            problems.setdefault(ev, []).append(f"{day}: {type(exc).__name__}")
    return frames[day]


def collect_counts(data_root: Path, samples: list[dict], cache: Path | None) -> tuple[dict, dict]:
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        counts = {k: DistanceCounts.from_arrays(z[k]) for k in z.files if k != "_problems"}
        problems = json.loads(str(z["_problems"])) if "_problems" in z.files else {}
        print(f"Stage A: loaded cached counts for {len(counts)} events from {cache}")
        return counts, problems
    by_event: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_event[s["event_id"]].append(s)
    counts, problems = {}, {}
    t0 = time.time()
    for k, (ev, rows) in enumerate(sorted(by_event.items()), 1):
        folder = data_root / rows[0]["event_year"] / ev
        frames: dict[str, np.ndarray | None] = {}
        parts = []
        for s in rows:
            today = cached_fire(frames, folder, s["last_input_date"], problems, ev)
            tomorrow = cached_fire(frames, folder, s["target_date"], problems, ev)
            if today is None or tomorrow is None:
                continue
            parts.append(DistanceCounts.from_frame(today, tomorrow))
        counts[ev] = DistanceCounts.merge(parts)
        if k % 25 == 0 or k == len(by_event):
            print(f"  Stage A {k}/{len(by_event)} events ({time.time() - t0:.0f}s)", flush=True)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, _problems=json.dumps(problems), **{e: c.to_arrays() for e, c in counts.items()}
        )
    return counts, problems


# ------------------------------------------------------------------------------------- stage B


def evaluate_fold(year: int, rows, counts: dict, cfg: dict, group_of: dict) -> dict:
    b = cfg["baselines"]
    train = DistanceCounts.merge(
        [counts[e] for e in fold_events(rows, year, "train") if e in counts]
    )
    val = DistanceCounts.merge(
        [counts[e] for e in fold_events(rows, year, "validation") if e in counts]
    )
    test_events = [e for e in fold_events(rows, year, "test") if e in counts]

    clim = fit_climatology(
        train,
        year,
        "train",
        tuple(b["distance_bin_climatology"]["edges_km"]),
        b["distance_bin_climatology"]["prior_strength"],
    )
    scale, amplitude, scale_diag = select_decay_scale(
        val, b["distance_decay"]["scale_km_candidates"], "validation"
    )
    radius_f1 = {
        r: f1_iou_at_threshold(val.to_score_counts(dilated_scores(val, r)), 0.5)["f1"]
        for r in b["dilated_persistence"]["radius_km_candidates"]
    }
    radius = max(radius_f1, key=lambda r: (np.nan_to_num(radius_f1[r], nan=-1), -r))

    def scores(name, c):
        return {
            "distance_decay": lambda: decay_scores(c, scale, amplitude),
            "climatology": lambda: clim.scores(c),
            "persistence": lambda: persistence_scores(c),
            "dilated_persistence": lambda: dilated_scores(c, radius),
        }[name]()

    thresholds = {n: best_f1_threshold(val.to_score_counts(scores(n, val)))[0] for n in SOFT}
    thresholds |= {n: 0.5 for n in BINARY}

    per_event = defaultdict(dict)
    pooled_parts = defaultdict(list)
    for e in test_events:
        c = counts[e]
        for name in SOFT + BINARY:
            sc = c.to_score_counts(scores(name, c))
            pooled_parts[name].append(sc)
            m = f1_iou_at_threshold(sc, thresholds[name])
            rec = {"f1": m["f1"], "iou": m["iou"], "n_pos": sc.n_pos, "n_valid": sc.n}
            if name in SOFT:
                rec["auprc"] = average_precision_from_counts(sc)
                rec["brier"] = brier_from_counts(sc)
            per_event[name][e] = rec

    from fireguard_ml.evaluation.metrics import merge_counts

    pooled = {}
    for name in SOFT + BINARY:
        pc = merge_counts(pooled_parts[name])
        m = f1_iou_at_threshold(pc, thresholds[name])
        pooled[name] = {"f1": m["f1"], "iou": m["iou"], "n_pos": pc.n_pos, "n_valid": pc.n}
        if name in SOFT:
            pooled[name]["auprc"] = average_precision_from_counts(pc)
            pooled[name]["brier"] = brier_from_counts(pc)
    ref = pooled["climatology"]["brier"]
    for name in SOFT:
        pooled[name]["bss_vs_climatology"] = brier_skill_score(pooled[name]["brier"], ref)

    def macro(name, key):
        v = np.array([r[key] for r in per_event[name].values()], dtype=float)
        v = v[np.isfinite(v)]
        return (float(v.mean()) if v.size else float("nan")), int(v.size)

    macro_tbl = {
        name: {
            key: macro(name, key)
            for key in (("auprc", "f1", "iou") if name in SOFT else ("f1", "iou"))
        }
        for name in SOFT + BINARY
    }
    boot_cfg = cfg["metrics"]["bootstrap"]
    boot = paired_group_bootstrap(
        {e: r["auprc"] for e, r in per_event["distance_decay"].items()},
        {e: r["auprc"] for e, r in per_event["climatology"].items()},
        group_of,
        n_boot=boot_cfg["n_boot"],
        seed=boot_cfg["seed"],
        ci=boot_cfg["ci"],
    )
    return {
        "test_events": len(test_events),
        "test_groups": len({group_of[e] for e in test_events}),
        "chosen": {
            "decay_scale_km": scale,
            "decay_amplitude": amplitude,
            "dilation_radius_km": radius,
            "thresholds": thresholds,
            "climatology_probs": clim.probs,
            "climatology_train_prevalence": clim.train_prevalence,
        },
        "validation_diagnostics": {"decay": scale_diag, "dilation_f1_by_radius": radius_f1},
        "pooled": pooled,
        "event_macro": macro_tbl,
        "paired_bootstrap_decay_minus_climatology_auprc": boot.__dict__,
        "per_event": per_event,
    }


# ------------------------------------------------------------------------------------- report


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.4f}"


def render(results: dict, meta: dict) -> str:
    out = [
        "# Baseline results, P1 leave-one-year-out (generated)",
        "",
        "> Generated by `scripts/run_baselines.py`; do not edit by hand. Protocol: "
        "`docs/evaluation_protocol.md`; metric definitions: `docs/metrics_contract.md`.",
        "",
        f"- Manifest status: **{meta['manifest_status']}**; manifest sha256 "
        f"`{meta['manifest_sha256'][:16]}...`",
        f"- Evaluation config sha256 `{meta['config_sha256'][:16]}...`; samples sha256 "
        f"`{meta['samples_sha256'][:16]}...` ({meta['n_samples']} scored samples)",
        f"- Events with unreadable TIFFs: {meta['events_with_problems']}",
        "- Downwind-weighted decay: **disabled** (no wind direction has passed the encoding and "
        "Level-A audits).",
        "",
    ]
    if meta["manifest_status"] != "FROZEN":
        out += [
            "> **PROVISIONAL**: computed on manifests built without real fire footprints. "
            "Re-run after the frozen manifests exist; do not quote these numbers.",
            "",
        ]
    out += [
        "## Pooled test metrics per fold (AUPRC, Brier and BSS for soft baselines only)",
        "",
        "| Test year | Events / groups | Prevalence | Decay AUPRC | Climatology AUPRC | "
        "Decay Brier | Climatology Brier | Decay BSS | Decay F1 / IoU | Climatology F1 / IoU | "
        "Persistence F1 / IoU | Dilated F1 / IoU |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for y, r in results.items():
        p = r["pooled"]
        prev = p["climatology"]["n_pos"] / max(p["climatology"]["n_valid"], 1)
        out.append(
            f"| {y} | {r['test_events']} / {r['test_groups']} | {prev:.3%} | "
            f"{fmt(p['distance_decay']['auprc'])} | {fmt(p['climatology']['auprc'])} | "
            f"{fmt(p['distance_decay']['brier'])} | {fmt(p['climatology']['brier'])} | "
            f"{fmt(p['distance_decay']['bss_vs_climatology'])} | "
            f"{fmt(p['distance_decay']['f1'])} / {fmt(p['distance_decay']['iou'])} | "
            f"{fmt(p['climatology']['f1'])} / {fmt(p['climatology']['iou'])} | "
            f"{fmt(p['persistence']['f1'])} / {fmt(p['persistence']['iou'])} | "
            f"{fmt(p['dilated_persistence']['f1'])} / {fmt(p['dilated_persistence']['iou'])} |"
        )
    out += [
        "",
        "## Event-macro test metrics per fold (mean over events; n = events with a defined value)",
        "",
        "| Test year | Decay AUPRC (n) | Climatology AUPRC (n) | Decay F1 | Climatology F1 | "
        "Persistence F1 | Dilated F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for y, r in results.items():
        m = r["event_macro"]
        out.append(
            f"| {y} | {fmt(m['distance_decay']['auprc'][0])} ({m['distance_decay']['auprc'][1]}) | "
            f"{fmt(m['climatology']['auprc'][0])} ({m['climatology']['auprc'][1]}) | "
            f"{fmt(m['distance_decay']['f1'][0])} | {fmt(m['climatology']['f1'][0])} | "
            f"{fmt(m['persistence']['f1'][0])} | {fmt(m['dilated_persistence']['f1'][0])} |"
        )
    out += [
        "",
        "## Paired event-group bootstrap: decay minus climatology, per-event AUPRC",
        "",
        "| Test year | Mean difference | 95 % CI | Groups | Events | Excluded (no next-day fire) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for y, r in results.items():
        b = r["paired_bootstrap_decay_minus_climatology_auprc"]
        out.append(
            f"| {y} | {b['estimate']:+.4f} | [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] | "
            f"{b['n_groups']} | {b['n_events']} | {b['excluded_events']} |"
        )
    out += [
        "",
        "## Parameters chosen on each fold's validation groups (never on test)",
        "",
        "| Test year | Decay scale (km) | Decay amplitude | Dilation radius (km) | "
        "Decay F1 threshold | Climatology F1 threshold |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for y, r in results.items():
        c = r["chosen"]
        out.append(
            f"| {y} | {c['decay_scale_km']:g} | {c['decay_amplitude']:.4f} | "
            f"{c['dilation_radius_km']:g} | "
            f"{c['thresholds']['distance_decay']:.4f} | {c['thresholds']['climatology']:.4f} |"
        )
    out += [
        "",
        "## Reading these numbers",
        "",
        "- **The strongest soft baseline is the bar M1 must clear at Gate 2**, fold by fold, "
        "with the paired event-group bootstrap (Plan v4 6.6).",
        "- Distance-decay AUPRC does not depend on its scale or amplitude (any decreasing "
        "function of distance ranks cells identically); both are fitted on validation by Brier.",
        "- Persistence and dilated persistence are binary maps, so only F1/IoU are reported.",
        "- **2019 caveat:** 2019 has 74 raw events versus 156-201 in other years; its fold has "
        "the fewest test groups and the widest intervals. Report it, but do not over-read it.",
        "- Event-macro AUPRC excludes events with no next-day fire in any scored sample (AUPRC "
        "is undefined there); the number used is shown as n.",
        "",
    ]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--events", nargs="*", default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    cfg_path = REPO / "configs" / "evaluation.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    manifest_path = REPO / cfg["manifests"]["folds"]
    rows = read_manifest(manifest_path)
    validate_manifest(rows)
    recorded = {
        name: h
        for h, name in (
            line.split("  ", 1)
            for line in (REPO / cfg["manifests"]["hashes"]).read_text().splitlines()
            if line and not line.startswith("#")
        )
    }
    if recorded.get(manifest_path.name) != sha256_file(manifest_path):
        raise SystemExit("P1 manifest does not match p1_split_hashes.txt; refusing to run")
    status = (
        "FROZEN"
        if "status=FROZEN" in (REPO / cfg["manifests"]["hashes"]).read_text()
        else "PROVISIONAL"
    )

    samples_path = REPO / cfg["manifests"]["eval_samples"]
    with samples_path.open(newline="", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))
    trial = bool(args.events)
    if trial:
        samples = [s for s in samples if s["event_id"] in set(args.events)]
    cache = (
        None
        if (args.no_cache or trial)
        else REPO / "outputs" / "role4" / "baseline_distance_counts.npz"
    )
    counts, problems = collect_counts(args.data_root, samples, cache)

    group_of = {r.event_id: r.group_id for r in rows if r.fold_test_year == rows[0].fold_test_year}
    results = {}
    for y in sorted({r.fold_test_year for r in rows}):
        if trial:
            have = {
                s: [e for e in fold_events(rows, y, s) if e in counts and counts[e].pos.sum()]
                for s in ("train", "validation", "test")
            }
            if not all(have.values()):
                print(
                    f"  fold {y}: skipped in trial (needs fire in train, validation and test; "
                    f"have {', '.join(f'{s} {len(v)}' for s, v in have.items())})"
                )
                continue
        results[y] = evaluate_fold(y, rows, counts, cfg, group_of)

    meta = {
        "manifest_status": status,
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "samples_sha256": sha256_file(samples_path),
        "n_samples": len(samples),
        "events_with_problems": len(problems),
        "problems": problems,
    }
    if trial:
        print(json.dumps({y: r["pooled"] for y, r in results.items()}, indent=1, default=float))
        print("Trial run: nothing written to docs/.")
        return 0
    (REPO / "docs" / "baseline_results.md").write_text(render(results, meta), encoding="utf-8")
    ev_dir = REPO / "docs" / "evidence" / "role4"
    ev_dir.mkdir(parents=True, exist_ok=True)
    slim = {y: {k: v for k, v in r.items() if k != "per_event"} for y, r in results.items()}
    (ev_dir / "baseline_results.json").write_text(
        json.dumps({"meta": meta, "folds": slim}, indent=2, default=float) + "\n", encoding="utf-8"
    )
    with (ev_dir / "baseline_per_event_test_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "fold_test_year",
                "event_id",
                "group_id",
                "baseline",
                "auprc",
                "brier",
                "f1",
                "iou",
                "n_pos",
                "n_valid",
            ]
        )
        for y, r in results.items():
            for name, evs in r["per_event"].items():
                for e, m in sorted(evs.items()):
                    w.writerow(
                        [
                            y,
                            e,
                            group_of[e],
                            name,
                            m.get("auprc", ""),
                            m.get("brier", ""),
                            m["f1"],
                            m["iou"],
                            m["n_pos"],
                            m["n_valid"],
                        ]
                    )
    print(f"Wrote docs/baseline_results.md ({status}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
