"""Next-day observability diagnostic, EVALUATION ONLY (Role 4, P3-R4-04).

    python scripts/run_observability_diagnostic.py --data-root <folder holding 2018..2021>

Step 1 (needs the data): for every scored sample, the share of cells in the TARGET day's file
whose three VIIRS reflectance bands (M11, I2, I1; VNP09GA) are all present. Missing reflectance
on day D+1 is the observability proxy available inside WSTS; the VNP09GA QA flags are not in the
files, so cloud/smoke flags cannot be used. Written to
``data/evaluation_only/next_day_observability.csv``. This is target-day information: no model,
normaliser, feature selection or threshold may read it (``test_observability_is_evaluation_only``).

Step 2: each test event gets a stratum from its mean next-day valid fraction with the threshold
pre-registered in Phase 2 (``data.observability.next_day_observability``, 0.5). Event-macro AUPRC
per stratum, model and fold is written to ``reports/phase3/observability_strata.csv``; a stratum
with fewer than 10 events on a fold is reported as INSUFFICIENT, not as a number. Primary scores
are never altered. ``--skip-compute`` reuses an existing step-1 file. ``--model name=csv`` adds
model per-event files (as in run_model_comparison.py).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from fireguard_ml.data.observability import next_day_observability

ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / "data" / "evaluation_only" / "next_day_observability.csv"
OUT = ROOT / "reports" / "phase3" / "observability_strata.csv"
BASE = ROOT / "docs" / "evidence" / "role4" / "baseline_per_event_test_scores.csv"
MIN_EVENTS = 10
REFLECTANCE_BANDS_1BASED = (1, 2, 3)


def compute(data_root: Path) -> None:
    import rasterio

    with (ROOT / "data" / "manifests" / "wsts_eval_samples.csv").open(encoding="utf-8") as f:
        samples = list(csv.DictReader(f))
    rows, t0 = [], time.time()
    for i, s in enumerate(samples, 1):
        p = data_root / s["event_year"] / s["event_id"] / f"{s['target_date']}.tif"
        if not p.exists():
            rows.append({**_key(s), "valid_fraction": "", "status": "missing_file"})
            continue
        with rasterio.open(p) as ds:
            r = ds.read(list(REFLECTANCE_BANDS_1BASED))
        frac = float(np.isfinite(r).all(axis=0).mean())
        rows.append({**_key(s), "valid_fraction": round(frac, 6), "status": "ok"})
        if i % 1000 == 0:
            print(f"  {i}/{len(samples)} ({time.time() - t0:.0f}s)", flush=True)
    OBS.parent.mkdir(parents=True, exist_ok=True)
    with OBS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _key(s: dict) -> dict:
    return {"event_id": s["event_id"], "target_date": s["target_date"], "evaluation_only": True}


def report(models: dict[str, str]) -> list[dict]:
    with OBS.open(encoding="utf-8") as f:
        obs = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    by_event = defaultdict(list)
    for r in obs:
        by_event[r["event_id"]].append(float(r["valid_fraction"]))
    tags = next_day_observability({e: float(np.mean(v)) for e, v in by_event.items()})
    per = defaultdict(lambda: defaultdict(dict))  # fold -> model -> event -> auprc
    for name, path in [("__baselines__", str(BASE)), *models.items()]:
        with Path(path).open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m = r["baseline"] if name == "__baselines__" else name
                if m in ("distance_decay", "climatology") or name != "__baselines__":
                    per[int(r["fold_test_year"])][m][r["event_id"]] = _num(r["auprc"])
    out = []
    for fold in sorted(per):
        for model, ev in sorted(per[fold].items()):
            for stratum in ("better_observed", "poorly_observed"):
                vals = [v for e, v in ev.items() if e in tags
                        and tags[e].value["stratum"] == stratum and math.isfinite(v)]  # fmt: skip
                n_ev = sum(1 for e in ev if e in tags and tags[e].value["stratum"] == stratum)
                ok = len(vals) >= MIN_EVENTS
                out.append({"fold": fold, "model": model, "stratum": stratum, "events": n_ev,
                            "events_with_defined_auprc": len(vals),
                            "event_macro_auprc": round(float(np.mean(vals)), 6) if ok else "",
                            "status": "OK" if ok else "INSUFFICIENT",
                            "label": "diagnostic, evaluation-only"})  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    return out


def _num(v) -> float:
    try:
        return float(v)
    except ValueError:
        return math.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--skip-compute", action="store_true")
    ap.add_argument("--model", action="append", default=[])
    a = ap.parse_args()
    if not a.skip_compute:
        if a.data_root is None:
            raise SystemExit("--data-root is required unless --skip-compute")
        compute(a.data_root)
    rows = report(dict(m.split("=", 1) for m in a.model))
    for r in rows:
        print(
            r["fold"], r["model"], r["stratum"], r["events"], r["event_macro_auprc"] or r["status"]
        )


if __name__ == "__main__":
    main()
