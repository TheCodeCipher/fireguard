"""Collect the M1 fold-2018 preview scores next to the baselines on the same fires.

    python scripts/build_m1_preview_results.py

Reads only files the pipeline wrote (nothing typed by hand):
    artifacts/runs/m1_fold2018_preview/post_reveal/per_event_test_scores.csv   (M1, Role 4 scoring)
    docs/evidence/role4/baseline_per_event_test_scores.csv                     (baselines)
    outputs/m1/fold_2018/m1_spatial_unet_T1_seed0.summary.json                (validation AUPRC)
Writes docs/evidence/role3/m1_fold2018_preview.json for the Results page.

PROTOTYPE: this is a preview on the five largest 2018 test fires (picked by fire size before any
scoring), not the Gate 2 evaluation, which needs every test fire of all four folds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "runs" / "m1_fold2018_preview"
M1 = RUN / "post_reveal" / "per_event_test_scores.csv"
BASE = ROOT / "docs" / "evidence" / "role4" / "baseline_per_event_test_scores.csv"
SUMMARY = ROOT / "outputs" / "m1" / "fold_2018" / "m1_spatial_unet_T1_seed0.summary.json"
OUT = ROOT / "docs" / "evidence" / "role3" / "m1_fold2018_preview.json"
MODELS = ("m1", "distance_decay", "climatology")
METRICS = ("auprc", "f1", "iou", "brier")


def _f(v: str) -> float | None:
    return float(v) if v not in ("", "nan") else None


def main() -> None:
    with M1.open(encoding="utf-8") as fh:
        m1 = {r["event_id"]: r for r in csv.DictReader(fh)}
    with BASE.open(encoding="utf-8") as fh:
        base = [r for r in csv.DictReader(fh)
                if r["event_id"] in m1 and r["fold_test_year"] == "2018"]
    rows = []
    for ev, r in sorted(m1.items()):
        row = {"event_id": ev, "n_pos": int(r["n_pos"]), "n_valid": int(r["n_valid"]),
               "m1": {k: _f(r[k]) for k in METRICS}}  # fmt: skip
        for b in ("distance_decay", "climatology"):
            br = next(x for x in base if x["event_id"] == ev and x["baseline"] == b)
            row[b] = {k: _f(br[k]) for k in METRICS}
        rows.append(row)
    mean = {m: {k: sum(r[m][k] for r in rows) / len(rows) for k in METRICS} for m in MODELS}
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    out = {
        "scope": "preview: M1 fold 2018 on 5 of the 220 test fires (the largest, chosen by size "
                 "before scoring); not the Gate 2 evaluation",
        "fold_test_year": 2018,
        "n_fires": len(rows),
        "n_test_fires_in_fold": 220,
        "validation_auprc": summary["val_auprc"],
        "best_epoch": summary["best_epoch"],
        "fires": rows,
        "mean_over_fires": mean,
        "m1_better_auprc_than_distance_decay": sum(
            r["m1"]["auprc"] > r["distance_decay"]["auprc"] for r in rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(OUT.relative_to(ROOT)), "mean_auprc":
                      {m: round(mean[m]["auprc"], 4) for m in MODELS}}))


if __name__ == "__main__":
    main()
