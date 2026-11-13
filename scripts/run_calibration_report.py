"""Fold-specific calibration report (Role 4, P3-R4-05).

    python scripts/run_calibration_report.py

For the soft baselines (and, with ``--model name=<run folder>``, for model runs) writes
``reports/phase3/calibration_report.md`` and ``reports/phase3/reliability_tables.csv``:
per fold, test reliability over 15 equal-mass bins and ECE, raw and after the fold's own
validation-fitted Platt calibrator. Thresholds and calibrators come from validation only; a
calibrator is applied only to its own fold (``FoldCalibrator.transform`` refuses others). For
model runs it also checks there is exactly one selection record per fold, fitted on validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from fireguard_ml.baselines.distance_decay import DistanceCounts, decay_scores, fit_climatology
from fireguard_ml.data.event_splits import fold_events, read_manifest
from fireguard_ml.evaluation.calibration import fit_platt
from fireguard_ml.evaluation.metrics import ScoreCounts, ece_equal_mass, reliability_table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase3"
CACHE = ROOT / "outputs" / "role4" / "baseline_distance_counts.npz"
BASE_JSON = ROOT / "docs" / "evidence" / "role4" / "baseline_results.json"


def selection_records(run_dir: Path) -> dict[int, dict]:
    """Exactly one validation-fitted selection per fold, matching its fold (P3-R4-05)."""
    out = {}
    for fdir in sorted(run_dir.glob("fold_*")):
        sels = list(fdir.glob("*.selection.json"))
        if len(sels) != 1:
            raise SystemExit(f"{fdir}: {len(sels)} selection records, expected exactly 1")
        s = json.loads(sels[0].read_text())
        fold = int(fdir.name.split("_")[1])
        if s["fitted_on"] != "validation" or int(s["fold"]) != fold:
            raise SystemExit(f"{sels[0]}: not a validation fit for fold {fold}")
        if int(s["threshold"]["fold"]) != fold or int(s["calibrator"]["fold"]) != fold:
            raise SystemExit(f"{sels[0]}: threshold/calibrator fold mismatch")
        out[fold] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", default=[], help="name=<training run folder>")
    a = ap.parse_args()
    rows_m = read_manifest(ROOT / "data" / "manifests" / "p1_loyo_fold_manifest.csv")
    z = np.load(CACHE, allow_pickle=False)
    counts = {k: DistanceCounts.from_arrays(z[k]) for k in z.files if k != "_problems"}
    base = json.loads(BASE_JSON.read_text())
    clim_cfg = yaml.safe_load((ROOT / "configs" / "evaluation.yaml").read_text())["baselines"][
        "distance_bin_climatology"]  # fmt: skip
    table, summary = [], []
    for fold_s, f in sorted(base["folds"].items()):
        y = int(fold_s)
        part = {
            s: DistanceCounts.merge([counts[e] for e in fold_events(rows_m, y, s) if e in counts])
            for s in ("train", "validation", "test")
        }
        ch = f["chosen"]
        clim = fit_climatology(part["train"], y, "train", tuple(clim_cfg["edges_km"]),
                               clim_cfg["prior_strength"])  # fmt: skip
        scale, amp = ch["decay_scale_km"], ch["decay_amplitude"]
        makers = {
            "distance_decay": lambda c, s_=scale, a_=amp: decay_scores(c, s_, a_),
            "climatology": clim.scores,
        }
        for name, mk in makers.items():
            val_c = part["validation"].to_score_counts(mk(part["validation"]))
            cal = fit_platt(val_c, "P1", y, "validation")
            for kind in ("raw", "platt_validation_fitted"):
                s = mk(part["test"])
                if kind != "raw":
                    s = cal.transform(np.clip(s, 1e-12, 1 - 1e-12), y)
                tc: ScoreCounts = part["test"].to_score_counts(s)
                ece = ece_equal_mass(tc, 15)
                summary.append({"fold": y, "model": name, "probabilities": kind, "ece_test": ece,
                                "platt_a": cal.a, "platt_b": cal.b})  # fmt: skip
                for i, r in enumerate(reliability_table(tc, 15)):
                    table.append({"fold": y, "model": name, "probabilities": kind, "bin": i, **r})
    for spec in a.model:
        name, run = spec.split("=", 1)
        recs = selection_records(Path(run))
        for fold, s in recs.items():
            summary.append({"fold": fold, "model": name, "probabilities": "selection_record",
                            "ece_test": "", "platt_a": s["calibrator"]["a"],
                            "platt_b": s["calibrator"]["b"]})  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "reliability_tables.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    L = ["# Calibration report (generated)", "",
         "> `scripts/run_calibration_report.py`. Calibrators and thresholds are fitted on each "
         "fold's validation groups at natural prevalence and applied only to that fold.", "",
         "Uncertainty: **no uncertainty layer exists.** FireGuard outputs one calibrated or "
         "uncalibrated probability per cell; there is no ensemble spread or predictive-interval "
         "product. A probability is not an uncertainty estimate of itself.", "",
         "| Fold | Model | Probabilities | Test ECE (15 equal-mass bins) | Platt a | Platt b |",
         "|---:|---|---|---:|---:|---:|"]  # fmt: skip
    for s in summary:
        e = f"{s['ece_test']:.5f}" if isinstance(s["ece_test"], float) else "-"
        L.append(f"| {s['fold']} | {s['model']} | {s['probabilities']} | {e} | "
                 f"{s['platt_a']:.3f} | {s['platt_b']:.3f} |")  # fmt: skip
    L += ["", "Reliability tables (mean predicted vs observed per bin): "
          "`reports/phase3/reliability_tables.csv`.", ""]  # fmt: skip
    (OUT / "calibration_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[8:]))


if __name__ == "__main__":
    main()
