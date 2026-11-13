"""Validate the selected OOF run and build the reveal package (Role 4, P4-R4-01/02/03).

    python scripts/validate_oof_and_build_reveal.py --artifact-root artifacts --run-id <id> \
        --data-root <WSTS folder> [--folds 2018 2019 2020 2021]

1. Coverage and mapping (P4-R4-01): every OOF record re-checked against the frozen manifests;
   writes ``runs/<id>/role4_oof_validation.json`` with coverage, rejected and missing
   (unavailable) forecasts. Exit 1 on any rejection.
2. Metric provenance (P4-R4-03): per-event metrics recomputed from the rasters and the WSTS
   target with each fold's own validation threshold; must equal Role 3's stored file.
3. Reveal package (P4-R4-02) under ``<artifact-root>/reveal_packages/<id>/``, deliberately
   outside ``runs/`` (which holds pre-reveal artifacts): ``pre_reveal/forecast_sessions.json``
   (no target content, checked) and ``reveal_store/`` (actual targets, observability,
   post-reveal metrics). Needs Role 1's verified ``replay_catalogue.csv`` in the run folder.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from fireguard_ml.data.event_splits import sha256_file
from fireguard_ml.evaluation.oof_validation import (
    approved_from_model_version,
    compare_metrics,
    recompute_event_metrics,
    thresholds_from_model_version,
    validate_oof,
)
from fireguard_ml.evaluation.reveal import build_reveal_package

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests"
STRICT = ("viirs_m11", "viirs_i2", "viirs_i1", "slope", "aspect", "elevation",
          "binary_active_fire_mask")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--manifest-dir", type=Path, default=MAN)
    a = ap.parse_args()
    run = a.artifact_root / "runs" / a.run_id
    oof = run / "oof_forecasts.jsonl"
    split_hash = sha256_file(a.manifest_dir / "p1_loyo_fold_manifest.csv")
    if "status=FROZEN" not in (a.manifest_dir / "p1_split_hashes.txt").read_text():
        raise SystemExit("P1 is not FROZEN")
    ok = approved_from_model_version(run, split_hash, STRICT)
    rep = validate_oof(oof, a.manifest_dir / "p1_loyo_fold_manifest.csv",
                       a.manifest_dir / "wsts_eval_samples.csv", ok,
                       set(a.folds) if a.folds else None)  # fmt: skip

    from fireguard_ml.data.wsts_dataset import TARGET_BAND_1BASED, scan_events, target_from_band

    events = scan_events(a.data_root)

    def target(ev: str, d: str):
        import rasterio

        with rasterio.open(events[ev].path(date.fromisoformat(d))) as ds:
            return target_from_band(ds.read(TARGET_BAND_1BASED))

    thr = thresholds_from_model_version(run)
    recomputed = recompute_event_metrics(oof, a.artifact_root, target, thr)
    metric_errors = compare_metrics(run / "post_reveal" / "per_event_test_scores.csv", recomputed)
    out = {"oof": rep.summary(), "rejected": rep.rejected[:500],
           "missing_unavailable": rep.missing[:2000], "duplicates": rep.duplicates,
           "metric_mismatches": metric_errors, "thresholds_by_fold": thr}  # fmt: skip
    if rep.ok and not metric_errors:
        obs = ROOT / "data" / "evaluation_only" / "next_day_observability.csv"
        out["reveal"] = build_reveal_package(
            run / "replay_catalogue.csv", oof, run, a.data_root,
            a.artifact_root / "reveal_packages" / a.run_id, obs,
            run / "post_reveal" / "per_event_test_scores.csv")  # fmt: skip
    (run / "role4_oof_validation.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({"oof": rep.summary(), "metric_mismatches": len(metric_errors),
                      "reveal_sessions": out.get("reveal", {}).get("sessions")}))  # fmt: skip
    raise SystemExit(0 if rep.ok and not metric_errors else 1)


if __name__ == "__main__":
    main()
