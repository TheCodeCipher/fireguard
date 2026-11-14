"""Bridge Role 4's reveal package to Role 5's replay registry (Integration, Prototype, P4 glue).

    python scripts/build_backend_registry.py --artifact-root artifacts --run-id <run> \
        --data-root <WSTS folder> --backend-artifact-root backend/artifacts --out <registry folder>

WHY THIS EXISTS
    The Phase 4 chain was built by four people against four documents. Every link existed except
    the last one: Role 4 writes a reveal package (pre_reveal/forecast_sessions.json + a separate
    reveal_store/), Role 5 registers artifacts from a per-forecast manifest
    (docs/interfaces/phase4_oof_artifact_contract.md) into its own storage layout. Nothing
    converted one into the other, and the two disagree in small ways that would each have
    rejected a real registration. This script is that conversion, with every mapping explicit:

    Role 3/4 output                              ->  Role 5 registry
    event_id (fire_24025599)                     ->  event_key, same string (catalogue CSV written)
    fold 2020                                    ->  fold_id "LOYO-2020" (p1_fold_id in catalogue)
    time safety STRICT_LEVEL_A_PASS_CHANNELS_ONLY ->  PASS_LEVEL_A (same meaning: every input
                                                     channel is Level-A PASS)
    one model_version, one checkpoint PER FOLD   ->  ModelVersion holds ONE checkpoint hash, so
                                                     one Role 5 version per fold "<v>@fold-2020"
    data_contract_version + Role 1 contract hash ->  data_schema_version + data_schema_sha256
    Role 1 phase3_input_manifest.csv sha256      ->  input_manifest_sha256
    runs/<id>/<event>/<origin>/probability.tif   ->  p1_loyo/<model>/<fold>/<event>/<origin>/
                                                     probability.tif (pre-reveal)
    reveal_store/actual/<forecast_id>.tif        ->  restricted/.../actual_target.tif (post-reveal)
    reveal_index event_metrics_post_reveal       ->  restricted/.../metrics.json + ReplayMetric rows

    It also fills one content gap: Role 4's reveal index carries EVENT-level metrics (all days of
    the event pooled), but the replay screen compares ONE forecast day with its target. The
    per-forecast-day metrics are computed here, offline, with Role 4's own metric functions on
    the exact two rasters being registered, so the API still only serves stored numbers.

MOCK VERSUS REAL (fail closed)
    --mode auto (default) chooses "mock" if ANY of these hold, otherwise "real":
      * the data root contains SYNTHETIC_NOT_WSTS.txt (scripts/make_synthetic_wsts.py);
      * the run folder contains PROTOTYPE_PLUMBING_RUN.json (scripts/run_prototype_pipeline.py);
      * configs/temporal_unet.yaml has training.enabled: false (Gate 2 not recorded);
      * model_version.json or any record says is_mock: true.
    --mode real is refused while any of them holds. Mock output uses Role 5's MOCK_ONLY protocol
    and "mock" artifact status, which Role 5's API shows only in development and labels as mock,
    and which Role 5's registry can never publish.

OUTPUTS
    <backend-artifact-root>/...        copies of the rasters in Role 5's layout
    <out>/event_catalogue.csv          for backend/scripts/import_wsts_event_catalogue.py
    <out>/registry_bundle.json         input of backend/scripts/load_registry_bundle.py
    <out>/manifests/<forecast_id>.json (real mode) Role 5 manifests (register_oof_artifacts.py)

Nothing here changes the ML outputs; the source files are copied, checksummed and re-verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
import yaml

from fireguard_ml.evaluation.metrics import counts_by_score, summarise
from fireguard_ml.evaluation.reveal import find_target_leaks

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_MARKER = "SYNTHETIC_NOT_WSTS.txt"
PLUMBING_MARKER = "PROTOTYPE_PLUMBING_RUN.json"
# Role 3's wording -> Role 5's wording. Only this one value maps to a publishable status.
TIME_SAFETY_MAP = {"STRICT_LEVEL_A_PASS_CHANNELS_ONLY": "PASS_LEVEL_A"}
METRIC_NAMES = ("auprc", "brier", "f1", "iou", "n_pos", "n_valid")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


P1_FOLDS = (2018, 2019, 2020, 2021)


def mock_reasons(
    data_root: Path, run_dir: Path, model_version: dict, records: list[dict]
) -> list:
    """Every reason the run must stay mock. Empty list = eligible for real registration."""
    reasons = []
    if (data_root / SYNTHETIC_MARKER).exists():
        reasons.append(
            f"data root holds {SYNTHETIC_MARKER} (synthetic pixels, not WSTS)"
        )
    if (run_dir / PLUMBING_MARKER).exists():
        reasons.append(f"run holds {PLUMBING_MARKER} (plumbing training budget)")
    cfg = yaml.safe_load(
        (ROOT / "configs" / "temporal_unet.yaml").read_text(encoding="utf-8")
    )
    # INTEGRATION (Prototype): the training gate blocks the temporal models only. An M1 run on
    # real WSTS data is Gate 2's own evidence, so a disabled gate is not a reason to mark it mock.
    architectures = {r.get("architecture") for r in records}
    if not cfg["training"]["enabled"] and architectures != {"m1_spatial_unet"}:
        reasons.append(
            "configs/temporal_unet.yaml training.enabled is false (Gate 2 not recorded)"
        )
    # INTEGRATION (Prototype, 2026-11-14): coverage 1.0 is measured within the exported folds, so a
    # one-fold export would otherwise register as real. A result needs all four P1 test years.
    exported = sorted(str(f) for f in model_version.get("folds", {}))
    if exported != sorted(str(y) for y in P1_FOLDS):
        reasons.append(f"only folds {exported} exported; a result needs all of {list(P1_FOLDS)}")
    if model_version.get("is_mock") or any(r.get("is_mock") for r in records):
        reasons.append("model_version.json or an OOF record has is_mock: true")
    # INTEGRATION (Prototype): a run on a handful of staged events is a smoke test, not a result.
    # Real registration needs Role 4's validation to have found every expected held-out forecast
    # of the exported folds (coverage 1.0, written by validate_oof_and_build_reveal.py).
    validation = run_dir / "role4_oof_validation.json"
    if validation.exists():
        cov = json.loads(validation.read_text(encoding="utf-8"))["oof"]["coverage"]
        if cov < 1.0:
            reasons.append(f"OOF coverage {cov:.4f} < 1.0 (subset or smoke run, not a full fold)")
    else:
        reasons.append("role4_oof_validation.json missing (Role 4 validation not run)")
    return reasons


def raster_metadata(path: Path, *, layer: str, units: str, nodata) -> dict:
    """Role 5's required raster metadata, read from the file itself (not trusted from records)."""
    with rasterio.open(path) as ds:
        width, height = ds.width, ds.height
        transform = [float(v) for v in tuple(ds.transform)[:6]]
        crs = ds.crs.to_string()
    return {
        "format": "GeoTIFF",
        "crs": crs,
        "affine_transform": transform,
        "width": width,
        "height": height,
        "nodata": nodata,  # None for float rasters (NaN marks invalid cells; JSON has no NaN)
        "valid_extent": [0, 0, width, height],
        "units": units,
        "layer_name": layer,
        "display_min": 0.0,
        "display_max": 1.0,
    }


def forecast_day_metrics(prob_path: Path, actual_path: Path, threshold: float) -> dict:
    """AUPRC/Brier/F1/IoU of ONE forecast against ITS target, on valid cells only.

    Valid = finite probability (Role 3 writes NaN on invalid cells) and target != 255 (Role 4's
    nodata). Uses Role 4's metric code and the fold's validation threshold, like the event metrics.
    """
    with rasterio.open(prob_path) as ds:
        p = ds.read(1)
    with rasterio.open(actual_path) as ds:
        y = ds.read(1)
    valid = np.isfinite(p) & (y != 255)
    m = summarise(
        counts_by_score(y == 1, np.where(valid, p, 0.0), valid), threshold=threshold
    )
    return {k: finite_or_none(m[k]) for k in METRIC_NAMES}


def finite_or_none(v) -> float | None:
    """INTEGRATION (Prototype, found on real WSTS data): a forecast day whose target has no fire
    has no defined AUPRC (NaN). NaN is neither valid JSON nor storable in Role 5's NOT NULL
    metric column, so an undefined metric is recorded as null in metrics.json and simply has no
    database row. The UI then shows "not defined (no fire that day)" instead of a number."""
    v = float(v)
    return v if np.isfinite(v) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--artifact-root", type=Path, required=True, help="ML artifact root"
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--data-root", type=Path, required=True, help="the WSTS root the run used"
    )
    ap.add_argument("--data-release", type=Path,
                    default=ROOT / "docs" / "evidence" / "role1" / "phase3_data_release.json",
                    help="Role 1 release JSON (input manifest + contract hashes)")  # fmt: skip
    ap.add_argument("--backend-artifact-root", type=Path, required=True,
                    help="Role 5 FIREGUARD_ARTIFACT_ROOT to copy the files into")  # fmt: skip
    ap.add_argument("--out", type=Path, required=True, help="registry output folder")
    ap.add_argument("--mode", choices=["auto", "mock", "real"], default="auto")
    a = ap.parse_args()

    run_dir = a.artifact_root / "runs" / a.run_id
    package = a.artifact_root / "reveal_packages" / a.run_id
    model_version = json.loads(
        (run_dir / "model_version.json").read_text(encoding="utf-8")
    )
    records = [
        json.loads(line)
        for line in (run_dir / "oof_forecasts.jsonl").open(encoding="utf-8")
    ]
    sessions = json.loads(
        (package / "pre_reveal" / "forecast_sessions.json").read_text()
    )["sessions"]
    reveal_index = json.loads(
        (package / "reveal_store" / "reveal_index.json").read_text()
    )
    release = json.loads(a.data_release.read_text(encoding="utf-8"))
    validation = json.loads(
        (run_dir / "role4_oof_validation.json").read_text(encoding="utf-8")
    )

    # Role 4 must have passed the run (no rejected forecasts, no metric mismatches); a reveal
    # package only exists when it did, but check the recorded result rather than assume it.
    if validation["oof"]["rejected"] or validation["metric_mismatches"]:
        raise SystemExit(
            "Role 4 validation did not pass for this run; nothing is bridged"
        )

    reasons = mock_reasons(a.data_root, run_dir, model_version, records)
    mode = ("mock" if reasons else "real") if a.mode == "auto" else a.mode
    if mode == "real" and reasons:
        raise SystemExit("refusing --mode real: " + "; ".join(reasons))

    # Values that are the same for every forecast of the run.
    input_manifest_sha = release["file_sha256_lf_normalised"][
        "phase3_input_manifest.csv"
    ]
    by_key = {(r["event_id"], r["origin_date"]): r for r in records}
    years = {}
    with (ROOT / "data" / "manifests" / "wsts_event_metadata.csv").open(
        encoding="utf-8-sig"
    ) as f:
        for row in csv.DictReader(f):
            years[row["event_id"]] = row["year"]

    prefix = "mock/" if mode == "mock" else ""
    bundle_forecasts, catalogue = [], {}
    for s in sessions:
        rec = by_key[(s["event_id"], s["origin_date"])]
        fold = str(s["fold"])
        fold_meta = model_version["folds"][fold]["meta"]
        T = int(fold_meta["T"])
        fold_id = f"LOYO-{fold}"
        model_slug = f"{s['model_version']}@fold-{fold}"
        # Role 5 layout (docs/backend_artifact_storage.md): protocol/model/fold/event/origin.
        rel = f"p1_loyo/{model_slug}/{fold_id}/{s['event_id']}/{s['origin_date']}"
        prob_uri = f"{prefix}{rel}/probability.tif"
        actual_uri = f"restricted/{prefix}{rel}/actual_target.tif"
        metrics_uri = f"restricted/{prefix}{rel}/metrics.json"

        # Copy, then re-verify the copied bytes against the checksums Roles 3 and 4 recorded.
        src_prob = a.artifact_root / s["probability_artifact_uri"]
        src_actual = package / reveal_index[s["forecast_id"]]["actual_target_uri"]
        dst_prob = a.backend_artifact_root / prob_uri
        dst_actual = a.backend_artifact_root / actual_uri
        for src, dst, expected in (
            (src_prob, dst_prob, s["probability_checksum_sha256"]),
            (
                src_actual,
                dst_actual,
                reveal_index[s["forecast_id"]]["actual_target_checksum_sha256"],
            ),
        ):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if sha256_file(dst) != expected:
                raise SystemExit(f"checksum mismatch after copying {src}")

        threshold = float(s["threshold"])
        day = forecast_day_metrics(dst_prob, dst_actual, threshold)
        event_level = {
            k: finite_or_none(v)
            for k, v in reveal_index[s["forecast_id"]][
                "event_metrics_post_reveal"
            ].items()
        }
        metrics_doc = {
            "mode": mode,
            "forecast_id": s["forecast_id"],
            "event_id": s["event_id"],
            "origin_date": s["origin_date"],
            "target_date": s["target_date"],
            "fold": fold,
            "threshold": threshold,
            "threshold_fold": s["threshold_fold"],
            "calibrator_fold": s["calibrator_fold"],
            "forecast_day": day,
            "event_pooled": event_level,
            "next_day_observability_valid_fraction": reveal_index[s["forecast_id"]][
                "next_day_observability_valid_fraction"
            ],
            "note": "post-reveal only; event_pooled = all scored days of the event (Role 4)",
        }
        dst_metrics = a.backend_artifact_root / metrics_uri
        dst_metrics.parent.mkdir(parents=True, exist_ok=True)
        dst_metrics.write_text(
            json.dumps(metrics_doc, indent=2) + "\n", encoding="utf-8"
        )

        prob_meta = raster_metadata(dst_prob, layer="OOF probability (uncalibrated sigmoid)",
                                    units="probability", nodata=None)  # fmt: skip
        actual_meta = raster_metadata(dst_actual, layer="Next-day VIIRS active fire (target)",
                                      units="binary_active_fire_detection", nodata=255)  # fmt: skip
        if (
            mode == "mock"
        ):  # Role 5 requires metadata.mode == "mock" on every mock artifact
            for m in (prob_meta, actual_meta):
                m["mode"] = "mock"
        artifacts = [
            {"artifact_type": "probability_map", "storage_uri": prob_uri,
             "content_type": "image/tiff", "checksum_sha256": sha256_file(dst_prob),
             "file_size_bytes": dst_prob.stat().st_size, "reveal_class": "pre_reveal",
             "metadata": prob_meta},
            {"artifact_type": "actual_target", "storage_uri": actual_uri,
             "content_type": "image/tiff", "checksum_sha256": sha256_file(dst_actual),
             "file_size_bytes": dst_actual.stat().st_size, "reveal_class": "post_reveal",
             "metadata": actual_meta},
            {"artifact_type": "metrics", "storage_uri": metrics_uri,
             "content_type": "application/json", "checksum_sha256": sha256_file(dst_metrics),
             "file_size_bytes": dst_metrics.stat().st_size, "reveal_class": "post_reveal",
             "metadata": {"format": "JSON", **({"mode": "mock"} if mode == "mock" else {})}},
        ]  # fmt: skip

        # Database metric rows (served by GET /replays/{event}/{origin}/metrics after reveal).
        calibrator_id = f"platt-P1-fold-{s['calibrator_fold']}-validation"
        metric_rows = [
            {"metric_name": k, "value": v, "aggregation": "forecast_day",
             "threshold": threshold if k in ("f1", "iou") else None, "calibrator_id": calibrator_id}
            for k, v in day.items() if v is not None
        ] + [
            {"metric_name": k, "value": v, "aggregation": "event",
             "threshold": threshold if k in ("f1", "iou") else None, "calibrator_id": calibrator_id}
            for k, v in event_level.items() if v is not None
        ]  # fmt: skip

        entry = {
            "forecast_id": s["forecast_id"],
            "event_key": s["event_id"],
            "held_out_group_id": s["group_id"],
            "protocol_id": "P1_LOYO" if mode == "real" else "MOCK_ONLY",
            "source_protocol_id": s[
                "protocol"
            ],  # what the ML side used, kept for provenance
            "fold_id": fold_id,
            "split_name": "test",
            "is_oof": True,
            "origin_date": s["origin_date"],
            "target_date": s["target_date"],
            "horizon_hours": int(s["horizon_hours"]),
            "model_name": ("[MOCK] " if mode == "mock" else "")
            + f"FireGuard {s['architecture']}",
            "model_version": model_slug,
            "architecture": s["architecture"],
            "checkpoint_sha256": s["checkpoint_sha256"],
            "config_hash": s["config_hash"],
            "split_manifest_sha256": s["split_manifest_sha256"],
            "input_manifest_sha256": input_manifest_sha,
            "data_schema_version": s["data_contract_version"],
            "data_schema_sha256": release[f"contract_hash_T{T}"],
            "channel_manifest_sha256": rec["channel_manifest_sha256"],
            "channels": rec["channels"],
            "time_safety_status": (
                TIME_SAFETY_MAP.get(
                    rec["time_safety_status"], rec["time_safety_status"]
                )
                if mode == "real"
                else "MOCK_NOT_A_RESEARCH_RESULT"
            ),
            "probability_kind": s["probability_kind"],
            "fold_validation_auprc": s["fold_validation_auprc"],
            "artifacts": artifacts,
            "metrics": metric_rows,
        }
        if mode == "real" and entry["time_safety_status"] != "PASS_LEVEL_A":
            raise SystemExit(f"{s['forecast_id']}: time safety {rec['time_safety_status']} is not "
                             "a Level-A PASS; Role 5 would refuse it")  # fmt: skip
        bundle_forecasts.append(entry)
        catalogue[s["event_id"]] = {
            "event_id": s["event_id"],
            "year": years[s["event_id"]],
            "name": "",
            "region": "western US (WSTS)",
            "start_date": "",
            "end_date": "",
            "complex_group_id": s["group_id"],
            "p1_fold_id": fold_id,
            "p2_fold_id": "",
            "manifest_sha256": s["split_manifest_sha256"],
        }

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "event_catalogue.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=list(next(iter(catalogue.values()))), lineterminator="\n"
        )
        w.writeheader()
        w.writerows(catalogue.values())

    bundle = {
        "bundle_version": "1.0",
        "mode": mode,
        "mock_reasons": reasons,
        "run_id": a.run_id,
        "source_model_version": model_version["version"],
        "forecasts": bundle_forecasts,
    }
    (a.out / "registry_bundle.json").write_text(
        json.dumps(bundle, indent=2) + "\n", encoding="utf-8"
    )

    if mode == "real":
        # One Role 5 manifest per forecast, exactly the keys of
        # docs/examples/oof_artifact_manifest.template.json (metrics rows are loaded separately).
        (a.out / "manifests").mkdir(exist_ok=True)
        keep = ("event_key", "model_name", "model_version", "architecture", "checkpoint_sha256",
                "config_hash", "protocol_id", "fold_id", "split_name", "is_oof", "origin_date",
                "target_date", "horizon_hours", "held_out_group_id", "split_manifest_sha256",
                "input_manifest_sha256", "data_schema_version", "data_schema_sha256",
                "channel_manifest_sha256", "time_safety_status", "artifacts")  # fmt: skip
        for e in bundle_forecasts:
            manifest = {
                "manifest_version": "1.0",
                "mode": "real",
                **{k: e[k] for k in keep},
            }
            (a.out / "manifests" / f"{e['forecast_id']}.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

    # Last guard: the pre-reveal part of each entry (everything a catalogue/forecast response may
    # be built from) must not carry target content. Post-reveal artifacts/metrics are excluded.
    pre_reveal_view = [
        {k: v for k, v in e.items() if k not in ("artifacts", "metrics")}
        | {
            "artifacts": [
                x for x in e["artifacts"] if x["reveal_class"] == "pre_reveal"
            ]
        }
        for e in bundle_forecasts
    ]
    leaks = find_target_leaks(pre_reveal_view)
    if leaks:
        raise SystemExit(f"pre-reveal view leaks target content: {leaks[:5]}")

    print(json.dumps({"mode": mode, "mock_reasons": reasons, "forecasts": len(bundle_forecasts),
                      "events": len(catalogue), "out": str(a.out)}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
