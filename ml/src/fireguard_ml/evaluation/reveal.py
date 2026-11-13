"""Leakage-safe reveal package for the replay app (Role 4, P4-R4-02).

Two separate trees, written by ``build_reveal_package``:

``pre_reveal/forecast_sessions.json``
    Approved event/origin metadata, OOF probability artifact references, model/fold/protocol
    provenance and the fold's validation metrics (they use no information about this forecast's
    target). Nothing else.

``reveal_store/``
    ``actual/<forecast_id>.tif`` (next-day target, uint8, nodata 255, on the source frame),
    ``reveal_index.json`` (forecast_id -> target artifact, next-day observability, post-reveal
    event metrics). Served only behind the reveal action.

``find_target_leaks`` scans any payload (API response, catalogue, browser bundle) for target or
observability content; Role 5/6 can call it in their own pre-reveal response tests.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import numpy as np

PRE_REVEAL_SESSION_FIELDS = (
    "forecast_id", "event_id", "group_id", "protocol", "fold", "origin_date", "target_date",
    "horizon_hours", "model_version", "architecture", "checkpoint_sha256", "config_hash",
    "split_manifest_sha256", "data_contract_version", "probability_artifact_uri",
    "probability_checksum_sha256", "probability_kind", "threshold_fold", "threshold",
    "calibrator_fold", "fold_validation_auprc", "height", "width", "crs", "transform",
)  # fmt: skip
REVEAL_AUDIT_FIELDS = (
    "session_id", "event_id", "origin_date", "forecast_artifact_id", "revealed_target_artifact_id",
    "revealed_at_utc", "model_version", "fold",
)  # fmt: skip
LEAK_MARKERS = ("actual", "target_uri", "target_path", "target_value", "y_true", "reveal_store",
                "observability", "auprc_event", "per_event", "n_pos", "iou", "f1")  # fmt: skip
ALLOWED_KEYS = {"target_date", "fold_validation_auprc"}  # a date and a validation-only number


def forecast_id(model_version: str, event_id: str, origin_date: str) -> str:
    return hashlib.sha256(f"{model_version}|{event_id}|{origin_date}".encode()).hexdigest()[:20]


def find_target_leaks(obj, path: str = "$") -> list[str]:
    """Paths in ``obj`` whose key or string value looks like target/observability content."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl not in ALLOWED_KEYS and any(m in kl for m in LEAK_MARKERS):
                hits.append(f"{path}.{k}")
            hits += find_target_leaks(v, f"{path}.{k}")
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            hits += find_target_leaks(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        sl = obj.lower()
        if any(m in sl for m in ("reveal_store", "actual_target", "/actual/")):
            hits.append(path)
    return hits


def _write_target(path: Path, y: np.ndarray, transform, crs) -> str:
    import rasterio
    from rasterio.transform import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=y.shape[0], width=y.shape[1], count=1,
                       dtype="uint8", nodata=255, transform=Affine(*transform[:6]), crs=crs,
                       compress="deflate") as ds:  # fmt: skip
        ds.write(y.astype(np.uint8), 1)
        ds.update_tags(layer="actual_target", rule="next-day active fire band: NaN->0, >0")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_reveal_package(catalogue_csv: Path, oof_jsonl: Path, run_dir: Path, data_root: Path,
                         out_dir: Path, observability_csv: Path | None = None,
                         event_metrics_csv: Path | None = None) -> dict:  # fmt: skip
    """Only forecasts present in Role 1's verified catalogue become sessions."""
    import rasterio

    from fireguard_ml.data.geo_artifacts import Frame, verify_artifact  # Role 1
    from fireguard_ml.data.wsts_dataset import TARGET_BAND_1BASED, scan_events, target_from_band

    mv = json.loads((Path(run_dir) / "model_version.json").read_text(encoding="utf-8"))
    with Path(catalogue_csv).open(encoding="utf-8") as f:
        approved = {(r["event_id"], r["origin_date"]) for r in csv.DictReader(f)
                    if r["artifact_verified"] == "True"}  # fmt: skip
    obs = {}
    if observability_csv and Path(observability_csv).exists():
        with Path(observability_csv).open(encoding="utf-8") as f:
            obs = {
                (r["event_id"], r["target_date"]): r["valid_fraction"] for r in csv.DictReader(f)
            }
    ev_metrics = {}
    if event_metrics_csv and Path(event_metrics_csv).exists():
        with Path(event_metrics_csv).open(encoding="utf-8") as f:
            ev_metrics = {r["event_id"]: r for r in csv.DictReader(f)}
    events = scan_events(Path(data_root))
    sessions, index, skipped = [], {}, []
    for line in Path(oof_jsonl).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = (r["event_id"], r["origin_date"])
        if key not in approved:
            skipped.append((*key, "not in the verified replay catalogue"))
            continue
        fold = str(r["fold"])
        fv = mv["folds"][fold]
        sel = fv["selection"]
        fid = forecast_id(r["model_version"], *key)
        s = {
            "forecast_id": fid, "event_id": r["event_id"], "group_id": r["group_id"],
            "protocol": r["protocol"], "fold": fold, "origin_date": r["origin_date"],
            "target_date": r["target_date"], "horizon_hours": r["horizon_hours"],
            "model_version": r["model_version"], "architecture": r["architecture"],
            "checkpoint_sha256": r["checkpoint_sha256"], "config_hash": r["config_hash"],
            "split_manifest_sha256": r["split_manifest_sha256"],
            "data_contract_version": r.get("data_contract_version"),
            "probability_artifact_uri": r["storage_uri"],
            "probability_checksum_sha256": r["checksum_sha256"],
            "probability_kind": r["probability_kind"],
            "threshold_fold": str(sel["threshold"]["fold"]),
            "threshold": sel["threshold"]["threshold"],
            "calibrator_fold": str(sel["calibrator"]["fold"]),
            "fold_validation_auprc": fv["meta"].get("val_auprc"),
            "height": r["height"], "width": r["width"], "crs": r["crs"],
            "transform": r["transform"],
        }  # fmt: skip
        assert tuple(s) == PRE_REVEAL_SESSION_FIELDS
        if s["threshold_fold"] != fold or s["calibrator_fold"] != fold:
            raise ValueError(f"{fid}: threshold/calibrator from another fold")
        sessions.append(s)
        ef = events[r["event_id"]]
        with rasterio.open(ef.path(date.fromisoformat(r["target_date"]))) as ds:
            y = target_from_band(ds.read(TARGET_BAND_1BASED))
            t_tr, t_crs = tuple(ds.transform)[:6], ds.crs.to_string()
        rel = Path("reveal_store") / "actual" / f"{fid}.tif"
        sha = _write_target(Path(out_dir) / rel, y, t_tr, t_crs)
        verify_artifact(Path(out_dir) / rel, Frame(r["height"], r["width"], r["crs"],
                                                   tuple(r["transform"])), sha,
                        layer="actual_target")  # fmt: skip
        m = ev_metrics.get(r["event_id"], {})
        index[fid] = {
            "actual_target_uri": rel.as_posix(), "actual_target_checksum_sha256": sha,
            "next_day_observability_valid_fraction": obs.get((r["event_id"], r["target_date"])),
            "event_metrics_post_reveal": {k: m[k] for k in ("auprc", "brier", "f1", "iou",
                                                           "n_pos", "n_valid") if k in m},
            "threshold": s["threshold"], "threshold_fold": fold,
        }  # fmt: skip
    pre = Path(out_dir) / "pre_reveal" / "forecast_sessions.json"
    pre.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sessions": sessions, "audit_fields": list(REVEAL_AUDIT_FIELDS)}
    leaks = find_target_leaks(payload)
    if leaks:
        raise ValueError(f"pre-reveal payload leaks target content at {leaks[:5]}")
    pre.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rs = Path(out_dir) / "reveal_store" / "reveal_index.json"
    rs.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return {"sessions": len(sessions), "skipped": skipped, "pre_reveal": str(pre),
            "reveal_index": str(rs)}  # fmt: skip


def audit_record(session_id: str, session: dict, revealed_at_utc: str) -> dict:
    """One row of the reveal-state audit log (Role 5 persists it)."""
    rec = {"session_id": session_id, "event_id": session["event_id"],
           "origin_date": session["origin_date"], "forecast_artifact_id": session["forecast_id"],
           "revealed_target_artifact_id": f"actual:{session['forecast_id']}",
           "revealed_at_utc": revealed_at_utc, "model_version": session["model_version"],
           "fold": session["fold"]}  # fmt: skip
    assert tuple(rec) == REVEAL_AUDIT_FIELDS
    return rec


def iter_sessions(path: Path) -> Iterable[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["sessions"]
