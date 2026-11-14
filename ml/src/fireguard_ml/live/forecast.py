"""Fold-mean live inference and static artefact publication.

Prediction is the OOF export's own code path per fold (``FoldNormalizer.apply`` ->
``pad_samples`` -> ``model.predict`` -> NaN on invalid cells), then the mean over folds.
``fold_spread`` (max - min over folds) is fold disagreement, not an uncertainty estimate.

Layout (relative to an artefact root; identical under the ML root and the backend root)::

    live/<live_run_id>/live_run.json                      (ML root only)
    live/<live_run_id>/live_forecasts.jsonl               (ML root only)
    live/<live_run_id>/<region_id>/<origin>/probability.tif
    live/<live_run_id>/<region_id>/<origin>/fold_spread.tif
    live/<live_run_id>/<region_id>/<origin>/inputs_raw.tif   (ML root only; WSTS raw format)
    live/index.json                                       (backend root only; written last)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from fireguard_ml.data.wsts_dataset import RAW_DESCRIPTIONS
from fireguard_ml.evaluation.calibration import FoldCalibrator
from fireguard_ml.inference.temporal_oof import (
    FORBIDDEN_PRE_REVEAL,
    _write_geotiff,
    channel_manifest_sha256,
)
from fireguard_ml.live.frames import LiveTemporalSource
from fireguard_ml.live.readiness import FoldBundle, ReadinessReport
from fireguard_ml.training.temporal_data import TemporalSample, pad_samples

INDEX_SCHEMA_VERSION = 1
LIVE_RUN_ID = re.compile(r"^live_\d{4}-\d{2}-\d{2}_[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_STALE_AFTER_HOURS = 36


def live_run_id_for(origin: date, run_id: str) -> str:
    rid = f"live_{origin.isoformat()}_{run_id}"
    if not LIVE_RUN_ID.match(rid):
        raise ValueError(f"run id {run_id!r} is not safe for paths and URLs")
    return rid


def predict_fold(sample: TemporalSample, bundle: FoldBundle, device: torch.device,
                 calibrated: bool = False) -> np.ndarray:
    """One fold's probability map ``[H, W]`` float32, NaN on invalid cells (as ``export_oof``)."""
    model = bundle.model.to(device)
    b = pad_samples([replace(sample, x=bundle.norm.apply(sample.x, sample.valid))],
                    2 ** model.depth)
    prob = model.predict(b.x.to(device), orig_hw=sample.hw)[0, 0].cpu().numpy()
    if calibrated:
        cal = bundle.selection["calibrator"]
        prob = FoldCalibrator(cal["protocol"], int(cal["fold"]), float(cal["a"]),
                              float(cal["b"])).transform(prob, bundle.fold, cal["protocol"])
    return np.where(sample.valid, prob, np.nan).astype(np.float32)


def predict_sample(sample: TemporalSample, bundles: Sequence[FoldBundle],
                   device: str | torch.device = "cpu", calibrated: bool = False) -> dict:
    """Fold-mean probability, fold spread and the per-fold maps for one sample."""
    if not bundles:
        raise ValueError("no fold models")
    dev = torch.device(device)
    per_fold = {b.fold: predict_fold(sample, b, dev, calibrated) for b in bundles}
    stack = np.stack(list(per_fold.values()))
    return {"mean": stack.mean(axis=0).astype(np.float32),
            "spread": (stack.max(axis=0) - stack.min(axis=0)).astype(np.float32),
            "per_fold": per_fold}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _retag(path: Path, **tags: str) -> None:
    import rasterio

    with rasterio.open(path, "r+") as ds:
        ds.update_tags(**tags)


def _write_raw_inputs(path: Path, raw: np.ndarray, transform, crs) -> None:
    """The 23-band raw frame in the WSTS file format (same band descriptions), for audits."""
    import rasterio
    from rasterio.transform import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=raw.shape[1], width=raw.shape[2],
                       count=raw.shape[0], dtype="float32", nodata=float("nan"),
                       compress="deflate", crs=crs, transform=Affine(*transform[:6])) as dst:
        dst.write(raw.astype(np.float32))
        for i, desc in enumerate(RAW_DESCRIPTIONS, start=1):
            dst.set_band_description(i, desc)


def write_json_atomic(path: Path, obj: dict) -> None:
    """Write through a temporary file and ``os.replace`` so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _artifact_entry(root: Path, uri: str) -> dict:
    p = root / uri
    return {"storage_uri": uri, "checksum_sha256": _sha256(p), "file_size_bytes": p.stat().st_size,
            "content_type": "image/tiff"}


def run_live_forecast(
    *,
    source: LiveTemporalSource,
    bundles: Sequence[FoldBundle],
    report: ReadinessReport,
    artifact_root: Path,
    backend_artifact_root: Path | None,
    n_detections: dict[str, int] | None = None,
    firms_source: str = "",
    firms_retrieved_at: str = "",
    extra_run_info: dict | None = None,
    device: str = "cpu",
    calibrated: bool = False,
    stale_after_hours: int = DEFAULT_STALE_AFTER_HOURS,
    write_inputs: bool = True,
    now: datetime | None = None,
) -> dict:
    """Forecast every region of ``source``; write the ML artefacts, then publish to the backend
    root (GeoTIFF copies first, ``live/index.json`` last and atomically)."""
    if source.sequence_length != report.T or tuple(source.channels) != tuple(report.channels):
        raise ValueError("the live source does not match the checkpoints' T / channel order")
    if bool(source.is_mock) != bool(report.is_mock):
        raise ValueError("mock flag of the source and of the checkpoints differ")
    artifact_root = Path(artifact_root)
    origin = source.origin_date
    target = origin + timedelta(days=1)
    live_run_id = live_run_id_for(origin, report.run_id)
    generated_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    kind = ("platt_calibrated_fold_mean" if calibrated else "model_sigmoid_uncalibrated_fold_mean")
    rel_run = f"live/{live_run_id}"
    out = artifact_root / rel_run
    if out.exists():  # a re-run of the same day replaces that day's files only
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n_detections = n_detections or {}

    records, regions_index = [], []
    for rid, _ in source.keys():
        sample = source.load((rid, target))
        region = source.region(rid)
        pred = predict_sample(sample, bundles, device, calibrated)
        rel = f"{rel_run}/{rid}/{origin.isoformat()}"
        prob_uri, spread_uri = f"{rel}/probability.tif", f"{rel}/fold_spread.tif"
        _write_geotiff(artifact_root / prob_uri, pred["mean"], sample.transform, sample.crs)
        _write_geotiff(artifact_root / spread_uri, pred["spread"], sample.transform, sample.crs)
        _retag(artifact_root / spread_uri, layer="fold_spread",
               units="max_minus_min_probability_over_folds")
        if write_inputs:
            _write_raw_inputs(artifact_root / rel / "inputs_raw.tif",
                              source.raw_frame(rid, origin), sample.transform, sample.crs)
        prov = source.provenance(rid)
        west, south, east, north = region.bbox_wgs84
        rec = {
            "region_id": rid,
            "protocol": "LIVE",
            "is_oof": False,
            "origin_date": origin.isoformat(),
            "target_date": target.isoformat(),
            "horizon_hours": 24,
            "model_version": report.run_id,
            "architecture": report.architecture,
            "folds": list(report.folds),
            "checkpoint_sha256_by_fold": dict(report.checkpoint_sha256_by_fold),
            "config_hash": report.config_hash,
            "split_manifest_sha256": report.split_manifest_hash,
            "channel_manifest_sha256": channel_manifest_sha256(report.channels),
            "channels": list(report.channels),
            "time_safety_status": "STRICT_LEVEL_A_PASS_CHANNELS_ONLY",
            "artifact_type": "probability_map",
            "storage_uri": prob_uri,
            "checksum_sha256": _sha256(artifact_root / prob_uri),
            "content_type": "image/tiff",
            "height": sample.hw[0],
            "width": sample.hw[1],
            "crs": sample.crs,
            "transform": list(sample.transform),
            "probability_kind": kind,
            "data_contract_version": report.preprocessing_version,
            "is_mock": bool(sample.is_mock),
            "center_lon": region.center_lon,
            "center_lat": region.center_lat,
            "bbox_wgs84": [west, south, east, north],
            "n_detections": int(n_detections.get(rid, 0)),
            "nan_fraction": prov["nan_fraction"],
            "reflectance_source": prov["reflectance_source"],
            "input_dates": prov["input_dates"],
            "firms_source": firms_source,
            "firms_retrieved_at": firms_retrieved_at,
            "gate_status": report.gate_status,
            "generated_at": generated_at,
        }
        bad = [k for k in rec if any(m in k.lower() for m in FORBIDDEN_PRE_REVEAL)]
        if bad:  # a live record can hold nothing target-derived: the target does not exist yet
            raise RuntimeError(f"forbidden keys in a live record: {bad}")
        records.append(rec)
        regions_index.append({
            **{k: rec[k] for k in ("region_id", "center_lon", "center_lat", "bbox_wgs84", "height",
                                   "width", "crs", "transform", "n_detections", "nan_fraction",
                                   "reflectance_source")},
            "artifacts": {"probability": _artifact_entry(artifact_root, prob_uri),
                          "fold_spread": _artifact_entry(artifact_root, spread_uri)},
        })

    with (out / "live_forecasts.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    index = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "live_run_id": live_run_id,
        "origin_date": origin.isoformat(),
        "target_date": target.isoformat(),
        "horizon_hours": 24,
        "generated_at": generated_at,
        "stale_after_hours": int(stale_after_hours),
        "model_version": report.run_id,
        "architecture": report.architecture,
        "folds": list(report.folds),
        "probability_kind": kind,
        "gate_status": report.gate_status,
        "is_mock": bool(report.is_mock),
        "data_contract_version": report.preprocessing_version,
        "firms_source": firms_source,
        "regions": regions_index,
    }
    write_json_atomic(out / "live_run.json", {
        "readiness": report.as_dict(), "index": index, "device": str(device),
        "calibrated": bool(calibrated), **(extra_run_info or {}),
    })

    if backend_artifact_root is not None:
        backend_root = Path(backend_artifact_root)
        dst_run = backend_root / rel_run
        if dst_run.exists():
            shutil.rmtree(dst_run)
        for reg in regions_index:
            for art in reg["artifacts"].values():
                dst = backend_root / art["storage_uri"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(artifact_root / art["storage_uri"], dst)
                if _sha256(dst) != art["checksum_sha256"]:
                    raise RuntimeError(f"copy of {art['storage_uri']} is corrupt")
        write_json_atomic(backend_root / "live" / "index.json", index)  # last: publishes the run

    return {"live_run_id": live_run_id, "regions": len(records), "gate_status": report.gate_status,
            "is_mock": bool(report.is_mock), "out_dir": str(out),
            "published": backend_artifact_root is not None}
