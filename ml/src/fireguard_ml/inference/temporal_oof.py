"""Offline out-of-fold (OOF) export for the selected temporal model (Role 3, P4-R3-02).

For every fold Y, the checkpoint trained for fold Y predicts exactly the events in fold Y's
``test`` split, i.e. the events whose group it never saw. Each assignment is checked mechanically
against the frozen manifest before any file is written.

Output layout (relative to the artifact root, matching Role 5's ``LocalArtifactStorage``
convention ``runs/<run-id>/...``)::

    runs/<run_id>/model_version.json                     model + per-fold checkpoint provenance
    runs/<run_id>/oof_forecasts.jsonl                    PRE-REVEAL records, one per forecast
    runs/<run_id>/<event_id>/<origin_date>/probability.tif   float32 GeoTIFF, original frame
    runs/<run_id>/post_reveal/per_event_test_scores.csv  metrics (need the target: post-reveal)

Pre-reveal records contain no target path, target value or target-derived metric. Actual-target
and next-day observability artifacts are Role 4's reveal manifest, not written here.

The probability layer is the model's sigmoid output. The fold's validation-fitted threshold and
Platt calibrator are recorded as provenance; no calibrated or uncertainty layer is produced.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from fireguard_ml.data.event_splits import fold_events, read_manifest
from fireguard_ml.evaluation.metrics import summarise
from fireguard_ml.models.checkpointing import load_checkpoint
from fireguard_ml.training.normalization import FoldNormalizer
from fireguard_ml.training.temporal_data import (
    TemporalSampleSource,
    pad_samples,
    scored_keys,
)
from fireguard_ml.training.temporal_trainer import ProtocolFiles, TemporalRunConfig, score_split

PRE_REVEAL_FIELDS = (
    "event_id",
    "group_id",
    "protocol",
    "fold",
    "split_name",
    "is_oof",
    "origin_date",
    "target_date",
    "horizon_hours",
    "model_version",
    "architecture",
    "checkpoint_sha256",
    "config_hash",
    "split_manifest_sha256",
    "channel_manifest_sha256",
    "channels",
    "time_safety_status",
    "artifact_type",
    "storage_uri",
    "checksum_sha256",
    "content_type",
    "height",
    "width",
    "crs",
    "transform",
    "probability_kind",
    "data_contract_version",
    "is_mock",
)
FORBIDDEN_PRE_REVEAL = ("target_uri", "target_path", "actual", "y_true", "auprc", "f1", "iou")


class OOFMappingError(RuntimeError):
    """A forecast would come from a checkpoint that saw the event's group."""


def channel_manifest_sha256(channels) -> str:
    return hashlib.sha256("\n".join(channels).encode()).hexdigest()


def assert_held_out(event_id: str, fold: int, rows) -> str:
    """Return the event's group id if fold ``fold`` holds it out as test; raise otherwise."""
    hits = [r for r in rows if r.fold_test_year == fold and r.event_id == event_id]
    if len(hits) != 1:
        raise OOFMappingError(f"{event_id}: not exactly one manifest row in fold {fold}")
    if hits[0].split != "test":
        raise OOFMappingError(
            f"{event_id}: is in '{hits[0].split}' for fold {fold}; its forecast must come from "
            "the checkpoint of the fold that tests it"
        )
    return hits[0].group_id


def _write_geotiff(path: Path, prob: np.ndarray, transform, crs) -> None:
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError('GeoTIFF export needs rasterio: pip install -e "ml[scripts]"') from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=prob.shape[0],
        width=prob.shape[1],
        count=1,
        dtype="float32",
        nodata=float("nan"),
        compress="deflate",
    )
    if transform is not None:
        profile["transform"] = Affine(*transform[:6])
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(prob.astype(np.float32), 1)
        dst.update_tags(layer="probability", units="probability_of_next_day_active_fire")


def export_oof(
    run_dir: Path,
    folds: dict[int, Path],
    source: TemporalSampleSource,
    protocol: ProtocolFiles,
    artifact_root: Path,
    run_id: str,
    device: str = "cpu",
) -> dict:
    """Export OOF forecasts for every fold in ``folds`` ({fold: checkpoint path})."""
    split_hash = protocol.verified_manifest_hash()
    rows = read_manifest(protocol.manifest)
    out = Path(artifact_root) / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    records, score_rows, versions, pooled_counts = [], [], {}, {}
    config_hashes = set()
    available = set(source.keys())
    for fold, ckpt in sorted(folds.items()):
        model, meta = load_checkpoint(ckpt)
        if not getattr(model, "strict_checkpoint", False):
            raise OOFMappingError(f"{ckpt}: not a strict checkpoint; strict artifacts only")
        if int(meta["fold"]) != fold:
            raise OOFMappingError(f"{ckpt}: trained for fold {meta['fold']}, asked for fold {fold}")
        if meta["split_manifest_hash"] != split_hash:
            raise OOFMappingError(f"{ckpt}: trained on a different split manifest")
        if tuple(meta["channels"]) != tuple(source.channels):
            raise OOFMappingError(f"{ckpt}: channel order differs from the source")
        config_hashes.add(meta["config_hash"])
        stem = Path(ckpt).name[: -len(".pt")]
        norm = FoldNormalizer.from_json(Path(ckpt).with_name(f"{stem}.normalizer.json"))
        if norm.fold != fold:
            raise OOFMappingError(f"normaliser for fold {norm.fold} paired with fold {fold}")
        selection = json.loads(Path(ckpt).with_name(f"{stem}.selection.json").read_text())
        ck_sha = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
        versions[str(fold)] = {
            "checkpoint": Path(ckpt).name,
            "checkpoint_sha256": ck_sha,
            "selection": selection,
            "meta": meta,
        }
        model = model.to(dev)
        test_events = fold_events(rows, fold, "test")
        keys = [k for k in scored_keys(protocol.eval_samples, test_events) if k in available]
        group_of = {ev: assert_held_out(ev, fold, rows) for ev in {k[0] for k in keys}}
        for k in keys:
            s = source.load(k)
            b = pad_samples([replace(s, x=norm.apply(s.x, s.valid))], 2**model.depth)
            prob = model.predict(b.x.to(dev), orig_hw=s.hw)[0, 0].cpu().numpy()
            prob = np.where(s.valid, prob, np.nan).astype(np.float32)  # invalid cells -> nodata
            rel = Path(s.event_id) / s.origin_date.isoformat() / "probability.tif"
            _write_geotiff(out / rel, prob, s.transform, s.crs)
            rec = {
                "event_id": s.event_id,
                "group_id": group_of[s.event_id],
                "protocol": "P1_LOYO",
                "fold": str(fold),
                "split_name": "test",
                "is_oof": True,
                "origin_date": s.origin_date.isoformat(),
                "target_date": s.target_date.isoformat(),
                "horizon_hours": 24,
                "model_version": run_id,
                "architecture": type(model).model_type,
                "checkpoint_sha256": ck_sha,
                "config_hash": meta["config_hash"],
                "split_manifest_sha256": split_hash,
                "channel_manifest_sha256": channel_manifest_sha256(meta["channels"]),
                "channels": list(meta["channels"]),
                "time_safety_status": "STRICT_LEVEL_A_PASS_CHANNELS_ONLY",
                "artifact_type": "probability_map",
                "storage_uri": f"runs/{run_id}/{rel.as_posix()}",
                "checksum_sha256": hashlib.sha256((out / rel).read_bytes()).hexdigest(),
                "content_type": "image/tiff",
                "height": s.hw[0],
                "width": s.hw[1],
                "crs": s.crs,
                "transform": list(s.transform) if s.transform else None,
                "probability_kind": "model_sigmoid_uncalibrated",
                "data_contract_version": meta["preprocessing_version"],
                "is_mock": bool(s.is_mock),
            }
            assert tuple(rec) == PRE_REVEAL_FIELDS
            records.append(rec)
        scored = score_split(model, source, norm, keys, dev, model.depth, amp=False)
        thr = selection["threshold"]["threshold"]
        pooled_counts[fold] = scored["pooled"].counts()
        for ev, hist in sorted(scored["per_event"].items()):
            m = summarise(hist.counts(), threshold=thr)
            score_rows.append(
                [
                    fold,
                    ev,
                    group_of[ev],
                    run_id,
                    m["auprc"],
                    m["brier"],
                    m["f1"],
                    m["iou"],
                    m["n_pos"],
                    m["n_valid"],
                ]
            )
    if len(config_hashes) != 1:
        raise OOFMappingError(f"folds were trained with different configs: {config_hashes}")
    with (out / "oof_forecasts.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out / "model_version.json").write_text(
        json.dumps(
            {
                "name": run_id,
                "version": run_id,
                "config_hash": config_hashes.pop(),
                "folds": versions,
                "is_mock": bool(source.is_mock),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    post = out / "post_reveal"
    post.mkdir(exist_ok=True)
    for fold, c in pooled_counts.items():  # pooled test counts for Role 4's pooled metrics
        np.savez_compressed(post / f"pooled_counts_fold_{fold}.npz", scores=c.scores, pos=c.pos,
                            neg=c.neg, sum_sq=c.sum_sq, sum_s_pos=c.sum_s_pos)  # fmt: skip
    with (post / "per_event_test_scores.csv").open("w", newline="", encoding="utf-8") as f:
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
        w.writerows(score_rows)
    return {
        "run_id": run_id,
        "forecasts": len(records),
        "events": len(score_rows),
        "folds": sorted(folds),
        "out_dir": str(out),
    }


def run_config_from_json(path: Path) -> TemporalRunConfig:
    d = json.loads(Path(path).read_text())["config"]
    d["channels"] = tuple(d["channels"])
    return TemporalRunConfig(**d)
