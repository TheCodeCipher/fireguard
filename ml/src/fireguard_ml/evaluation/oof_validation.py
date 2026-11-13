"""Independent validation of OOF forecasts and replay metrics (Role 4, P4-R4-01 / P4-R4-03).

Role 3's exporter already refuses wrong-fold checkpoints. This module re-checks every exported
record from the frozen manifests alone, so a defect in the exporter cannot slip through:

* coverage: every scored (event, origin) of ``wsts_eval_samples.csv`` should have exactly one
  forecast; missing ones are listed as unavailable, never silently dropped;
* mapping: the record's fold must be the fold whose ``test`` split holds the event's group, and
  the group id must match the manifest;
* provenance: split hash, config hash, checkpoint hash (against ``model_version.json``), data
  contract version, OOF flag and strict channels must match the approved run;
* metrics: stored per-event metrics are recomputed from the probability rasters and the target
  and must agree; the threshold used must be the record's own fold's validation threshold.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fireguard_ml.data.event_splits import read_manifest
from fireguard_ml.evaluation.metrics import StreamingScoreHistogram, summarise


@dataclass
class Approved:
    """What the selected run must match (from Role 3's model_version.json + frozen manifests)."""

    split_manifest_sha256: str
    config_hash: str
    checkpoint_sha256_by_fold: dict[str, str]
    data_contract_version: str
    channels: tuple[str, ...]


@dataclass
class OOFReport:
    expected: int = 0
    accepted: int = 0
    rejected: list = field(default_factory=list)  # (event, origin, reason)
    missing: list = field(default_factory=list)  # (event, origin)
    duplicates: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected and not self.duplicates

    def summary(self) -> dict:
        cov = self.accepted / self.expected if self.expected else math.nan
        return {"expected": self.expected, "accepted": self.accepted,
                "rejected": len(self.rejected), "missing": len(self.missing),
                "duplicates": len(self.duplicates), "coverage": cov}  # fmt: skip


def approved_from_model_version(run_dir: Path, split_hash: str, channels) -> Approved:
    mv = json.loads((Path(run_dir) / "model_version.json").read_text(encoding="utf-8"))
    metas = [v["meta"] for v in mv["folds"].values()]
    versions = {m["preprocessing_version"] for m in metas}
    if len(versions) != 1:
        raise ValueError(f"folds use different data contracts: {versions}")
    return Approved(split_hash, mv["config_hash"],
                    {f: v["checkpoint_sha256"] for f, v in mv["folds"].items()},
                    versions.pop(), tuple(channels))  # fmt: skip


def validate_oof(oof_jsonl: Path, manifest_csv: Path, samples_csv: Path, ok: Approved,
                 folds_evaluated: set[int] | None = None) -> OOFReport:  # fmt: skip
    rows = read_manifest(Path(manifest_csv))
    test_fold = {}
    group = {}
    for r in rows:
        group[r.event_id] = r.group_id
        if r.split == "test":
            if r.event_id in test_fold:
                raise ValueError(f"{r.event_id} is test in two folds; manifest invalid")
            test_fold[r.event_id] = r.fold_test_year
    with Path(samples_csv).open(encoding="utf-8") as f:
        expected = {(r["event_id"], r["last_input_date"]): r["target_date"]
                    for r in csv.DictReader(f)}  # fmt: skip
    if folds_evaluated is not None:
        expected = {k: v for k, v in expected.items() if test_fold[k[0]] in folds_evaluated}
    rep = OOFReport(expected=len(expected))
    seen = set()
    for line in Path(oof_jsonl).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = (r["event_id"], r["origin_date"])
        reasons = []
        if key in seen:
            rep.duplicates.append(key)
            continue
        seen.add(key)
        if key not in expected:
            reasons.append("not a scored (event, origin) of the frozen sample list")
        elif expected[key] != r["target_date"]:
            reasons.append("target date differs from the sample list")
        ev = r["event_id"]
        if ev not in test_fold:
            reasons.append("event not in the frozen manifest")
        else:
            if int(r["fold"]) != test_fold[ev]:
                reasons.append(f"predicted by fold {r['fold']} but its group is held out in "
                               f"fold {test_fold[ev]} (that checkpoint saw the group)")  # fmt: skip
            if r.get("group_id") != group[ev]:
                reasons.append("group id differs from the manifest")
        if not r.get("is_oof") or r.get("split_name") != "test":
            reasons.append("not flagged as an out-of-fold test forecast")
        if r.get("split_manifest_sha256") != ok.split_manifest_sha256:
            reasons.append("split hash does not match the frozen P1 manifest")
        if r.get("config_hash") != ok.config_hash:
            reasons.append("config hash does not match the approved run")
        if r.get("checkpoint_sha256") != ok.checkpoint_sha256_by_fold.get(str(r["fold"])):
            reasons.append("checkpoint hash does not match the approved fold checkpoint")
        if r.get("data_contract_version") != ok.data_contract_version:
            reasons.append("data contract version differs")
        if tuple(r.get("channels", ())) != ok.channels:
            reasons.append("channels differ from the strict contract")
        if r.get("is_mock"):
            reasons.append("mock forecast")
        if reasons:
            rep.rejected.append((*key, "; ".join(reasons)))
        else:
            rep.accepted += 1
    rep.missing = sorted(set(expected) - seen)
    return rep


def recompute_event_metrics(oof_jsonl: Path, artifact_root: Path, target_loader,
                            threshold_by_fold: dict[str, float]) -> dict:  # fmt: skip
    """Per-event metrics from the stored rasters; ``target_loader(event, target_date)`` returns
    the binary target on the forecast's frame (Role 1's reader)."""
    import rasterio

    hists: dict = defaultdict(lambda: StreamingScoreHistogram(n_bins=2**16))
    fold_of = {}
    for line in Path(oof_jsonl).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        with rasterio.open(Path(artifact_root) / r["storage_uri"]) as ds:
            p = ds.read(1)
        y = target_loader(r["event_id"], r["target_date"])
        valid = np.isfinite(p)
        hists[r["event_id"]].add(y, np.where(valid, p, 0.0), valid)
        fold_of[r["event_id"]] = str(r["fold"])
    return {ev: summarise(h.counts(), threshold=threshold_by_fold[fold_of[ev]])
            for ev, h in hists.items()}  # fmt: skip


def compare_metrics(stored_csv: Path, recomputed: dict, tol: float = 1e-6) -> list[tuple]:
    """Rows where the stored replay metric differs from the recomputation."""
    bad = []
    with Path(stored_csv).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = recomputed.get(r["event_id"])
            if m is None:
                bad.append((r["event_id"], "no forecasts to recompute from"))
                continue
            for k in ("auprc", "brier", "f1", "iou", "n_pos", "n_valid"):
                a, b = float(r[k]), float(m[k])
                if not (math.isnan(a) and math.isnan(b)) and abs(a - b) > tol * max(1.0, abs(b)):
                    bad.append((r["event_id"], f"{k}: stored {a} vs recomputed {b}"))
    return bad


def thresholds_from_model_version(run_dir: Path) -> dict[str, float]:
    """Each fold's own validation threshold; refuses records that are not validation fits."""
    mv = json.loads((Path(run_dir) / "model_version.json").read_text(encoding="utf-8"))
    out = {}
    for fold, v in mv["folds"].items():
        sel = v["selection"]
        if sel["fitted_on"] != "validation" or str(sel["fold"]) != str(fold):
            raise ValueError(f"fold {fold}: selection record is not this fold's validation fit")
        if str(sel["threshold"]["fold"]) != str(fold):
            raise ValueError(f"fold {fold}: threshold belongs to fold {sel['threshold']['fold']}")
        out[str(fold)] = float(sel["threshold"]["threshold"])
    return out
