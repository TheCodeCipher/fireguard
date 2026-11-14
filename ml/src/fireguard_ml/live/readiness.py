"""Readiness gate: refuse to forecast unless the trained artefacts are exactly the validated ones.

Hard checks (never skippable): every fold's checkpoint and sidecars exist, load, are strict, real,
mutually consistent, on the frozen split, and are byte-identical to what the OOF export recorded
in ``runs/<run_id>/model_version.json``.

Gate checks (skippable with ``allow_ungated``, which stamps the output ``UNGATED``): the backend
registry bundle is in ``real`` mode and Gate 2 is recorded as ``BEATS`` or ``MATCHES``.

File names are the ones ``training/temporal_trainer.train_fold`` and
``inference/temporal_oof.export_oof`` write; nothing is renamed or rewritten here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fireguard_ml.data.event_splits import sha256_file
from fireguard_ml.data.wsts_dataset import DATA_CONTRACT_VERSION
from fireguard_ml.models.checkpointing import load_checkpoint
from fireguard_ml.training.normalization import FoldNormalizer

GATE2_PASSING = ("BEATS", "MATCHES")
SIDECARS = ("normalizer.json", "selection.json", "summary.json")


class LiveNotReadyError(RuntimeError):
    """The trained artefacts do not satisfy the readiness gate."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("live forecasting is not ready:\n  - " + "\n  - ".join(self.problems))


@dataclass
class FoldBundle:
    fold: int
    checkpoint: Path
    checkpoint_sha256: str
    model: object
    meta: dict
    norm: FoldNormalizer
    selection: dict
    summary: dict


@dataclass
class ReadinessReport:
    run: str
    run_id: str
    folds: list[int]
    gate_status: str  # "GATED" | "UNGATED"
    gate_problems: list[str] = field(default_factory=list)
    is_mock: bool = False
    architecture: str = ""
    T: int = 0
    channels: tuple[str, ...] = ()
    config_hash: str = ""
    split_manifest_hash: str = ""
    preprocessing_version: str = ""
    checkpoint_sha256_by_fold: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["channels"] = list(self.channels)
        return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_live_readiness(
    *,
    run_dir: Path,
    run: str,
    folds: Sequence[int],
    run_id: str,
    artifact_root: Path,
    repo_root: Path,
    allow_ungated: bool = False,
    allow_mock: bool = False,
) -> tuple[ReadinessReport, list[FoldBundle]]:
    """Run every check; raise ``LiveNotReadyError`` listing all problems, else return the folds."""
    run_dir, artifact_root, repo_root = Path(run_dir), Path(artifact_root), Path(repo_root)
    folds = sorted(int(f) for f in folds)
    hard: list[str] = []
    if not folds:
        raise LiveNotReadyError(["no folds requested"])
    if allow_mock and "mock" not in run_id.lower():
        raise LiveNotReadyError(
            ["--allow-mock needs 'mock' in --run-id so nobody mistakes the output"])

    cfg_path = repo_root / "configs" / "temporal_unet.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    expected_channels = tuple(cfg["data"]["channels"])
    frozen_split = sha256_file(repo_root / cfg["data"]["split_manifest"])

    # 4 (first half): what the OOF export recorded
    mv_path = artifact_root / "runs" / run_id / "model_version.json"
    model_version = _json(mv_path) if mv_path.is_file() else None
    if model_version is None:
        hard.append(f"{mv_path} is missing (run scripts/export_temporal_oof.py for all folds)")
    elif bool(model_version.get("is_mock")) and not allow_mock:
        hard.append(f"{mv_path}: is_mock is true (synthetic export)")

    bundles: list[FoldBundle] = []
    for fold in folds:
        fdir = run_dir / f"fold_{fold}"
        ckpt = fdir / f"{run}.pt"
        # 1. files
        absent = [p.name for p in [ckpt, *(fdir / f"{run}.{s}" for s in SIDECARS)]
                  if not p.is_file()]
        if absent:
            hard.append(f"fold {fold}: missing in {fdir}: {', '.join(absent)}")
            continue
        # 2. checkpoint
        try:
            model, meta = load_checkpoint(ckpt)
        except Exception as exc:  # torch raises several types for damaged files
            hard.append(f"fold {fold}: {ckpt.name} does not load ({type(exc).__name__}: {exc})")
            continue
        if not getattr(model, "strict_checkpoint", False):
            hard.append(f"fold {fold}: not a strict checkpoint")
        if bool(meta.get("is_mock")) and not allow_mock:
            hard.append(f"fold {fold}: checkpoint was trained on synthetic (mock) data")
        if int(meta["fold"]) != fold:
            hard.append(f"fold {fold}: checkpoint says fold {meta['fold']}")
        if meta["preprocessing_version"] != DATA_CONTRACT_VERSION:
            hard.append(f"fold {fold}: preprocessing_version {meta['preprocessing_version']!r} "
                        f"!= {DATA_CONTRACT_VERSION!r}")
        if tuple(meta["channels"]) != expected_channels:
            hard.append(f"fold {fold}: channel order differs from configs/temporal_unet.yaml")
        if meta["split_manifest_hash"] != frozen_split:
            hard.append(f"fold {fold}: trained on a different split manifest than the frozen one")
        # 3. sidecars
        norm = FoldNormalizer.from_json(fdir / f"{run}.normalizer.json")
        selection = _json(fdir / f"{run}.selection.json")
        summary = _json(fdir / f"{run}.summary.json")
        if norm.fold != fold or tuple(norm.channels) != tuple(meta["channels"]):
            hard.append(f"fold {fold}: normaliser does not belong to this checkpoint")
        if norm.split_manifest_hash != meta["split_manifest_hash"]:
            hard.append(f"fold {fold}: normaliser was fitted on a different split manifest")
        if int(selection.get("fold", -1)) != fold:
            hard.append(f"fold {fold}: selection.json belongs to fold {selection.get('fold')}")
        if int(summary["config"]["T"]) != int(meta["T"]):
            hard.append(f"fold {fold}: summary T differs from the checkpoint")
        # 4 (second half): byte identity with the exported model
        sha = _sha256(ckpt)
        if model_version is not None:
            rec = model_version.get("folds", {}).get(str(fold))
            if rec is None:
                hard.append(f"fold {fold}: not in {mv_path.name} (export all folds first)")
            elif rec.get("checkpoint_sha256") != sha:
                hard.append(f"fold {fold}: checkpoint differs from the one exported as {run_id} "
                            "(retrained since the export?)")
        bundles.append(FoldBundle(fold, ckpt, sha, model, dict(meta), norm, selection, summary))

    if len({b.meta["config_hash"] for b in bundles}) > 1:
        hard.append("folds were trained with different configs")
    if len({int(b.meta["T"]) for b in bundles}) > 1:
        hard.append("folds have different context lengths T")
    if len({type(b.model).model_type for b in bundles}) > 1:
        hard.append("folds have different architectures")
    if hard:
        raise LiveNotReadyError(hard)

    # 5 and 6: gates
    gate: list[str] = []
    bundle_path = artifact_root / "registry" / run_id / "registry_bundle.json"
    if not bundle_path.is_file():
        gate.append(f"{bundle_path} is missing (run scripts/build_backend_registry.py)")
    elif _json(bundle_path).get("mode") != "real":
        gate.append(f"{bundle_path.name}: mode is not 'real'")
    gd_path = repo_root / "reports" / "phase3" / "gate_decisions.json"
    if not gd_path.is_file():
        gate.append(f"{gd_path} is missing (run scripts/run_model_comparison.py)")
    else:
        gd = _json(gd_path)
        decision = gd.get("gate2", {}).get("decision")
        if decision not in GATE2_PASSING:
            gate.append(f"Gate 2 decision is {decision!r}, needs one of {GATE2_PASSING}")
        if gd.get("split_manifest_sha256") != bundles[0].meta["split_manifest_hash"]:
            gate.append("Gate 2 was decided on a different split manifest")
    if gate and not allow_ungated:
        raise LiveNotReadyError(gate)

    m0 = bundles[0]
    report = ReadinessReport(
        run=run, run_id=run_id, folds=folds,
        gate_status="UNGATED" if gate else "GATED", gate_problems=gate,
        is_mock=any(bool(b.meta.get("is_mock")) for b in bundles),
        architecture=type(m0.model).model_type, T=int(m0.meta["T"]),
        channels=tuple(m0.meta["channels"]), config_hash=m0.meta["config_hash"],
        split_manifest_hash=m0.meta["split_manifest_hash"],
        preprocessing_version=m0.meta["preprocessing_version"],
        checkpoint_sha256_by_fold={str(b.fold): b.checkpoint_sha256 for b in bundles},
    )
    return report, bundles
