"""Paired model comparisons, Gate 2 and Gate 3 decisions (Role 4, P3-R4-02/03).

Inputs are per-event test-score files in the contract format (``docs/metrics_contract.md`` 7):
``fold_test_year, event_id, group_id, baseline, auprc, brier, f1, iou, n_pos, n_valid`` where the
``baseline`` column names the model. Baselines and models share the format, so one engine compares
everything. Rules are fixed in ``docs/model_comparison_protocol.md`` before any model result.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fireguard_ml.evaluation.bootstrap import paired_group_bootstrap

METRICS = ("auprc", "brier", "f1", "iou", "n_pos", "n_valid")
# Gate 3 comparisons fixed in advance: (candidate, reference, what the candidate adds)
GATE3_PAIRS = (
    ("m3_stacked_unet", "m1_unet", "temporal context (5 days, stacked)"),
    ("m4_convlstm_unet", "m3_stacked_unet", "recurrence (ConvLSTM)"),
    ("m4_convlstm_unet", "m3_stacked_unet_base40", "recurrence at matched capacity (control)"),
    ("m1_unet", "m2_unet_no_terrain", "terrain channels (M1 with vs M2 without)"),
)
# Sequence-length ablation: not an added module. The shorter context is preferred if it is not
# WORSE on any fold (cheaper at equal skill); otherwise T=5 stays.
CONTEXT_ABLATION = ("m3_stacked_unet_T3", "m3_stacked_unet", "context 3 days vs 5 days")
SOFT_BASELINES = ("distance_decay", "climatology")
MIN_FOLDS_ABOVE_ZERO = 2  # v4 Gate 3


class ComparisonError(ValueError):
    pass


def read_scores(paths: Iterable[Path]) -> dict:
    """``{fold: {model: {event: {metric: value}}}}`` plus ``group_of``."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    group_of: dict[str, str] = {}
    for p in paths:
        with Path(p).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                fold, model, ev = int(r["fold_test_year"]), r["baseline"], r["event_id"]
                if ev in out[fold][model]:
                    raise ComparisonError(f"{p}: duplicate row for {model} {fold} {ev}")
                if group_of.setdefault(ev, r["group_id"]) != r["group_id"]:
                    raise ComparisonError(f"{ev}: two different group ids across files")
                out[fold][model][ev] = {k: _num(r.get(k)) for k in METRICS}
    return {"scores": {f: dict(m) for f, m in out.items()}, "group_of": group_of}


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def check_identical_events(scores: Mapping, fold: int, models: Iterable[str]) -> set[str]:
    """Every compared model must score exactly the same events on a fold (P3-R4-02)."""
    sets = {m: set(scores[fold][m]) for m in models}
    first = next(iter(sets.values()))
    bad = {m: len(s ^ first) for m, s in sets.items() if s != first}
    if bad:
        raise ComparisonError(f"fold {fold}: models score different events {bad}")
    return first


@dataclass(frozen=True)
class PairedResult:
    fold: int
    candidate: str
    reference: str
    metric: str
    mean_diff: float
    ci_low: float
    ci_high: float
    n_events: int
    n_groups: int
    excluded_events: int
    share_above_zero: float

    @property
    def verdict(self) -> str:
        if self.ci_low > 0:
            return "BETTER"
        if self.ci_high < 0:
            return "WORSE"
        return "NO_DIFFERENCE_SHOWN"


def paired(scores, group_of, fold, candidate, reference, metric="auprc", cfg=None) -> PairedResult:
    cfg = cfg or {"n_boot": 2000, "seed": 0, "ci": 0.95}
    check_identical_events(scores, fold, (candidate, reference))
    a = {e: v[metric] for e, v in scores[fold][candidate].items()}
    b = {e: v[metric] for e, v in scores[fold][reference].items()}
    r = paired_group_bootstrap(a, b, group_of, n_boot=cfg["n_boot"], seed=cfg["seed"], ci=cfg["ci"])
    return PairedResult(fold, candidate, reference, metric, r.estimate, r.ci_low, r.ci_high,
                        r.n_events, r.n_groups, r.excluded_events, r.share_above_zero)  # fmt: skip


def per_event_differences(scores, fold, candidate, reference, metric="auprc") -> list[dict]:
    out = []
    for e in sorted(scores[fold][candidate]):
        a, b = scores[fold][candidate][e][metric], scores[fold][reference][e][metric]
        diff = a - b if np.isfinite(a) and np.isfinite(b) else math.nan
        out.append({"fold": fold, "event_id": e, "candidate": candidate, "reference": reference,
                    "metric": metric, "candidate_value": a, "reference_value": b,
                    "difference": diff})  # fmt: skip
    return out


def event_macro(scores, fold, model, metric) -> tuple[float, int]:
    v = np.array([r[metric] for r in scores[fold][model].values()], float)
    v = v[np.isfinite(v)]
    return (float(v.mean()) if v.size else math.nan), int(v.size)


def strongest_baseline(validation_auprc: Mapping[str, float]) -> str:
    """Strongest soft baseline of a fold = higher VALIDATION pooled AUPRC (never test)."""
    cands = {b: validation_auprc[b] for b in SOFT_BASELINES if b in validation_auprc}
    if not cands:
        raise ComparisonError("no validation AUPRC for the soft baselines")
    return max(sorted(cands), key=lambda b: cands[b])


def gate2(results: Iterable[PairedResult]) -> dict:
    """M1 vs the strongest soft baseline of each fold.

    BEATS: no fold WORSE and >= 2 folds BETTER. MATCHES: no fold WORSE. FAILS: any fold WORSE.
    """
    rs = list(results)
    if not rs:
        return {"decision": "NOT_RUN"}
    worse = [r.fold for r in rs if r.verdict == "WORSE"]
    better = [r.fold for r in rs if r.verdict == "BETTER"]
    if worse:
        d = "FAILS"
    elif len(better) >= MIN_FOLDS_ABOVE_ZERO:
        d = "BEATS"
    else:
        d = "MATCHES"
    return {"decision": d, "passes": d in ("BEATS", "MATCHES"), "better_folds": better,
            "worse_folds": worse, "folds": sorted(r.fold for r in rs)}  # fmt: skip


def context_ablation(results: Iterable[PairedResult]) -> dict:
    rs = list(results)
    if not rs:
        return {"decision": "NOT_RUN"}
    worse = sorted(r.fold for r in rs if r.verdict == "WORSE")
    return {"decision": "KEEP_T5" if worse else "PREFER_T3", "worse_folds": worse,
            "folds": sorted(r.fold for r in rs)}  # fmt: skip


def gate3(results: Iterable[PairedResult]) -> dict:
    """Retain the added module only if its interval is above zero on >= 2 evaluated folds."""
    rs = list(results)
    if not rs:
        return {"decision": "NOT_RUN"}
    above = sorted(r.fold for r in rs if r.ci_low > 0)
    keep = len(above) >= MIN_FOLDS_ABOVE_ZERO
    return {"decision": "RETAIN" if keep else "REMOVE", "folds_above_zero": above,
            "folds": sorted(r.fold for r in rs)}  # fmt: skip
