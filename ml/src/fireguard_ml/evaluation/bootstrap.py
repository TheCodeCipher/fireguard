"""Paired bootstrap over event groups (Plan v4 6.5-6.6).

Resampling unit = fire-complex group. Pixels are never resampled: neighbouring pixels and the
days of one fire are strongly correlated, so a pixel bootstrap would give falsely narrow intervals.
Each replicate draws groups with replacement and keeps *all* events of every drawn group.

The headline statistic is the mean per-event difference (model minus reference) of a metric,
usually AUPRC, computed on the same events for both models ("paired").
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    n_boot: int
    n_groups: int
    n_events: int
    share_above_zero: float
    excluded_events: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def group_bootstrap(
    value_by_event: Mapping[str, float],
    group_of: Mapping[str, str],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapResult:
    """Cluster bootstrap of ``statistic`` over per-event values, resampling whole groups.

    Events whose value is NaN (e.g. AUPRC is undefined for an event with no next-day fire) are
    excluded before resampling, and the count is reported.
    """
    finite = {e: v for e, v in value_by_event.items() if np.isfinite(v)}
    missing = [e for e in finite if e not in group_of]
    if missing:
        raise KeyError(f"events without a group id: {missing[:5]}")
    by_group: dict[str, list[float]] = defaultdict(list)
    for e, v in finite.items():
        by_group[group_of[e]].append(v)
    groups = sorted(by_group)
    if not groups:
        raise ValueError("no finite per-event values to bootstrap")
    blocks = [np.asarray(by_group[g], dtype=np.float64) for g in groups]
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        reps[b] = statistic(np.concatenate([blocks[i] for i in pick]))
    alpha = (1 - ci) / 2
    return BootstrapResult(
        estimate=float(statistic(np.concatenate(blocks))),
        ci_low=float(np.quantile(reps, alpha)),
        ci_high=float(np.quantile(reps, 1 - alpha)),
        n_boot=n_boot,
        n_groups=len(groups),
        n_events=len(finite),
        share_above_zero=float((reps > 0).mean()),
        excluded_events=len(value_by_event) - len(finite),
    )


def paired_group_bootstrap(
    metric_model: Mapping[str, float],
    metric_reference: Mapping[str, float],
    group_of: Mapping[str, str],
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapResult:
    """Bootstrap the mean per-event difference ``model - reference`` on events scored by both."""
    if set(metric_model) != set(metric_reference):
        raise ValueError("paired comparison needs exactly the same events for both models")
    diff = {e: metric_model[e] - metric_reference[e] for e in metric_model}
    return group_bootstrap(diff, group_of, np.mean, n_boot, seed, ci)


def bootstrap_event_metric(
    metric_fn: Callable[[np.ndarray], float],
    per_event_scores: Mapping[str, float],
    group_of: Mapping[str, str],
    n_boot: int = 2000,
    seed: int = 0,
) -> BootstrapResult:
    """Protocol-named wrapper (validation_protocols.md 13.5) around ``group_bootstrap``."""
    return group_bootstrap(per_event_scores, group_of, metric_fn, n_boot, seed)
