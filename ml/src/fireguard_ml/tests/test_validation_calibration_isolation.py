"""Thresholds, calibrators and fitted baseline parameters can only come from the right split."""

from __future__ import annotations

import numpy as np
import pytest

from fireguard_ml.evaluation.bootstrap import group_bootstrap, paired_group_bootstrap
from fireguard_ml.evaluation.calibration import (
    FoldArtifactStore,
    IsolationError,
    fit_platt,
    select_f1_threshold,
)
from fireguard_ml.evaluation.metrics import brier_from_counts, counts_by_score

rng = np.random.default_rng(3)


def miscalibrated(n=50_000):
    true_p = rng.random(n) * 0.2
    y = (rng.random(n) < true_p).astype(int)
    over = np.clip(true_p * 3, 0, 1)  # overconfident model
    return y, over


@pytest.mark.parametrize("split", ["train", "test"])
def test_thresholds_and_calibrators_refuse_non_validation_splits(split):
    y, p = miscalibrated()
    c = counts_by_score(y, p)
    with pytest.raises(IsolationError):
        select_f1_threshold(c, "P1", 2019, split)
    with pytest.raises(IsolationError):
        fit_platt(c, "P1", 2019, split)


def test_platt_on_validation_improves_brier_and_is_fold_locked():
    y, p = miscalibrated()
    cal = fit_platt(counts_by_score(y, p), "P1", 2020, "validation")
    before = brier_from_counts(counts_by_score(y, p))
    after = brier_from_counts(counts_by_score(y, cal.transform(p, fold=2020)))
    assert after < before
    with pytest.raises(IsolationError):
        cal.transform(p, fold=2021)


def test_store_keeps_one_artifact_per_fold_and_never_overwrites():
    y, p = miscalibrated()
    c = counts_by_score(y, p)
    store = FoldArtifactStore()
    store.put_threshold(select_f1_threshold(c, "P1", 2018, "validation"))
    store.put_calibrator(fit_platt(c, "P1", 2018, "validation"))
    with pytest.raises(IsolationError):
        store.put_threshold(select_f1_threshold(c, "P1", 2018, "validation"))
    with pytest.raises(KeyError):
        store.calibrator("P1", 2019)
    assert not hasattr(store, "pooled_calibrator")


def test_bootstrap_resamples_whole_groups_not_events():
    # two groups: one with 9 identical events at +1, one with 1 event at -1. An event-level
    # bootstrap would almost never draw a sample with mean < 0; a group bootstrap often does.
    values = {f"a{i}": 1.0 for i in range(9)} | {"b0": -1.0}
    groups = {f"a{i}": "GA" for i in range(9)} | {"b0": "GB"}
    r = group_bootstrap(values, groups, n_boot=4000, seed=1)
    assert r.n_groups == 2 and r.n_events == 10
    assert r.ci_low == pytest.approx(-1.0)  # replicates drawing only GB exist
    assert 0.6 < r.share_above_zero < 0.9  # ~75 % of draws contain GA at least once


def test_paired_bootstrap_requires_identical_events_and_reports_nan_exclusions():
    groups = {"e1": "G1", "e2": "G2", "e3": "G3"}
    model = {"e1": 0.5, "e2": 0.6, "e3": float("nan")}
    ref = {"e1": 0.3, "e2": 0.4, "e3": 0.2}
    r = paired_group_bootstrap(model, ref, groups, n_boot=500)
    assert r.estimate == pytest.approx(0.2)
    assert r.excluded_events == 1
    with pytest.raises(ValueError, match="same events"):
        paired_group_bootstrap({"e1": 0.5}, ref, groups)


def test_platt_converges_on_distance_decay_shaped_scores():
    """Regression: plain Newton diverged (slope ~1e18) on the distance-decay baseline, whose
    scores underflow to exactly 0 for most cells. Shape below reproduces that failure."""
    from fireguard_ml.evaluation.metrics import ScoreCounts

    d = np.arange(0, 200, 0.375)
    s = np.append(0.4069 * np.exp(-d / 0.5), 0.0)
    rate = np.append(0.4 * np.exp(-d / 2.0) + 2e-4, 1.2e-4)
    tot = np.append(np.full(d.size, 100_000.0), 3.5e7)
    pos = np.round(rate * tot)
    o = np.argsort(-s, kind="stable")
    c = ScoreCounts(s[o], pos[o].astype(np.int64), (tot - pos)[o].astype(np.int64))
    cal = fit_platt(c, "P1", 2018, "validation")
    assert np.isfinite(cal.a) and 0.1 < cal.a < 5
    q = cal.transform(np.clip(c.scores, 1e-12, 1), 2018)
    brier = lambda sc: float((c.pos * (1 - sc) ** 2 + c.neg * sc**2).sum() / c.n)  # noqa: E731
    assert brier(q) <= brier(c.scores)
