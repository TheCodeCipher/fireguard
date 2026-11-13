"""Role 4 P3-R4-02/03: paired comparison engine and gate rules."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from fireguard_ml.evaluation.comparison import (
    ComparisonError,
    PairedResult,
    context_ablation,
    gate2,
    gate3,
    paired,
    per_event_differences,
    read_scores,
    strongest_baseline,
)

FIELDS = ["fold_test_year", "event_id", "group_id", "baseline", "auprc", "brier", "f1", "iou",
          "n_pos", "n_valid"]  # fmt: skip


def write(path, model, values, fold=2020, groups=None):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for i, v in enumerate(values):
            g = groups[i] if groups else f"G{i // 2}"
            w.writerow([fold, f"e{i}", g, model, v, 0.1, 0.2, 0.1, 5, 100])
    return path


def pr(fold, lo, hi):
    return PairedResult(fold, "a", "b", "auprc", (lo + hi) / 2, lo, hi, 10, 5, 0, 0.5)


def test_paired_bootstrap_on_identical_events_and_its_direction(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.uniform(0.1, 0.4, 40)
    a = write(tmp_path / "a.csv", "m1_unet", base + 0.05)
    b = write(tmp_path / "b.csv", "distance_decay", base)
    d = read_scores([a, b])
    r = paired(d["scores"], d["group_of"], 2020, "m1_unet", "distance_decay")
    assert r.mean_diff == pytest.approx(0.05) and r.verdict == "BETTER"
    assert r.n_groups == 20 and r.n_events == 40
    diffs = per_event_differences(d["scores"], 2020, "m1_unet", "distance_decay")
    assert all(x["difference"] == pytest.approx(0.05) for x in diffs)


def test_models_must_score_the_same_events(tmp_path):
    a = write(tmp_path / "a.csv", "m1_unet", [0.2] * 6)
    b = write(tmp_path / "b.csv", "distance_decay", [0.2] * 5)
    d = read_scores([a, b])
    with pytest.raises(ComparisonError, match="different events"):
        paired(d["scores"], d["group_of"], 2020, "m1_unet", "distance_decay")


def test_conflicting_group_ids_or_duplicates_are_refused(tmp_path):
    a = write(tmp_path / "a.csv", "m1_unet", [0.2, 0.3], groups=["G1", "G1"])
    b = write(tmp_path / "b.csv", "distance_decay", [0.2, 0.3], groups=["G1", "G2"])
    with pytest.raises(ComparisonError, match="group"):
        read_scores([a, b])
    with pytest.raises(ComparisonError, match="duplicate"):
        read_scores([a, a])


def test_undefined_event_auprc_is_excluded_not_zeroed(tmp_path):
    a = write(tmp_path / "a.csv", "m1_unet", [0.3, "nan", 0.5, 0.4])
    b = write(tmp_path / "b.csv", "distance_decay", [0.2, "nan", 0.4, 0.3])
    d = read_scores([a, b])
    r = paired(d["scores"], d["group_of"], 2020, "m1_unet", "distance_decay", cfg={
        "n_boot": 200, "seed": 0, "ci": 0.95})  # fmt: skip
    assert r.excluded_events == 1 and r.n_events == 3


def test_strongest_baseline_uses_validation_values_only():
    assert strongest_baseline({"distance_decay": 0.31, "climatology": 0.30}) == "distance_decay"
    assert strongest_baseline({"distance_decay": 0.29, "climatology": 0.30}) == "climatology"


def test_gate2_rules():
    assert gate2([pr(y, 0.01, 0.03) for y in (2018, 2019, 2020, 2021)])["decision"] == "BEATS"
    assert gate2([pr(2018, 0.01, 0.03), *[pr(y, -0.01, 0.02) for y in (2019, 2020, 2021)]])[
        "decision"] == "MATCHES"  # fmt: skip
    g = gate2([pr(2018, 0.01, 0.03), pr(2019, 0.01, 0.02), pr(2020, -0.03, -0.01)])
    assert g["decision"] == "FAILS" and not g["passes"]
    assert gate2([])["decision"] == "NOT_RUN"


def test_gate3_needs_two_folds_above_zero():
    assert gate3([pr(2018, 0.01, 0.02), pr(2019, 0.001, 0.02), pr(2020, -0.01, 0.01)])[
        "decision"] == "RETAIN"  # fmt: skip
    one = gate3([pr(2018, 0.01, 0.02), pr(2019, -0.001, 0.02), pr(2020, -0.01, 0.01)])
    assert one["decision"] == "REMOVE" and one["folds_above_zero"] == [2018]


def test_context_ablation_prefers_shorter_context_unless_worse():
    assert context_ablation([pr(2018, -0.01, 0.01)])["decision"] == "PREFER_T3"
    assert context_ablation([pr(2018, -0.02, -0.01)])["decision"] == "KEEP_T5"
