"""Next-day scoring uses the training target semantics and Role 4's metric code (layer 1)."""

from __future__ import annotations

import numpy as np

from fireguard_ml.baselines.persistence import dilated_persistence, fire_mask
from fireguard_ml.data.wsts_dataset import target_from_band
from fireguard_ml.evaluation.metrics import counts_by_score, f1_iou, summarise
from fireguard_ml.live.scoring import SCORE_FIELDS, score_forecast


def _bands(hw=(30, 30)):
    rng = np.random.default_rng(0)
    target = np.full(hw, np.nan, np.float32)
    target[10:14, 10:15] = 2030.0
    target[2, 2] = 30.0  # 00:30 UTC: still a positive target (> 0)
    origin = np.full(hw, np.nan, np.float32)
    origin[10:13, 9:13] = 1400.0
    prob = rng.uniform(0, 0.3, hw).astype(np.float32)
    prob[9:15, 9:16] += 0.6
    prob[0, :] = np.nan
    return prob, target, origin


def test_matches_target_from_band_and_role4_metrics():
    prob, target, origin = _bands()
    m = score_forecast(prob, target, origin, threshold=0.5)
    assert tuple(m) == SCORE_FIELDS
    y = target_from_band(target).astype(bool)
    valid = np.isfinite(prob)
    ref = summarise(counts_by_score(y, np.where(valid, prob, 0.0), valid), threshold=0.5)
    assert m["auprc"] == ref["auprc"] and m["brier"] == ref["brier"]
    assert m["f1"] == ref["f1"] and m["iou"] == ref["iou"]
    assert m["n_pos"] == 21 and m["n_valid"] == 870  # 20 cells + the 00:30 one; row 0 invalid
    p = f1_iou(y, dilated_persistence(fire_mask(origin), 0.375).astype(bool), valid)
    assert m["persistence_f1"] == p["f1"] and m["persistence_iou"] == p["iou"]


def test_day_without_fire_gives_none_not_nan():
    prob, _, origin = _bands()
    m = score_forecast(prob, np.full(prob.shape, np.nan, np.float32), origin, threshold=0.5)
    assert m["n_pos"] == 0 and m["auprc"] is None


def test_grids_must_match():
    prob, target, origin = _bands()
    try:
        score_forecast(prob, target[:-1], origin, 0.5)
    except ValueError:
        return
    raise AssertionError("mismatched grids must be refused")
