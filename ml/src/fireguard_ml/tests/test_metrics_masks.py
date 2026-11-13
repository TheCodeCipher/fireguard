"""Metric correctness and mask handling: padded / invalid cells must never affect any metric."""

from __future__ import annotations

import numpy as np
import pytest

from fireguard_ml.evaluation.metrics import (
    StreamingScoreHistogram,
    average_precision,
    best_f1_threshold,
    brier_from_counts,
    brier_score,
    brier_skill_score,
    calibration_ece,
    counts_by_score,
    f1_iou,
    f1_iou_at_threshold,
    merge_counts,
    summarise,
)

rng = np.random.default_rng(7)


def frame(h=30, w=40, prevalence=0.05):
    y = (rng.random((h, w)) < prevalence).astype(np.uint8)
    s = np.clip(0.3 * y + 0.8 * rng.random((h, w)), 0, 1)  # overlapping classes
    return y, s


def padded(y, s, pad_h=2, pad_w=8, y_fill=0, s_fill=0.99):
    """Pad bottom/right like pad-to-32 and fill the padding with confident false positives."""
    yp = np.pad(y, ((0, pad_h), (0, pad_w)), constant_values=y_fill)
    sp = np.pad(s, ((0, pad_h), (0, pad_w)), constant_values=s_fill)
    mask = np.pad(np.ones_like(y, dtype=bool), ((0, pad_h), (0, pad_w)), constant_values=False)
    return yp, sp, mask


def test_average_precision_hand_example():
    # scores 0.9(pos) 0.8(neg) 0.7(pos) 0.1(neg): AP = 0.5*1 + 0.5*(2/3)
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    assert average_precision(y, s) == pytest.approx(0.5 + 1 / 3)


def test_average_precision_matches_scikit_learn_including_ties():
    sk = pytest.importorskip("sklearn.metrics")
    for _ in range(5):
        y = (rng.random(2000) < 0.03).astype(int)
        s = np.round(rng.random(2000) + 0.3 * y, 2)  # rounding creates many ties
        assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s))


def test_every_metric_ignores_padding_and_invalid_cells():
    y, s = frame()
    yp, sp, mask = padded(y, s)
    assert average_precision(yp, sp, mask) == pytest.approx(average_precision(y, s))
    assert brier_score(yp, sp, mask) == pytest.approx(brier_score(y, s))
    assert calibration_ece(yp, sp, mask) == pytest.approx(calibration_ece(y, s))
    pred, predp = (s >= 0.5).astype(int), (sp >= 0.5).astype(int)
    assert f1_iou(yp, predp, mask) == f1_iou(y, pred)
    # the padding really is adversarial: without the mask results change
    assert average_precision(yp, sp) != pytest.approx(average_precision(y, s))


def test_invalid_cells_inside_the_frame_are_excluded_too():
    y, s = frame()
    mask = np.ones_like(y, dtype=bool)
    mask[5:10, 5:10] = False
    y2, s2 = y.copy(), s.copy()
    y2[5:10, 5:10], s2[5:10, 5:10] = 1, 0.0  # garbage where invalid
    assert average_precision(y2, s2, mask) == pytest.approx(average_precision(y[mask], s[mask]))


def test_nan_scores_are_allowed_only_outside_the_mask():
    y, s = frame()
    s = s.copy()
    s[0, 0] = np.nan
    mask = np.ones_like(y, dtype=bool)
    with pytest.raises(ValueError, match="finite"):
        average_precision(y, s, mask)
    mask[0, 0] = False
    assert np.isfinite(average_precision(y, s, mask)) or y[mask].sum() == 0


def test_shape_mismatch_and_non_binary_target_are_rejected():
    with pytest.raises(ValueError):
        average_precision(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="binary"):
        average_precision(np.full((2, 2), 2), np.zeros((2, 2)))


def test_ap_is_nan_without_positives_and_f1_iou_handle_empty_cases():
    assert np.isnan(average_precision(np.zeros(5), rng.random(5)))
    r = f1_iou(np.zeros(5), np.zeros(5))
    assert np.isnan(r["f1"]) and np.isnan(r["iou"])
    r = f1_iou(np.array([1, 0]), np.array([1, 0]))
    assert r["f1"] == 1 and r["iou"] == 1


def test_pooled_counts_equal_concatenated_pixels():
    frames = [frame() for _ in range(4)]
    merged = merge_counts([counts_by_score(y, s) for y, s in frames])
    ys = np.concatenate([y.ravel() for y, _ in frames])
    ss = np.concatenate([s.ravel() for _, s in frames])
    assert summarise(merged)["auprc"] == pytest.approx(average_precision(ys, ss))
    assert brier_from_counts(merged) == pytest.approx(brier_score(ys, ss))


def test_threshold_metrics_from_counts_equal_pixel_metrics():
    y, s = frame()
    c = counts_by_score(y, s)
    t, f1 = best_f1_threshold(c)
    direct = f1_iou(y, (s >= t).astype(int))
    assert f1_iou_at_threshold(c, t) == direct
    assert direct["f1"] == pytest.approx(f1)


def test_streaming_histogram_matches_exact_metrics():
    hist = StreamingScoreHistogram()
    exact = []
    for _ in range(3):
        y, s = frame(60, 70)
        yp, sp, mask = padded(y, s)
        hist.add(yp, sp, mask)
        exact.append(counts_by_score(y, s))
    exact_c = merge_counts(exact)
    approx_c = hist.counts()
    assert summarise(approx_c)["auprc"] == pytest.approx(summarise(exact_c)["auprc"], abs=1e-4)
    assert brier_from_counts(approx_c) == pytest.approx(brier_from_counts(exact_c), rel=1e-12)


def test_brier_skill_score():
    assert brier_skill_score(0.05, 0.10) == pytest.approx(0.5)
    assert np.isnan(brier_skill_score(0.05, 0.0))


def test_perfectly_calibrated_scores_have_small_ece():
    p = rng.random(200_000)
    y = (rng.random(p.size) < p).astype(int)
    assert calibration_ece(y, p) < 0.01
