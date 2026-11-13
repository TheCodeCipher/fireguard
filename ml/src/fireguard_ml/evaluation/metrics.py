"""Masked evaluation metrics for FireGuard (contract: ``docs/metrics_contract.md``).

Every metric takes a ``valid_mask``; cells where it is False (padding, invalid data) are dropped
before anything is computed. All metrics are defined at natural prevalence on full frames.

Internally every score-based metric works on *counts grouped by distinct score*:
``(scores, pos, neg)`` with ``scores`` strictly decreasing. That representation

* reproduces scikit-learn's ``average_precision_score`` exactly (ties included),
* lets pooled metrics be built by adding per-event counts instead of concatenating pixels, and
* lets thresholds be chosen on validation without revisiting the pixels.

For model outputs over a whole fold (hundreds of millions of pixels) use
``StreamingScoreHistogram``, which bins scores in [0, 1] into 2**20 bins; ranking metrics are then
exact up to a score resolution of ~1e-6 and the Brier score is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------------------------
# input handling


def _flatten(y_true, y_score, valid_mask=None) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"y_true {y.shape} and scores {s.shape} differ")
    if valid_mask is None:
        m = np.ones(y.shape, dtype=bool)
    else:
        m = np.asarray(valid_mask, dtype=bool)
        if m.shape != y.shape:
            raise ValueError(f"valid_mask {m.shape} does not match {y.shape}")
    y, s = y[m], s[m]
    if y.size and not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be binary inside the valid mask")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite inside the valid mask")
    return y.astype(bool), s


@dataclass
class ScoreCounts:
    """Positive / negative counts per distinct score (scores strictly decreasing)."""

    scores: np.ndarray
    pos: np.ndarray
    neg: np.ndarray
    sum_sq: float = 0.0  # sum of s**2 over all cells (exact Brier)
    sum_s_pos: float = 0.0  # sum of s over positive cells
    extras: dict = field(default_factory=dict)

    @property
    def n_pos(self) -> int:
        return int(self.pos.sum())

    @property
    def n(self) -> int:
        return int(self.pos.sum() + self.neg.sum())


def counts_by_score(y_true, y_score, valid_mask=None) -> ScoreCounts:
    y, s = _flatten(y_true, y_score, valid_mask)
    uniq, inv = np.unique(s, return_inverse=True)
    pos = np.bincount(inv, weights=y, minlength=uniq.size)
    tot = np.bincount(inv, minlength=uniq.size)
    order = slice(None, None, -1)
    return ScoreCounts(
        uniq[order],
        pos[order].astype(np.int64),
        (tot - pos)[order].astype(np.int64),
        float((s**2).sum()),
        float(s[y].sum()),
    )


def merge_counts(parts: list[ScoreCounts]) -> ScoreCounts:
    """Pool several ScoreCounts (e.g. all events of a fold) exactly."""
    parts = [p for p in parts if p.scores.size]
    if not parts:
        return ScoreCounts(np.array([]), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    s = np.concatenate([p.scores for p in parts])
    uniq, inv = np.unique(s, return_inverse=True)
    pos = np.bincount(inv, weights=np.concatenate([p.pos for p in parts]), minlength=uniq.size)
    neg = np.bincount(inv, weights=np.concatenate([p.neg for p in parts]), minlength=uniq.size)
    return ScoreCounts(
        uniq[::-1],
        pos[::-1].astype(np.int64),
        neg[::-1].astype(np.int64),
        sum(p.sum_sq for p in parts),
        sum(p.sum_s_pos for p in parts),
    )


# ---------------------------------------------------------------------------------------------
# ranking


def average_precision_from_counts(c: ScoreCounts) -> float:
    """AP = sum_k (R_k - R_{k-1}) * P_k over distinct thresholds; NaN without positives."""
    n_pos = c.pos.sum()
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(c.pos)
    fp = np.cumsum(c.neg)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(c.pos / n_pos * precision))


def average_precision(y_true, y_score, valid_mask=None) -> float:
    """Exact masked average precision (AUPRC), identical to scikit-learn's definition."""
    return average_precision_from_counts(counts_by_score(y_true, y_score, valid_mask))


auprc = average_precision  # name used in the Role 4 protocol


# ---------------------------------------------------------------------------------------------
# thresholded


def f1_iou_from_confusion(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Positive-class F1 and IoU; NaN when there are no positives and no predictions."""
    denom_f1 = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    return {
        "f1": 2 * tp / denom_f1 if denom_f1 else float("nan"),
        "iou": tp / denom_iou if denom_iou else float("nan"),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def f1_iou(y_true, y_pred, valid_mask=None) -> dict[str, float]:
    y, p = _flatten(y_true, y_pred, valid_mask)
    if p.size and not np.isin(p, (0, 1)).all():
        raise ValueError("y_pred must be binary; threshold scores first")
    p = p.astype(bool)
    return f1_iou_from_confusion(int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum()))


def f1_iou_at_threshold(c: ScoreCounts, threshold: float) -> dict[str, float]:
    """F1/IoU when every cell with score >= threshold is predicted positive."""
    keep = c.scores >= threshold
    tp = int(c.pos[keep].sum())
    fp = int(c.neg[keep].sum())
    return f1_iou_from_confusion(tp, fp, c.n_pos - tp)


def best_f1_threshold(c: ScoreCounts) -> tuple[float, float]:
    """Threshold maximising F1 (predict positive when score >= threshold). Validation use only."""
    if c.n_pos == 0:
        raise ValueError("cannot choose an F1 threshold without positives")
    tp = np.cumsum(c.pos)
    fp = np.cumsum(c.neg)
    f1 = 2 * tp / (tp + fp + c.n_pos)
    k = int(np.argmax(f1))
    return float(c.scores[k]), float(f1[k])


# ---------------------------------------------------------------------------------------------
# probability quality


def brier_from_counts(c: ScoreCounts) -> float:
    """Mean squared error between probability and outcome: (sum s^2 - 2 sum_pos s + n_pos) / n."""
    if c.n == 0:
        return float("nan")
    return (c.sum_sq - 2 * c.sum_s_pos + c.n_pos) / c.n


def brier_score(y_true, y_prob, valid_mask=None) -> float:
    y, p = _flatten(y_true, y_prob, valid_mask)
    if p.size and (p.min() < 0 or p.max() > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def brier_skill_score(brier_model: float, brier_reference: float) -> float:
    """BSS = 1 - BS_model / BS_reference (Plan v4 reference: training-fold climatology)."""
    if not brier_reference > 0:
        return float("nan")
    return 1.0 - brier_model / brier_reference


def reliability_table(c: ScoreCounts, n_bins: int = 15) -> list[dict]:
    """Equal-mass bins (each holds ~1/n_bins of the cells), highest scores first."""
    if c.n == 0:
        return []
    tot = c.pos + c.neg
    cum = np.cumsum(tot)
    edges = np.searchsorted(cum, np.linspace(0, c.n, n_bins + 1)[1:-1], side="left")
    rows, start = [], 0
    for end in list(edges + 1) + [tot.size]:
        end = max(end, start)
        if end == start:
            continue
        sl = slice(start, end)
        n = int(tot[sl].sum())
        rows.append(
            {
                "n": n,
                "mean_prob": float((c.scores[sl] * tot[sl]).sum() / n),
                "observed": float(c.pos[sl].sum() / n),
                "score_high": float(c.scores[sl][0]),
                "score_low": float(c.scores[sl][-1]),
            }
        )
        start = end
    return rows


def ece_equal_mass(c: ScoreCounts, n_bins: int = 15) -> float:
    """Expected calibration error over equal-mass bins (Plan v4 6.5)."""
    rows = reliability_table(c, n_bins)
    if not rows:
        return float("nan")
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["mean_prob"] - r["observed"]) for r in rows) / n)


def calibration_ece(y_true, y_prob, valid_mask=None, bins: int = 15) -> float:
    return ece_equal_mass(counts_by_score(y_true, y_prob, valid_mask), bins)


# ---------------------------------------------------------------------------------------------
# streaming accumulator for model outputs


class StreamingScoreHistogram:
    """Accumulate masked probabilities in [0, 1] over many frames without storing pixels.

    Ranking metrics use 2**20 equal-width bins (resolution ~9.5e-7); Brier sums are exact.
    """

    def __init__(self, n_bins: int = 2**20) -> None:
        self.n_bins = n_bins
        self.pos = np.zeros(n_bins, dtype=np.int64)
        self.neg = np.zeros(n_bins, dtype=np.int64)
        self.sum_sq = 0.0
        self.sum_s_pos = 0.0

    def add(self, y_true, y_prob, valid_mask=None) -> None:
        y, p = _flatten(y_true, y_prob, valid_mask)
        if p.size and (p.min() < 0 or p.max() > 1):
            raise ValueError("probabilities must lie in [0, 1]")
        idx = np.minimum((p * self.n_bins).astype(np.int64), self.n_bins - 1)
        self.pos += np.bincount(idx[y], minlength=self.n_bins)
        self.neg += np.bincount(idx[~y], minlength=self.n_bins)
        self.sum_sq += float((p**2).sum())
        self.sum_s_pos += float(p[y].sum())

    def merge(self, other: StreamingScoreHistogram) -> None:
        self.pos += other.pos
        self.neg += other.neg
        self.sum_sq += other.sum_sq
        self.sum_s_pos += other.sum_s_pos

    def counts(self) -> ScoreCounts:
        used = np.nonzero(self.pos + self.neg)[0][::-1]
        centres = (used + 0.5) / self.n_bins
        return ScoreCounts(centres, self.pos[used], self.neg[used], self.sum_sq, self.sum_s_pos)


# ---------------------------------------------------------------------------------------------
# one-call summary


def summarise(c: ScoreCounts, threshold: float | None = None, ece_bins: int = 15) -> dict:
    """All headline metrics from one ScoreCounts. F1/IoU only when a threshold is supplied."""
    out = {
        "n_valid": c.n,
        "n_pos": c.n_pos,
        "prevalence": c.n_pos / c.n if c.n else float("nan"),
        "auprc": average_precision_from_counts(c),
        "brier": brier_from_counts(c),
        "ece": ece_equal_mass(c, ece_bins),
    }
    if threshold is not None:
        out.update(f1_iou_at_threshold(c, threshold))
        out["threshold"] = threshold
    return out
