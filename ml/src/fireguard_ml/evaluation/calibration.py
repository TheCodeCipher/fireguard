"""Fold-specific F1 thresholds and probability calibrators (Plan v4 4.6, 6.3).

Rules this module enforces in code, not only in documentation:

* thresholds and calibrators are fitted on the ``validation`` split only;
* each is tied to one (protocol, fold) and is never reused for another fold;
* there is no API for pooling calibrators across folds.

The calibrator is Platt scaling on the logit of the input probability, fitted by weighted
Newton-Raphson on grouped counts, so it scales to full-frame validation sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fireguard_ml.evaluation.metrics import ScoreCounts, best_f1_threshold

FIT_SPLIT = "validation"
EPS = 1e-6


class IsolationError(RuntimeError):
    """Raised when a threshold or calibrator would be fitted or reused outside its own fold."""


def _require_validation(split: str) -> None:
    if split != FIT_SPLIT:
        raise IsolationError(
            f"thresholds/calibrators may only be fitted on '{FIT_SPLIT}', not '{split}'"
        )


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass(frozen=True)
class FoldThreshold:
    protocol: str
    fold: int
    threshold: float
    validation_f1: float


def select_f1_threshold(counts: ScoreCounts, protocol: str, fold: int, split: str) -> FoldThreshold:
    """Choose the F1-maximising threshold on this fold's validation groups."""
    _require_validation(split)
    t, f1 = best_f1_threshold(counts)
    return FoldThreshold(protocol, fold, t, f1)


@dataclass(frozen=True)
class FoldCalibrator:
    protocol: str
    fold: int
    a: float  # slope on logit(p)
    b: float  # intercept

    def transform(self, p, fold: int, protocol: str = "P1") -> np.ndarray:
        if (protocol, fold) != (self.protocol, self.fold):
            raise IsolationError(
                f"calibrator for {self.protocol}/{self.fold} applied to {protocol}/{fold}"
            )
        z = self.a * _logit(np.asarray(p, dtype=np.float64)) + self.b
        return 1.0 / (1.0 + np.exp(-z))


def fit_platt(
    counts: ScoreCounts, protocol: str, fold: int, split: str, max_iter: int = 50
) -> FoldCalibrator:
    """Weighted logistic regression of the outcome on logit(score), grouped by distinct score."""
    _require_validation(split)
    if counts.n_pos == 0 or counts.n_pos == counts.n:
        raise ValueError("calibration needs both positives and negatives")
    x = _logit(counts.scores)
    w_pos, w_neg = counts.pos.astype(float), counts.neg.astype(float)

    def nll(a_: float, b_: float) -> float:
        z = a_ * x + b_
        # -sum w_pos*log(sigmoid(z)) + w_neg*log(1-sigmoid(z)), computed stably
        return float(np.sum(w_pos * np.logaddexp(0.0, -z) + w_neg * np.logaddexp(0.0, z)))

    a, b = 1.0, 0.0
    cur = nll(a, b)
    converged = False
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        w = w_pos + w_neg
        g_a = np.sum((w_pos - w * p) * x)
        g_b = np.sum(w_pos - w * p)
        h = w * p * (1 - p)
        hess = np.array([[np.sum(h * x * x), np.sum(h * x)], [np.sum(h * x), np.sum(h)]])
        step = np.linalg.solve(hess + 1e-9 * np.eye(2), np.array([g_a, g_b]))
        # Backtracking: plain Newton overshoots on heavily tied scores (e.g. many exact zeros),
        # which once drove the slope to 1e18 on the distance-decay baseline.
        t_ = 1.0
        while t_ > 1e-8:
            na, nb = a + t_ * step[0], b + t_ * step[1]
            new = nll(na, nb)
            if np.isfinite(new) and new <= cur:
                break
            t_ /= 2
        else:
            converged = True  # no descent direction left
            break
        a, b, prev, cur = na, nb, cur, new
        if abs(prev - cur) <= 1e-10 * max(1.0, abs(cur)) or np.max(np.abs(t_ * step)) < 1e-10:
            converged = True
            break
    if not (converged and np.isfinite(a) and np.isfinite(b) and abs(a) < 1e3):
        raise ValueError(f"Platt fit did not converge to a usable calibrator (a={a}, b={b})")
    return FoldCalibrator(protocol, fold, float(a), float(b))


class FoldArtifactStore:
    """Holds one threshold and one calibrator per (protocol, fold). No pooling API exists."""

    def __init__(self) -> None:
        self._thresholds: dict[tuple[str, int], FoldThreshold] = {}
        self._calibrators: dict[tuple[str, int], FoldCalibrator] = {}

    def put_threshold(self, t: FoldThreshold) -> None:
        key = (t.protocol, t.fold)
        if key in self._thresholds:
            raise IsolationError(f"threshold for {key} already fitted; refit means a new run")
        self._thresholds[key] = t

    def put_calibrator(self, c: FoldCalibrator) -> None:
        key = (c.protocol, c.fold)
        if key in self._calibrators:
            raise IsolationError(f"calibrator for {key} already fitted; refit means a new run")
        self._calibrators[key] = c

    def threshold(self, protocol: str, fold: int) -> FoldThreshold:
        return self._thresholds[(protocol, fold)]

    def calibrator(self, protocol: str, fold: int) -> FoldCalibrator:
        return self._calibrators[(protocol, fold)]
