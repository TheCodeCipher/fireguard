"""Continuous distance baselines (Plan v4 6.1): distance-to-fire decay and the training-fold
distance-bin climatology ``P(next-day fire | distance-to-current-fire bin)``.

Both scores are functions of one quantity, the distance from a cell to today's nearest fire. A run
over a dataset therefore only needs, per event, the number of positive and negative target cells
at each distance (``DistanceCounts``). Every baseline score, threshold and metric is then derived
from those counts without touching pixels again.

Two facts worth knowing:

* The decay score ``amplitude * exp(-d / scale)`` is strictly decreasing in d, so its ranking, and
  hence its AUPRC, depends on neither parameter. Both are chosen on validation by Brier score, the
  metric they do affect: without an amplitude every burning cell would get probability 1 although
  only a fraction of them burn the next day. For each candidate scale the Brier-optimal amplitude
  has a closed form, ``sum(e * y) / sum(e ** 2)`` clipped to [0, 1].
* The climatology is not forced to be monotone in distance; it is whatever the training groups
  show, smoothed towards the training prevalence when a bin is sparse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fireguard_ml.baselines.persistence import distance_to_fire_km
from fireguard_ml.evaluation.metrics import ScoreCounts, brier_from_counts

KEY_PER_KM = 100  # distances stored at 10 m resolution
NO_FIRE_KEY = -1  # frame has no fire on the input day
DEFAULT_BIN_EDGES_KM = (0.0, 0.375, 0.75, 1.5, 3.0, 5.0, 10.0, 20.0, 50.0)
FIT_SPLIT = "train"
SELECT_SPLIT = "validation"


class SplitError(RuntimeError):
    """Raised when a baseline parameter would be fitted on the wrong split."""


@dataclass
class DistanceCounts:
    """Positive / negative next-day cells per distance key (10 m units; -1 = no fire in frame)."""

    keys: np.ndarray
    pos: np.ndarray
    neg: np.ndarray

    @classmethod
    def empty(cls) -> DistanceCounts:
        z = np.array([], dtype=np.int64)
        return cls(z, z.copy(), z.copy())

    @classmethod
    def from_frame(cls, fire_today, target_tomorrow, valid_mask=None) -> DistanceCounts:
        fire = np.asarray(fire_today, dtype=bool)
        y = np.asarray(target_tomorrow, dtype=bool)
        m = np.ones(y.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, bool)
        d = distance_to_fire_km(fire)
        keys = np.where(np.isinf(d), NO_FIRE_KEY, np.rint(d * KEY_PER_KM)).astype(np.int64)
        k, y = keys[m], y[m]
        uniq, inv = np.unique(k, return_inverse=True)
        pos = np.bincount(inv, weights=y, minlength=uniq.size).astype(np.int64)
        tot = np.bincount(inv, minlength=uniq.size).astype(np.int64)
        return cls(uniq, pos, tot - pos)

    @classmethod
    def merge(cls, parts: list[DistanceCounts]) -> DistanceCounts:
        parts = [p for p in parts if p.keys.size]
        if not parts:
            return cls.empty()
        k = np.concatenate([p.keys for p in parts])
        uniq, inv = np.unique(k, return_inverse=True)
        pos = np.bincount(inv, weights=np.concatenate([p.pos for p in parts]), minlength=uniq.size)
        neg = np.bincount(inv, weights=np.concatenate([p.neg for p in parts]), minlength=uniq.size)
        return cls(uniq, pos.astype(np.int64), neg.astype(np.int64))

    def distance_km(self) -> np.ndarray:
        return np.where(self.keys == NO_FIRE_KEY, np.inf, self.keys / KEY_PER_KM)

    def to_score_counts(self, scores_per_key: np.ndarray) -> ScoreCounts:
        """Group cells by the score a baseline gives them (for the metric functions)."""
        s = np.asarray(scores_per_key, dtype=np.float64)
        uniq, inv = np.unique(s, return_inverse=True)
        pos = np.bincount(inv, weights=self.pos, minlength=uniq.size)
        neg = np.bincount(inv, weights=self.neg, minlength=uniq.size)
        n = self.pos + self.neg
        return ScoreCounts(
            uniq[::-1],
            pos[::-1].astype(np.int64),
            neg[::-1].astype(np.int64),
            float((s**2 * n).sum()),
            float((s * self.pos).sum()),
        )

    def to_arrays(self) -> np.ndarray:
        return np.stack([self.keys, self.pos, self.neg])

    @classmethod
    def from_arrays(cls, a: np.ndarray) -> DistanceCounts:
        return cls(a[0].astype(np.int64), a[1].astype(np.int64), a[2].astype(np.int64))


# ---------------------------------------------------------------------------------------------
# scores per distance key


def decay_scores(counts: DistanceCounts, scale_km: float, amplitude: float = 1.0) -> np.ndarray:
    if scale_km <= 0:
        raise ValueError("scale must be positive")
    if not 0 < amplitude <= 1:
        raise ValueError("amplitude must lie in (0, 1]")
    return amplitude * np.exp(-counts.distance_km() / scale_km)  # exp(-inf) = 0 without fire


def persistence_scores(counts: DistanceCounts) -> np.ndarray:
    return (counts.keys == 0).astype(np.float64)


def dilated_scores(counts: DistanceCounts, radius_km: float) -> np.ndarray:
    return (counts.distance_km() <= radius_km + 1e-9).astype(np.float64)


def best_amplitude(counts: DistanceCounts, scale_km: float) -> float:
    """Brier-optimal amplitude for one scale: argmin_a sum((a * e - y) ** 2), clipped to (0, 1]."""
    e = decay_scores(counts, scale_km)
    denom = float((e**2 * (counts.pos + counts.neg)).sum())
    a = float((e * counts.pos).sum()) / denom if denom > 0 else 1.0
    return float(np.clip(a, 1e-12, 1.0))


def select_decay_scale(
    validation: DistanceCounts, candidates_km, split: str
) -> tuple[float, float, dict]:
    """Pick (scale, amplitude) with the lowest pooled validation Brier score."""
    if split != SELECT_SPLIT:
        raise SplitError(f"decay scale is chosen on '{SELECT_SPLIT}', not '{split}'")
    briers, amps = {}, {}
    for s in candidates_km:
        s = float(s)
        amps[s] = best_amplitude(validation, s)
        sc = validation.to_score_counts(decay_scores(validation, s, amps[s]))
        briers[s] = brier_from_counts(sc)
    best = min(briers, key=briers.get)
    return best, amps[best], {"brier_by_scale": briers, "amplitude_by_scale": amps}


@dataclass(frozen=True)
class DistanceBinClimatology:
    """``P(next-day fire | distance bin)`` fitted on one fold's training groups only."""

    edges_km: tuple[float, ...]
    probs: tuple[float, ...]  # one per distance bin, then one for "no fire in frame"
    fold: int
    train_prevalence: float

    def bin_of(self, distance_km: np.ndarray) -> np.ndarray:
        d = np.asarray(distance_km, dtype=np.float64)
        b = np.searchsorted(np.asarray(self.edges_km), d - 1e-9, side="left")
        return np.where(np.isinf(d), len(self.edges_km) + 1, b)

    def predict_distance(self, distance_km) -> np.ndarray:
        return np.asarray(self.probs)[self.bin_of(distance_km)]

    def predict_frame(self, fire_today) -> np.ndarray:
        return self.predict_distance(distance_to_fire_km(fire_today))

    def scores(self, counts: DistanceCounts) -> np.ndarray:
        return self.predict_distance(counts.distance_km())


def fit_climatology(
    train: DistanceCounts,
    fold: int,
    split: str,
    edges_km=DEFAULT_BIN_EDGES_KM,
    prior_strength: float = 1.0,
) -> DistanceBinClimatology:
    """Fit bin probabilities on training groups; sparse bins shrink towards the training rate."""
    if split != FIT_SPLIT:
        raise SplitError(f"climatology is fitted on '{FIT_SPLIT}' groups only, not '{split}'")
    n_total = train.pos.sum() + train.neg.sum()
    if n_total == 0:
        raise ValueError("no training cells")
    p0 = train.pos.sum() / n_total
    edges = tuple(float(e) for e in edges_km)
    dummy = DistanceBinClimatology(edges, (0.0,) * (len(edges) + 2), fold, p0)
    bins = dummy.bin_of(train.distance_km())
    n_bins = len(edges) + 2
    pos = np.bincount(bins, weights=train.pos, minlength=n_bins)
    tot = np.bincount(bins, weights=train.pos + train.neg, minlength=n_bins)
    probs = (pos + prior_strength * p0) / (tot + prior_strength)
    return DistanceBinClimatology(edges, tuple(float(p) for p in probs), fold, float(p0))
