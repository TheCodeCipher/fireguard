"""Next-day verification of a live forecast, with training-target semantics.

The observed target is built from the next day's raw active-fire band with the loader's own
``target_from_band`` (NaN -> 0, then ``> 0``), and scored with Role 4's metric code. The reference
is dilated persistence of the origin day's fire (binary, so F1 / IoU only: its AUPRC is degenerate).
"""

from __future__ import annotations

import math

import numpy as np

from fireguard_ml.baselines.persistence import dilated_persistence, fire_mask
from fireguard_ml.data.wsts_dataset import target_from_band
from fireguard_ml.evaluation.metrics import counts_by_score, f1_iou, summarise

SCORE_FIELDS = ("auprc", "brier", "f1", "iou", "n_pos", "n_valid",
                "persistence_f1", "persistence_iou")


def _finite_or_none(v) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def score_forecast(prob: np.ndarray, target_fire_band: np.ndarray, origin_fire_band: np.ndarray,
                   threshold: float, persistence_radius_km: float = 0.375) -> dict:
    """Metrics of one probability map against the observed next-day fire.

    ``prob`` is NaN on invalid cells; ``*_fire_band`` are raw active-fire bands (HHMM / NaN).
    Undefined metrics (e.g. AUPRC on a day without fire) are ``None``, as in the replay registry.
    """
    if prob.shape != target_fire_band.shape or prob.shape != origin_fire_band.shape:
        raise ValueError("probability, target and origin rasters must share one grid")
    y = target_from_band(target_fire_band).astype(bool)
    valid = np.isfinite(prob)
    m = summarise(counts_by_score(y, np.where(valid, prob, 0.0), valid), threshold=threshold)
    ref = dilated_persistence(fire_mask(origin_fire_band), persistence_radius_km).astype(bool)
    p = f1_iou(y, ref, valid)
    return {
        "auprc": _finite_or_none(m["auprc"]), "brier": _finite_or_none(m["brier"]),
        "f1": _finite_or_none(m["f1"]), "iou": _finite_or_none(m["iou"]),
        "n_pos": int(m["n_pos"]), "n_valid": int(m["n_valid"]),
        "persistence_f1": _finite_or_none(p["f1"]), "persistence_iou": _finite_or_none(p["iou"]),
    }
