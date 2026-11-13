"""Binary persistence baselines (Plan v4 6.1). Use for F1/IoU only: a binary map has a single
operating point, so its AUPRC is degenerate and must not be the only AUPRC reference."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

PIXEL_KM = 0.375


def fire_mask(active_fire_band) -> np.ndarray:
    """Binary fire state from the WSTS active-fire band (HHMM values, NaN = no detection).

    Matches the official loader: NaN -> 0, then ``> 0``.
    """
    a = np.asarray(active_fire_band, dtype=np.float64)
    return np.isfinite(a) & (a > 0)


def distance_to_fire_km(fire: np.ndarray) -> np.ndarray:
    """Euclidean distance (km) from every cell to the nearest fire cell; ``inf`` if no fire."""
    fire = np.asarray(fire, dtype=bool)
    if not fire.any():
        return np.full(fire.shape, np.inf)
    return ndimage.distance_transform_edt(~fire) * PIXEL_KM


def persistence(fire: np.ndarray) -> np.ndarray:
    """Tomorrow's fire = today's fire."""
    return np.asarray(fire, dtype=np.uint8)


def dilated_persistence(fire: np.ndarray, radius_km: float) -> np.ndarray:
    """Today's fire grown by ``radius_km`` in every direction (Euclidean disk)."""
    if radius_km < 0:
        raise ValueError("radius must be non-negative")
    return (distance_to_fire_km(fire) <= radius_km).astype(np.uint8)
