"""VIIRS surface reflectance (raw bands 0-2: M11, I2, I1) for live frames.

Training data used VNP09GA Collection 1 (``NOAA/VIIRS/001/VNP09GA``), stored as scaled integers
(reflectance x 10,000). That collection is decommissioned; current data is Collection 2, stored in
reflectance units. A provider must return values **on the Collection 1 scale**; a per-band linear
map (``LinearReflectanceMapping``, fitted by ``scripts/check_live_input_parity.py`` from input
pixels only) absorbs the remaining calibration difference.

A missing observation is NaN. The fold normaliser turns NaN into the fold's training mean, exactly
as it does for cloudy training pixels; the forecast record reports the NaN fraction per channel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np

from fireguard_ml.live.grid import LiveRegion

REFLECTANCE_CHANNELS = ("viirs_m11", "viirs_i2", "viirs_i1")  # raw bands 0, 1, 2
V2_TO_V1_SCALE = 10_000.0


class ReflectanceUnavailable(RuntimeError):
    """No reflectance could be obtained for this region and day (the caller falls back to NaN)."""


class ReflectanceProvider(Protocol):
    name: str

    def fetch(self, region: LiveRegion, day: date) -> np.ndarray:
        """``[3, H, W]`` float32 (M11, I2, I1) on the Collection 1 scale; NaN where missing."""


class MissingReflectanceProvider:
    """All NaN: the model then sees the fold's training-mean reflectance. Always labelled."""

    name = "missing"

    def fetch(self, region: LiveRegion, day: date) -> np.ndarray:
        return np.full((len(REFLECTANCE_CHANNELS), *region.hw), np.nan, np.float32)


@dataclass(frozen=True)
class LinearReflectanceMapping:
    """``v1 = scale * x + offset`` per band, where ``x`` is already on the x10,000 scale."""

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fitted_on: str = "identity"

    def apply(self, x: np.ndarray) -> np.ndarray:
        s = np.asarray(self.scale, np.float32)[:, None, None]
        o = np.asarray(self.offset, np.float32)[:, None, None]
        return (x * s + o).astype(np.float32)

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "channels": list(REFLECTANCE_CHANNELS), "scale": list(self.scale),
            "offset": list(self.offset), "fitted_on": self.fitted_on,
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> LinearReflectanceMapping:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(d["channels"]) != REFLECTANCE_CHANNELS:
            raise ValueError(f"{path}: channels {d['channels']} != {REFLECTANCE_CHANNELS}")
        return cls(tuple(map(float, d["scale"])), tuple(map(float, d["offset"])),
                   str(d.get("fitted_on", "unknown")))


def fit_linear_mapping(x: np.ndarray, y: np.ndarray, fitted_on: str) -> LinearReflectanceMapping:
    """Least-squares ``y ~ scale * x + offset`` per band over cells finite in both. Inputs only:
    ``x`` and ``y`` are reflectance stacks ``[N, 3, ...]``; no label is involved."""
    scale, offset = [], []
    for b in range(len(REFLECTANCE_CHANNELS)):
        xb, yb = x[:, b].ravel().astype(np.float64), y[:, b].ravel().astype(np.float64)
        ok = np.isfinite(xb) & np.isfinite(yb)
        if ok.sum() < 100 or np.var(xb[ok]) == 0:
            raise ValueError(f"not enough paired cells to fit {REFLECTANCE_CHANNELS[b]}")
        s, o = np.polyfit(xb[ok], yb[ok], 1)
        scale.append(float(s))
        offset.append(float(o))
    return LinearReflectanceMapping(tuple(scale), tuple(offset), fitted_on)  # type: ignore[arg-type]
