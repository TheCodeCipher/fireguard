"""Live frames as a ``TemporalSampleSource`` (the interface the trainer and the OOF export use).

A live frame is assembled as a 23-band raw WSTS frame (NaN in every band the strict model does not
read) and converted with the training loader's own ``process_raw`` and ``channel_indices``. The
result plugs into ``FoldNormalizer.apply`` -> ``pad_samples`` -> ``model.predict`` unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np

from fireguard_ml.data.wsts_dataset import (
    ACTIVE_FIRE_BAND,
    RAW_BANDS,
    channel_indices,
    process_raw,
    raw_bands_needed,
)
from fireguard_ml.live.active_fire import ActiveFireRasterProvider
from fireguard_ml.live.grid import LiveRegion
from fireguard_ml.live.reflectance import (
    REFLECTANCE_CHANNELS,
    MissingReflectanceProvider,
    ReflectanceProvider,
    ReflectanceUnavailable,
)
from fireguard_ml.live.terrain import TERRAIN_BANDS, TerrainProvider
from fireguard_ml.training.temporal_data import SampleKey, TemporalSample, validate_sample

REFLECTANCE_RAW = tuple(RAW_BANDS.index(c) for c in REFLECTANCE_CHANNELS)  # (0, 1, 2)
TERRAIN_RAW = tuple(RAW_BANDS.index(c) for c in TERRAIN_BANDS)  # (12, 13, 14)
LIVE_RAW_BANDS = tuple(sorted({*REFLECTANCE_RAW, *TERRAIN_RAW, ACTIVE_FIRE_BAND}))

assert REFLECTANCE_RAW == (0, 1, 2) and TERRAIN_RAW == (12, 13, 14) and ACTIVE_FIRE_BAND == 22


class LiveInputError(RuntimeError):
    """A live input does not have the shape or content the strict contract needs."""


class FutureDataError(LiveInputError):
    """An input dated after the forecast origin was requested."""


def build_raw_frame(region: LiveRegion, fire_hhmm: np.ndarray, reflectance: np.ndarray,
                    terrain: np.ndarray) -> np.ndarray:
    """``[23, H, W]`` float32 in ``RAW_BANDS`` order; NaN in every band not supplied."""
    h, w = region.hw
    if fire_hhmm.shape != (h, w):
        raise LiveInputError(f"active fire is {fire_hhmm.shape}, region is {(h, w)}")
    if reflectance.shape != (3, h, w):
        raise LiveInputError(f"reflectance is {reflectance.shape}, expected {(3, h, w)}")
    if terrain.shape != (3, h, w):
        raise LiveInputError(f"terrain is {terrain.shape}, expected {(3, h, w)}")
    # Single cells may be NaN (SRTM is masked over open water, as in WSTS frames; the normaliser
    # imputes them like any missing measurement). A band with no data at all is a failed fetch.
    if any(not np.isfinite(band).any() for band in terrain):
        raise LiveInputError(f"{region.region_id}: a terrain band is entirely missing")
    raw = np.full((len(RAW_BANDS), h, w), np.nan, np.float32)
    raw[list(REFLECTANCE_RAW)] = reflectance
    raw[list(TERRAIN_RAW)] = terrain
    raw[ACTIVE_FIRE_BAND] = fire_hhmm
    return raw


def strict_from_raw(raw: np.ndarray, channels: Sequence[str]) -> np.ndarray:
    """Raw ``[23, H, W]`` -> strict ``[C, H, W]``, by the training loader's own code path."""
    return np.ascontiguousarray(process_raw(raw)[channel_indices(channels)], dtype=np.float32)


class LiveTemporalSource:
    """Live samples for a set of regions and one forecast origin day ``D`` (target ``D + 1``).

    ``y`` is an all-zero placeholder (the target does not exist yet); nothing in the live path
    reads it and it is never written to disk.
    """

    def __init__(
        self,
        regions: Sequence[LiveRegion],
        origin_date: date,
        channels: Sequence[str],
        T: int,
        active_fire: ActiveFireRasterProvider,
        reflectance: ReflectanceProvider,
        terrain: TerrainProvider,
        is_mock: bool = False,
    ) -> None:
        self.channels = tuple(channels)
        self.sequence_length = int(T)
        self.is_mock = bool(is_mock)
        self.origin_date = origin_date
        self._regions = {r.region_id: r for r in regions}
        if len(self._regions) != len(regions):
            raise LiveInputError("duplicate region ids")
        missing = set(raw_bands_needed(self.channels)) - set(LIVE_RAW_BANDS)
        if missing:
            names = [RAW_BANDS[b] for b in sorted(missing)]
            raise LiveInputError(f"no live source for raw bands {names}; the live builder supplies "
                                 f"{[RAW_BANDS[b] for b in LIVE_RAW_BANDS]} only")
        self._fire, self._refl, self._terrain = active_fire, reflectance, terrain
        self._raw: dict[tuple[str, date], np.ndarray] = {}
        self._refl_source: dict[tuple[str, date], str] = {}

    # -- TemporalSampleSource -------------------------------------------------------------------

    def keys(self) -> list[SampleKey]:
        target = self.origin_date + timedelta(days=1)
        return sorted((rid, target) for rid in self._regions)

    def load(self, key: SampleKey) -> TemporalSample:
        rid, target = key
        if rid not in self._regions or target != self.origin_date + timedelta(days=1):
            raise KeyError(key)
        region = self._regions[rid]
        T = self.sequence_length
        input_dates = tuple(target - timedelta(days=T - i) for i in range(T))
        x = np.stack([strict_from_raw(self.raw_frame(rid, d), self.channels) for d in input_dates])
        s = TemporalSample(
            event_id=rid,
            input_dates=input_dates,
            target_date=target,
            x=x.astype(np.float32, copy=False),
            y=np.zeros(region.hw, np.uint8),
            valid=np.ones(region.hw, bool),  # every native cell, as in WSTSTemporalSource
            channels=self.channels,
            transform=tuple(region.transform),
            crs=region.crs,
            is_mock=self.is_mock,
        )
        validate_sample(s, T, self.channels)
        return s

    # -- live extras ----------------------------------------------------------------------------

    def region(self, region_id: str) -> LiveRegion:
        return self._regions[region_id]

    def raw_frame(self, region_id: str, day: date) -> np.ndarray:
        """The 23-band raw frame of one input day (cached). Refuses any day after the origin."""
        if day > self.origin_date:
            raise FutureDataError(f"{region_id}: {day} is after the forecast origin "
                                  f"{self.origin_date}")
        k = (region_id, day)
        if k not in self._raw:
            region = self._regions[region_id]
            try:
                refl = np.asarray(self._refl.fetch(region, day), dtype=np.float32)
                source = self._refl.name
            except ReflectanceUnavailable:
                refl = MissingReflectanceProvider().fetch(region, day)
                source = MissingReflectanceProvider.name
            raw = build_raw_frame(region, np.asarray(self._fire.fetch(region, day), np.float32),
                                  refl, np.asarray(self._terrain.fetch(region), np.float32))
            raw.flags.writeable = False
            self._raw[k] = raw
            self._refl_source[k] = source
        return self._raw[k]

    def provenance(self, region_id: str) -> dict:
        """Input completeness of one sample: NaN fraction per strict channel, reflectance source."""
        sample = self.load((region_id, self.origin_date + timedelta(days=1)))
        nan_fraction = {c: float(np.isnan(sample.x[:, i]).mean())
                        for i, c in enumerate(self.channels)}
        sources = sorted({self._refl_source[(region_id, d)] for d in sample.input_dates})
        return {"nan_fraction": nan_fraction,
                "reflectance_source": sources[0] if len(sources) == 1 else "+".join(sources),
                "input_dates": [d.isoformat() for d in sample.input_dates]}
