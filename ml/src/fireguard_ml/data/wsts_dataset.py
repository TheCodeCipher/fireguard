"""WSTS GeoTIFF dataset interface for FireGuard Phase 3 (Role 1, P3-R1-05).

Reads the extracted WildfireSpreadTS GeoTIFFs (``<root>/<year>/<event_id>/<YYYY-MM-DD>.tif``,
23 float32 bands, EPSG:32610, 375 m) and emits ``TemporalSample`` objects that satisfy Role 3's
``TemporalSampleSource`` contract (``training/temporal_data.py``). With ``T=1`` the same source
serves Role 2's M1.

Semantics are those of the pinned baseline loader (WildfireSpreadTS @ f62dc49,
``src/dataloader/FireSpreadDataset.py``), minus its training-time and evaluation-time spatial
transforms and its fixed standardisation statistics:

* sequence: ``T`` consecutive daily files as input, the next file's band 23 as target
  (loader lines 136-158);
* target: band 23 (``active fire``, 0-based 22), NaN -> 0, then ``> 0`` (lines 323, 328);
* input active-fire band: NaN -> 0, HHMM -> floor(HHMM / 100) (lines 322, 326);
* degree bands 7, 13, 19 (0-based) -> ``sin(deg2rad(.))`` (line 339, utils.py line 162);
* binary active-fire mask = processed active-fire > 0, appended last (lines 343, 348);
* land cover (band 16) -> 17 one-hot planes in place (lines 353-361).

Deliberately **not** reproduced, by FireGuard decision (plan v4):

* random crops, flips and rotations (training) and the centre crop to a multiple of 32
  (evaluation). Frames keep their native extent; padding to a multiple of 32 is done at batching
  (``training/temporal_data.pad_samples``) and is marked invalid;
* standardisation with the loader's year-pair statistics. Standardisation is fold-specific and
  fitted on training groups only (``training/normalization.py``). Missing measurements stay NaN
  here and are imputed by that normaliser with the standardised mean, which is what the loader's
  final ``nan_to_num`` does.

The target file contributes **only** band 23, read on its own; no other band of the target day is
ever read (``test_wsts_dataset::test_target_day_contributes_only_the_target_band``).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from fireguard_ml.training.temporal_data import (
    SampleKey,
    TemporalContractError,
    TemporalSample,
    validate_sample,
)

PINNED_LOADER_REPO = "https://github.com/SebastianGer/WildfireSpreadTS"
PINNED_LOADER_COMMIT = "f62dc49f2c2a6d249c071533ae5340c0d01bfb95"
DATA_CONTRACT_VERSION = "wsts_strict_input_v1"

# Raw GeoTIFF bands, 0-based, named as in Role 4's leakage register.
RAW_BANDS: tuple[str, ...] = (
    "viirs_m11", "viirs_i2", "viirs_i1", "ndvi", "evi2",
    "total_precipitation", "wind_speed", "wind_direction", "minimum_temperature",
    "maximum_temperature", "energy_release_component", "specific_humidity",
    "slope", "aspect", "elevation", "palmer_drought_severity_index", "land_cover_class",
    "gfs_total_precipitation", "gfs_wind_speed", "gfs_wind_direction", "gfs_temperature",
    "gfs_specific_humidity", "active_fire",
)  # fmt: skip
# Band descriptions stored in the TIFFs, in the same order (checked on every file read). Band 18
# (GFS precipitation, excluded) is named ``..._sum`` before the 2019-11-07 change in the creation
# code and ``..._last`` after it; both are accepted (ALT_DESCRIPTIONS).
ALT_DESCRIPTIONS = {17: frozenset({"total_precipitation_surface_sum",
                                    "total_precipitation_surface_last"})}  # fmt: skip
RAW_DESCRIPTIONS: tuple[str, ...] = (
    "M11", "I2", "I1", "NDVI_last", "EVI2_last", "total precipitation", "wind speed",
    "wind direction", "minimum temperature", "maximum temperature", "energy release component",
    "specific humidity", "slope", "aspect", "elevation", "pdsi", "LC_Type1",
    "total_precipitation_surface_sum", "forecast wind speed", "forecast wind direction",
    "forecast temperature", "forecast specific humidity", "active fire",
)  # fmt: skip
LAND_COVER_CLASSES: tuple[str, ...] = (
    "evergreen_needleleaf", "evergreen_broadleaf", "deciduous_needleleaf", "deciduous_broadleaf",
    "mixed_forest", "closed_shrubland", "open_shrubland", "woody_savanna", "savanna",
    "grassland", "permanent_wetland", "cropland", "urban", "cropland_natural_mosaic",
    "snow_ice", "barren", "water",
)  # fmt: skip
PROCESSED_CHANNELS: tuple[str, ...] = (
    RAW_BANDS[:16]
    + tuple(f"land_cover_{c}" for c in LAND_COVER_CLASSES)
    + RAW_BANDS[17:22]
    + ("active_fire_time", "binary_active_fire_mask")
)
DEGREE_BANDS = (7, 13, 19)  # sin() applied; pinned utils.get_indices_of_degree_features
LAND_COVER_BAND = 16
ACTIVE_FIRE_BAND = 22
TARGET_BAND_1BASED = ACTIVE_FIRE_BAND + 1
# processed index -> raw band it comes from (None = derived, not a band of its own)
PROCESSED_TO_RAW: tuple[int | None, ...] = (
    tuple(range(16)) + (LAND_COVER_BAND,) * 17 + tuple(range(17, 22)) + (22, 22)
)
DATE_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.tif$")

assert len(PROCESSED_CHANNELS) == 40 and len(PROCESSED_TO_RAW) == 40


class WSTSAlignmentError(TemporalContractError):
    """Frames of one sample do not share shape, transform and CRS."""


class WSTSFileError(TemporalContractError):
    """A file the sample needs is missing, unreadable or has the wrong band layout."""


def process_raw(raw: np.ndarray) -> np.ndarray:
    """23 raw bands -> 40 processed channels, pinned-loader semantics, before standardisation.

    ``raw`` is ``[..., 23, H, W]`` float32. NaN in measured bands is preserved (missing value).
    """
    if raw.shape[-3] != len(RAW_BANDS):
        raise ValueError(f"expected {len(RAW_BANDS)} bands, got {raw.shape[-3]}")
    x = raw.astype(np.float32, copy=True)
    af = np.nan_to_num(x[..., ACTIVE_FIRE_BAND, :, :], nan=0.0)
    x[..., ACTIVE_FIRE_BAND, :, :] = np.floor_divide(af, 100)
    for b in DEGREE_BANDS:
        x[..., b, :, :] = np.sin(np.deg2rad(x[..., b, :, :]))
    binary = (x[..., ACTIVE_FIRE_BAND : ACTIVE_FIRE_BAND + 1, :, :] > 0).astype(np.float32)
    # Land cover exactly as the loader: NaN -> 0 happens before the one-hot lookup, so class
    # index -1 selects the LAST row (water). Recorded as a loader defect; land cover is excluded
    # from the strict inputs, so it does not affect FireGuard models.
    lc = np.nan_to_num(x[..., LAND_COVER_BAND, :, :], nan=0.0).astype(np.int64) - 1
    onehot = np.eye(17, dtype=np.float32)[lc]  # [..., H, W, 17]
    onehot = np.moveaxis(onehot, -1, -3)
    return np.concatenate(
        [x[..., :LAND_COVER_BAND, :, :], onehot, x[..., LAND_COVER_BAND + 1 :, :, :], binary],
        axis=-3,
    )


def target_from_band(af_band: np.ndarray) -> np.ndarray:
    """Next-day target from the raw active-fire band: NaN -> 0, then ``> 0`` (uint8)."""
    return (np.nan_to_num(af_band, nan=0.0) > 0).astype(np.uint8)


def channel_indices(channels: Sequence[str]) -> list[int]:
    unknown = [c for c in channels if c not in PROCESSED_CHANNELS]
    if unknown:
        raise ValueError(f"unknown processed channels {unknown}")
    return [PROCESSED_CHANNELS.index(c) for c in channels]


def raw_bands_needed(channels: Sequence[str]) -> list[int]:
    """0-based raw bands that must be read to build ``channels`` (processing is per band)."""
    return sorted({PROCESSED_TO_RAW[i] for i in channel_indices(channels)})


# ------------------------------------------------------------------------------------- index


@dataclass(frozen=True)
class EventFiles:
    event_id: str
    year: int
    folder: Path
    dates: tuple[date, ...]

    def path(self, d: date) -> Path:
        return self.folder / f"{d.isoformat()}.tif"


def scan_events(data_root: Path, years: Iterable[int] = (2018, 2019, 2020, 2021)) -> dict:
    """``{event_id: EventFiles}`` with dates sorted exactly as the loader sorts file names."""
    out = {}
    for y in years:
        ydir = Path(data_root) / str(y)
        if not ydir.is_dir():
            continue
        for ev in sorted(p for p in ydir.iterdir() if p.is_dir()):
            ds = sorted(date.fromisoformat(m.group(1)) for f in ev.iterdir()
                        if (m := DATE_FILE.match(f.name)))  # fmt: skip
            if ev.name in out:
                raise ValueError(f"event {ev.name} appears in more than one year")
            out[ev.name] = EventFiles(ev.name, y, ev, tuple(ds))
    return out


# ------------------------------------------------------------------------------------- reading


def descriptions_ok(desc: Sequence[str | None]) -> bool:
    if len(desc) != len(RAW_DESCRIPTIONS):
        return False
    return all(d in ALT_DESCRIPTIONS.get(i, {RAW_DESCRIPTIONS[i]}) for i, d in enumerate(desc))


def _read(path: Path, bands_1based: Sequence[int]):
    import rasterio

    try:
        with rasterio.open(path) as ds:
            if ds.count != len(RAW_BANDS):
                raise WSTSFileError(f"{path}: {ds.count} bands, expected {len(RAW_BANDS)}")
            if not descriptions_ok(ds.descriptions):
                raise WSTSFileError(f"{path}: band descriptions differ from the WSTS schema")
            arr = ds.read(list(bands_1based)).astype(np.float32)
            return arr, tuple(ds.transform)[:6], ds.crs.to_string() if ds.crs else None
    except WSTSFileError:
        raise
    except Exception as exc:  # rasterio raises several types for damaged/partial files
        raise WSTSFileError(f"{path}: unreadable ({exc})") from exc


class WSTSTemporalSource:
    """Real WSTS samples for the strict contract. ``is_mock`` is False.

    Keys are the rows of ``wsts_eval_samples.csv`` (identical for M1, M3, M4) restricted to
    events present on disk and, optionally, to ``events``.
    """

    is_mock = False

    def __init__(
        self,
        data_root: Path,
        channels: Sequence[str],
        T: int,
        samples_csv: Path,
        events: Iterable[str] | None = None,
        cache_frames: int = 96,
    ) -> None:
        self.data_root = Path(data_root)
        self.channels = tuple(channels)
        self.sequence_length = int(T)
        self._proc_idx = channel_indices(self.channels)
        self._bands = raw_bands_needed(self.channels)
        # positions of the needed bands inside a full 23-band layout
        self._index = scan_events(self.data_root)
        wanted = set(events) if events is not None else None
        keys = []
        with Path(samples_csv).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ev = r["event_id"]
                if ev in self._index and (wanted is None or ev in wanted):
                    keys.append((ev, date.fromisoformat(r["target_date"])))
        self._keys = sorted(keys)
        self._cache: OrderedDict = OrderedDict()
        self._cache_frames = cache_frames

    def keys(self) -> list[SampleKey]:
        return list(self._keys)

    def _frame(self, ev: EventFiles, d: date):
        """Processed strict channels for one input day, plus its grid (cached)."""
        k = (ev.event_id, d)
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        p = ev.path(d)
        if not p.exists():
            raise WSTSFileError(f"{ev.event_id}: missing input file {p.name}")
        arr, transform, crs = _read(p, [b + 1 for b in self._bands])
        full = np.full((len(RAW_BANDS), *arr.shape[1:]), np.nan, np.float32)
        full[self._bands] = arr
        proc = process_raw(full)[self._proc_idx]
        item = (proc, transform, crs)
        self._cache[k] = item
        if len(self._cache) > self._cache_frames:
            self._cache.popitem(last=False)
        return item

    def load(self, key: SampleKey) -> TemporalSample:
        ev_id, target = key
        ev = self._index[ev_id]
        T = self.sequence_length
        input_dates = tuple(target - timedelta(days=T - i) for i in range(T))
        frames = [self._frame(ev, d) for d in input_dates]
        tp = ev.path(target)
        if not tp.exists():
            raise WSTSFileError(f"{ev_id}: missing target file {tp.name}")
        y_arr, t_transform, t_crs = _read(tp, [TARGET_BAND_1BASED])  # band 23 only
        grids = {(f[0].shape[-2:], f[1], f[2]) for f in frames} | {
            (y_arr.shape[-2:], t_transform, t_crs)
        }
        if len(grids) != 1:
            raise WSTSAlignmentError(f"{ev_id} {target}: frames differ in shape/transform/CRS")
        x = np.stack([f[0] for f in frames]).astype(np.float32)
        y = target_from_band(y_arr[0])
        s = TemporalSample(
            event_id=ev_id,
            input_dates=input_dates,
            target_date=target,
            x=x,
            y=y,
            valid=np.ones(y.shape, bool),  # every native cell is scored, as in the loader
            channels=self.channels,
            transform=t_transform,
            crs=t_crs,
            is_mock=False,
        )
        validate_sample(s, T, self.channels)
        return s


# ------------------------------------------------------------------------------------- padding


def padding_metadata(hw: tuple[int, int], multiple: int = 32) -> dict:
    """Bottom/right padding for a native ``(H, W)`` frame.

    Padding only adds rows below and columns to the right, so the affine transform (whose origin
    is the top-left corner) is unchanged and every native cell keeps its coordinates.
    """
    h, w = hw
    ph, pw = -(-h // multiple) * multiple, -(-w // multiple) * multiple
    return {"orig_h": h, "orig_w": w, "padded_h": ph, "padded_w": pw,
            "pad_bottom": ph - h, "pad_right": pw - w, "pad_top": 0, "pad_left": 0,
            "transform_changed": False}  # fmt: skip


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def contract_dict(channels: Sequence[str], T: int) -> dict:
    """The data-contract facts every experiment records (P3-R1-06)."""
    return {
        "data_contract_version": DATA_CONTRACT_VERSION,
        "pinned_loader": {"repo": PINNED_LOADER_REPO, "commit": PINNED_LOADER_COMMIT},
        "channels": list(channels),
        "processed_indices": channel_indices(channels),
        "raw_bands_0based": raw_bands_needed(channels),
        "T": T,
        "target": "raw band 23 (active fire) of day D+1: NaN->0, >0",
        "valid_mask": "all native cells; padding invalid",
        "padding": "bottom/right to a multiple of 32; transform unchanged",
        "standardisation": "fold-specific, train groups only; NaN -> standardised mean",
    }


def contract_hash(channels: Sequence[str], T: int) -> str:
    blob = json.dumps(contract_dict(channels, T), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
