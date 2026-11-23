"""Fast on-disk cache of the strict WSTS channels, and a sample source that reads from it.

Why it exists
    Decoding GeoTIFFs dominated training time (each epoch re-read every frame), which left the
    GPU mostly idle. The cache is built once from the GeoTIFFs with exactly the same processing
    as ``WSTSTemporalSource`` (Role 1's reader), stored as one memory-mapped array per event, and
    then every epoch reads plain arrays.

Layout (one folder per event, ``<cache_dir>/<event_id>/``)
    ``x.npy``      float16 ``[D, C, H, W]``   processed strict channels for each file date
    ``y.npy``      uint8   ``[D, H, W]``      next-day target rule applied to that date's band 23
    ``meta.json``  dates, channels, grid (transform, CRS), data-contract version
    ``<cache_dir>/cache_info.json``  channels and contract version shared by all events

float16 keeps 3 significant digits (e.g. reflectance 6694 is stored as 6692); NaN (missing
measurement) is preserved and imputed later by the fold normaliser, exactly as before.

``CachedTemporalSource`` has the same interface and returns the same samples as
``WSTSTemporalSource`` (checked by ``tests/test_frame_cache.py``), so every other module is
unchanged. It is picklable, so it works with multi-process data loading on Windows.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from fireguard_ml.data.wsts_dataset import (
    DATA_CONTRACT_VERSION,
    RAW_BANDS,
    TARGET_BAND_1BASED,
    WSTSAlignmentError,
    WSTSFileError,
    _read,
    channel_indices,
    process_raw,
    raw_bands_needed,
    scan_events,
    target_from_band,
)
from fireguard_ml.training.temporal_data import SampleKey, TemporalSample, validate_sample

CACHE_FORMAT = 1
SYNTHETIC_MARKER = "SYNTHETIC_NOT_WSTS.txt"


def _write_npy_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def cache_event(ev, out_dir: Path, channels: Sequence[str]) -> dict:
    """Process every file date of one event into ``out_dir`` (x.npy, y.npy, meta.json)."""
    bands = raw_bands_needed(channels)
    proc_idx = channel_indices(channels)
    xs, ys, grids = [], [], []
    for d in ev.dates:
        p = ev.path(d)
        arr, transform, crs = _read(p, [b + 1 for b in bands])
        full = np.full((len(RAW_BANDS), *arr.shape[1:]), np.nan, np.float32)
        full[bands] = arr
        xs.append(process_raw(full)[proc_idx].astype(np.float16))
        y_arr, _, _ = _read(p, [TARGET_BAND_1BASED])
        ys.append(target_from_band(y_arr[0]))
        grids.append((tuple(int(v) for v in arr.shape[1:]), [float(v) for v in transform],
                      str(crs) if crs is not None else None))
    shapes = {g[0] for g in grids}
    if len(shapes) != 1:
        raise WSTSAlignmentError(f"{ev.event_id}: frames differ in shape {sorted(shapes)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_npy_atomic(out_dir / "x.npy", np.stack(xs))
    _write_npy_atomic(out_dir / "y.npy", np.stack(ys).astype(np.uint8))
    meta = {
        "cache_format": CACHE_FORMAT,
        "event_id": ev.event_id,
        "year": ev.year,
        "dates": [d.isoformat() for d in ev.dates],
        "channels": list(channels),
        "data_contract_version": DATA_CONTRACT_VERSION,
        "transforms": [g[1] for g in grids],
        "crs": [g[2] for g in grids],
        "height": grids[0][0][0],
        "width": grids[0][0][1],
    }
    tmp = out_dir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    os.replace(tmp, out_dir / "meta.json")  # written last: its presence marks the event as done
    return meta


def build_frame_cache(data_root: Path, cache_dir: Path, channels: Sequence[str],
                      events: Iterable[str] | None = None, log=print) -> dict:
    """Build (or finish building) the cache. Resumable: finished events are skipped."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    info_path = cache_dir / "cache_info.json"
    info = {"cache_format": CACHE_FORMAT, "channels": list(channels),
            "data_contract_version": DATA_CONTRACT_VERSION, "data_root": str(data_root),
            # synthetic demo data (scripts/data/make_synthetic_wsts.py) stays labelled as mock
            "synthetic": (Path(data_root) / SYNTHETIC_MARKER).exists()}
    if info_path.exists():
        old = json.loads(info_path.read_text(encoding="utf-8"))
        keys = ("cache_format", "channels", "data_contract_version", "synthetic")
        if [old.get(k) for k in keys] != [info[k] for k in keys]:
            raise SystemExit(f"{cache_dir} holds a cache for other channels or another contract "
                             "version; delete the folder or choose another --cache-dir")
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    index = scan_events(Path(data_root))
    if not index:
        raise SystemExit(f"no WSTS events found under {data_root} (expected <root>/2018/fire_*/)")
    wanted = sorted(set(events) & set(index)) if events is not None else sorted(index)
    todo = [e for e in wanted if not (cache_dir / e / "meta.json").exists()]
    log(f"[cache] {len(wanted) - len(todo)}/{len(wanted)} events already cached in {cache_dir}")
    t0 = time.time()
    for i, e in enumerate(todo, 1):
        cache_event(index[e], cache_dir / e, channels)
        if i % 25 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            log(f"[cache] {i}/{len(todo)} events, about {rate * (len(todo) - i) / 60:.0f} min left")
    return {"events": len(wanted), "built": len(todo), "cache_dir": str(cache_dir)}


class CachedTemporalSource:
    """``TemporalSampleSource`` over the frame cache; same samples as ``WSTSTemporalSource``."""

    def __init__(self, cache_dir: Path, channels: Sequence[str], T: int, samples_csv: Path,
                 events: Iterable[str] | None = None, max_open_events: int = 64) -> None:
        self.cache_dir = Path(cache_dir)
        info = json.loads((self.cache_dir / "cache_info.json").read_text(encoding="utf-8"))
        self.is_mock = bool(info.get("synthetic", False))
        self.channels = tuple(channels)
        self.sequence_length = int(T)
        missing = [c for c in self.channels if c not in info["channels"]]
        if missing:
            raise ValueError(f"channels {missing} are not in the cache ({info['channels']})")
        self._sel = [info["channels"].index(c) for c in self.channels]
        self._meta: dict[str, dict] = {}
        for d in sorted(p for p in self.cache_dir.iterdir() if (p / "meta.json").exists()):
            self._meta[d.name] = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self._pos = {ev: {d: i for i, d in enumerate(m["dates"])} for ev, m in self._meta.items()}
        wanted = set(events) if events is not None else None
        keys = []
        with Path(samples_csv).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ev = r["event_id"]
                if ev in self._meta and (wanted is None or ev in wanted):
                    keys.append((ev, date.fromisoformat(r["target_date"])))
        self._keys = sorted(keys)
        self._max_open = max_open_events
        self._open: OrderedDict = OrderedDict()

    def __getstate__(self):  # memory maps are reopened lazily in each worker process
        state = dict(self.__dict__)
        state["_open"] = OrderedDict()
        return state

    def keys(self) -> list[SampleKey]:
        return list(self._keys)

    def event_ids(self) -> list[str]:
        return sorted(self._meta)

    def _arrays(self, ev: str):
        if ev in self._open:
            self._open.move_to_end(ev)
            return self._open[ev]
        d = self.cache_dir / ev
        arrs = (np.load(d / "x.npy", mmap_mode="r"), np.load(d / "y.npy", mmap_mode="r"))
        self._open[ev] = arrs
        if len(self._open) > self._max_open:
            self._open.popitem(last=False)
        return arrs

    def load(self, key: SampleKey) -> TemporalSample:
        ev, target = key
        meta = self._meta[ev]
        pos = self._pos[ev]
        T = self.sequence_length
        input_dates = tuple(target - timedelta(days=T - i) for i in range(T))
        try:
            idx = [pos[d.isoformat()] for d in input_dates]
            t_idx = pos[target.isoformat()]
        except KeyError as exc:
            raise WSTSFileError(f"{ev}: file for {exc.args[0]} is missing") from None
        grids = {(tuple(meta["transforms"][i]), meta["crs"][i]) for i in [*idx, t_idx]}
        if len(grids) != 1:
            raise WSTSAlignmentError(f"{ev} {target}: frames differ in transform/CRS")
        xa, ya = self._arrays(ev)
        x = np.asarray(xa[idx][:, self._sel], dtype=np.float32)
        y = np.array(ya[t_idx], dtype=np.uint8)
        s = TemporalSample(event_id=ev, input_dates=input_dates, target_date=target, x=x, y=y,
                           valid=np.ones(y.shape, bool), channels=self.channels,
                           transform=tuple(meta["transforms"][t_idx]), crs=meta["crs"][t_idx],
                           is_mock=self.is_mock)
        validate_sample(s, T, self.channels)
        return s
