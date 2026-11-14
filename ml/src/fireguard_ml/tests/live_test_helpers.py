"""Shared builders for the live-forecast tests (not a test module)."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml

from fireguard_ml.data.event_splits import sha256_file
from fireguard_ml.data.wsts_dataset import DATA_CONTRACT_VERSION, RAW_BANDS, RAW_DESCRIPTIONS
from fireguard_ml.live.grid import LiveRegion, region_for_center
from fireguard_ml.models.checkpointing import save_checkpoint
from fireguard_ml.models.temporal_unet import SpatialUNet, StackedDaysUNet
from fireguard_ml.training.normalization import DEFAULT_NO_STANDARDIZE, FoldNormalizer

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "ml").is_dir() and (p / "docs").is_dir())
CHANNELS = tuple(yaml.safe_load((REPO / "configs" / "temporal_unet.yaml").read_text())
                 ["data"]["channels"])
RUN = "m1_spatial_unet_base8_T1_seed0"
RUN_ID = "m1_t1_test"
CENTER = (-120.2, 38.4)  # lon, lat inside the training domain
ORIGIN = date(2026, 11, 1)


def small_region(h: int = 40, w: int = 52, rid: str = "r_test") -> LiveRegion:
    """A small frame on the WSTS lattice (fast tests)."""
    return LiveRegion(rid, CENTER[0], CENTER[1], (375.0, 0.0, 697500.0, 0.0, -375.0, 4136625.0),
                      h, w)


def wsts_region() -> LiveRegion:
    return region_for_center(*CENTER)


class ArrayFire:
    """Raw active-fire bands per day from a dict; days not given have no detections."""

    source = "TEST_FIRE"

    def __init__(self, by_day: dict):
        self.by_day = by_day
        self.requested: list[date] = []

    def fetch(self, region, day):
        self.requested.append(day)
        return self.by_day.get(day, np.full(region.hw, np.nan, np.float32))


class ArrayReflectance:
    name = "test_reflectance"

    def __init__(self, by_day: dict | None = None, seed: int = 0):
        self.by_day = by_day or {}
        self.seed = seed
        self.requested: list[date] = []

    def fetch(self, region, day):
        self.requested.append(day)
        if day in self.by_day:
            return self.by_day[day]
        rng = np.random.default_rng(self.seed + day.toordinal())
        return rng.uniform(500, 5000, size=(3, *region.hw)).astype(np.float32)


class ArrayTerrain:
    name = "test_terrain"

    def __init__(self, arr: np.ndarray | None = None, seed: int = 1):
        self.arr = arr
        self.seed = seed

    def fetch(self, region):
        if self.arr is not None:
            return self.arr
        rng = np.random.default_rng(self.seed)
        h, w = region.hw
        return np.stack([rng.uniform(0, 30, (h, w)), rng.uniform(0, 360, (h, w)),
                         rng.uniform(100, 3000, (h, w))]).astype(np.float32)


def fire_band(hw, cells: dict) -> np.ndarray:
    """Raw active-fire band with HHMM at ``{(row, col): hhmm}`` and NaN elsewhere."""
    out = np.full(hw, np.nan, np.float32)
    for (r, c), v in cells.items():
        out[r, c] = v
    return out


def write_wsts_tif(path: Path, raw: np.ndarray, transform, crs: str = "EPSG:32610") -> None:
    import rasterio
    from rasterio.transform import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=raw.shape[1], width=raw.shape[2],
                       count=len(RAW_BANDS), dtype="float32", crs=crs,
                       transform=Affine(*transform[:6])) as dst:
        dst.write(raw.astype(np.float32))
        for i, desc in enumerate(RAW_DESCRIPTIONS, start=1):
            dst.set_band_description(i, desc)


def make_wsts_event(root: Path, event: str = "fire_99999999", first: date = date(2020, 7, 1),
                    n_days: int = 6, region: LiveRegion | None = None, seed: int = 0):
    """A small WSTS-format event with every band filled, NaN gaps in reflectance and a
    00:30 UTC detection. Returns ``(region, samples_csv, {day: raw})``."""
    region = region or small_region(rid=event)
    h, w = region.hw
    rng = np.random.default_rng(seed)
    terrain = ArrayTerrain(seed=seed + 5).fetch(region)
    frames = {}
    for i in range(n_days):
        d = first + timedelta(days=i)
        raw = rng.normal(size=(len(RAW_BANDS), h, w)).astype(np.float32)
        raw[0:3] = rng.uniform(500, 5000, size=(3, h, w))
        raw[0:3, 3:6, 4:9] = np.nan  # cloud gap
        raw[7], raw[19] = rng.uniform(0, 360, (h, w)), rng.uniform(0, 360, (h, w))
        raw[12:15] = terrain
        raw[16] = rng.integers(1, 18, (h, w))
        fire = np.full((h, w), np.nan, np.float32)
        fire[10:14, 5 + i : 9 + i] = 2030.0
        fire[20, 20 + i] = 30.0  # 00:30 UTC: a positive target, absent from the input mask
        raw[22] = fire
        frames[d] = raw
        write_wsts_tif(root / str(first.year) / event / f"{d.isoformat()}.tif", raw,
                       region.transform)
    samples = root / "samples.csv"
    with samples.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["event_id", "target_date"])
        for i in range(1, n_days):
            wr.writerow([event, (first + timedelta(days=i)).isoformat()])
    return region, samples, frames


class RawFrameFire:
    source = "WSTS_FILE"

    def __init__(self, frames):
        self.frames = frames

    def fetch(self, region, day):
        return self.frames[day][22]


class RawFrameReflectance:
    name = "wsts_file"

    def __init__(self, frames):
        self.frames = frames

    def fetch(self, region, day):
        return self.frames[day][0:3]


def split_hash() -> str:
    return sha256_file(REPO / "data" / "manifests" / "p1_loyo_fold_manifest.csv")


def make_repo_root(tmp: Path, gate2: str | None = "BEATS", split: str | None = None) -> Path:
    """A minimal fake repository root holding what the readiness gate reads."""
    root = tmp / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "data" / "manifests").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "temporal_unet.yaml", root / "configs")
    shutil.copy(REPO / "data" / "manifests" / "p1_loyo_fold_manifest.csv",
                root / "data" / "manifests")
    if gate2 is not None:
        (root / "reports" / "phase3").mkdir(parents=True)
        (root / "reports" / "phase3" / "gate_decisions.json").write_text(json.dumps(
            {"split_manifest_sha256": split or split_hash(), "gate2": {"decision": gate2}}))
    return root


def make_trained_run(tmp: Path, folds=(2020, 2021), *, T: int = 1, is_mock: bool = False,
                     strict: bool = True, channels=CHANNELS, run_id: str = RUN_ID,
                     registry_mode: str | None = "real", export: bool = True,
                     preprocessing_version: str = DATA_CONTRACT_VERSION,
                     config_hashes: dict | None = None):
    """Tiny fold checkpoints + sidecars + export record, named exactly as the trainer names them.

    Returns ``(run_dir, run, artifact_root)``.
    """
    run = RUN if T == 1 else f"m3_stacked_unet_base8_T{T}_seed0"
    run_dir, artifact_root = tmp / "outputs" / "m1", tmp / "artifacts"
    versions = {}
    for fold in folds:
        torch.manual_seed(fold)
        model = (SpatialUNet(len(channels), 1, base=8) if T == 1
                 else StackedDaysUNet(len(channels), T, base=8))
        cfg_hash = (config_hashes or {}).get(fold, "c" * 64)
        meta = {"channels": list(channels), "T": T, "preprocessing_version": preprocessing_version,
                "config_hash": cfg_hash, "fold": fold, "split_manifest_hash": split_hash(),
                "protocol": "P1_LOYO", "seed": 0, "is_mock": is_mock}
        fdir = run_dir / f"fold_{fold}"
        ckpt = fdir / f"{run}.pt"
        save_checkpoint(model, ckpt, meta, strict=strict)
        std_on = tuple(c not in DEFAULT_NO_STANDARDIZE for c in channels)
        rng = np.random.default_rng(fold)
        FoldNormalizer(
            fold, tuple(channels),
            tuple(float(rng.uniform(100, 2000)) if on else 0.0 for on in std_on),
            tuple(float(rng.uniform(50, 800)) if on else 1.0 for on in std_on),
            std_on, 1000, split_hash(),
        ).to_json(fdir / f"{run}.normalizer.json")
        selection = {"protocol": "P1", "fold": fold, "fitted_on": "validation",
                     "threshold": {"protocol": "P1", "fold": fold, "threshold": 0.4,
                                   "validation_f1": 0.5},
                     "calibrator": {"protocol": "P1", "fold": fold, "a": 0.8, "b": -1.0}}
        (fdir / f"{run}.selection.json").write_text(json.dumps(selection))
        (fdir / f"{run}.summary.json").write_text(json.dumps(
            {"run": run, "fold": fold, "is_mock": is_mock, "config": {"T": T}}))
        versions[str(fold)] = {"checkpoint": ckpt.name, "selection": selection, "meta": meta,
                               "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}
    if export:
        out = artifact_root / "runs" / run_id
        out.mkdir(parents=True)
        (out / "model_version.json").write_text(json.dumps(
            {"name": run_id, "version": run_id, "config_hash": "c" * 64, "folds": versions,
             "is_mock": is_mock}))
    if registry_mode is not None:
        reg = artifact_root / "registry" / run_id
        reg.mkdir(parents=True)
        (reg / "registry_bundle.json").write_text(json.dumps({"mode": registry_mode}))
    return run_dir, run, artifact_root
