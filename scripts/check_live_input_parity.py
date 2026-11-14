"""Layer-3 check: can the live recipe rebuild real WSTS input frames?

    python scripts/check_live_input_parity.py --data-root $WSTS [--events-per-year 10] \
        [--fit-v2-mapping]

For a sample of WSTS events (2018-2021), every scored input day is rebuilt from the live sources
on the event's own grid and compared with the WSTS GeoTIFF:

* active fire: FIRMS ``VIIRS_SNPP_SP`` (archive) rasterised with every candidate rule / day
  offset; cell F1 of "detected" against WSTS band 22. The best setting is reported; copy it to
  ``configs/live_forecast.yaml`` (``active_fire.rasterize_rule`` / ``day_utc_offset_hours``).
* terrain: Earth Engine SRTM recipe vs bands 12-14 (MAE elevation, slope, sin(aspect)).
* reflectance: Earth Engine VNP09GA Collection 1 (what WSTS used) and Collection 2 x 10,000
  vs bands 0-2 (Pearson r and relative bias per band). ``--fit-v2-mapping`` fits the per-band
  linear map Collection 2 -> Collection 1 from these input pixels (no label is used) and writes
  ``configs/live/reflectance_v2_to_v1.json``; Collection 2 is then scored after the mapping.

Reads the WSTS archive; calls FIRMS (``FIRMS_MAP_KEY``) and Earth Engine. Writes only
``docs/evidence/live/live_input_parity.json`` / ``live_input_parity_per_day.csv`` and, with
``--fit-v2-mapping``, the mapping file. Uses no GPU and no training output, so it can run while
training continues. Exit code 1 if any threshold fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

from fireguard_ml.data.event_splits import read_manifest
from fireguard_ml.data.wsts_dataset import ACTIVE_FIRE_BAND, RAW_BANDS, _read, scan_events
from fireguard_ml.live.active_fire import FirmsAreaProvider, FirmsError, rasterize_active_fire
from fireguard_ml.live.earth_engine import EarthEngineTerrain, EarthEngineVNP09GA
from fireguard_ml.live.grid import region_from_grid
from fireguard_ml.live.reflectance import REFLECTANCE_CHANNELS, fit_linear_mapping
from fireguard_ml.live.terrain import TerrainUnavailable

ROOT = Path(__file__).resolve().parents[1]
RULES = ("intersects", "containing")
OFFSETS = (0.0, -7.0, -8.0)  # UTC, Pacific daylight, Pacific standard
THRESHOLDS = {
    "fire_f1_min": 0.95,
    "elevation_mae_m_max": 5.0,
    "slope_mae_deg_max": 0.5,
    "sin_aspect_mae_max": 0.05,
    "v1_pearson_min": 0.98,
    "v1_rel_bias_max": 0.02,
    "v2_pearson_min": 0.95,
    "v2_rel_bias_max": 0.05,
}


def f1(pred: np.ndarray, ref: np.ndarray) -> tuple[int, int, int]:
    return (int((pred & ref).sum()), int((pred & ~ref).sum()), int((~pred & ref).sum()))


def corr_bias(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100:
        return float("nan"), float("nan"), int(ok.sum())
    xv, yv = x[ok].astype(np.float64), y[ok].astype(np.float64)
    r = float(np.corrcoef(xv, yv)[0, 1])
    bias = float((xv.mean() - yv.mean()) / max(abs(yv.mean()), 1e-9))
    return r, bias, int(ok.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--events-per-year", type=int, default=10)
    ap.add_argument("--max-days-per-event", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--firms-source", default="VIIRS_SNPP_SP")
    ap.add_argument("--fit-v2-mapping", action="store_true")
    ap.add_argument("--skip-reflectance", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "evidence" / "live")
    a = ap.parse_args()

    index = scan_events(a.data_root)
    rows = read_manifest(ROOT / "data" / "manifests" / "p1_loyo_fold_manifest.csv")
    by_year = defaultdict(set)
    for r in rows:
        if r.split == "test" and r.event_id in index:
            by_year[r.fold_test_year].add(r.event_id)
    rng = random.Random(a.seed)
    events = [ev for y in sorted(by_year)
              for ev in rng.sample(sorted(by_year[y]), min(a.events_per_year, len(by_year[y])))]
    firms = FirmsAreaProvider(os.environ.get("FIRMS_MAP_KEY", ""), a.firms_source)
    terrain = EarthEngineTerrain()
    v1, v2 = EarthEngineVNP09GA("v1"), EarthEngineVNP09GA("v2")

    fire_conf = defaultdict(lambda: [0, 0, 0])
    terr_err = defaultdict(list)
    refl_pairs = {"v1": ([], []), "v2": ([], [])}
    per_day = []
    for ev in events:
        files = index[ev]
        days = sorted(files.dates)[: a.max_days_per_event]
        raw0, transform, crs = _read(files.path(days[0]), range(1, len(RAW_BANDS) + 1))
        region = region_from_grid(ev, transform, raw0.shape[1], raw0.shape[2], crs or "EPSG:32610")
        try:
            t = terrain.fetch(region)
            for name, i, live in (("elevation", 14, t[2]), ("slope", 12, t[0])):
                d = np.abs(live - raw0[i])
                terr_err[name].append(float(np.nanmean(d)))
            terr_err["sin_aspect"].append(float(np.nanmean(np.abs(
                np.sin(np.deg2rad(t[1])) - np.sin(np.deg2rad(raw0[13]))))))
        except TerrainUnavailable as exc:
            print(f"{ev}: terrain unavailable ({exc})")
        try:
            dets = firms.detections(region.bbox_wgs84, days[0] - timedelta(days=1),
                                    days[-1] + timedelta(days=1))
        except FirmsError as exc:
            raise SystemExit(f"FIRMS: {exc}") from None
        for d in days:
            raw, _, _ = _read(files.path(d), range(1, len(RAW_BANDS) + 1))
            ref = np.isfinite(raw[ACTIVE_FIRE_BAND]) & (raw[ACTIVE_FIRE_BAND] > 0)
            row = {"event_id": ev, "date": d.isoformat(), "wsts_fire_cells": int(ref.sum())}
            for rule in RULES:
                for off in OFFSETS:
                    pred = np.isfinite(rasterize_active_fire(dets, region, d, rule=rule,
                                                             utc_offset_hours=off))
                    c = f1(pred, ref)
                    k = f"{rule}@{off:+.0f}h"
                    fire_conf[k] = [x + y for x, y in zip(fire_conf[k], c, strict=True)]
                    row[f"fire_tp_{k}"], row[f"fire_fp_{k}"], row[f"fire_fn_{k}"] = c
            if not a.skip_reflectance:
                for name, prov in (("v1", v1), ("v2", v2)):
                    try:
                        live = prov.fetch(region, d)
                    except Exception as exc:  # unavailable day or collection
                        row[f"{name}_status"] = f"unavailable: {type(exc).__name__}"
                        continue
                    refl_pairs[name][0].append(live)
                    refl_pairs[name][1].append(raw[0:3])
                    for b, ch in enumerate(REFLECTANCE_CHANNELS):
                        r, bias, n = corr_bias(live[b], raw[b])
                        row[f"{name}_{ch}_r"], row[f"{name}_{ch}_bias"] = r, bias
            per_day.append(row)
        print(f"{ev}: {len(days)} days checked", flush=True)

    def score(k):
        tp, fp, fn = fire_conf[k]
        return 2 * tp / max(2 * tp + fp + fn, 1)

    fire = {k: {"f1": score(k), "tp": v[0], "fp": v[1], "fn": v[2]} for k, v in fire_conf.items()}
    best = max(fire, key=lambda k: fire[k]["f1"]) if fire else None
    result = {"events": events, "n_days": len(per_day), "thresholds": THRESHOLDS,
              "firms_source": a.firms_source, "active_fire": fire, "active_fire_best": best,
              "terrain": {k: float(np.mean(v)) for k, v in terr_err.items()}, "reflectance": {}}
    checks = {"fire_f1": best is not None and fire[best]["f1"] >= THRESHOLDS["fire_f1_min"],
              "elevation": result["terrain"].get("elevation", 1e9)
              <= THRESHOLDS["elevation_mae_m_max"],
              "slope": result["terrain"].get("slope", 1e9) <= THRESHOLDS["slope_mae_deg_max"],
              "sin_aspect": result["terrain"].get("sin_aspect", 1e9)
              <= THRESHOLDS["sin_aspect_mae_max"]}

    if not a.skip_reflectance:
        for name in ("v1", "v2"):
            xs, ys = refl_pairs[name]
            if not xs:
                result["reflectance"][name] = "unavailable"
                checks[f"reflectance_{name}"] = False
                continue
            x, y = np.stack(xs), np.stack(ys)
            if name == "v2" and a.fit_v2_mapping:
                mapping = fit_linear_mapping(x, y, f"wsts {len(xs)} frames, seed {a.seed}")
                mapping.to_json(ROOT / "configs" / "live" / "reflectance_v2_to_v1.json")
                x = np.stack([mapping.apply(f) for f in xs])
                result["reflectance"]["v2_mapping"] = {"scale": mapping.scale,
                                                       "offset": mapping.offset}
            stats = {}
            for b, ch in enumerate(REFLECTANCE_CHANNELS):
                r, bias, n = corr_bias(x[:, b], y[:, b])
                stats[ch] = {"pearson_r": r, "rel_bias": bias, "n_cells": n}
            result["reflectance"][name] = stats
            lim_r, lim_b = THRESHOLDS[f"{name}_pearson_min"], THRESHOLDS[f"{name}_rel_bias_max"]
            checks[f"reflectance_{name}"] = all(
                s["pearson_r"] >= lim_r and abs(s["rel_bias"]) <= lim_b for s in stats.values())
    result["checks"] = checks
    result["status"] = "PASS" if all(checks.values()) else "FAIL"

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "live_input_parity.json").write_text(json.dumps(result, indent=2) + "\n",
                                                  encoding="utf-8")
    cols = sorted({k for r in per_day for k in r}, key=lambda k: (k not in ("event_id", "date"), k))
    with (a.out / "live_input_parity_per_day.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(per_day)
    print(json.dumps({"status": result["status"], "checks": checks, "active_fire_best": best},
                     indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
