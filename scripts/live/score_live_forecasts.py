"""Score one published live forecast against the next day's observed VIIRS fire.

    python scripts/live/score_live_forecasts.py --artifact-root artifacts \
        --live-run-id live_2026-11-20_m1_t1_v1

Run it once the target day (origin + 1) is a completed UTC day. The observed target is the
target day's FIRMS detections rasterised with the same recipe as the inputs, then the loader's
``target_from_band``. Reference: dilated persistence of the origin-day fire (F1 / IoU).

Appends to ``<artifact-root>/live/live_scores.csv`` (rows of the same run are replaced, so the
script is safe to re-run). It writes nothing else and reads no training output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio

from fireguard_ml.data.wsts_dataset import ACTIVE_FIRE_BAND
from fireguard_ml.live.active_fire import (
    CsvDetectionProvider,
    FirmsAreaProvider,
    FirmsError,
    rasterize_active_fire,
)
from fireguard_ml.live.grid import region_from_grid
from fireguard_ml.live.scoring import SCORE_FIELDS, score_forecast
from fireguard_ml.live.settings import load_live_settings

COLUMNS = ("live_run_id", "region_id", "origin_date", "target_date", *SCORE_FIELDS,
           "threshold", "firms_source", "is_mock", "gate_status", "scored_at")


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    load_live_settings(ROOT)  # FIRMS_MAP_KEY from backend/.env
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--live-run-id", default=None,
                    help="default: every published run whose target day is over and that has no "
                         "score yet")
    ap.add_argument("--firms-source", default=None, help="default: the source the forecast used")
    ap.add_argument("--firms-csv", type=Path, default=None,
                    help="saved FIRMS CSV instead of the API")
    a = ap.parse_args()
    if a.live_run_id:
        score_run(a, a.live_run_id)
        return
    out = a.artifact_root / "live" / "live_scores.csv"
    done = set()
    if out.is_file():
        with out.open(newline="", encoding="utf-8") as f:
            done = {r["live_run_id"] for r in csv.DictReader(f)}
    today = datetime.now(UTC).date()
    for info_path in sorted((a.artifact_root / "live").glob("live_*/live_run.json")):
        rid = info_path.parent.name
        target = date.fromisoformat(json.loads(info_path.read_text())["index"]["target_date"])
        if rid not in done and target < today:
            score_run(a, rid)


def score_run(a, live_run_id: str) -> None:
    a.live_run_id = live_run_id
    run_dir = a.artifact_root / "live" / a.live_run_id
    info = json.loads((run_dir / "live_run.json").read_text(encoding="utf-8"))
    af = info["active_fire_config"]
    offset = float(af["day_utc_offset_hours"])
    threshold = float(info["threshold_fold_mean"])
    with (run_dir / "live_forecasts.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    target = date.fromisoformat(info["index"]["target_date"])
    if target >= datetime.now(UTC).date() and a.firms_csv is None:
        raise SystemExit(f"target day {target} is not a completed UTC day yet")

    try:
        if a.firms_csv is not None:
            provider = CsvDetectionProvider(a.firms_csv)
        else:
            source = a.firms_source or af["firms_source"]
            provider = FirmsAreaProvider(os.environ.get("FIRMS_MAP_KEY", ""), source)
    except FirmsError as exc:
        raise SystemExit(f"FIRMS: {exc}") from None

    rows = []
    for rec in records:
        region = region_from_grid(rec["region_id"], rec["transform"], rec["height"], rec["width"],
                                  rec["crs"])
        origin = date.fromisoformat(rec["origin_date"])
        try:
            dets = provider.detections(region.bbox_wgs84, origin - timedelta(days=1),
                                       target + timedelta(days=1))
        except FirmsError as exc:
            raise SystemExit(f"FIRMS: {exc}") from None
        kw = dict(exclude_confidence=tuple(af["exclude_confidence"]),
                  buffer_m=float(af["buffer_m"]), rule=af["rasterize_rule"],
                  utc_offset_hours=offset)
        target_band = rasterize_active_fire(dets, region, target, **kw)
        raw_inputs = a.artifact_root / Path(rec["storage_uri"]).with_name("inputs_raw.tif")
        if raw_inputs.is_file():  # the exact origin-day fire the model saw
            with rasterio.open(raw_inputs) as ds:
                origin_band = ds.read(ACTIVE_FIRE_BAND + 1)
        else:
            origin_band = rasterize_active_fire(dets, region, origin, **kw)
        with rasterio.open(a.artifact_root / rec["storage_uri"]) as ds:
            prob = ds.read(1)
        m = score_forecast(prob, target_band.astype(np.float32), origin_band, threshold)
        rows.append({"live_run_id": a.live_run_id, "region_id": rec["region_id"],
                     "origin_date": rec["origin_date"], "target_date": rec["target_date"], **m,
                     "threshold": threshold, "firms_source": provider.source,
                     "is_mock": rec["is_mock"], "gate_status": rec["gate_status"],
                     "scored_at": datetime.now(UTC).isoformat(timespec="seconds")})

    out = a.artifact_root / "live" / "live_scores.csv"
    kept = []
    if out.is_file():
        with out.open(newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if r["live_run_id"] != a.live_run_id]
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(kept)
        w.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)
    os.replace(tmp, out)
    print(json.dumps({"live_run_id": a.live_run_id, "regions_scored": len(rows),
                      "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
