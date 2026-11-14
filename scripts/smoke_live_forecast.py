"""End-to-end smoke test of live forecasting on SYNTHETIC (mock) checkpoints.

    python scripts/smoke_live_forecast.py --work prototype_work/live_smoke

Proves the pieces connect; says nothing about forecast quality. Steps:

    1. train two tiny M1 folds on ``SyntheticTemporalSource`` (CPU, base width 8, 1 epoch)
    2. export them with ``export_oof`` under the run id ``m1_live_smoke_mock``
    3. write a small FIRMS CSV fixture (one in-domain fire cluster, one low-confidence detection,
       one detection dated after the origin)
    4. run ``scripts/run_live_forecast.py --allow-mock --allow-ungated`` on it
    5. check the published index, checksums and rasters
    6. if the backend package is importable, serve the result through ``/api/v1/live`` in-process
       and compare the bytes

Everything is written under ``--work`` (git-ignored ``prototype_work/`` by default). It never
touches ``outputs/``, ``artifacts/`` or ``backend/artifacts``, and it runs on CPU. Do not start it
while a real training run needs the machine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
import yaml

from fireguard_ml.data.event_splits import fold_events, read_manifest
from fireguard_ml.data.wsts_dataset import DATA_CONTRACT_VERSION
from fireguard_ml.inference.temporal_oof import export_oof
from fireguard_ml.training.temporal_data import SyntheticTemporalSource
from fireguard_ml.training.temporal_trainer import ProtocolFiles, TemporalRunConfig, train_fold

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "m1_live_smoke_mock"
ORIGIN = date(2026, 11, 1)
CENTER = (-120.2, 38.4)  # lon, lat: inside the training domain (Sierra Nevada)


def subset_source(proto: ProtocolFiles, folds, channels) -> SyntheticTemporalSource:
    """Mock source over a few events per fold: 2 train, 1 validation, 1 test."""
    rows = read_manifest(proto.manifest)
    wanted: set[str] = set()
    for fold in folds:
        for split, n in (("train", 2), ("validation", 1), ("test", 1)):
            wanted.update(fold_events(rows, fold, split)[:n])
    first, last = {}, {}
    with proto.eval_samples.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["event_id"] in wanted:
                k = int(r["target_index"])
                d0 = date.fromisoformat(r["target_date"]) - timedelta(days=k)
                first.setdefault(r["event_id"], d0)
                last[r["event_id"]] = max(last.get(r["event_id"], 0), k)
    return SyntheticTemporalSource({ev: (d0, last[ev] + 1) for ev, d0 in first.items()},
                                   channels, T=1)


def write_firms_fixture(path: Path) -> int:
    rng = np.random.default_rng(0)
    lon0, lat0 = CENTER
    rows = [(lat0 + dy, lon0 + dx, ORIGIN, 2030, "n")
            for dy, dx in rng.uniform(-0.02, 0.02, size=(30, 2))]
    rows.append((lat0 + 0.05, lon0 + 0.05, ORIGIN, 2030, "l"))  # low confidence: must be dropped
    rows.append((lat0, lon0, ORIGIN + timedelta(days=1), 2030, "h"))  # future: must be dropped
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["latitude", "longitude", "bright_ti4", "scan", "track", "acq_date", "acq_time",
                    "satellite", "instrument", "confidence", "version", "bright_ti5", "frp",
                    "daynight"])
        for lat, lon, d, t, c in rows:
            w.writerow([f"{lat:.5f}", f"{lon:.5f}", 330.0, 0.4, 0.4, d.isoformat(), f"{t:04d}",
                        "N", "VIIRS", c, "2.0NRT", 290.0, 5.0, "D"])
    return len(rows)


def check(cond: bool, what: str, failures: list[str]) -> None:
    print(("    ok   " if cond else "    FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=ROOT / "prototype_work" / "live_smoke")
    ap.add_argument("--folds", type=int, nargs="+", default=[2020, 2021])
    a = ap.parse_args()
    work = a.work.resolve()
    for forbidden in ("outputs", "artifacts", "backend"):
        if work == (ROOT / forbidden).resolve() or (ROOT / forbidden).resolve() in work.parents:
            raise SystemExit(f"--work must not be inside {forbidden}/")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    failures: list[str] = []

    data = yaml.safe_load((ROOT / "configs" / "temporal_unet.yaml").read_text())["data"]
    proto = ProtocolFiles(ROOT / data["split_manifest"], ROOT / data["split_hashes"],
                          ROOT / data["eval_samples"], ROOT / data["strict_channel_manifest"])
    channels = tuple(data["channels"])
    source = subset_source(proto, a.folds, channels)
    cfg = TemporalRunConfig(model="m1_spatial_unet", T=1, channels=channels, base=8, max_epochs=1,
                            grad_accum=1, amp=False, preprocessing_version=DATA_CONTRACT_VERSION,
                            extra={"live_smoke": True, "synthetic_data": True})
    run_dir, artifacts = work / "m1", work / "artifacts"
    backend_artifacts = work / "backend_artifacts"

    print("=== [1/6] train tiny MOCK M1 folds", flush=True)
    for fold in a.folds:
        train_fold(cfg, source, proto, fold, run_dir, "cpu")
    print("=== [2/6] export OOF (mock)", flush=True)
    folds = {f: run_dir / f"fold_{f}" / f"{cfg.run_name()}.pt" for f in a.folds}
    export_oof(run_dir, folds, source, proto, artifacts, RUN_ID, "cpu")
    print("=== [3/6] FIRMS fixture", flush=True)
    fixture = work / "firms_fixture.csv"
    write_firms_fixture(fixture)

    print("=== [4/6] run_live_forecast.py", flush=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "run_live_forecast.py"),
           "--run-dir", str(run_dir), "--run", cfg.run_name(), "--run-id", RUN_ID,
           "--artifact-root", str(artifacts), "--backend-artifact-root", str(backend_artifacts),
           "--origin-date", ORIGIN.isoformat(), "--firms-csv", str(fixture),
           "--folds", *map(str, a.folds), "--allow-ungated", "--allow-mock", "--device", "cpu"]
    subprocess.run(cmd, check=True, cwd=ROOT)

    print("=== [5/6] check the published files", flush=True)
    index = json.loads((backend_artifacts / "live" / "index.json").read_text(encoding="utf-8"))
    check(index["live_run_id"] == f"live_{ORIGIN.isoformat()}_{RUN_ID}", "live run id", failures)
    check(index["is_mock"] is True, "index is marked mock", failures)
    check(index["gate_status"] == "UNGATED", "index is marked UNGATED", failures)
    check(index["target_date"] == (ORIGIN + timedelta(days=1)).isoformat(), "target = D+1",
          failures)
    check(len(index["regions"]) == 1, "exactly one region (one cluster)", failures)
    for reg in index["regions"]:
        check(reg["n_detections"] == 30, "30 detections (low-confidence and future dropped)",
              failures)
        check(all(reg["nan_fraction"][c] == 1.0 for c in ("viirs_m11", "viirs_i2", "viirs_i1")),
              "missing reflectance is reported", failures)
        for name, art in reg["artifacts"].items():
            p = backend_artifacts / art["storage_uri"]
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            check(sha == art["checksum_sha256"], f"{name}: checksum matches", failures)
            with rasterio.open(p) as ds:
                arr = ds.read(1)
                check((ds.height, ds.width) == (reg["height"], reg["width"])
                      and ds.crs.to_string() == reg["crs"], f"{name}: grid matches", failures)
            check(bool(np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1),
                  f"{name}: finite values in [0, 1]", failures)

    print("=== [6/6] serve through the backend route (in-process)", flush=True)
    try:
        os.environ["FIREGUARD_ARTIFACT_ROOT"] = str(backend_artifacts)
        os.environ.setdefault("DATABASE_URL", f"sqlite:///{(work / 'smoke.db').as_posix()}")
        from fastapi.testclient import TestClient
        from fireguard.main import app
    except ImportError as exc:
        print(f"    skipped: backend package not importable here ({exc.name})")
    else:
        client = TestClient(app)
        r = client.get("/api/v1/live/forecasts")
        check(r.status_code == 200, "GET /api/v1/live/forecasts is 200", failures)
        if r.status_code == 200:
            body = r.json()
            check(body["live_run_id"] == index["live_run_id"], "API serves the published run",
                  failures)
            reg = body["regions"][0]
            tif = client.get(reg["artifacts"]["probability"]["url"])
            uri = index["regions"][0]["artifacts"]["probability"]["storage_uri"]
            local = backend_artifacts / uri
            check(tif.status_code == 200 and tif.content == local.read_bytes(),
                  "API returns the exact probability GeoTIFF", failures)

    print(json.dumps({"status": "FAILED" if failures else "PASSED", "failures": failures,
                      "work": str(work)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
