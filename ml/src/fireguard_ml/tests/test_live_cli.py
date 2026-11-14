"""The live CLI wires readiness, FIRMS input, regions, inference and publication (layer 1)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta

import numpy as np
import pytest
import yaml

pytest.importorskip("rasterio")

from live_test_helpers import (  # noqa: E402
    CENTER,
    ORIGIN,
    REPO,
    RUN,
    RUN_ID,
    make_trained_run,
    wsts_region,
)

from fireguard_ml.live.grid import LiveRegion  # noqa: E402
from fireguard_ml.live.terrain import write_terrain_tif  # noqa: E402

SCRIPT = REPO / "scripts" / "run_live_forecast.py"
HEADER = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,"
          "confidence,version,bright_ti5,frp,daynight")


def _fixture(tmp_path):
    rng = np.random.default_rng(0)
    lines = [HEADER]
    for dy, dx in rng.uniform(-0.02, 0.02, size=(25, 2)):
        lines.append(f"{CENTER[1] + dy:.5f},{CENTER[0] + dx:.5f},330,0.4,0.4,"
                     f"{ORIGIN.isoformat()},2030,N,VIIRS,n,2.0NRT,290,5,D")
    lines.append(f"{CENTER[1]:.5f},{CENTER[0]:.5f},330,0.4,0.4,"
                 f"{(ORIGIN + timedelta(days=1)).isoformat()},0100,N,VIIRS,h,2.0NRT,290,5,D")
    lines.append("35.0,-80.0,330,0.4,0.4," + ORIGIN.isoformat() + ",2030,N,VIIRS,h,2,290,5,D")
    f = tmp_path / "firms.csv"
    f.write_text("\n".join(lines) + "\n")
    return f


def _config(tmp_path):
    cfg = yaml.safe_load((REPO / "configs" / "live_forecast.yaml").read_text())
    reg = wsts_region()
    left, bottom, right, top = reg.bounds
    pad = 40 * 375.0
    big = LiveRegion("mosaic", 0, 0, (375.0, 0.0, left - pad, 0.0, -375.0, top + pad),
                     reg.height + 80, reg.width + 80)
    rng = np.random.default_rng(2)
    h, w = big.hw
    write_terrain_tif(tmp_path / "terrain.tif", np.stack(
        [rng.uniform(0, 30, (h, w)), rng.uniform(0, 360, (h, w)),
         rng.uniform(100, 3000, (h, w))]).astype(np.float32), big)
    cfg["terrain"] = {"provider": "mosaic", "mosaic_path": str(tmp_path / "terrain.tif"),
                      "cache_dir": str(tmp_path / "cache")}
    cfg["reflectance"]["provider"] = "missing"
    cfg["inference"]["folds"] = [2020, 2021]
    cfg["regions"]["min_cluster_detections"] = 10
    p = tmp_path / "live.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _cli(tmp_path, *extra):
    run_dir, run, artifact_root = make_trained_run(tmp_path, (2020, 2021))
    assert run == RUN
    cmd = [sys.executable, str(SCRIPT), "--run-dir", str(run_dir), "--run", run,
           "--run-id", RUN_ID, "--artifact-root", str(artifact_root),
           "--origin-date", ORIGIN.isoformat(), "--firms-csv", str(_fixture(tmp_path)),
           "--config", str(_config(tmp_path)), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO), artifact_root


def test_cli_refuses_when_gate2_is_not_passed(tmp_path):
    gd = json.loads((REPO / "reports" / "phase3" / "gate_decisions.json").read_text())
    if gd.get("gate2", {}).get("decision") in ("BEATS", "MATCHES"):
        pytest.skip("Gate 2 has been recorded as passed in this checkout")
    r, _ = _cli(tmp_path)
    assert r.returncode != 0 and "NOT READY" in r.stderr and "Gate 2" in r.stderr


def test_cli_dry_run_writes_nothing(tmp_path):
    r, artifact_root = _cli(tmp_path, "--allow-ungated", "--dry-run")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout[r.stdout.index("{"):])
    assert out["dry_run"] is True and out["gate_status"] == "UNGATED"
    # the D+1 row and the out-of-domain row are never even requested from FIRMS
    assert out["detections_fetched"] == 25 and out["detections_after_origin_dropped"] == 0
    assert out["detections_on_origin_day"] == 25
    assert out["region_report"]["n_clusters"] == 1 and len(out["region_ids"]) == 1
    assert not (artifact_root / "live").exists()


def test_cli_full_run_publishes_one_region(tmp_path):
    backend = tmp_path / "backend_artifacts"
    r, artifact_root = _cli(tmp_path, "--allow-ungated", "--backend-artifact-root", str(backend))
    assert r.returncode == 0, r.stderr
    assert "UNGATED" in r.stdout
    index = json.loads((backend / "live" / "index.json").read_text())
    assert index["gate_status"] == "UNGATED" and index["is_mock"] is False
    (reg,) = index["regions"]
    assert reg["n_detections"] == 25 and reg["reflectance_source"] == "missing"
    assert reg["nan_fraction"]["viirs_m11"] == 1.0 and reg["nan_fraction"]["slope"] == 0.0
    assert (reg["height"], reg["width"]) == wsts_region().hw
    info = json.loads((artifact_root / "live" / index["live_run_id"] / "live_run.json").read_text())
    assert info["threshold_fold_mean"] == pytest.approx(0.4)
    assert info["summary"]["detections_fetched"] == 25


def test_cli_refuses_an_origin_that_is_not_a_completed_day(tmp_path):
    run_dir, run, artifact_root = make_trained_run(tmp_path, (2020, 2021))
    r = subprocess.run([sys.executable, str(SCRIPT), "--run-dir", str(run_dir), "--run", run,
                        "--run-id", RUN_ID, "--artifact-root", str(artifact_root),
                        "--origin-date", "2999-01-01", "--allow-ungated"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "not a completed UTC day" in r.stderr
