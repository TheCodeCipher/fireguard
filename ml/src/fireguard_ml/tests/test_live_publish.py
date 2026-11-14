"""Publication: time safety, no target-derived content, atomic index, safe URIs (layer 1)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

import numpy as np
import pytest

pytest.importorskip("rasterio")

import rasterio  # noqa: E402
from live_test_helpers import (  # noqa: E402
    CHANNELS,
    ORIGIN,
    RUN_ID,
    ArrayFire,
    ArrayReflectance,
    ArrayTerrain,
    fire_band,
    make_repo_root,
    make_trained_run,
    small_region,
)

from fireguard_ml.evaluation.reveal import find_target_leaks  # noqa: E402
from fireguard_ml.inference.temporal_oof import FORBIDDEN_PRE_REVEAL  # noqa: E402
from fireguard_ml.live import forecast as live_forecast  # noqa: E402
from fireguard_ml.live.forecast import (  # noqa: E402
    LIVE_RUN_ID,
    live_run_id_for,
    run_live_forecast,
    write_json_atomic,
)
from fireguard_ml.live.frames import FutureDataError, LiveTemporalSource  # noqa: E402
from fireguard_ml.live.readiness import check_live_readiness  # noqa: E402

NOW = datetime(2026, 11, 2, 6, 0, tzinfo=UTC)


def _run(tmp_path, regions=None, publish=True, T=1, **kw):
    run_dir, run, artifact_root = make_trained_run(tmp_path, (2020, 2021), T=T)
    report, bundles = check_live_readiness(run_dir=run_dir, run=run, folds=(2020, 2021),
                                           run_id=RUN_ID, artifact_root=artifact_root,
                                           repo_root=make_repo_root(tmp_path))
    regions = regions or [small_region(rid="r_a"), small_region(rid="r_b")]
    fire = ArrayFire({ORIGIN: fire_band((40, 52), {(5, 5): 2030, (6, 6): 1400})})
    src = LiveTemporalSource(regions, ORIGIN, CHANNELS, T, fire, ArrayReflectance(),
                             ArrayTerrain())
    backend = tmp_path / "backend_artifacts" if publish else None
    out = run_live_forecast(source=src, bundles=bundles, report=report,
                            artifact_root=artifact_root, backend_artifact_root=backend,
                            n_detections={"r_a": 12}, firms_source="TEST", now=NOW, **kw)
    return out, artifact_root, backend, src, fire


def test_inputs_never_come_from_after_the_origin(tmp_path):
    for T in (1, 4):
        out, _, _, src, fire = _run(tmp_path / f"T{T}", T=T)
        assert fire.requested and max(fire.requested) == ORIGIN
        assert min(fire.requested) == ORIGIN - timedelta(days=T - 1)
        with pytest.raises(FutureDataError):
            src.raw_frame("r_a", ORIGIN + timedelta(days=1))


def test_records_and_index_hold_nothing_target_derived(tmp_path):
    out, artifact_root, backend, _, _ = _run(tmp_path)
    run_dir = artifact_root / "live" / out["live_run_id"]
    records = [json.loads(x) for x in (run_dir / "live_forecasts.jsonl").read_text().splitlines()]
    index = json.loads((backend / "live" / "index.json").read_text())
    assert len(records) == 2
    assert find_target_leaks(records) == [] and find_target_leaks(index) == []
    for rec in records:
        assert not [k for k in rec if any(m in k.lower() for m in FORBIDDEN_PRE_REVEAL)]
        assert rec["protocol"] == "LIVE" and rec["is_oof"] is False and rec["is_mock"] is False
        assert rec["origin_date"] == ORIGIN.isoformat()
        assert rec["target_date"] == (ORIGIN + timedelta(days=1)).isoformat()
        assert rec["channels"] == list(CHANNELS) and rec["gate_status"] == "GATED"
        assert rec["probability_kind"] == "model_sigmoid_uncalibrated_fold_mean"
    # the y placeholder is never written: only the three expected rasters exist per region
    tifs = sorted(p.name for p in run_dir.rglob("*.tif"))
    assert tifs == sorted(["probability.tif", "fold_spread.tif", "inputs_raw.tif"] * 2)


def test_index_is_complete_checksummed_and_uris_are_safe(tmp_path):
    out, artifact_root, backend, _, _ = _run(tmp_path)
    index = json.loads((backend / "live" / "index.json").read_text())
    assert LIVE_RUN_ID.match(index["live_run_id"])
    assert index["live_run_id"] == f"live_{ORIGIN.isoformat()}_{RUN_ID}"
    assert index["generated_at"] == NOW.isoformat(timespec="seconds")
    assert index["stale_after_hours"] == 36 and index["folds"] == [2020, 2021]
    assert [r["region_id"] for r in index["regions"]] == ["r_a", "r_b"]
    assert index["regions"][0]["n_detections"] == 12 and index["regions"][1]["n_detections"] == 0
    for reg in index["regions"]:
        for art in reg["artifacts"].values():
            uri = art["storage_uri"]
            assert "\\" not in uri and not uri.startswith("/") and ".." not in uri
            assert PurePosixPath(uri).parts[:2] == ("live", index["live_run_id"])
            for root in (artifact_root, backend):
                data = (root / uri).read_bytes()
                assert hashlib.sha256(data).hexdigest() == art["checksum_sha256"]
                assert len(data) == art["file_size_bytes"]
            with rasterio.open(backend / uri) as ds:
                assert (ds.height, ds.width) == (reg["height"], reg["width"])
                assert ds.crs.to_string() == reg["crs"] == "EPSG:32610"
                assert tuple(ds.transform)[:6] == tuple(reg["transform"])
    # nothing but the published run and the index under the backend root
    assert sorted(p.name for p in (backend / "live").iterdir()) == ["index.json",
                                                                    index["live_run_id"]]


def test_raw_inputs_are_kept_in_wsts_format(tmp_path):
    out, artifact_root, _, src, _ = _run(tmp_path)
    p = artifact_root / "live" / out["live_run_id"] / "r_a" / ORIGIN.isoformat() / "inputs_raw.tif"
    with rasterio.open(p) as ds:
        assert ds.count == 23 and ds.descriptions[22] == "active fire"
        np.testing.assert_array_equal(ds.read(), src.raw_frame("r_a", ORIGIN))


def test_without_backend_root_nothing_is_published(tmp_path):
    out, artifact_root, _, _, _ = _run(tmp_path, publish=False)
    assert out["published"] is False
    assert not (tmp_path / "backend_artifacts").exists()
    assert (artifact_root / "live" / out["live_run_id"] / "live_run.json").is_file()


def test_rerun_replaces_only_that_run(tmp_path):
    out, artifact_root, backend, _, _ = _run(tmp_path)
    other = backend / "live" / "live_2026-10-31_m1_t1_test"
    other.mkdir(parents=True)
    (other / "keep.txt").write_text("x")
    (backend / "live" / out["live_run_id"] / "stale.txt").write_text("x")
    run_dir, run, _ = tmp_path / "outputs" / "m1", "m1_spatial_unet_base8_T1_seed0", None
    report, bundles = check_live_readiness(run_dir=run_dir, run=run, folds=(2020, 2021),
                                           run_id=RUN_ID, artifact_root=artifact_root,
                                           repo_root=tmp_path / "repo")
    src = LiveTemporalSource([small_region(rid="r_a")], ORIGIN, CHANNELS, 1, ArrayFire({}),
                             ArrayReflectance(), ArrayTerrain())
    run_live_forecast(source=src, bundles=bundles, report=report, artifact_root=artifact_root,
                      backend_artifact_root=backend, now=NOW)
    assert (other / "keep.txt").is_file()
    assert not (backend / "live" / out["live_run_id"] / "stale.txt").exists()
    index = json.loads((backend / "live" / "index.json").read_text())
    assert [r["region_id"] for r in index["regions"]] == ["r_a"]


def test_index_is_written_atomically(tmp_path, monkeypatch):
    target = tmp_path / "live" / "index.json"
    write_json_atomic(target, {"v": 1})
    calls = []
    real = live_forecast.os.replace

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        assert json.loads(open(src, encoding="utf-8").read()) == {"v": 2}
        assert json.loads(target.read_text()) == {"v": 1}  # old index readable until the swap
        real(src, dst)

    monkeypatch.setattr(live_forecast.os, "replace", spy)
    write_json_atomic(target, {"v": 2})
    assert calls == [(str(target) + ".tmp", str(target))]
    assert json.loads(target.read_text()) == {"v": 2}
    assert not (tmp_path / "live" / "index.json.tmp").exists()


def test_mismatched_source_is_refused(tmp_path):
    run_dir, run, artifact_root = make_trained_run(tmp_path, (2020, 2021))
    report, bundles = check_live_readiness(run_dir=run_dir, run=run, folds=(2020, 2021),
                                           run_id=RUN_ID, artifact_root=artifact_root,
                                           repo_root=make_repo_root(tmp_path))
    for src in (
        LiveTemporalSource([small_region()], ORIGIN, CHANNELS, 3, ArrayFire({}),
                           ArrayReflectance(), ArrayTerrain()),
        LiveTemporalSource([small_region()], ORIGIN, CHANNELS, 1, ArrayFire({}),
                           ArrayReflectance(), ArrayTerrain(), is_mock=True),
    ):
        with pytest.raises(ValueError):
            run_live_forecast(source=src, bundles=bundles, report=report,
                              artifact_root=artifact_root, backend_artifact_root=None)


def test_live_run_id_rejects_unsafe_run_ids():
    assert live_run_id_for(ORIGIN, "m1_t1_v1") == "live_2026-11-01_m1_t1_v1"
    for bad in ("../x", "a/b", "a b", "", "a\\b"):
        with pytest.raises(ValueError):
            live_run_id_for(ORIGIN, bad)
