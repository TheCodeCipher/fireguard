"""Integration tests for scripts/load_registry_bundle.py (added during the Prototype integration).

The bundle is what scripts/build_backend_registry.py writes from Role 4's reveal package. These
tests build small bundles by hand (byte files; Role 5's registry checks bytes, checksums and
metadata, it never parses rasters) and check that loading them goes through Role 5's rules:

* mock bundles become "mock" artifacts, carry metric rows, and can never be published;
* real bundles become "validated" (invisible to replay) and only --publish makes them
  replayable, after which the full forecast-first / reveal-after flow works through the API;
* a real bundle whose time-safety status is not PASS_LEVEL_A is refused by Role 5's validator.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fireguard.core.config import get_settings
from fireguard.core.database import Base, get_db
from fireguard.main import app
from fireguard.models import ForecastArtifact, ReplayMetric
from fireguard.services.oof_registry import RegistryValidationError

BACKEND = Path(__file__).resolve().parents[2]
HEX = "a" * 64


def _load_loader():
    spec = importlib.util.spec_from_file_location(
        "load_registry_bundle", BACKEND / "scripts" / "load_registry_bundle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root: Path, uri: str, payload: bytes) -> dict:
    path = root / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"checksum_sha256": hashlib.sha256(payload).hexdigest(), "file_size_bytes": len(payload)}


def _raster_meta(mock: bool) -> dict:
    meta = {
        "format": "GeoTIFF", "crs": "EPSG:32610", "affine_transform": [375, 0, 0, 0, -375, 0],
        "width": 4, "height": 3, "nodata": None, "valid_extent": [0, 0, 4, 3],
        "units": "probability", "display_min": 0.0, "display_max": 1.0,
    }  # fmt: skip
    if mock:
        meta["mode"] = "mock"
    return meta


def _bundle(tmp: Path, artifact_root: Path, *, mode: str, time_safety: str = "PASS_LEVEL_A"):
    """A one-forecast bundle shaped exactly like build_backend_registry.py output."""
    mock = mode == "mock"
    prefix = "mock/" if mock else ""
    rel = "p1_loyo/m@fold-2020/LOYO-2020/fire_1/2020-08-05"
    artifacts = []
    for kind, uri, ctype, reveal, payload in (
        ("probability_map", f"{prefix}{rel}/probability.tif", "image/tiff", "pre_reveal", b"P"),
        ("actual_target", f"restricted/{prefix}{rel}/actual_target.tif", "image/tiff",
         "post_reveal", b"T"),
        ("metrics", f"restricted/{prefix}{rel}/metrics.json", "application/json",
         "post_reveal", b"{}"),
    ):  # fmt: skip
        spec = {"artifact_type": kind, "storage_uri": uri, "content_type": ctype,
                "reveal_class": reveal, **_write(artifact_root, uri, payload)}  # fmt: skip
        if kind == "metrics":
            spec["metadata"] = {"format": "JSON", **({"mode": "mock"} if mock else {})}
        else:
            spec["metadata"] = _raster_meta(mock)
        artifacts.append(spec)
    entry = {
        "forecast_id": "f1", "event_key": "fire_1", "held_out_group_id": "G2020_0001",
        "protocol_id": "MOCK_ONLY" if mock else "P1_LOYO", "source_protocol_id": "P1_LOYO",
        "fold_id": "LOYO-2020", "split_name": "test", "is_oof": True,
        "origin_date": "2020-08-05", "target_date": "2020-08-06", "horizon_hours": 24,
        "model_name": ("[MOCK] " if mock else "") + "FireGuard m3_stacked_unet",
        "model_version": "m@fold-2020", "architecture": "m3_stacked_unet",
        "checkpoint_sha256": HEX, "config_hash": HEX, "split_manifest_sha256": HEX,
        "input_manifest_sha256": HEX, "data_schema_version": "wsts_strict_input_v1",
        "data_schema_sha256": HEX, "channel_manifest_sha256": HEX, "channels": ["a", "b"],
        "time_safety_status": time_safety if not mock else "MOCK_NOT_A_RESEARCH_RESULT",
        "probability_kind": "model_sigmoid_uncalibrated", "fold_validation_auprc": 0.1,
        "artifacts": artifacts,
        "metrics": [
            {"metric_name": "auprc", "value": 0.5, "aggregation": "forecast_day",
             "threshold": None, "calibrator_id": "platt-P1-fold-2020-validation"},
            {"metric_name": "f1", "value": 0.4, "aggregation": "event",
             "threshold": 0.3, "calibrator_id": "platt-P1-fold-2020-validation"},
        ],
    }  # fmt: skip
    out = tmp / f"registry_{mode}"
    out.mkdir()
    with (out / "event_catalogue.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "year", "complex_group_id", "p1_fold_id", "manifest_sha256"])
        w.writerow(["fire_1", "2020", "G2020_0001", "LOYO-2020", HEX])
    path = out / "registry_bundle.json"
    path.write_text(json.dumps({"bundle_version": "1.0", "mode": mode, "forecasts": [entry]}))
    return path


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated artifact root + in-memory DB shared by the loader, Role 5 CLI and the API."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.setenv("FIREGUARD_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FIREGUARD_APP_ENV", "development")
    get_settings.cache_clear()
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    loader = _load_loader()
    # The loader, Role 5's registration CLI and catalogue importer each hold SessionLocal.
    for module in (loader, loader.register_cli, loader.catalogue_cli):
        monkeypatch.setattr(module, "SessionLocal", session)

    def override_get_db():
        db = session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield loader, session, artifact_root, tmp_path
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_mock_bundle_loads_as_mock_with_metrics_and_cannot_publish(env):
    loader, session, artifact_root, tmp = env
    bundle = _bundle(tmp, artifact_root, mode="mock")
    summary = loader.load(bundle, publish=False)
    assert summary["states"] == ["mock"]
    with session() as db:
        statuses = {a.status for a in db.scalars(select(ForecastArtifact))}
        assert statuses == {"mock"}
        assert len(db.scalars(select(ReplayMetric)).all()) == 2
    # Loading twice replaces metric rows instead of duplicating them.
    loader.load(bundle, publish=False)
    with session() as db:
        assert len(db.scalars(select(ReplayMetric)).all()) == 2
    with pytest.raises(RegistryValidationError, match="never be published"):
        loader.load(bundle, publish=True)


def test_real_bundle_needs_publish_then_serves_the_reveal_flow(env):
    loader, session, artifact_root, tmp = env
    bundle = _bundle(tmp, artifact_root, mode="real")
    assert loader.load(bundle, publish=False)["states"] == ["validated"]

    client = TestClient(app)
    origins = client.get("/api/v1/replays/fire_1/origins").json()
    assert origins["count"] == 0  # validated is not replay-visible

    assert loader.load(bundle, publish=True)["states"] == ["published"]
    origins = client.get("/api/v1/replays/fire_1/origins").json()
    assert origins["count"] == 1 and origins["origins"][0]["is_mock"] is False

    sid = client.post("/api/v1/replays/fire_1/2020-08-05/session").json()["session_id"]
    h = {"X-Replay-Session-ID": sid}
    assert client.get("/api/v1/replays/fire_1/2020-08-05/metrics", headers=h).status_code == 403
    forecast = client.get("/api/v1/replays/fire_1/2020-08-05/forecast", headers=h).json()
    assert [a["artifact_type"] for a in forecast["artifacts"]] == ["probability_map"]
    assert client.post("/api/v1/replays/fire_1/2020-08-05/reveal", headers=h).status_code == 200
    metrics = client.get("/api/v1/replays/fire_1/2020-08-05/metrics", headers=h).json()["metrics"]
    assert {(m["metric_name"], m["aggregation"]) for m in metrics} == {
        ("auprc", "forecast_day"),
        ("f1", "event"),
    }


def test_real_bundle_without_level_a_pass_is_refused_by_role5(env):
    loader, _session, artifact_root, tmp = env
    bundle = _bundle(tmp, artifact_root, mode="real", time_safety="STRICT_LEVEL_A_PASS")
    with pytest.raises(RegistryValidationError, match="PASS_LEVEL_A"):
        loader.load(bundle, publish=False)
