"""Live-forecast routes: retrieve-only, checksum-verified, no session, no model."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from fireguard.core.config import get_settings
from fireguard.main import app

RUN = "live_2026-11-01_m1_t1_v1"
REGION = "r_38p400n_120p200w"


def _publish(root, generated_at=None, **overrides):
    base = root / "live" / RUN / REGION / "2026-11-01"
    base.mkdir(parents=True)
    arts = {}
    for layer, payload in (("probability", b"II*\x00prob"), ("fold_spread", b"II*\x00spread")):
        (base / f"{layer}.tif").write_bytes(payload)
        arts[layer] = {"storage_uri": f"live/{RUN}/{REGION}/2026-11-01/{layer}.tif",
                       "checksum_sha256": hashlib.sha256(payload).hexdigest(),
                       "file_size_bytes": len(payload), "content_type": "image/tiff"}
    index = {
        "schema_version": 1, "live_run_id": RUN, "origin_date": "2026-11-01",
        "target_date": "2026-11-02", "horizon_hours": 24,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(timespec="seconds"),
        "stale_after_hours": 36, "model_version": "m1_t1_v1", "architecture": "m1_spatial_unet",
        "folds": [2018, 2019, 2020, 2021],
        "probability_kind": "model_sigmoid_uncalibrated_fold_mean", "gate_status": "GATED",
        "is_mock": False, "data_contract_version": "wsts_strict_input_v1",
        "firms_source": "VIIRS_SNPP_NRT",
        "regions": [{
            "region_id": REGION, "center_lon": -120.2, "center_lat": 38.4,
            "bbox_wgs84": [-120.7, 37.9, -119.7, 38.9], "height": 304, "width": 215,
            "crs": "EPSG:32610", "transform": [375.0, 0.0, 697500.0, 0.0, -375.0, 4136625.0],
            "n_detections": 25,
            "nan_fraction": {"viirs_m11": 0.1, "binary_active_fire_mask": 0.0},
            "reflectance_source": "earth_engine_vnp09ga_v2", "artifacts": arts,
        }],
    }
    index.update(overrides)
    (root / "live" / "index.json").write_text(json.dumps(index))
    return index


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREGUARD_ARTIFACT_ROOT", str(tmp_path))
    get_settings.cache_clear()
    yield TestClient(app), tmp_path
    get_settings.cache_clear()


def test_no_published_forecast_is_a_clean_404(client):
    c, _ = client
    r = c.get("/api/v1/live/forecasts")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "LIVE_FORECAST_NOT_AVAILABLE"


def test_latest_forecast_is_served_with_urls_and_disclaimer(client):
    c, root = client
    _publish(root)
    r = c.get("/api/v1/live/forecasts")
    assert r.status_code == 200
    body = r.json()
    assert body["live_run_id"] == RUN and body["stale"] is False
    assert body["gate_status"] == "GATED" and body["is_mock"] is False
    assert "not an official warning" in body["disclaimer"]
    (reg,) = body["regions"]
    assert reg["region_id"] == REGION and reg["n_detections"] == 25
    assert reg["artifacts"]["probability"]["url"] == (
        f"/api/v1/live/forecasts/{RUN}/{REGION}/probability")
    assert "storage_uri" not in json.dumps(body)  # storage layout stays private
    for layer in ("probability", "fold_spread"):
        f = c.get(reg["artifacts"][layer]["url"])
        assert f.status_code == 200 and f.headers["content-type"] == "image/tiff"
        assert hashlib.sha256(f.content).hexdigest() == reg["artifacts"][layer]["checksum_sha256"]


def test_no_replay_session_header_is_needed(client):
    c, root = client
    _publish(root)
    r = c.get(f"/api/v1/live/forecasts/{RUN}/{REGION}/probability")
    assert r.status_code == 200


def test_old_index_is_marked_stale(client):
    c, root = client
    _publish(root, generated_at=datetime.now(UTC) - timedelta(hours=40))
    assert c.get("/api/v1/live/forecasts").json()["stale"] is True


def test_edited_geotiff_is_refused(client):
    c, root = client
    _publish(root)
    (root / "live" / RUN / REGION / "2026-11-01" / "probability.tif").write_bytes(b"II*\x00evil")
    r = c.get(f"/api/v1/live/forecasts/{RUN}/{REGION}/probability")
    assert r.status_code == 409 and r.json()["error"]["code"] == "ARTIFACT_CHECKSUM_MISMATCH"


def test_path_traversal_in_the_index_is_refused(client, tmp_path_factory):
    c, root = client
    index = _publish(root)
    outside = tmp_path_factory.mktemp("outside") / "secret.tif"
    outside.write_bytes(b"secret")
    index["regions"][0]["artifacts"]["probability"]["storage_uri"] = "../outside/secret.tif"
    (root / "live" / "index.json").write_text(json.dumps(index))
    r = c.get(f"/api/v1/live/forecasts/{RUN}/{REGION}/probability")
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_ARTIFACT_URI"


def test_unknown_run_region_or_layer(client):
    c, root = client
    _publish(root)
    for url in (f"/api/v1/live/forecasts/live_2026-10-01_old/{REGION}/probability",
                f"/api/v1/live/forecasts/{RUN}/r_other/probability"):
        r = c.get(url)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "LIVE_FORECAST_NOT_AVAILABLE"
    assert c.get(f"/api/v1/live/forecasts/{RUN}/{REGION}/inputs_raw").status_code == 422
    assert c.get(f"/api/v1/live/forecasts/{RUN}/..%2F..%2Fx/probability").status_code in (404, 422)


def test_unreadable_or_incomplete_index_is_503(client):
    c, root = client
    (root / "live").mkdir()
    (root / "live" / "index.json").write_text("{not json")
    assert c.get("/api/v1/live/forecasts").json()["error"]["code"] == "LIVE_INDEX_UNREADABLE"
    (root / "live" / "index.json").write_text(json.dumps({"schema_version": 2}))
    assert c.get("/api/v1/live/forecasts").status_code == 503
    (root / "live" / "index.json").write_text(json.dumps({"schema_version": 1}))
    assert c.get("/api/v1/live/forecasts").status_code == 503


def test_live_routes_accept_no_writes(client):
    c, root = client
    _publish(root)
    assert c.post("/api/v1/live/forecasts").status_code == 405
    assert c.delete(f"/api/v1/live/forecasts/{RUN}/{REGION}/probability").status_code == 405
