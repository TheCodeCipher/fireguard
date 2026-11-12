from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fireguard.core.config import get_settings
from fireguard.core.database import Base, get_db
from fireguard.main import app
from fireguard.models import ForecastArtifact, ForecastRun, ModelVersion, WildfireEvent


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    mock_dir = artifact_root / "mock"
    mock_dir.mkdir(parents=True)
    (mock_dir / "probability.json").write_text(
        '{"mode":"mock","grid":[[0.2,0.8]]}', encoding="utf-8"
    )
    (mock_dir / "actual.json").write_text('{"mode":"mock","target":[[0,1]]}', encoding="utf-8")

    monkeypatch.setenv("FIREGUARD_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    get_settings.cache_clear()

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestSession() as db:
        event = WildfireEvent(
            event_key="mock-event-001", name="Synthetic event", year=2020,
            region="western-us", source_dataset="MOCK_FIXTURE",
        )
        model = ModelVersion(
            name="Synthetic U-Net", version="0.0-mock", architecture="UNet2D",
            model_config_json={"config_hash": "mock-hash"},
        )
        db.add_all([event, model])
        db.flush()
        run = ForecastRun(
            event_id=event.id, model_version_id=model.id,
            origin_date=date(2020, 8, 1), target_date=date(2020, 8, 2),
            horizon_hours=24, fold="LOYO-2020", split_name="test", is_oof=True,
            protocol="P1_LOYO", channel_manifest_sha256="a" * 64,
            time_safety_status="PASS_LEVEL_A",
            metadata_json={
                "input_window_start": "2020-07-31",
                "input_window_end": "2020-08-01",
                "channel_manifest": "strict-no-gfs-v1",
                "target_uri": "must-not-be-returned",
            },
        )
        db.add(run)
        db.flush()
        db.add_all([
            ForecastArtifact(
                id=100, forecast_run_id=run.id, artifact_type="probability_map",
                storage_uri="mock/probability.json", content_type="application/json",
            ),
            ForecastArtifact(
                id=101, forecast_run_id=run.id, artifact_type="actual_target",
                storage_uri="mock/actual.json", content_type="application/json",
            ),
        ])
        db.commit()

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(engine)
        engine.dispose()
        get_settings.cache_clear()


def test_event_catalogue_returns_metadata_only(api_client):
    response = api_client.get("/api/v1/wsts/events")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["events"][0]["event_id"] == "mock-event-001"
    assert payload["events"][0]["has_replay"] is True
    assert "target" not in response.text.lower()


def test_replay_actual_locked_until_forecast_then_reveal(api_client):
    base = "/api/v1/replays/mock-event-001/2020-08-01"
    start = api_client.post(f"{base}/session")
    assert start.status_code == 200
    session_id = start.json()["session_id"]
    headers = {"X-Replay-Session-ID": session_id}

    locked = api_client.get(f"{base}/actual", headers=headers)
    assert locked.status_code == 403
    early_reveal = api_client.post(f"{base}/reveal", headers=headers)
    assert early_reveal.status_code == 409

    inputs = api_client.get(f"{base}/inputs", headers=headers)
    assert inputs.status_code == 200
    assert "target_uri" not in inputs.text

    forecast = api_client.get(f"{base}/forecast", headers=headers)
    assert forecast.status_code == 200
    assert forecast.json()["fold_id"] == "LOYO-2020"
    artifact_id = forecast.json()["artifacts"][0]["artifact_id"]

    actual_hidden_file = api_client.get("/api/v1/artifacts/101", headers=headers)
    assert actual_hidden_file.status_code == 403

    hidden_artifact = api_client.get(f"/api/v1/artifacts/{artifact_id}", headers=headers)
    assert hidden_artifact.status_code == 200
    assert b'"mode":"mock"' in hidden_artifact.content

    revealed = api_client.post(f"{base}/reveal", headers=headers)
    assert revealed.status_code == 200
    assert revealed.json()["revealed"] is True

    actual = api_client.get(f"{base}/actual", headers=headers)
    assert actual.status_code == 200
    actual_id = actual.json()["artifacts"][0]["artifact_id"]
    actual_file = api_client.get(f"/api/v1/artifacts/{actual_id}", headers=headers)
    assert actual_file.status_code == 200
    assert b'"target"' in actual_file.content


def test_event_catalogue_pagination_reports_page_and_total_counts(api_client):
    response = api_client.get("/api/v1/wsts/events?limit=1&offset=0")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["total_count"] == 1
    assert payload["limit"] == 1
    assert payload["offset"] == 0


def test_event_catalogue_does_not_create_fake_events(api_client):
    response = api_client.get("/api/v1/wsts/events?year=2019")
    assert response.status_code == 200
    assert response.json()["events"] == []


def test_live_firms_fails_closed_without_server_key(api_client):
    response = api_client.get(
        "/api/v1/firms/availability",
        params={"source": "VIIRS_SNPP_SP"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "FIRMS_NOT_CONFIGURED"
    assert "MAP_KEY" not in response.text or "not configured" in response.text


def test_fires_invalid_bbox_rejected_before_upstream_call(api_client):
    response = api_client.get(
        "/api/v1/fires",
        params={
            "source": "VIIRS_SNPP_SP",
            "start_date": "2026-11-01",
            "end_date": "2026-11-02",
            "bbox": "north,south,east,west",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_BBOX"


def test_fires_date_range_is_bounded(api_client):
    response = api_client.get(
        "/api/v1/fires",
        params={
            "source": "VIIRS_SNPP_SP",
            "start_date": "2026-09-01",
            "end_date": "2026-10-15",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "DATE_RANGE_TOO_LARGE"



def test_cors_allows_configured_frontend_origin(api_client):
    response = api_client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
