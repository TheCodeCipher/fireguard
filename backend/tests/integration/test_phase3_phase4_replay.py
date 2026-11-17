from datetime import date

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fireguard.core.config import get_settings
from fireguard.core.database import Base, get_db
from fireguard.main import app
from fireguard.models import ForecastArtifact, ForecastRun, ModelVersion, WildfireEvent


def test_only_published_or_explicit_dev_mock_oof_artifacts_are_catalogued(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "mock").mkdir(parents=True)
    (artifact_root / "mock/probability.json").write_text('{"mode":"mock"}', encoding="utf-8")
    monkeypatch.setenv("FIREGUARD_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FIREGUARD_APP_ENV", "development")
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
        event = WildfireEvent(event_key="mock-only", source_dataset="MOCK_FIXTURE")
        model = ModelVersion(name="mock", version="0", architecture="mock")
        db.add_all([event, model])
        db.flush()
        run = ForecastRun(
            event_id=event.id,
            model_version_id=model.id,
            origin_date=date(2020, 1, 1),
            target_date=date(2020, 1, 2),
            horizon_hours=24,
            fold="MOCK-FOLD",
            split_name="test",
            is_oof=True,
            protocol="MOCK_ONLY",
            time_safety_status="MOCK_NOT_A_RESEARCH_RESULT",
            metadata_json={"mode": "mock"},
        )
        db.add(run)
        db.flush()
        db.add(
            ForecastArtifact(
                forecast_run_id=run.id,
                artifact_type="probability_map",
                storage_uri="mock/probability.json",
                content_type="application/json",
                checksum_sha256="0" * 64,
                status="generated",
                metadata_json={"mode": "mock"},
            )
        )
        db.commit()

    try:
        with TestClient(app) as client:
            hidden = client.get("/api/v1/replays/mock-only/origins")
            assert hidden.status_code == 200
            assert hidden.json()["count"] == 0
            with TestSession() as db:
                artifact = db.query(ForecastArtifact).one()
                artifact.status = "mock"
                artifact.checksum_sha256 = __import__("hashlib").sha256(
                    (artifact_root / "mock/probability.json").read_bytes()
                ).hexdigest()
                artifact.metadata_json = {
                    "mode": "mock",
                    "reveal_class": "pre_reveal",
                    "file_size_bytes": (
                        artifact_root / "mock/probability.json"
                    ).stat().st_size,
                    "metadata": {"format": "JSON"},
                }
                db.commit()
            visible = client.get("/api/v1/replays/mock-only/origins")
            assert visible.status_code == 200
            assert visible.json()["count"] == 1
            assert visible.json()["origins"][0]["is_mock"] is True
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(engine)
        engine.dispose()
        get_settings.cache_clear()


def test_static_artifact_rejects_checksum_tampering(tmp_path, monkeypatch):
    import hashlib
    from datetime import date

    artifact_root = tmp_path / "artifacts"
    mock_dir = artifact_root / "mock"
    mock_dir.mkdir(parents=True)
    target = mock_dir / "probability.json"
    target.write_text('{"mode":"mock"}', encoding="utf-8")
    monkeypatch.setenv("FIREGUARD_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("FIREGUARD_APP_ENV", "development")
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
        event = WildfireEvent(event_key="checksum-event", source_dataset="MOCK_FIXTURE")
        model = ModelVersion(name="mock", version="0", architecture="mock")
        db.add_all([event, model])
        db.flush()
        run = ForecastRun(
            event_id=event.id, model_version_id=model.id, origin_date=date(2020, 1, 1),
            target_date=date(2020, 1, 2), horizon_hours=24, fold="MOCK-FOLD",
            split_name="test", is_oof=True, protocol="MOCK_ONLY",
            time_safety_status="MOCK_NOT_A_RESEARCH_RESULT", metadata_json={"mode": "mock"},
        )
        db.add(run)
        db.flush()
        checksum = hashlib.sha256(target.read_bytes()).hexdigest()
        db.add(ForecastArtifact(
            forecast_run_id=run.id,
            artifact_type="probability_map",
            storage_uri="mock/probability.json",
            content_type="application/json", checksum_sha256=checksum, status="mock",
            metadata_json={
                "mode": "mock",
                "reveal_class": "pre_reveal",
                "file_size_bytes": target.stat().st_size,
                "metadata": {"format": "JSON"},
            },
        ))
        db.commit()

    try:
        with TestClient(app) as client:
            base = "/api/v1/replays/checksum-event/2020-01-01"
            session = client.post(f"{base}/session").json()["session_id"]
            headers = {"X-Replay-Session-ID": session}
            forecast = client.get(f"{base}/forecast", headers=headers)
            assert forecast.status_code == 200
            artifact_id = forecast.json()["artifacts"][0]["artifact_id"]
            target.write_text('{"mode":"tampered"}', encoding="utf-8")
            response = client.get(f"/api/v1/artifacts/{artifact_id}", headers=headers)
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "ARTIFACT_CHECKSUM_MISMATCH"
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(engine)
        engine.dispose()
        get_settings.cache_clear()
