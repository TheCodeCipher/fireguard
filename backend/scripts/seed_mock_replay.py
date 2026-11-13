"""Seed one clearly synthetic replay for local UI integration only.

Run from backend/: python scripts/seed_mock_replay.py
Never use this fixture as a research result or real WSTS event.
"""
import json
from datetime import date
from pathlib import Path

from fireguard.core.database import Base, SessionLocal, engine
from fireguard.models import ForecastArtifact, ForecastRun, ModelVersion, WildfireEvent


ARTIFACT_DIR = Path("artifacts/mock")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACT_DIR / "probability_map.json").write_text(
    json.dumps(
        {"mode": "mock", "grid": [[0.1, 0.3], [0.6, 0.8]], "units": "probability"},
        indent=2,
    ),
    encoding="utf-8",
)
(ARTIFACT_DIR / "actual_target.json").write_text(
    json.dumps(
        {"mode": "mock", "grid": [[0, 0], [1, 1]], "target": "synthetic_binary_observation"},
        indent=2,
    ),
    encoding="utf-8",
)

Base.metadata.create_all(bind=engine)
with SessionLocal() as db:
    event = db.query(WildfireEvent).filter_by(event_key="mock-fire-0001").one_or_none()
    if event is None:
        event = WildfireEvent(
            event_key="mock-fire-0001",
            name="Synthetic replay fixture (not a WSTS event)",
            year=2020,
            region="western-us (illustrative only)",
            source_dataset="MOCK_FIXTURE",
        )
        db.add(event)
        db.flush()

    model = (
        db.query(ModelVersion)
        .filter_by(name="FireGuard mock model", version="0.0.0-mock")
        .one_or_none()
    )
    if model is None:
        model = ModelVersion(
            name="FireGuard mock model",
            version="0.0.0-mock",
            architecture="synthetic-fixture",
            model_config_json={"mode": "mock", "config_hash": "not-a-real-experiment"},
        )
        db.add(model)
        db.flush()

    run = db.query(ForecastRun).filter_by(
        event_id=event.id, model_version_id=model.id,
        origin_date=date(2020, 8, 1), fold="MOCK-FOLD",
    ).one_or_none()
    if run is None:
        run = ForecastRun(
            event_id=event.id,
            model_version_id=model.id,
            origin_date=date(2020, 8, 1),
            target_date=date(2020, 8, 2),
            horizon_hours=24,
            fold="MOCK-FOLD",
            split_name="test",
            is_oof=True,
            protocol="MOCK_ONLY",
            time_safety_status="MOCK_NOT_A_RESEARCH_RESULT",
            metadata_json={
                "input_window_start": "2020-07-31",
                "input_window_end": "2020-08-01",
                "channel_manifest": "synthetic-fixture",
                "target_uri": "must-not-be-returned-before-reveal",
            },
        )
        db.add(run)
        db.flush()
        db.add_all([
            ForecastArtifact(
                forecast_run_id=run.id,
                artifact_type="probability_map",
                storage_uri="mock/probability_map.json",
                content_type="application/json",
                metadata_json={"mode": "mock"},
            ),
            ForecastArtifact(
                forecast_run_id=run.id,
                artifact_type="actual_target",
                storage_uri="mock/actual_target.json",
                content_type="application/json",
                metadata_json={"mode": "mock"},
            ),
        ])
    db.commit()

print("Synthetic replay fixture seeded. Do not report these values as model results.")
