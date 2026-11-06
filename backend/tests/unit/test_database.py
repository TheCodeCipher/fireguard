from datetime import date

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from fireguard.core.database import Base
from fireguard.models import (
    ForecastArtifact,
    ForecastRun,
    ModelVersion,
    WildfireEvent,
)


def test_database_tables_are_created():
    test_engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(bind=test_engine)

    inspector = inspect(test_engine)

    expected_tables = {
        "wildfire_events",
        "model_versions",
        "forecast_runs",
        "forecast_artifacts",
    }

    assert expected_tables.issubset(set(inspector.get_table_names()))

    test_engine.dispose()


def test_forecast_relationships():
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)

    with Session(test_engine) as session:
        event = WildfireEvent(
            event_key="WSTS-TEST-001",
            name="Test wildfire",
            region="Western US",
            source_dataset="WSTS",
        )

        model = ModelVersion(
            name="FireGuard baseline",
            version="0.1.0",
            architecture="UNet2D",
        )

        session.add_all([event, model])
        session.flush()

        forecast = ForecastRun(
            event_id=event.id,
            model_version_id=model.id,
            origin_date=date(2020, 7, 1),   # Updated to use date object
            target_date=date(2020, 7, 2),   # Updated to use date object
            horizon_hours=24,
            fold="LOYO-2020",
            split_name="test",
        )

        session.add(forecast)
        session.flush()

        artifact = ForecastArtifact(
            forecast_run_id=forecast.id,
            artifact_type="probability_map",
            storage_uri="s3://fireguard-test/forecast-001.tif",
            content_type="image/tiff",
        )

        session.add(artifact)
        session.commit()

        assert forecast.event.event_key == "WSTS-TEST-001"
        assert forecast.model_version.architecture == "UNet2D"
        assert len(forecast.artifacts) == 1

    test_engine.dispose()
