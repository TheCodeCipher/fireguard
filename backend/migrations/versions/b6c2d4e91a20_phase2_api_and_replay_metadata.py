"""Add Phase 2 FIRMS cache, replay-session and provenance metadata.

Revision ID: b6c2d4e91a20
Revises: a4291d9fbbaa
"""
from alembic import op
import sqlalchemy as sa

revision = "b6c2d4e91a20"
down_revision = "a4291d9fbbaa"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("wildfire_events", sa.Column("year", sa.Integer(), nullable=True))
    op.add_column(
        "wildfire_events", sa.Column("complex_group_id", sa.String(length=100), nullable=True)
    )
    op.add_column("wildfire_events", sa.Column("p1_fold_id", sa.String(length=50), nullable=True))
    op.add_column("wildfire_events", sa.Column("p2_fold_id", sa.String(length=50), nullable=True))
    op.add_column(
        "wildfire_events", sa.Column("manifest_sha256", sa.String(length=64), nullable=True)
    )
    op.create_index("ix_wildfire_events_year", "wildfire_events", ["year"])
    op.create_index("ix_wildfire_events_complex_group_id", "wildfire_events", ["complex_group_id"])

    op.add_column(
        "forecast_runs",
        sa.Column("is_oof", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.add_column(
        "forecast_runs",
        sa.Column("protocol", sa.String(length=50), server_default="P1_LOYO", nullable=False),
    )
    op.add_column(
        "forecast_runs",
        sa.Column("channel_manifest_sha256", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "forecast_runs",
        sa.Column(
            "time_safety_status", sa.String(length=40),
            server_default="UNKNOWN", nullable=False,
        ),
    )
    op.add_column(
        "forecast_runs",
        sa.Column("metadata_json", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
    )
    op.create_index("ix_forecast_runs_is_oof", "forecast_runs", ["is_oof"])

    op.add_column(
        "forecast_artifacts",
        sa.Column("status", sa.String(length=30), server_default="available", nullable=False),
    )
    op.add_column(
        "forecast_artifacts",
        sa.Column("metadata_json", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
    )

    op.create_table(
        "fire_observations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=40), nullable=False),
        sa.Column("platform", sa.String(length=30), nullable=False),
        sa.Column("product", sa.String(length=10), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("frp", sa.Float(), nullable=True),
        sa.Column("confidence", sa.String(length=20), nullable=False),
        sa.Column("raw_attributes", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_fire_observations_id", "fire_observations", ["id"])
    op.create_index(
        "ix_fire_observations_source_id", "fire_observations", ["source_id"], unique=True
    )
    op.create_index("ix_fire_observations_source", "fire_observations", ["source"])
    op.create_index("ix_fire_observations_acquired_at", "fire_observations", ["acquired_at"])

    op.create_table(
        "replay_metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "forecast_run_id", sa.Integer(),
            sa.ForeignKey("forecast_runs.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("metric_name", sa.String(length=60), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("aggregation", sa.String(length=30), server_default="event", nullable=False),
        sa.Column("calibrator_id", sa.String(length=100), nullable=True),
    )
    op.create_index("ix_replay_metrics_id", "replay_metrics", ["id"])
    op.create_index("ix_replay_metrics_forecast_run_id", "replay_metrics", ["forecast_run_id"])

    op.create_table(
        "replay_sessions",
        sa.Column("session_id", sa.String(length=36), primary_key=True),
        sa.Column(
            "forecast_run_id", sa.Integer(),
            sa.ForeignKey("forecast_runs.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("forecast_viewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revealed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_replay_sessions_forecast_run_id", "replay_sessions", ["forecast_run_id"])
    op.create_index("ix_replay_sessions_expires_at", "replay_sessions", ["expires_at"])

    op.create_table(
        "data_assets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset", sa.String(length=100), nullable=False),
        sa.Column("product_version", sa.String(length=100), nullable=True),
        sa.Column("sensing_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("available_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=True),
        sa.Column("storage_uri", sa.String(length=1000), nullable=False),
        sa.Column("quality_summary", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_data_assets_id", "data_assets", ["id"])
    op.create_index("ix_data_assets_dataset", "data_assets", ["dataset"])


def downgrade() -> None:
    op.drop_index("ix_data_assets_dataset", table_name="data_assets")
    op.drop_index("ix_data_assets_id", table_name="data_assets")
    op.drop_table("data_assets")
    op.drop_index("ix_replay_sessions_expires_at", table_name="replay_sessions")
    op.drop_index("ix_replay_sessions_forecast_run_id", table_name="replay_sessions")
    op.drop_table("replay_sessions")
    op.drop_index("ix_replay_metrics_forecast_run_id", table_name="replay_metrics")
    op.drop_index("ix_replay_metrics_id", table_name="replay_metrics")
    op.drop_table("replay_metrics")
    op.drop_index("ix_fire_observations_acquired_at", table_name="fire_observations")
    op.drop_index("ix_fire_observations_id", table_name="fire_observations")
    op.drop_index("ix_fire_observations_source", table_name="fire_observations")
    op.drop_index("ix_fire_observations_source_id", table_name="fire_observations")
    op.drop_table("fire_observations")
    op.drop_column("forecast_artifacts", "metadata_json")
    op.drop_column("forecast_artifacts", "status")
    op.drop_index("ix_forecast_runs_is_oof", table_name="forecast_runs")
    op.drop_column("forecast_runs", "metadata_json")
    op.drop_column("forecast_runs", "time_safety_status")
    op.drop_column("forecast_runs", "channel_manifest_sha256")
    op.drop_column("forecast_runs", "protocol")
    op.drop_column("forecast_runs", "is_oof")
    op.drop_index("ix_wildfire_events_complex_group_id", table_name="wildfire_events")
    op.drop_index("ix_wildfire_events_year", table_name="wildfire_events")
    op.drop_column("wildfire_events", "manifest_sha256")
    op.drop_column("wildfire_events", "p2_fold_id")
    op.drop_column("wildfire_events", "p1_fold_id")
    op.drop_column("wildfire_events", "complex_group_id")
    op.drop_column("wildfire_events", "year")
