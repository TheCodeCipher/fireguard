"""Add Phase 3 artifact lifecycle audit support.

Revision ID: c31f6a2b9d40
Revises: b6c2d4e91a20
"""
from alembic import op
import sqlalchemy as sa

revision = "c31f6a2b9d40"
down_revision = "b6c2d4e91a20"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Phase 2 used `available`; existing rows are unverified, so demote them
    # rather than accidentally exposing them as published research artifacts.
    op.execute("UPDATE forecast_artifacts SET status = 'generated' WHERE status = 'available'")
    with op.batch_alter_table("forecast_artifacts") as batch_op:
        batch_op.alter_column(
            "status",
            existing_type=sa.String(length=30),
            existing_nullable=False,
            server_default="generated",
        )
    op.add_column(
        "replay_sessions",
        sa.Column("audit_json", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("replay_sessions", "audit_json")
    # Do not promote unvalidated `generated` artifacts back to Phase 2's
    # replay-visible `available` state. A downgrade must remain fail-closed.
    with op.batch_alter_table("forecast_artifacts") as batch_op:
        batch_op.alter_column(
            "status",
            existing_type=sa.String(length=30),
            existing_nullable=False,
            server_default="available",
        )
