from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from fireguard.core.config import get_settings
from fireguard.core.database import get_db

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check(db: Annotated[Session, Depends(get_db)]) -> dict:
    settings = get_settings()
    database_status = "healthy"
    try:
        db.execute(text("SELECT 1"))
    except Exception:  # health endpoint reports degraded status; do not leak DB details
        database_status = "unavailable"
    overall = "healthy" if database_status == "healthy" else "degraded"
    return {
        "status": overall,
        "application": settings.app_name,
        "environment": settings.app_env,
        "version": settings.app_version,
        "timestamp": datetime.now(UTC).isoformat(),
        "dependencies": {"database": database_status},
    }
