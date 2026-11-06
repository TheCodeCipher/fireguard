
from datetime import UTC, datetime

from fastapi import APIRouter

from fireguard.core.config import get_settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict:
    settings = get_settings()

    return {
        "status": "healthy",
        "application": settings.app_name,
        "environment": settings.app_env,
        "version": settings.app_version,
        "timestamp": datetime.now(UTC).isoformat(),
    }
