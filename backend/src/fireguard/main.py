
from fastapi import FastAPI

from fireguard.api.routes import firms, health
from fireguard.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Backend API for the FireGuard wildfire intelligence platform.",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(
    health.router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    firms.router,
    prefix=settings.api_v1_prefix,
)


@app.get("/", tags=["Root"])
async def root() -> dict:
    return {
        "message": "Welcome to the FireGuard API",
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
        "firms_observations": f"{settings.api_v1_prefix}/firms/observations",
    }
