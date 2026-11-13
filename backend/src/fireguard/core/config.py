from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FIREGUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "FireGuard API"
    app_env: str = "development"
    app_version: str = "0.2.0"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    database_url: str = Field(
        default="sqlite:///./fireguard_dev.db",
        validation_alias="DATABASE_URL",
    )
    firms_map_key: str | None = Field(default=None, validation_alias="FIRMS_MAP_KEY")
    firms_base_url: str = "https://firms.modaps.eosdis.nasa.gov"
    firms_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    firms_observation_cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)
    firms_availability_cache_ttl_seconds: int = Field(default=1800, ge=0, le=86400)

    artifact_root: Path = Path("artifacts")
    replay_session_ttl_seconds: int = Field(default=900, ge=60, le=86400)
    event_catalogue_page_limit: int = Field(default=500, ge=1, le=5000)
    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
