
from functools import lru_cache

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
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    api_v1_prefix: str = "/api/v1"

    database_url: str = Field(
        default="sqlite:///./fireguard_dev.db",
        validation_alias="DATABASE_URL",
    )

    firms_map_key: str | None = Field(
        default=None,
        validation_alias="FIRMS_MAP_KEY",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
