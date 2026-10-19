"""Centralized, type-safe configuration using Pydantic Settings."""
from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_name: str = "ML Prediction API"
    app_version: str = "2.0.0"
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    log_level: str = "INFO"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 2
    reload: bool = False

    # --- Model ---
    model_path: Path = Path("models/model.pkl")
    expected_feature_count: int = 4
    max_batch_size: int = 1000

    # --- CORS ---
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()