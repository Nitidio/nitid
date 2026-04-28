from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 8

    database_url: str = "sqlite:///./web/storage/nitid.db"
    models_dir: str = "./models"
    uploads_dir: str = "./web/storage/uploads"
    results_dir: str = "./web/storage/results"

    model_config = SettingsConfigDict(env_prefix="NITID_", env_file=".env", extra="ignore")


settings = Settings()
