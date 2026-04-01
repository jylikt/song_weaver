"""
Application configuration loaded from environment variables.
All settings have sensible defaults so the service can start without a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from enum import Enum


class GeneratorProvider(str, Enum):
    LOCAL = "local"
    REMOTE_GPU = "remote_gpu"


class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: str = "INFO"
    port: int = Field(default=8000, alias="PORT")

    # ── Generator selection ──────────────────────────────────────────────────
    generator_provider: GeneratorProvider = GeneratorProvider.LOCAL

    # ── Output storage ───────────────────────────────────────────────────────
    output_dir: str = "generated"

    # ── Remote GPU worker ────────────────────────────────────────────────────
    # URL of the gpu-worker service. Must end without a trailing slash.
    # Example: http://localhost:9000   (local development)
    #          https://gpu.mycompany.com  (production)
    remote_worker_url: str = "http://localhost:9000"

    # Bearer token sent to the worker in: Authorization: Bearer <token>
    # Must match WORKER_TOKEN set in the gpu-worker's environment.
    # Leave blank when the worker has auth disabled (development only).
    remote_worker_token: str = ""

    # Seconds to wait for the worker to respond before giving up.
    remote_worker_timeout_sec: int = 300


# Single shared settings instance used across the application
settings = Settings()
