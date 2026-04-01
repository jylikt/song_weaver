"""
GPU Worker configuration — loaded from environment variables / .env file.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Process ───────────────────────────────────────────────────────────────
    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: str = "INFO"
    port: int = Field(default=9000, alias="PORT")

    # ── Auth ──────────────────────────────────────────────────────────────────
    # If empty, authentication is skipped (development only).
    worker_token: str = ""

    # ── Model defaults ────────────────────────────────────────────────────────
    # Used when /load-model is called without a model_name body.
    default_model_name: str = "yue-base"

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "output"

    # ── YuE model configuration ───────────────────────────────────────────────
    # Local filesystem path (or HuggingFace repo id) to the YuE checkpoint.
    # Example: /models/yue-s2-lyric  or  m-a-p/YuE-s2-lyric-audiovae-24k-v1.1
    # Leave blank to run in stub mode (silent WAV, no GPU required).
    yue_model_path: str = ""

    # Compute device. Set to "cuda" on a GPU server, "cpu" for testing.
    yue_device: str = "cuda"

    # Floating-point precision for the model.
    # fp16 is standard for inference on NVIDIA GPUs.
    # Use fp32 when running on CPU.
    yue_dtype: str = "fp16"

    # Maximum generation duration the worker will accept (hard cap).
    yue_max_duration_sec: int = 300

    # Sample rate of the YuE audio output. Must match the checkpoint.
    # YuE-s2-lyric uses 24000 Hz.
    yue_sample_rate: int = 24000

    # Number of inference steps (higher = better quality, slower).
    yue_num_steps: int = 50

    # Classifier-free guidance scale. Higher values follow the prompt
    # more strictly. Typical range: 1.5–5.0 for music generation.
    yue_cfg_scale: float = 3.0

    # Timeout for a single generation call in seconds.
    # Song-gen's REMOTE_WORKER_TIMEOUT_SEC should be set higher than this.
    generation_timeout_sec: int = 240


settings = Settings()
