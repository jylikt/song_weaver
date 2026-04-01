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


class ModelFamily(str, Enum):
    AUTO = "auto"
    CAUSAL = "causal"
    SEQ2SEQ = "seq2seq"
    PIPELINE = "pipeline"


class GenerationBackend(str, Enum):
    TRANSFORMERS = "transformers"
    YUE_NATIVE = "yue_native"


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
    # Example: HKUSTAudio/YuE-s1-7B-anneal-en-icl
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
    # YuE-s1-7B uses 24000 Hz.
    yue_sample_rate: int = 24000

    # Number of inference steps (higher = better quality, slower).
    yue_num_steps: int = 50

    # Classifier-free guidance scale. Higher values follow the prompt
    # more strictly. Typical range: 1.5–5.0 for music generation.
    yue_cfg_scale: float = 3.0

    # Timeout for a single generation call in seconds.
    # Song-gen's REMOTE_WORKER_TIMEOUT_SEC should be set higher than this.
    generation_timeout_sec: int = 240

    # ── Codec configuration ───────────────────────────────────────────────────
    # XCodec2 model used to decode the audio codec tokens produced by the LLM.
    # HKUSTAudio/YuE uses m-a-p/xcodec2. Set to a local path if already
    # downloaded, or leave as the HF repo id to download on first use.
    # Set to empty string to disable codec loading (generation will fall back
    # to a silent WAV with a clear error in the logs).
    yue_codec_path: str = "m-a-p/xcodec2"

    # Number of RVQ codebooks in the codec (8 for standard xcodec2).
    yue_codec_n_codebooks: int = 8

    # ── Architecture / backend overrides ─────────────────────────────────────
    # Set to true if the checkpoint requires custom code (e.g. HKUSTAudio/YuE).
    # When AUTO, the worker will attempt to detect this from the model config.
    yue_trust_remote_code: bool = False

    # Force a specific model family instead of auto-detecting from config.
    # auto   — detect from AutoConfig (recommended)
    # causal — force AutoModelForCausalLM (Llama-based YuE checkpoints)
    # seq2seq — force AutoModelForSeq2SeqLM
    # pipeline — force pipeline() API
    yue_model_family: ModelFamily = ModelFamily.AUTO

    # Select which generation backend to use.
    # transformers — use HuggingFace transformers generate() API
    # yue_native   — use the YuE package's own pipeline (requires yue installed)
    yue_generation_backend: GenerationBackend = GenerationBackend.TRANSFORMERS


settings = Settings()
