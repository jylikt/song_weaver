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
    # XCodec2 codec — decodes audio codec tokens produced by the YuE LLM.
    # Priority: local path > HF repo id.
    # Examples:
    #   /root/xcodec2          (pre-downloaded local directory — recommended)
    #   HKUSTAudio/xcodec2     (HuggingFace repo, requires internet + optional HF_TOKEN)
    # Leave empty to skip codec loading (generations fall back to silent WAV).
    yue_codec_path: str = "HKUSTAudio/xcodec2"

    # HuggingFace token for gated/private model repos (codec or LLM).
    # Set via HF_TOKEN env var or here. Leave blank for public repos.
    yue_hf_token: str = ""

    # Number of RVQ codebooks in the codec (8 for standard xcodec2).
    yue_codec_n_codebooks: int = 8

    # ── Audio token extraction ────────────────────────────────────────────────
    # YuE extends the Llama-2 vocabulary with audio codec tokens.
    # Text tokens occupy IDs [0, yue_text_vocab_size).
    # Audio codec tokens occupy IDs [yue_text_vocab_size, full_vocab_size).
    #
    # IMPORTANT: tokenizer.vocab_size returns the FULL extended vocabulary
    # (e.g. 83734) which includes audio tokens — do NOT use it as this threshold.
    # The correct value is the original Llama-2 text vocabulary size: 32000.
    yue_text_vocab_size: int = 32000

    # ── Native YuE subprocess inference ──────────────────────────────────────
    # Set to the path of a cloned HKUSTAudio/YuE GitHub repository.
    # When set, the worker calls infer.py from the repo as a subprocess instead
    # of the built-in transformers path. This is the most accurate and fastest
    # inference method as it uses the official YuE pipeline directly.
    # Example: /root/YuE
    # Leave blank to use the built-in transformers inference path.
    yue_repo_path: str = ""

    # ── Architecture / backend overrides ─────────────────────────────────────
    # Set to true if the checkpoint requires custom code (e.g. HKUSTAudio/YuE).
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
