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


class CausalMmCapWindow(str, Enum):
    """Which part of the mm codec stream to keep when longer than requested."""

    HEAD = "head"  # first N interleaved tokens (chronological intro)
    TAIL = "tail"  # last N tokens — often less empty intro on causal generate()


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
    # xcodec_mini_infer — the audio codec YuE was designed for.
    # It has 8 RVQ codebooks, a 24 kHz sample rate, and is publicly available.
    # Priority: local path > HF repo id.
    # Examples:
    #   /root/xcodec_mini_infer   (pre-downloaded local snapshot — fastest)
    #   m-a-p/xcodec_mini_infer   (HuggingFace repo, requires internet)
    # Leave empty to skip codec loading (generations fall back to silent WAV).
    yue_codec_path: str = "m-a-p/xcodec_mini_infer"

    # HuggingFace token for gated/private model repos (codec or LLM).
    # Set via HF_TOKEN env var or here. Leave blank for public repos.
    yue_hf_token: str = ""

    # Number of RVQ codebooks in the codec.
    # xcodec_mini_infer uses 8 codebooks (same as the original xcodec2 config).
    # The worker auto-detects this from the loaded model and overrides this value
    # when the detected count differs — so this is only the fallback default.
    yue_codec_n_codebooks: int = 8

    # ── Audio token extraction ────────────────────────────────────────────────
    # YuE mmtokenizer v0.2 (YuE inference/codecmanipulator.py):
    #   text 0–31999, specials 32000–32021, then per-codec ranges.
    # xcodec uses 1024 entries per codebook; tokenizer IDs are NOT (id - 32000).
    # Flat interleave order matches (time × codebook): t0_cb0…t0_cb{K-1}, t1_cb0…
    # Defaults match CodecManipulator("xcodec"): global_offset 45334, K=8 for stage-2 decode.
    yue_text_vocab_size: int = 32000
    yue_mm_xcodec_global_offset: int = 45334
    yue_mm_xcodec_codebook_size: int = 1024
    # Frames per second of **audio** implied by the codec (used for token budget
    # and duration cap). YuE mm tokenizer often uses ~50 steps/s, but m-a-p
    # SoundStream in the research xcodec repo is ~75 frames/s at 24 kHz
    # (~320 samples/frame). When 0, the worker probes SoundStream.decode;
    # otherwise this value is used if probing fails or for non-repo codecs.
    yue_xcodec_tokens_fps: int = 50
    # Override samples per codec time-step for duration math (0 = auto-probe
    # for _XCodecRepoWrapper only). Example: 320 for m-a-p SoundStream @ 24 kHz.
    yue_codec_samples_per_frame: int = 0
    # Inflate max_new_tokens for causal mm path (not every token is xcodec ID).
    # Output length is still capped to requested duration (tokens + waveform trim).
    yue_causal_mm_budget_multiplier: float = 1.45
    # After decode, trim WAV to duration_sec × sample_rate (fixes 10s request → 15s file).
    yue_trim_waveform_to_request_duration: bool = True
    # When generated mm tokens exceed the duration cap: keep first (head) or last
    # (tail) window. Tail often reduces long low-energy intros on built-in causal.
    yue_causal_mm_cap_window: CausalMmCapWindow = CausalMmCapWindow.TAIL

    # ── Native YuE subprocess inference ──────────────────────────────────────
    # Path to a cloned https://github.com/multimodal-art-projection/YuE repo root.
    # When set together with yue_model_path, the worker runs inference/infer.py
    # (official pipeline). Install YuE env + xcodec_mini_infer per upstream README.
    # Example: /root/YuE
    # Leave blank to use the built-in transformers inference path.
    yue_repo_path: str = ""
    # Arguments passed to official infer.py (see YuE README).
    yue_native_stage2_model: str = "m-a-p/YuE-s2-1B-general"
    yue_native_run_n_segments: int = 2
    yue_native_stage2_batch_size: int = 4
    # Directory whose *child* is ``models/`` (infer.py: ``from models...``).
    # Default: YuE/inference/xcodec_mini_infer after ``git clone m-a-p/xcodec_mini_infer``.
    # If the codec lives elsewhere (e.g. research repo /root/xcodec), set this path.
    yue_native_xcodec_mini_path: str = ""
    # Passed to native infer.py subprocess (Transformers). Use ``sdpa`` when
    # flash-attn is not installed (avoids ImportError on FlashAttention2).
    # Empty = do not set (checkpoint may request flash_attention_2).
    yue_native_attn_implementation: str = "sdpa"

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
