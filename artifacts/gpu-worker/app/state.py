"""
Global worker state — YuE model lifecycle and GPU telemetry.

This module owns the single source of truth for:
  - the loaded YuE model instance and tokenizer/processor
  - model lifecycle metadata (loaded/unloaded, timestamps, generation count)
  - live GPU telemetry (VRAM, device name, temperature)

All mutations go through async methods that hold self.lock, so concurrent
/generate or /load-model requests cannot corrupt state.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YUE INTEGRATION POINTS IN THIS FILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  load_model()   — architecture-aware YuE weight loading into VRAM
  unload_model() — delete references and call torch.cuda.empty_cache()
  _refresh_gpu_telemetry() — reads real VRAM/temperature via torch.cuda

Architecture detection uses AutoConfig to inspect the checkpoint before
loading any weights. LlamaConfig (and other decoder-only configs) route
to AutoModelForCausalLM; seq2seq configs route to AutoModelForSeq2SeqLM.
Override with YUE_MODEL_FAMILY env var if auto-detection is wrong.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Telemetry dataclasses ─────────────────────────────────────────────────────

@dataclass
class GpuStatus:
    available: bool = False
    device: str = "cpu"
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    temperature_c: int = 0

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "device": self.device,
            "vram_total_gb": self.vram_total_gb,
            "vram_used_gb": self.vram_used_gb,
            "temperature_c": self.temperature_c,
        }


@dataclass
class ModelState:
    loaded: bool = False
    model_name: Optional[str] = None
    stub_mode: bool = False      # True when YuE is not installed / path not configured
    model_family: Optional[str] = None  # 'causal', 'seq2seq', or None in stub mode
    loaded_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    generation_count: int = 0
    load_duration_sec: float = 0.0


# ─── Worker state singleton ────────────────────────────────────────────────────

class WorkerState:
    """
    Thread-safe holder for the live YuE model and all mutable worker state.

    Attributes set by load_model() and accessed by _run_inference():
        _model        : the loaded YuE / HuggingFace model object
        _processor    : the matching tokenizer / audio processor
        _device       : torch.device the model lives on
        _torch_dtype  : dtype used during inference (float16 / bfloat16 / float32)
        _model_family : resolved architecture family ('causal' or 'seq2seq')
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.model = ModelState()
        self.gpu = GpuStatus()
        self._started_at: datetime = datetime.utcnow()

        # Populated by load_model(); consumed by _run_inference() in generate.py
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._codec: Optional[Any] = None          # audio codec (xcodec_mini_infer)
        self._codec_n_codebooks: Optional[int] = None  # detected from codec; overrides config
        self._device: Optional[Any] = None         # torch.device
        self._torch_dtype: Optional[Any] = None    # torch dtype
        self._model_family: Optional[str] = None   # 'causal' | 'seq2seq'

    @property
    def uptime_seconds(self) -> float:
        return (datetime.utcnow() - self._started_at).total_seconds()

    # ── Public lifecycle API ───────────────────────────────────────────────────

    async def load_model(self, model_name: str) -> None:
        """
        Load the YuE model into GPU (or CPU) memory using architecture-aware
        loader selection.

        Steps:
          1. If YUE_MODEL_PATH is unset → stub mode (silent WAV, no weights).
          2. Resolve device and dtype from settings.
          3. Detect model family via AutoConfig (or use YUE_MODEL_FAMILY override).
          4. Load weights with the correct loader class.
          5. Store references and update metadata.

        Raises RuntimeError with actionable messages on any failure.
        """
        from app.config import ModelFamily, settings
        from app.yue_adapter import detect_model_family, load_codec, load_model_and_processor

        async with self.lock:
            model_path = settings.yue_model_path or ""

            if not model_path:
                # ── Stub mode — no checkpoint configured ─────────────────────
                logger.warning(
                    "YUE_MODEL_PATH is not set — running in stub mode. "
                    "Generation will produce silent WAV files."
                )
                self._model = None
                self._processor = None
                self._model_family = None
                self.model.loaded = True
                self.model.stub_mode = True
                self.model.model_family = None
                self.model.model_name = model_name
                self.model.loaded_at = datetime.utcnow()
                return

            # ── Real YuE model loading ─────────────────────────────────────────
            t_start = time.perf_counter()
            logger.info(
                "Loading YuE model | path=%s | device=%s | dtype=%s | trust_remote_code=%s",
                model_path,
                settings.yue_device,
                settings.yue_dtype,
                settings.yue_trust_remote_code,
            )

            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "PyTorch is not installed. Run: pip install torch torchaudio"
                ) from exc

            # ── Resolve device ────────────────────────────────────────────────
            requested_device = settings.yue_device.lower()
            if requested_device == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    "YUE_DEVICE=cuda but CUDA is unavailable — falling back to CPU. "
                    "Check your NVIDIA driver and CUDA installation."
                )
                requested_device = "cpu"
            device = torch.device(requested_device)

            # ── Resolve dtype ─────────────────────────────────────────────────
            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            torch_dtype = dtype_map.get(settings.yue_dtype, torch.float16)
            if device.type == "cpu" and torch_dtype != torch.float32:
                logger.warning(
                    "CPU device does not support %s — switching to fp32.", settings.yue_dtype
                )
                torch_dtype = torch.float32

            # ── Resolve model family ───────────────────────────────────────────
            family_override = settings.yue_model_family
            if family_override != ModelFamily.AUTO:
                family = family_override.value
                logger.info(
                    "Model family override via YUE_MODEL_FAMILY: %s", family
                )
            else:
                family = detect_model_family(
                    model_path,
                    trust_remote_code=settings.yue_trust_remote_code,
                )
                if family == "unknown":
                    raise RuntimeError(
                        f"Cannot determine model family for checkpoint '{model_path}'. "
                        f"The detected config class is not in any known architecture mapping. "
                        f"Set YUE_MODEL_FAMILY=causal or YUE_MODEL_FAMILY=seq2seq to override."
                    )

            logger.info("Resolved model family: %s", family)

            # ── Load weights ──────────────────────────────────────────────────
            model, processor = load_model_and_processor(
                model_path=model_path,
                family=family,
                device=device,
                torch_dtype=torch_dtype,
                trust_remote_code=settings.yue_trust_remote_code,
            )

            # ── Reset meta-device leak from LLM loading ───────────────────────
            # from_pretrained with low_cpu_mem_usage=True leaves
            # torch.set_default_device('meta') set globally on some accelerate
            # versions. Reset it now so the codec loading is not affected.
            from app.yue_adapter import _reset_default_device
            _reset_default_device()

            # ── Store references ──────────────────────────────────────────────
            self._model = model
            self._processor = processor
            self._device = device
            self._torch_dtype = torch_dtype
            self._model_family = family

            # ── Load audio codec (xcodec_mini_infer, needed for YuE decoding) ──
            # load_codec() returns None on failure and logs a clear error — the
            # worker stays functional in stub-style degraded mode.
            if family == "causal" and settings.yue_codec_path:
                self._codec = load_codec(
                    codec_path=settings.yue_codec_path,
                    device=device,
                    hf_token=settings.yue_hf_token,
                    n_codebooks=settings.yue_codec_n_codebooks,
                    sample_rate=settings.yue_sample_rate,
                    codec_samples_per_frame=settings.yue_codec_samples_per_frame,
                    xcodec_tokens_fps=settings.yue_xcodec_tokens_fps,
                )
            else:
                self._codec = None

            # ── Detect actual codec quantizer count ──────────────────────────
            # The codec's RVQ/FSQ may have a different number of quantizer
            # stages than yue_codec_n_codebooks. Read it from the model's
            # internal quantizer structure so that both the token budget and
            # the decode reshape use the correct value.
            self._codec_n_codebooks = None
            if self._codec is not None:
                from app.yue_adapter import detect_codec_n_quantizers
                detected = detect_codec_n_quantizers(self._codec)
                if detected is not None:
                    logger.info(
                        "Codec n_quantizers detected from model: %d "
                        "(config yue_codec_n_codebooks=%d%s)",
                        detected,
                        settings.yue_codec_n_codebooks,
                        "" if detected == settings.yue_codec_n_codebooks
                        else " — will use detected value",
                    )
                    self._codec_n_codebooks = detected
                else:
                    logger.info(
                        "Using yue_codec_n_codebooks=%d (auto-detect skipped or unavailable; "
                        "SoundStream repo uses mm tokenizer offsets — see yue_mm_xcodec_*).",
                        settings.yue_codec_n_codebooks,
                    )

            elapsed = time.perf_counter() - t_start

            self.model.loaded = True
            self.model.stub_mode = False
            self.model.model_family = family
            self.model.model_name = model_name
            self.model.loaded_at = datetime.utcnow()
            self.model.load_duration_sec = round(elapsed, 2)

            self._refresh_gpu_telemetry()

            logger.info(
                "YuE model ready | model=%s | family=%s | device=%s | dtype=%s | load_time=%.1fs",
                model_name,
                family,
                device,
                settings.yue_dtype,
                elapsed,
            )

    async def unload_model(self) -> None:
        """
        Release the YuE model from memory and clear the CUDA cache.
        Safe to call even if no model is loaded.
        """
        async with self.lock:
            if not self.model.loaded:
                return

            model_name = self.model.model_name
            logger.info("Unloading model: %s", model_name)

            if self._model is not None:
                del self._model
                self._model = None
            if self._processor is not None:
                del self._processor
                self._processor = None
            if self._codec is not None:
                del self._codec
                self._codec = None
            self._codec_n_codebooks = None
            self._device = None
            self._torch_dtype = None
            self._model_family = None

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("CUDA cache cleared after unload")
            except ImportError:
                pass

            self.model.loaded = False
            self.model.stub_mode = False
            self.model.model_family = None
            self.model.model_name = None
            self.model.loaded_at = None
            self.gpu.vram_used_gb = 0.0

            logger.info("Model unloaded: %s", model_name)

    async def record_generation(self) -> None:
        """
        Update counters and refresh GPU telemetry after a completed generation.
        Called by the /generate route after inference returns.
        """
        async with self.lock:
            self.model.last_used_at = datetime.utcnow()
            self.model.generation_count += 1
            self._refresh_gpu_telemetry()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _refresh_gpu_telemetry(self) -> None:
        """
        Update self.gpu with real VRAM usage and device info from torch.cuda.
        Falls back to zeroed-out values when CUDA is unavailable.
        """
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self.gpu.available = True
                self.gpu.device = props.name
                self.gpu.vram_total_gb = round(props.total_memory / 1e9, 1)
                self.gpu.vram_used_gb = round(
                    torch.cuda.memory_allocated(0) / 1e9, 2
                )
                # For real GPU temperatures, replace with:
                #   result = subprocess.run(
                #       ["nvidia-smi", "--query-gpu=temperature.gpu",
                #        "--format=csv,noheader,nounits"],
                #       capture_output=True, text=True)
                #   self.gpu.temperature_c = int(result.stdout.strip())
                self.gpu.temperature_c = 0
        except (ImportError, RuntimeError):
            self.gpu.available = False
            self.gpu.device = "cpu"


# ── Singleton ─────────────────────────────────────────────────────────────────
worker_state = WorkerState()
