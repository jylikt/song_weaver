"""
Global worker state — YuE model lifecycle and GPU telemetry.

This module owns the single source of truth for:
  - the loaded YuE model instance and tokenizer/processor
  - model lifecycle metadata (loaded/unloaded, timestamps, generation count)
  - live GPU telemetry (VRAM, device name, temperature)

All mutations go through async methods that hold self.lock, so concurrent
/generate or /load-model requests cannot corrupt state. On a multi-GPU or
multi-process deployment, replace this in-process lock with Redis or a
shared-memory backend.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YUE INTEGRATION POINTS IN THIS FILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  load_model()   — import and load YuE weights into VRAM
  unload_model() — delete references and call torch.cuda.empty_cache()
  _refresh_gpu_telemetry() — reads real VRAM/temperature via torch.cuda
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
    loaded_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    generation_count: int = 0
    load_duration_sec: float = 0.0


# ─── Worker state singleton ────────────────────────────────────────────────────

class WorkerState:
    """
    Thread-safe holder for the live YuE model and all mutable worker state.

    Attributes set by load_model() and accessed by _run_inference():
        _model      : the loaded YuE / HuggingFace model object
        _processor  : the matching tokenizer / audio processor
        _device     : torch.device the model lives on
        _torch_dtype: dtype used during inference (float16 / bfloat16 / float32)
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.model = ModelState()
        self.gpu = GpuStatus()
        self._started_at: datetime = datetime.utcnow()

        # Populated by load_model(); consumed by _run_inference() in generate.py
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._device: Optional[Any] = None        # torch.device
        self._torch_dtype: Optional[Any] = None   # torch dtype

    @property
    def uptime_seconds(self) -> float:
        return (datetime.utcnow() - self._started_at).total_seconds()

    # ── Public lifecycle API ───────────────────────────────────────────────────

    async def load_model(self, model_name: str) -> None:
        """
        Load the YuE model into GPU (or CPU) memory.

        If YUE_MODEL_PATH is not set, falls back to stub mode so the worker
        can be exercised in development without a GPU or model checkpoint.

        The asyncio lock is held for the duration of loading so concurrent
        /generate calls block cleanly rather than racing.
        """
        from app.config import settings

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
                self.model.loaded = True
                self.model.stub_mode = True
                self.model.model_name = model_name
                self.model.loaded_at = datetime.utcnow()
                return

            # ── Real YuE model loading ─────────────────────────────────────────
            t_start = time.perf_counter()
            logger.info(
                "Loading YuE model | path=%s | device=%s | dtype=%s",
                model_path,
                settings.yue_device,
                settings.yue_dtype,
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

            # ── Resolve dtype ──────────────────────────────────────────────────
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

            # ── Load YuE ──────────────────────────────────────────────────────
            #
            # YuE (M-A-P) can be loaded via the HuggingFace Transformers API.
            # Adjust the import and class names to match the version you have
            # installed.  Three common patterns:
            #
            # Pattern A — Official YuE pip package:
            #   from yue.models import YuEForConditionalGeneration
            #   from yue.tokenization import YuETokenizer
            #   model = YuEForConditionalGeneration.from_pretrained(
            #       model_path, torch_dtype=torch_dtype
            #   )
            #   processor = YuETokenizer.from_pretrained(model_path)
            #
            # Pattern B — HuggingFace AutoModel (if YuE is published on HF Hub):
            #   from transformers import AutoModelForSeq2SeqLM, AutoProcessor
            #   model = AutoModelForSeq2SeqLM.from_pretrained(
            #       model_path, torch_dtype=torch_dtype
            #   )
            #   processor = AutoProcessor.from_pretrained(model_path)
            #
            # Pattern C — YuE source checkout (clone the repo alongside this worker):
            #   import sys
            #   sys.path.insert(0, "/opt/YuE")           # path to cloned repo
            #   from models.yue import YuEPipeline
            #   model = YuEPipeline.from_pretrained(model_path, device=device, dtype=torch_dtype)
            #   processor = model.processor                # pipeline owns the processor
            #
            # ─────────────────────────────────────────────────────────────────
            # ACTIVE INTEGRATION (update to match your installed package):
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoProcessor

                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
                model = model.to(device)
                model.eval()

                processor = AutoProcessor.from_pretrained(model_path)

            except ImportError as exc:
                raise RuntimeError(
                    "transformers is not installed. Run: pip install transformers accelerate"
                ) from exc
            except OSError as exc:
                raise RuntimeError(
                    f"Could not load YuE checkpoint from '{model_path}'. "
                    f"Check that YUE_MODEL_PATH points to a valid directory or HF repo id. "
                    f"Original error: {exc}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Unexpected error loading YuE model: {exc}"
                ) from exc

            # ── Store references on state ──────────────────────────────────────
            self._model = model
            self._processor = processor
            self._device = device
            self._torch_dtype = torch_dtype

            elapsed = time.perf_counter() - t_start

            # ── Update model metadata ──────────────────────────────────────────
            self.model.loaded = True
            self.model.stub_mode = False
            self.model.model_name = model_name
            self.model.loaded_at = datetime.utcnow()
            self.model.load_duration_sec = round(elapsed, 2)

            # ── Refresh GPU telemetry ──────────────────────────────────────────
            self._refresh_gpu_telemetry()

            logger.info(
                "YuE model ready | model=%s | device=%s | dtype=%s | load_time=%.1fs",
                model_name,
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

            # ── Release model references ──────────────────────────────────────
            # Python will GC the model once all references are dropped.
            # torch.cuda.empty_cache() then releases the freed VRAM back to the
            # OS so it's available for the next model load.
            if self._model is not None:
                del self._model
                self._model = None
            if self._processor is not None:
                del self._processor
                self._processor = None
            self._device = None
            self._torch_dtype = None

            # ── Clear CUDA cache ──────────────────────────────────────────────
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("CUDA cache cleared after unload")
            except ImportError:
                pass  # torch not installed — stub mode

            # ── Reset state ────────────────────────────────────────────────────
            self.model.loaded = False
            self.model.stub_mode = False
            self.model.model_name = None
            self.model.loaded_at = None
            self.gpu.vram_used_gb = 0.0

            logger.info("Model unloaded: %s", model_name)

    async def record_generation(self) -> None:
        """
        Update counters and refresh GPU telemetry after a completed generation.
        Called by the /generate route after _run_inference() returns.
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

        Called inside the lock from load_model() and record_generation().
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
                # torch.cuda does not expose temperature directly.
                # To get real temperatures, call nvidia-smi via subprocess:
                #   result = subprocess.run(
                #       ["nvidia-smi", "--query-gpu=temperature.gpu",
                #        "--format=csv,noheader,nounits"],
                #       capture_output=True, text=True
                #   )
                #   self.gpu.temperature_c = int(result.stdout.strip())
                self.gpu.temperature_c = 0  # replace with nvidia-smi call above
        except (ImportError, RuntimeError):
            # CUDA not available or torch not installed
            self.gpu.available = False
            self.gpu.device = "cpu"


# ── Singleton ─────────────────────────────────────────────────────────────────
worker_state = WorkerState()
