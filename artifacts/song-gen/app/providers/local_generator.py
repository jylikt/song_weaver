"""
Local music generator provider.

This provider runs inference directly inside the main FastAPI process.
It is currently a STUB — it writes a silent WAV file so the full job
pipeline (queue, status polling, result metadata, UI audio player) can
be exercised end-to-end without any GPU or model dependency.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTEGRATION GUIDE — plugging in a real open-source model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Install your model's dependencies in requirements.txt:

    audiocraft          # Meta's MusicGen / AudioGen
    bark                # Suno's Bark (text-to-audio)
    torch torchaudio    # Required by both

STEP 2 — Load the model once in __init__ (not per request):

    from audiocraft.models import MusicGen
    self._model = MusicGen.get_pretrained("medium")
    # Options: "small" (300M), "medium" (1.5B), "large" (3.3B), "melody"
    # For lyrics-conditioned output, load a fine-tuned checkpoint here:
    #   self._model = MusicGen.get_pretrained("/models/lyrics-finetuned-v1")

STEP 3 — Replace _run_model() with real inference (see the method below).

STEP 4 — Fine-tuning / task-specific adaptation:
    The style_preset and lyrics fields provide structured supervision
    signals. To fine-tune on your own dataset:
    - Export past job requests (prompt + lyrics + style_preset) as a corpus.
    - Fine-tune using AudioCraft's train.py or a LoRA wrapper.
    - Point self._model to your checkpoint path in __init__.
    - No other code changes are needed — the job pipeline, API, and UI
      all remain the same.

STEP 5 — Adding a second local provider variant:
    Create a new class (e.g. BarkMusicGenerator) in this folder,
    subclass BaseMusicGenerator, and register it in factory.py.
    Per-request mode routing is already in place.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import logging
import struct
import wave
from pathlib import Path

from app.models.generation import GenerationRequest
from app.providers.base import BaseMusicGenerator, GenerationError

logger = logging.getLogger(__name__)


class LocalMusicGenerator(BaseMusicGenerator):
    """
    Runs music generation in the local process.

    Currently a stub — replace `_run_model` with real inference.
    The rest of the pipeline (job queue, status polling, UI) is fully functional.
    """

    def __init__(self) -> None:
        # ── TODO: load your model here (runs once, cached via lru_cache) ────
        # Example:
        #   from audiocraft.models import MusicGen
        #   self._model = MusicGen.get_pretrained("medium")
        #   self._model.set_generation_params(duration=30)
        #
        # For a fine-tuned lyrics-conditioned model:
        #   self._model = MusicGen.get_pretrained("/models/lyrics-finetuned-v1")
        # ─────────────────────────────────────────────────────────────────────
        logger.info("LocalMusicGenerator initialised (stub mode — no real model loaded)")

    async def generate_song(
        self,
        request: GenerationRequest,
        output_dir: Path,
        job_id: str,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}.wav"

        logger.info(
            "LocalMusicGenerator: starting generation",
            extra={
                "job_id": job_id,
                "style_preset": request.style_preset.value,
                "duration_sec": request.duration_sec,
                "seed": request.seed,
            },
        )

        # Run the (potentially blocking) model call in a thread pool so we
        # don't block the event loop during inference.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._run_model, request, output_path)

        logger.info(
            "LocalMusicGenerator: generation complete",
            extra={"output": str(output_path)},
        )
        return output_path

    # ─────────────────────────────────────────────────────────────────────────
    # Inference — replace this method with real model inference
    # ─────────────────────────────────────────────────────────────────────────

    def _run_model(self, request: GenerationRequest, output_path: Path) -> None:
        """
        Blocking model call — runs inside a thread pool executor so the
        async event loop is not blocked during generation.

        ══════════════════════════════════════════════════════════════════════
        STUB IMPLEMENTATION
        Currently writes a silent WAV file.
        Replace the entire body with real model inference.
        ══════════════════════════════════════════════════════════════════════

        HOW TO REPLACE — AudioCraft MusicGen example:

            import torch, torchaudio

            # ── Reproducibility ───────────────────────────────────────────
            if request.seed is not None:
                torch.manual_seed(request.seed)

            # ── Set generation parameters ─────────────────────────────────
            self._model.set_generation_params(
                duration=request.duration_sec,
                top_k=250,
                top_p=0.0,
                temperature=1.0,
                cfg_coef=3.0,   # classifier-free guidance strength
            )

            # ── Lyrics-conditioned generation (fine-tuned model) ──────────
            # TODO: This path requires a model fine-tuned for lyrics input.
            # Use this once you have trained/loaded a lyrics-aware checkpoint.
            wav = self._model.generate_with_lyrics(
                descriptions=[f"{request.prompt} | style: {request.style_preset.value}"],
                lyrics=[request.lyrics],
                progress=True,
            )

            # ── Prompt-only generation (base MusicGen, no fine-tuning) ────
            # Use this for out-of-the-box MusicGen without lyrics conditioning.
            # wav = self._model.generate([request.prompt], progress=True)

            # ── Save output ───────────────────────────────────────────────
            torchaudio.save(
                str(output_path),
                wav[0].cpu(),               # shape: [channels, samples]
                self._model.sample_rate,
            )
            return  # ← exit; skip the silent stub below

        ══════════════════════════════════════════════════════════════════════
        FINE-TUNING ADAPTATION POINT
        To adapt this for task-specific fine-tuning:
          1. Train a LoRA or full fine-tune on your lyrics+prompt corpus.
          2. Update __init__ to load your checkpoint instead of "medium".
          3. Pass request.style_preset.value as part of the description
             so the model learns style conditioning.
          4. The seed field enables reproducible output — use torch.manual_seed
             before every inference call.
        ══════════════════════════════════════════════════════════════════════
        """
        # ── TODO: replace below with real inference ───────────────────────
        self._write_silent_wav(output_path, duration_sec=request.duration_sec)

    @staticmethod
    def _write_silent_wav(path: Path, duration_sec: int, sample_rate: int = 44100) -> None:
        """Write a silent PCM WAV so the pipeline can be tested without a model."""
        num_samples = sample_rate * duration_sec
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)      # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack("<h", 0) * num_samples)
