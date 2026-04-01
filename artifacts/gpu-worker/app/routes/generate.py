"""
POST /generate — run YuE music generation.

Flow:
  1. Auto-load the default model if none is loaded.
  2. Run _run_inference() in a thread-pool executor (blocking call).
  3. Update generation counters and refresh GPU telemetry.
  4. Return the audio_url for the main app to download.

Error handling (per spec FR-6):
  - Model not loaded / load failure  → 503 with detail message
  - Invalid payload                  → 422 (Pydantic validation, automatic)
  - Inference runtime failure        → 500 with detail message
  - Output write failure             → 500 with detail message
"""

from __future__ import annotations

import asyncio
import logging
import random
import struct
import wave
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.models import GenerateRequest, GenerateResponse
from app.state import worker_state

logger = logging.getLogger(__name__)
router = APIRouter(tags=["generation"])


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate a song with YuE",
    description=(
        "Runs lyrics-based music generation using the loaded YuE model. "
        "If no model is loaded, the default model is loaded automatically. "
        "Returns a direct URL to download the generated WAV file."
    ),
    responses={
        200: {"description": "Generation successful. `audio_url` points to the WAV file."},
        500: {"description": "Inference failed. See `detail` for the error message."},
        503: {"description": "Model failed to load or worker is unavailable."},
    },
)
async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
    # ── Validate duration cap ──────────────────────────────────────────────────
    if body.duration_sec > settings.yue_max_duration_sec:
        raise HTTPException(
            status_code=422,
            detail=(
                f"duration_sec={body.duration_sec} exceeds worker maximum "
                f"of {settings.yue_max_duration_sec}s. Reduce the duration."
            ),
        )

    # ── Auto-load model if not yet loaded ──────────────────────────────────────
    if not worker_state.model.loaded:
        logger.info(
            "No model loaded — auto-loading default: %s",
            settings.default_model_name,
        )
        try:
            await worker_state.load_model(settings.default_model_name)
        except RuntimeError as exc:
            logger.error("Auto-load failed: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load model '{settings.default_model_name}': {exc}",
            ) from exc

    model_name = worker_state.model.model_name
    stub_mode = worker_state.model.stub_mode

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{body.job_id}.wav"

    logger.info(
        "Generation start | job=%s | model=%s | mode=%s | style=%s | duration=%ds | seed=%s",
        body.job_id,
        model_name,
        "stub" if stub_mode else "yue",
        body.style_preset,
        body.duration_sec,
        body.seed,
    )

    # ── Run inference in thread pool (never block the event loop) ─────────────
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            _run_inference,
            body,
            output_path,
            stub_mode,
        )
    except Exception as exc:
        logger.exception("Inference failed | job=%s | error=%s", body.job_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise HTTPException(
            status_code=500,
            detail="Inference completed but output file is missing or empty.",
        )

    # ── Update counters + GPU telemetry ───────────────────────────────────────
    await worker_state.record_generation()

    # ── Build download URL ─────────────────────────────────────────────────────
    base_url = str(request.base_url).rstrip("/")
    audio_url = f"{base_url}/output/{output_path.name}"

    logger.info(
        "Generation complete | job=%s | size=%d bytes | url=%s",
        body.job_id,
        output_path.stat().st_size,
        audio_url,
    )

    return GenerateResponse(
        job_id=body.job_id,
        audio_url=audio_url,
        duration_sec=body.duration_sec,
        model_name=model_name,
        provider="GpuWorker",
    )


# ─── Inference ────────────────────────────────────────────────────────────────

def _run_inference(body: GenerateRequest, output_path: Path, stub_mode: bool) -> None:
    """
    Blocking inference function — called inside a ThreadPoolExecutor.

    When stub_mode is True (YUE_MODEL_PATH not set), writes a silent WAV so
    the full pipeline can be tested without a GPU or model checkpoint.

    When stub_mode is False, runs real YuE inference using the model instance
    stored on worker_state.
    """
    if stub_mode:
        logger.warning(
            "Stub mode — generating silent WAV for job %s. "
            "Set YUE_MODEL_PATH to enable real inference.",
            body.job_id,
        )
        _write_silent_wav(output_path, duration_sec=body.duration_sec)
        return

    _run_yue_inference(body, output_path)


def _run_yue_inference(body: GenerateRequest, output_path: Path) -> None:
    """
    Real YuE inference.

    Reads the model and processor from worker_state (loaded by load_model()).
    Maps the API request fields to YuE generation parameters and saves the
    output WAV to output_path.

    ──────────────────────────────────────────────────────────────────────────
    ADAPTING TO YOUR YuE VERSION
    ──────────────────────────────────────────────────────────────────────────
    YuE's Python interface depends on which package/checkout you have.
    The code below follows the HuggingFace-style pattern that is most common.
    Adjust the generate() call to match the exact method signature of your
    installed version.  All three common patterns are documented inline.
    ──────────────────────────────────────────────────────────────────────────
    """
    import torch

    model = worker_state._model
    processor = worker_state._processor
    device = worker_state._device

    if model is None or processor is None:
        raise RuntimeError(
            "Model or processor is None — was load_model() called successfully?"
        )

    # ── Reproducibility ────────────────────────────────────────────────────────
    if body.seed is not None:
        torch.manual_seed(body.seed)
        random.seed(body.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(body.seed)
        try:
            import numpy as np
            np.random.seed(body.seed)
        except ImportError:
            pass

    # ── Build genre / style prompt ─────────────────────────────────────────────
    # YuE expects a genre description that controls the musical style.
    # Combine the user's prompt with the style_preset for richer conditioning.
    genre_prompt = _build_genre_prompt(body.style_preset, body.prompt)

    logger.debug(
        "YuE inference | job=%s | genre_prompt=%r | lyrics_len=%d | duration=%ds",
        body.job_id,
        genre_prompt,
        len(body.lyrics),
        body.duration_sec,
    )

    # ── Tokenise / process inputs ──────────────────────────────────────────────
    #
    # Pattern A — HuggingFace AutoProcessor:
    #   inputs = processor(
    #       text=genre_prompt,
    #       lyrics=body.lyrics,
    #       return_tensors="pt",
    #       padding=True,
    #   ).to(device)
    #
    # Pattern B — YuE custom tokenizer (if processor has separate methods):
    #   genre_ids = processor.encode_genre(genre_prompt, return_tensors="pt").to(device)
    #   lyric_ids = processor.encode_lyrics(body.lyrics, return_tensors="pt").to(device)
    #   inputs = {"genre_input_ids": genre_ids, "lyric_input_ids": lyric_ids}
    #
    # ── Active integration ─────────────────────────────────────────────────────
    inputs = processor(
        text=genre_prompt,
        lyrics=body.lyrics,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # ── Run generation ─────────────────────────────────────────────────────────
    #
    # Pattern A — HuggingFace generate() (Seq2Seq / encoder-decoder):
    #   with torch.no_grad():
    #       outputs = model.generate(
    #           **inputs,
    #           max_new_tokens=_duration_to_tokens(body.duration_sec),
    #           do_sample=True,
    #           guidance_scale=settings.yue_cfg_scale,
    #           num_inference_steps=settings.yue_num_steps,
    #       )
    #   audio_values = outputs.audio_values           # shape [batch, channels, samples]
    #   sample_rate  = outputs.sample_rate            # or settings.yue_sample_rate
    #
    # Pattern B — YuE pipeline-style inference:
    #   result = model(
    #       genre_description=genre_prompt,
    #       lyrics=body.lyrics,
    #       duration_sec=body.duration_sec,
    #       seed=body.seed,
    #       cfg_scale=settings.yue_cfg_scale,
    #   )
    #   audio_values = result.audio                   # shape [channels, samples]
    #   sample_rate  = result.sample_rate
    #
    # Pattern C — YuE source checkout (infer.py-style):
    #   audio_values, sample_rate = model.generate(
    #       genre=genre_prompt,
    #       lyrics=body.lyrics,
    #       duration=body.duration_sec,
    #       steps=settings.yue_num_steps,
    #       cfg=settings.yue_cfg_scale,
    #   )
    #
    # ── Active integration ─────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=_duration_to_tokens(body.duration_sec, settings.yue_sample_rate),
            do_sample=True,
            guidance_scale=settings.yue_cfg_scale,
            num_inference_steps=settings.yue_num_steps,
        )

    audio_values = outputs.audio_values   # [batch, channels, samples]
    sample_rate = getattr(outputs, "sample_rate", settings.yue_sample_rate)

    # ── Save output ────────────────────────────────────────────────────────────
    try:
        import torchaudio
        # Take the first item in the batch, move to CPU for saving
        audio_cpu = audio_values[0].cpu()  # shape [channels, samples]
        torchaudio.save(str(output_path), audio_cpu, sample_rate)
    except ImportError:
        # torchaudio not available — fall back to soundfile
        import soundfile as sf
        import numpy as np
        audio_np = audio_values[0].cpu().numpy()  # [channels, samples]
        audio_np = audio_np.T                      # soundfile expects [samples, channels]
        sf.write(str(output_path), audio_np, sample_rate, subtype="PCM_16")

    logger.info(
        "YuE inference saved | job=%s | path=%s | sample_rate=%d",
        body.job_id,
        output_path,
        sample_rate,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_genre_prompt(style_preset: str, user_prompt: str) -> str:
    """
    Combine the style_preset enum value and the user's free-text prompt into
    a single genre description string for YuE.

    YuE expects a natural-language genre description like:
        "pop, upbeat, female vocalist, acoustic guitar"

    The style_preset provides a strong musical genre anchor; the user's prompt
    adds texture and emotional direction.
    """
    style_labels = {
        "pop":        "pop, contemporary, radio-friendly",
        "rock":       "rock, electric guitar, drums",
        "hip_hop":    "hip-hop, rap, urban, trap",
        "rnb":        "r&b, soul, smooth, groove",
        "electronic": "electronic, synthesizer, EDM, dance",
        "jazz":       "jazz, improvisation, swing, acoustic",
        "classical":  "classical, orchestral, symphonic, strings",
        "country":    "country, acoustic guitar, folk, storytelling",
        "folk":       "folk, acoustic, singer-songwriter, organic",
        "metal":      "metal, heavy, distorted guitar, aggressive",
        "custom":     "",
    }
    base = style_labels.get(style_preset, style_preset)
    parts = [p.strip() for p in [base, user_prompt] if p.strip()]
    return ", ".join(parts)


def _duration_to_tokens(duration_sec: int, sample_rate: int) -> int:
    """
    Convert a target duration in seconds to an approximate token count.

    YuE's codec compresses audio at a fixed frame rate (typically 75 tokens/s
    for EnCodec at 24kHz).  Adjust the frame_rate constant to match the codec
    your YuE checkpoint uses.
    """
    # EnCodec / DAC frame rate at 24kHz ≈ 75 tokens/second
    # Adjust if your checkpoint uses a different codec or sample rate.
    codec_frame_rate = 75
    return duration_sec * codec_frame_rate


def _write_silent_wav(path: Path, duration_sec: int, sample_rate: int = 44100) -> None:
    """Write a silent mono PCM WAV — used only in stub mode."""
    num_samples = sample_rate * duration_sec
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<h", 0) * num_samples)
