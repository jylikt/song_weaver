"""
POST /generate — run YuE music generation.

Flow:
  1. Validate the request (duration cap, model loaded).
  2. Auto-load the default model if none is loaded.
  3. Run inference in a thread-pool executor (blocking call).
     - Stub mode  → write a silent WAV (no GPU required).
     - Real mode  → delegate to yue_adapter.run_yue_inference().
  4. Update generation counters and refresh GPU telemetry.
  5. Return the audio_url for the main app to download.

Error handling (per spec FR-6):
  - Model not loaded / load failure  → 503 with detail message
  - Invalid payload                  → 422 (Pydantic validation, automatic)
  - Unsupported model family         → 500 with actionable detail message
  - Inference runtime failure        → 500 with detail message
  - Output write failure             → 500 with detail message
"""

from __future__ import annotations

import asyncio
import logging
import struct
import wave
from pathlib import Path

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
        "Returns a direct URL to download the generated WAV file.\n\n"
        "Supported checkpoints: HKUSTAudio/YuE-s1-7B-anneal-en-icl (causal/Llama) "
        "and compatible seq2seq audio models."
    ),
    responses={
        200: {"description": "Generation successful. `audio_url` points to the WAV file."},
        422: {"description": "Validation error (e.g. duration_sec exceeds cap)."},
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
    model_family = worker_state._model_family

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{body.job_id}.wav"

    logger.info(
        "Generation start | job=%s | model=%s | family=%s | stub=%s | style=%s | duration=%ds | seed=%s",
        body.job_id,
        model_name,
        model_family or "n/a",
        stub_mode,
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
            model_family,
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


# ─── Inference dispatcher ─────────────────────────────────────────────────────

def _run_inference(
    body: GenerateRequest,
    output_path: Path,
    stub_mode: bool,
    model_family: str | None,
) -> None:
    """
    Blocking inference dispatcher — called inside a ThreadPoolExecutor.

    Routes to:
      - Stub path   (stub_mode=True): writes a silent WAV, no model required.
      - YuE adapter (stub_mode=False): delegates to yue_adapter.run_yue_inference()
        which handles both 'causal' (Llama-based) and 'seq2seq' paths.
    """
    if stub_mode:
        logger.warning(
            "Stub mode — generating silent WAV for job %s. "
            "Set YUE_MODEL_PATH to enable real inference.",
            body.job_id,
        )
        _write_silent_wav(output_path, duration_sec=body.duration_sec)
        return

    if model_family is None:
        raise RuntimeError(
            "Model is marked as loaded but model_family is not set. "
            "This is an internal state error — unload and reload the model."
        )

    from app.yue_adapter import run_yue_inference

    # Use the codec's auto-detected quantizer count when available.
    # This handles the common mismatch between yue_codec_n_codebooks=8 (config
    # default) and xcodec2 models that use a single FSQ quantizer (n_q=1).
    # detect_codec_n_quantizers() reads codec.generator.quantizer.codebooks
    # shape[0] at load time and stores it in worker_state._codec_n_codebooks.
    n_codebooks = (
        worker_state._codec_n_codebooks
        if worker_state._codec_n_codebooks is not None
        else settings.yue_codec_n_codebooks
    )

    run_yue_inference(
        body=body,
        output_path=output_path,
        model=worker_state._model,
        processor=worker_state._processor,
        device=worker_state._device,
        family=model_family,
        codec=worker_state._codec,
        n_codebooks=n_codebooks,
        text_vocab_size=settings.yue_text_vocab_size,
        sample_rate=settings.yue_sample_rate,
        cfg_scale=settings.yue_cfg_scale,
        num_steps=settings.yue_num_steps,
        yue_repo_path=settings.yue_repo_path,
        yue_model_path=settings.yue_model_path,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _write_silent_wav(path: Path, duration_sec: int, sample_rate: int = 44100) -> None:
    """Write a silent mono PCM WAV — used only in stub mode."""
    num_samples = sample_rate * duration_sec
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<h", 0) * num_samples)
