"""
Generation service — orchestrates the full job lifecycle.

Flow:
  1. create_job  — validate request, persist job record, enqueue background task
  2. _run_job    — select provider, call generate_song, update job status
  3. get_job     — return current job state for status polling
  4. get_result  — assemble full result response (only when completed/failed)

Status transitions:
  QUEUED → RUNNING → COMPLETED
                   → FAILED
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.models.generation import (
    GenerationJob,
    GenerationRequest,
    JobStatus,
    ResultMetadata,
)
from app.providers.base import GenerationError
from app.providers.factory import get_generator
from app.services.job_store import job_store

logger = logging.getLogger(__name__)


# ─── Public API ────────────────────────────────────────────────────────────────

async def create_job(request: GenerationRequest) -> GenerationJob:
    """
    Persist a new job record and kick off background processing.
    Returns the job immediately with status=QUEUED — do not wait for generation.
    """
    job = GenerationJob(request=request)
    await job_store.put(job)

    logger.info(
        "Job queued",
        extra={"job_id": job.job_id, "mode": request.mode.value, "style": request.style_preset.value},
    )

    # Fire-and-forget — the event loop runs _run_job concurrently
    asyncio.create_task(_run_job(job.job_id))

    return job


async def get_job(job_id: str) -> GenerationJob | None:
    """Return the current job record, or None if the job does not exist."""
    return await job_store.get(job_id)


async def build_result_metadata(job: GenerationJob, base_url: str) -> tuple[str, ResultMetadata]:
    """
    Build the download URL and result metadata for a completed job.

    Args:
        job:      A completed GenerationJob.
        base_url: Request base URL used to construct the download link.

    Returns:
        (download_url, ResultMetadata)
    """
    download_url = f"{base_url.rstrip('/')}/api/v1/jobs/{job.job_id}/download"
    metadata = ResultMetadata(
        job_id=job.job_id,
        provider=job.provider or "unknown",
        output_path=job.output_file or "",
        duration_sec=job.request.duration_sec,
        created_at=job.created_at,
        lyrics_used=job.request.lyrics,
        style_preset=job.request.style_preset,
        seed=job.request.seed,
    )
    return download_url, metadata


# ─── Background task ───────────────────────────────────────────────────────────

async def _run_job(job_id: str) -> None:
    """
    Drive a single job from QUEUED to COMPLETED or FAILED.
    Runs as an asyncio background task — never call this directly.
    """
    job = await job_store.get(job_id)
    if job is None:
        logger.error("_run_job: job not found — was it deleted?", extra={"job_id": job_id})
        return

    # ── QUEUED → RUNNING ─────────────────────────────────────────────────────
    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow()
    job.progress = 0
    await job_store.put(job)
    logger.info("Job started", extra={"job_id": job_id, "mode": job.request.mode.value})

    output_dir = Path(settings.output_dir)

    try:
        # Select backend based on the per-request mode field
        generator = get_generator(mode=job.request.mode)
        job.provider = generator.name

        output_path = await generator.generate_song(
            request=job.request,
            output_dir=output_dir,
            job_id=job_id,
        )

        # ── RUNNING → COMPLETED ──────────────────────────────────────────────
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.output_file = str(output_path)
        job.progress = 100
        await job_store.put(job)
        logger.info(
            "Job completed",
            extra={"job_id": job_id, "provider": job.provider, "output": str(output_path)},
        )

    except GenerationError as exc:
        logger.error(
            "Job failed (GenerationError)",
            extra={"job_id": job_id, "provider": exc.provider, "error": str(exc)},
        )
        _mark_failed(job, str(exc))
        await job_store.put(job)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Job failed (unexpected error)", extra={"job_id": job_id})
        _mark_failed(job, f"Unexpected error: {exc}")
        await job_store.put(job)


def _mark_failed(job: GenerationJob, error: str) -> None:
    """Helper: transition a job to FAILED state."""
    job.status = JobStatus.FAILED
    job.completed_at = datetime.utcnow()
    job.error = error
