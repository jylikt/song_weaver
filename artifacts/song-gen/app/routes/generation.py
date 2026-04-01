"""
Song generation API routes.

Mounted at /api/v1 (see app/main.py).

Endpoints:
  POST   /generate                  — submit a new song generation job
  GET    /jobs/{job_id}             — poll job status
  GET    /jobs/{job_id}/result      — fetch completed result + metadata
  GET    /jobs/{job_id}/download    — stream the generated WAV file
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.models.generation import (
    GenerationMode,
    GenerationRequest,
    JobCreatedResponse,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
)
from app.services.generation_service import build_result_metadata, create_job, get_job

logger = logging.getLogger(__name__)
router = APIRouter(tags=["generation"])


# ─── Submit a generation job ───────────────────────────────────────────────────

@router.post(
    "/generate",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Submit a song generation job",
    description=(
        "Accepts a generation request and immediately returns a `job_id`. "
        "Generation runs asynchronously in the background. "
        "Poll `GET /api/v1/jobs/{job_id}` to track progress, "
        "then call `GET /api/v1/jobs/{job_id}/result` to retrieve the output."
    ),
    responses={
        202: {"description": "Job accepted and queued for processing."},
        422: {"description": "Invalid request payload — see error detail for field-level validation errors."},
    },
)
async def submit_generation(payload: GenerationRequest) -> JobCreatedResponse:
    job = await create_job(payload)
    return JobCreatedResponse(
        job_id=job.job_id,
        status=job.status,
        mode=payload.mode,
    )


# ─── Poll job status ───────────────────────────────────────────────────────────

@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
    description=(
        "Returns the current lifecycle state of a generation job. "
        "Poll this endpoint until `status` is `completed` or `failed`. "
        "Recommended polling interval: every 2–5 seconds."
    ),
    responses={
        200: {"description": "Current job state."},
        404: {"description": "No job found with the given ID."},
    },
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have never been created or may have expired.",
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        mode=job.request.mode,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
    )


# ─── Fetch result + metadata ───────────────────────────────────────────────────

@router.get(
    "/jobs/{job_id}/result",
    response_model=JobResultResponse,
    summary="Get generation result and metadata",
    description=(
        "Returns the full result for a completed job, including a direct download URL "
        "and metadata (provider used, duration, lyrics, seed, output path). "
        "Returns HTTP 409 if the job is still running, and HTTP 422 if it failed."
    ),
    responses={
        200: {"description": "Job completed — result and metadata returned."},
        404: {"description": "No job found with the given ID."},
        409: {"description": "Job is still queued or running. Try again later."},
        422: {"description": "Job failed — error detail is included in the response body."},
        500: {"description": "Job completed but the output file is missing from the server."},
    },
)
async def get_job_result(job_id: str, request: Request) -> JobResultResponse:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )

    if job.status == JobStatus.FAILED:
        return JobResultResponse(
            job_id=job.job_id,
            status=job.status,
            error=job.error,
        )

    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job '{job_id}' is not yet finished (status={job.status.value}). "
                "Continue polling GET /api/v1/jobs/{job_id} and retry this endpoint once completed."
            ),
        )

    # COMPLETED — validate output file exists before responding
    output_path = Path(job.output_file) if job.output_file else None
    if output_path is None or not output_path.exists():
        logger.error(
            "Completed job missing output file",
            extra={"job_id": job_id, "expected_path": str(output_path)},
        )
        raise HTTPException(
            status_code=500,
            detail="Job completed but the output audio file is missing from the server.",
        )

    download_url, metadata = await build_result_metadata(job, str(request.base_url))

    return JobResultResponse(
        job_id=job.job_id,
        status=job.status,
        download_url=download_url,
        metadata=metadata,
    )


# ─── Stream / download audio file ─────────────────────────────────────────────

@router.get(
    "/jobs/{job_id}/download",
    summary="Download the generated audio file",
    description="Streams the generated WAV file for a completed job. Can be played or saved directly.",
    response_class=FileResponse,
    responses={
        200: {"description": "WAV audio file stream.", "content": {"audio/wav": {}}},
        404: {"description": "Job not found."},
        409: {"description": "Job output not yet available."},
        500: {"description": "File missing from server."},
    },
)
async def download_audio(job_id: str) -> FileResponse:
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.status != JobStatus.COMPLETED or not job.output_file:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' has no completed output (status={job.status.value}).",
        )

    output_path = Path(job.output_file)
    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Audio file is missing from the server.")

    return FileResponse(
        path=str(output_path),
        media_type="audio/wav",
        filename=f"song_{job_id}.wav",
    )
