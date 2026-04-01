"""
Pydantic models for generation requests, jobs, and API responses.

These are the public API contracts — keep field names stable across versions.
Internal fields (e.g. GenerationJob) are storage-oriented and not exposed directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    """Lifecycle states a generation job moves through."""

    QUEUED = "queued"
    """Job has been accepted and is waiting for a worker slot."""

    RUNNING = "running"
    """Generation is actively in progress."""

    COMPLETED = "completed"
    """Generation finished successfully. Result is available."""

    FAILED = "failed"
    """Generation encountered an unrecoverable error."""


class GenerationMode(str, Enum):
    """
    Selects which generation backend processes this job.
    Overrides the server-level GENERATOR_PROVIDER env var for this request.
    """

    LOCAL = "local"
    """Run inference in the local process (requires a model to be loaded)."""

    REMOTE_GPU = "remote_gpu"
    """Dispatch to a remote GPU inference worker over HTTP."""


class StylePreset(str, Enum):
    """
    Broad musical genre/style presets.
    These are passed as hints to the generation model.
    Use CUSTOM and rely on `prompt` for fine-grained control.
    """

    POP = "pop"
    ROCK = "rock"
    HIP_HOP = "hip_hop"
    RNB = "rnb"
    ELECTRONIC = "electronic"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    COUNTRY = "country"
    FOLK = "folk"
    METAL = "metal"
    CUSTOM = "custom"


# ─── Request ──────────────────────────────────────────────────────────────────

class GenerationRequest(BaseModel):
    """
    Payload for submitting a new song generation job.

    All text fields accept UTF-8. Lyrics should include structural markers
    such as `[Verse 1]`, `[Chorus]`, `[Bridge]` on their own lines so the
    model can learn the song structure.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Natural-language description of the desired song. "
            "Include mood, theme, instrumentation, tempo, and any sonic references. "
            "The more specific, the better."
        ),
        examples=["An uplifting pop anthem about summer freedom with a driving beat and catchy chorus hook"],
    )
    lyrics: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description=(
            "Full song lyrics to be performed. "
            "Use `[Verse 1]`, `[Chorus]`, `[Bridge]`, etc. as section markers on their own lines. "
            "Use `\\n` for line breaks within sections."
        ),
        examples=[
            "[Verse 1]\nSun is shining bright\nFreedom in the air\n\n[Chorus]\nWe are alive tonight\nNothing can compare"
        ],
    )
    style_preset: StylePreset = Field(
        default=StylePreset.POP,
        description=(
            "Broad musical genre preset applied as a generation hint. "
            "Use `custom` to rely entirely on the `prompt` field for style direction."
        ),
    )
    duration_sec: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Target output duration in seconds. Must be between 5 and 300.",
    )
    seed: Optional[int] = Field(
        default=None,
        description=(
            "Optional integer seed for deterministic generation. "
            "Using the same seed with the same inputs will produce the same output. "
            "Omit (or set to null) for a random result."
        ),
        examples=[42],
    )
    mode: GenerationMode = Field(
        default=GenerationMode.LOCAL,
        description=(
            "Which generation backend to use for this job. "
            "`local` runs inference in-process; `remote_gpu` dispatches to an external worker."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Uplifting pop song about chasing dreams with a soaring chorus",
                    "lyrics": "[Verse 1]\nNever gonna stop, keep climbing high\nEvery step I take beneath this open sky\n\n[Chorus]\nChase your dreams until you touch the stars\nNothing in this world can hold us back",
                    "style_preset": "pop",
                    "duration_sec": 60,
                    "seed": 42,
                    "mode": "local",
                }
            ]
        }
    }


# ─── Internal job record ──────────────────────────────────────────────────────

class GenerationJob(BaseModel):
    """
    Internal job record. Stored in the job store and never returned directly.
    API responses are built from this via response model classes.

    In production, replace the in-memory store with a DB table and map
    this model to a Drizzle/SQLAlchemy schema.
    """

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.QUEUED
    request: GenerationRequest

    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Populated on completion
    output_file: Optional[str] = None
    # Name of the provider class that ran this job (e.g. "LocalMusicGenerator")
    provider: Optional[str] = None
    # Human-readable error message, set on failure
    error: Optional[str] = None
    # 0–100 progress hint, optionally updated by the provider during generation
    progress: int = 0


# ─── API Response schemas ──────────────────────────────────────────────────────

class JobCreatedResponse(BaseModel):
    """Returned immediately after a job is accepted (HTTP 202)."""

    job_id: str = Field(description="Unique identifier for the queued job.")
    status: JobStatus = Field(description="Initial job status — always `queued` on creation.")
    mode: GenerationMode = Field(description="Backend that will process this job.")
    message: str = Field(
        default="Job queued. Poll GET /api/v1/jobs/{job_id} for status updates.",
        description="Human-readable next-step hint.",
    )


class JobStatusResponse(BaseModel):
    """
    Current state snapshot for a job.
    Poll this endpoint until `status` is `completed` or `failed`.
    """

    job_id: str = Field(description="Unique job identifier.")
    status: JobStatus = Field(description="Current lifecycle state of the job.")
    mode: GenerationMode = Field(description="Backend processing this job.")
    progress: int = Field(description="Estimated completion percentage (0–100).")
    created_at: datetime = Field(description="UTC timestamp when the job was created.")
    started_at: Optional[datetime] = Field(None, description="UTC timestamp when processing started.")
    completed_at: Optional[datetime] = Field(None, description="UTC timestamp when processing finished.")
    error: Optional[str] = Field(None, description="Error message if the job failed, otherwise null.")


class ResultMetadata(BaseModel):
    """
    Full metadata included in a completed job result.
    Designed to be stored alongside the audio file for traceability.
    """

    job_id: str = Field(description="Job that produced this result.")
    provider: str = Field(description="Name of the generator class that produced the audio.")
    output_path: str = Field(description="File-system path where the audio file is saved.")
    duration_sec: int = Field(description="Requested duration that was passed to the generator.")
    created_at: datetime = Field(description="UTC timestamp when the job was created.")
    lyrics_used: str = Field(description="The exact lyrics that were passed to the generator.")
    style_preset: StylePreset = Field(description="Style preset used for this generation.")
    seed: Optional[int] = Field(None, description="Seed used, if any.")


class JobResultResponse(BaseModel):
    """
    Full result payload returned once a job reaches `completed` or `failed`.
    """

    job_id: str = Field(description="Unique job identifier.")
    status: JobStatus = Field(description="Final job status.")
    download_url: Optional[str] = Field(
        None,
        description="Direct URL to stream or download the generated WAV file. Null if the job failed.",
    )
    metadata: Optional[ResultMetadata] = Field(
        None,
        description="Generation metadata. Null if the job failed.",
    )
    error: Optional[str] = Field(
        None,
        description="Error detail if the job failed, otherwise null.",
    )
