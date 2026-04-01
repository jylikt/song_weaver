"""
Request and response schemas for the GPU worker API.

These define the exact contract the main SongGen app expects.
Keep field names stable — the RemoteGpuMusicGenerator client
in the main app depends on this schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── /health ──────────────────────────────────────────────────────────────────

class GpuStatusSchema(BaseModel):
    available: bool = Field(description="Whether a real GPU device is accessible.")
    device: str = Field(description="Device name (e.g. 'NVIDIA A100 80GB' or 'cpu (simulated)').")
    vram_total_gb: float = Field(description="Total GPU VRAM in gigabytes.")
    vram_used_gb: float = Field(description="Currently allocated VRAM in gigabytes.")
    temperature_c: int = Field(description="GPU core temperature in Celsius.")


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when the worker is healthy.")
    version: str = Field(description="Worker API version.")
    model_loaded: bool = Field(description="Whether a model is currently loaded in memory.")
    model_name: Optional[str] = Field(None, description="Name of the currently loaded model, if any.")
    model_family: Optional[str] = Field(
        None,
        description=(
            "Resolved model architecture family: 'causal' (Llama-based, e.g. YuE-s1-7B) "
            "or 'seq2seq' (encoder-decoder). Null when no model is loaded or in stub mode."
        ),
    )
    stub_mode: bool = Field(
        description=(
            "True when YUE_MODEL_PATH is not set and the worker is generating "
            "silent WAV placeholders. False when real YuE inference is active."
        )
    )
    loaded_at: Optional[datetime] = Field(None, description="UTC timestamp when the model was loaded.")
    load_duration_sec: float = Field(description="How long the last model load took in seconds.")
    generation_count: int = Field(description="Number of completed generation calls since startup.")
    uptime_seconds: float = Field(description="Seconds since the worker process started.")
    gpu: GpuStatusSchema = Field(description="GPU telemetry.")


# ─── /load-model ──────────────────────────────────────────────────────────────

class LoadModelRequest(BaseModel):
    model_name: Optional[str] = Field(
        None,
        description=(
            "Name or path of the model to load. "
            "Omit to use the DEFAULT_MODEL_NAME configured in the worker's env."
        ),
        examples=["yue-base", "m-a-p/YuE-s2-lyric-audiovae-24k-v1.1"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"model_name": "yue-base"}]
        }
    }


class LoadModelResponse(BaseModel):
    success: bool
    model_name: str = Field(description="The model that was loaded.")
    loaded_at: datetime = Field(description="UTC timestamp of when loading completed.")
    message: str = Field(description="Human-readable status message.")


# ─── /unload-model ────────────────────────────────────────────────────────────

class UnloadModelResponse(BaseModel):
    success: bool
    message: str


# ─── /generate ────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """
    Generation payload sent by the main SongGen app.
    All fields mirror GenerationRequest in the main app's models.
    """

    job_id: str = Field(
        description="Job ID from the main app. Used as the output filename stem for traceability.",
        examples=["a31040db-3087-4339-82e4-71c60d48d5d5"],
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language song description passed to the model.",
    )
    lyrics: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Full song lyrics with [Verse]/[Chorus] markers.",
    )
    style_preset: str = Field(
        default="pop",
        description="Musical genre hint (e.g. 'pop', 'rock', 'electronic').",
    )
    duration_sec: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Target output duration in seconds.",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional seed for reproducible generation.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "a31040db-3087-4339-82e4-71c60d48d5d5",
                    "prompt": "Uplifting pop song about chasing dreams",
                    "lyrics": "[Verse 1]\nNever gonna stop\n\n[Chorus]\nChase your dreams",
                    "style_preset": "pop",
                    "duration_sec": 30,
                    "seed": 42,
                }
            ]
        }
    }


class GenerateResponse(BaseModel):
    """
    Response returned to the main app after successful generation.

    The `audio_url` field is the direct download URL for the WAV file.
    The main app fetches this URL and saves the audio locally.
    """

    job_id: str = Field(description="Echo of the job_id from the request.")
    audio_url: str = Field(
        description=(
            "Absolute URL to download the generated audio file. "
            "The main app GETs this URL to retrieve the WAV."
        )
    )
    duration_sec: int = Field(description="Duration of the generated audio.")
    model_name: str = Field(description="Model that produced this output.")
    provider: str = Field(description="Provider name for traceability ('GpuWorker').")


# ─── Error responses ──────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    detail: str
    hint: Optional[str] = None
