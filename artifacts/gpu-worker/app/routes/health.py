"""
GET /health — returns worker health, model state, and GPU telemetry.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.models import GpuStatusSchema, HealthResponse
from app.state import worker_state

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Worker health check",
    description=(
        "Returns the current status of the GPU worker, including YuE model load state, "
        "stub_mode flag (True = silent WAV placeholder, False = real YuE inference), "
        "and GPU VRAM telemetry. A 200 response means the worker is accepting requests."
    ),
)
async def health() -> HealthResponse:
    gpu = worker_state.gpu
    model = worker_state.model

    return HealthResponse(
        status="ok",
        version="1.0.0",
        model_loaded=model.loaded,
        model_name=model.model_name,
        model_family=model.model_family,
        stub_mode=model.stub_mode,
        loaded_at=model.loaded_at,
        load_duration_sec=model.load_duration_sec,
        generation_count=model.generation_count,
        uptime_seconds=round(worker_state.uptime_seconds, 2),
        gpu=GpuStatusSchema(
            available=gpu.available,
            device=gpu.device,
            vram_total_gb=gpu.vram_total_gb,
            vram_used_gb=gpu.vram_used_gb,
            temperature_c=gpu.temperature_c,
        ),
    )
