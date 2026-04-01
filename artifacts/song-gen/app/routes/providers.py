"""
Providers route — lists available generation backends and their capabilities.

GET /api/v1/providers
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import settings

router = APIRouter(tags=["providers"])


class ProviderInfo(BaseModel):
    id: str = Field(description="Provider identifier — use this as the `mode` value in a generation request.")
    name: str = Field(description="Human-readable provider name.")
    description: str = Field(description="What this provider does and when to use it.")
    available: bool = Field(description="Whether this provider is ready to accept jobs.")
    notes: Optional[str] = Field(None, description="Extra caveats or setup requirements.")


class ProvidersResponse(BaseModel):
    active_provider: str = Field(
        description="Default provider used when no `mode` is specified in the request."
    )
    providers: list[ProviderInfo] = Field(description="All registered generation backends.")


@router.get(
    "/providers",
    response_model=ProvidersResponse,
    summary="List available generation providers",
    description=(
        "Returns all registered generation backends with their current availability. "
        "Use the provider `id` as the `mode` field in a generation request to target a specific backend."
    ),
)
async def list_providers() -> ProvidersResponse:
    remote_configured = bool(settings.remote_worker_url)

    providers = [
        ProviderInfo(
            id="local",
            name="Local Music Generator",
            description=(
                "Runs inference directly in the server process. "
                "Currently operates in stub mode — produces a silent WAV file as a placeholder. "
                "Integrate a real model (e.g. AudioCraft, Bark, Suno) in "
                "app/providers/local_generator.py → _run_model()."
            ),
            available=True,
            notes="Stub mode active. Replace _run_model() in local_generator.py for real audio output.",
        ),
        ProviderInfo(
            id="remote_gpu",
            name="Remote GPU Worker",
            description=(
                "Dispatches generation to the gpu-worker FastAPI service. "
                "Designed for large models that cannot run in the Replit CPU environment. "
                "The worker auto-loads its default model and returns an audio_url for download."
            ),
            available=remote_configured,
            notes=(
                "Set REMOTE_WORKER_URL (and optionally REMOTE_WORKER_TOKEN) in your .env to enable."
                if not remote_configured
                else f"Worker endpoint: {settings.remote_worker_url}"
            ),
        ),
    ]

    return ProvidersResponse(
        active_provider=settings.generator_provider.value,
        providers=providers,
    )
