"""
Health check route.

GET /api/v1/health — used by load balancers, uptime monitors, and readiness probes.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["health"])

# Track when the process started so we can report uptime
_started_at: datetime = datetime.now(tz=timezone.utc)


class HealthResponse(BaseModel):
    status: str = Field(
        description="Service status. `ok` means the server is healthy and accepting requests."
    )
    version: str = Field(description="API version string.")
    active_provider: str = Field(
        description="Currently configured generation backend (matches GENERATOR_PROVIDER env var)."
    )
    uptime_seconds: float = Field(description="Seconds since the server process started.")
    python_version: str = Field(description="Python runtime version.")
    timestamp: datetime = Field(description="UTC timestamp of this health response.")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns the current health status of the Song Generation Service. "
        "A 200 response with `status: ok` means the service is running and accepting requests. "
        "Used by load balancers and monitoring tools."
    ),
)
async def health_check() -> HealthResponse:
    from app.config import settings

    now = datetime.now(tz=timezone.utc)
    uptime = (now - _started_at).total_seconds()

    return HealthResponse(
        status="ok",
        version="1.0.0",
        active_provider=settings.generator_provider.value,
        uptime_seconds=round(uptime, 2),
        python_version=sys.version.split()[0],
        timestamp=now,
    )
