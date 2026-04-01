"""
FastAPI application factory.

Route layout (all under /api/v1):
  POST   /api/v1/generate                  — submit generation job
  GET    /api/v1/jobs/{job_id}             — poll job status
  GET    /api/v1/jobs/{job_id}/result      — fetch result + metadata
  GET    /api/v1/jobs/{job_id}/download    — stream WAV file
  GET    /api/v1/health                    — service health check
  GET    /api/v1/providers                 — list available backends

OpenAPI docs:
  /docs       — Swagger UI
  /redoc      — ReDoc
  /openapi.json — raw spec
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes.generation import router as generation_router
from app.routes.health import router as health_router
from app.routes.providers import router as providers_router
from app.routes.ui import router as ui_router

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Song Generation Service",
        version="1.0.0",
        description=(
            "## AI Song Generation Service\n\n"
            "Async REST API for generating songs with lyrics using pluggable AI backends.\n\n"
            "### How it works\n"
            "1. **Submit** a job via `POST /api/v1/generate` with your prompt, lyrics, and style.\n"
            "2. **Poll** `GET /api/v1/jobs/{job_id}` until status is `completed` or `failed`.\n"
            "3. **Fetch** the result via `GET /api/v1/jobs/{job_id}/result` to get the download URL and metadata.\n"
            "4. **Download** the WAV file from `GET /api/v1/jobs/{job_id}/download`.\n\n"
            "### Backends\n"
            "Use `GET /api/v1/providers` to list available backends. "
            "Pass `mode: local` or `mode: remote_gpu` in your request to target a specific backend.\n\n"
            "### Current status\n"
            "The local backend runs in **stub mode** — it writes a silent WAV as a placeholder. "
            "Connect a real model in `app/providers/local_generator.py → _run_model()`."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "Song Generation Service",
        },
        license_info={
            "name": "MIT",
        },
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict to known domains in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global error handler — always return JSON ─────────────────────────────
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected server error occurred. Please try again later."},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    # UI at root — must be included BEFORE the static mount
    app.include_router(ui_router)
    # API endpoints under /api/v1
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(providers_router, prefix="/api/v1")
    app.include_router(generation_router, prefix="/api/v1")

    # ── Static files — serve generated audio at /generated/<filename> ─────────
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/generated", StaticFiles(directory=str(output_dir)), name="generated")

    # ── Startup log ───────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def _startup() -> None:
        logger.info(
            "Song Generation Service started | env=%s | provider=%s | output_dir=%s",
            settings.app_env.value,
            settings.generator_provider.value,
            settings.output_dir,
        )

    return app


app = create_app()
