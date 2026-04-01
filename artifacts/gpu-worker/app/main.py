"""
GPU Worker FastAPI application.

Endpoints:
  GET  /health         — worker health + model + GPU telemetry
  POST /load-model     — load a model into (simulated) GPU memory
  POST /unload-model   — release model from memory
  POST /generate       — run music generation

Static files:
  GET  /output/{filename} — download generated WAV files

Swagger UI: /docs
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.auth import require_auth
from app.config import settings
from app.routes.generate import router as generate_router
from app.routes.health import router as health_router
from app.routes.model import router as model_router
from app.state import worker_state

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Auth dependency applied to all protected routes
_auth = Depends(require_auth)


def create_app() -> FastAPI:
    auth_note = (
        "Authentication is **disabled** (WORKER_TOKEN not set)."
        if not settings.worker_token
        else "All requests require `Authorization: Bearer <token>`."
    )

    app = FastAPI(
        title="SongGen GPU Worker",
        version="1.0.0",
        description=(
            "## SongGen GPU Worker\n\n"
            "Remote inference worker for the SongGen AI music generation service.\n\n"
            "This service simulates (or will run) heavy model inference on GPU infrastructure. "
            "The main SongGen app routes `mode: remote_gpu` jobs to this worker.\n\n"
            "### Workflow\n"
            "1. `POST /load-model` — load the desired model into GPU memory.\n"
            "2. `POST /generate` — run inference. Model is auto-loaded if not already ready.\n"
            "3. `POST /unload-model` — release GPU memory when done.\n\n"
            f"### Authentication\n{auth_note}\n\n"
            "### Integration\n"
            "Replace `_run_inference()` in `app/routes/generate.py` with real model code. "
            "Replace `load_model()` / `unload_model()` in `app/state.py` with real "
            "torch model loading / `cuda.empty_cache()` calls."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global error handler ──────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Unexpected server error."},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    # Health is unauthenticated — used by the main app to probe worker status
    app.include_router(health_router)
    # All other routes require auth (if token is configured)
    app.include_router(model_router, dependencies=[_auth])
    app.include_router(generate_router, dependencies=[_auth])

    # ── Static files — serve generated WAV files ──────────────────────────────
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

    # ── Startup ───────────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def _startup() -> None:
        yue_path = settings.yue_model_path or "(not set — stub mode)"
        logger.info(
            "GPU Worker started | env=%s | default_model=%s | yue_path=%s | "
            "device=%s | dtype=%s | auth=%s",
            settings.app_env.value,
            settings.default_model_name,
            yue_path,
            settings.yue_device,
            settings.yue_dtype,
            "enabled" if settings.worker_token else "disabled",
        )
        if not settings.yue_model_path:
            logger.warning(
                "YUE_MODEL_PATH is not configured. Worker will run in stub mode "
                "(silent WAV). Set YUE_MODEL_PATH to a checkpoint path or HF repo id "
                "to enable real YuE inference."
            )

    return app


app = create_app()
