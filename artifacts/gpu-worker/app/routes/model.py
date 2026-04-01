"""
Model lifecycle routes.

POST /load-model   — load a model into (simulated) GPU memory
POST /unload-model — free model from memory
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models import LoadModelRequest, LoadModelResponse, UnloadModelResponse
from app.state import worker_state

logger = logging.getLogger(__name__)
router = APIRouter(tags=["model"])


@router.post(
    "/load-model",
    response_model=LoadModelResponse,
    summary="Load a model into memory",
    description=(
        "Loads the specified model into GPU memory (simulated in stub mode). "
        "If a different model is already loaded, it is unloaded first. "
        "Omit `model_name` to use the worker's DEFAULT_MODEL_NAME."
    ),
    responses={
        200: {"description": "Model loaded successfully."},
        409: {"description": "Model is already loaded with the same name."},
    },
)
async def load_model(body: LoadModelRequest) -> LoadModelResponse:
    model_name = body.model_name or settings.default_model_name

    # If a different model is loaded, unload it first
    if worker_state.model.loaded and worker_state.model.model_name != model_name:
        logger.info(
            "Swapping model: %s → %s",
            worker_state.model.model_name,
            model_name,
        )
        await worker_state.unload_model()

    # Already loaded with the same name — idempotent OK
    if worker_state.model.loaded and worker_state.model.model_name == model_name:
        return LoadModelResponse(
            success=True,
            model_name=model_name,
            loaded_at=worker_state.model.loaded_at,
            message=f"Model '{model_name}' is already loaded.",
        )

    await worker_state.load_model(model_name)

    return LoadModelResponse(
        success=True,
        model_name=model_name,
        loaded_at=worker_state.model.loaded_at,
        message=f"Model '{model_name}' loaded successfully.",
    )


@router.post(
    "/unload-model",
    response_model=UnloadModelResponse,
    summary="Unload the current model",
    description=(
        "Releases the loaded model from memory (simulated in stub mode). "
        "Safe to call even if no model is currently loaded."
    ),
)
async def unload_model() -> UnloadModelResponse:
    if not worker_state.model.loaded:
        return UnloadModelResponse(success=True, message="No model was loaded.")

    model_name = worker_state.model.model_name
    await worker_state.unload_model()

    return UnloadModelResponse(
        success=True,
        message=f"Model '{model_name}' unloaded.",
    )
