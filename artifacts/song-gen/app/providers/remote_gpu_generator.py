"""
Remote GPU music generator provider.

Delegates generation to the gpu-worker FastAPI service over HTTP.

Worker contract (defined by artifacts/gpu-worker):
┌─────────────────────────────────────────────────────────────────────────┐
│  GET  /health        → HealthResponse   (no auth required)              │
│  POST /load-model    → LoadModelResponse                                │
│  POST /generate      → GenerateResponse                                 │
│  POST /unload-model  → UnloadModelResponse                              │
└─────────────────────────────────────────────────────────────────────────┘

Flow used by this provider:
  1. GET /health  — check whether the model is already loaded.
  2. If not loaded → POST /load-model (auto-triggers with default model).
  3. POST /generate — run inference, receive audio_url.
  4. GET <audio_url> — stream and save the WAV locally.

Configuration (environment variables):
  REMOTE_WORKER_URL      Base URL of the gpu-worker (default: http://localhost:9000)
  REMOTE_WORKER_TOKEN    Bearer token; leave blank if worker auth is disabled
  REMOTE_WORKER_TIMEOUT_SEC  Request timeout in seconds (default: 300)

──────────────────────────────────────────────────────────────────────────
ADAPTING TO A DIFFERENT WORKER API
──────────────────────────────────────────────────────────────────────────
1. Adjust `_build_payload` if your worker expects a different JSON schema.
2. Adjust `_parse_audio_url` if your worker returns the URL under a
   different key, or returns raw bytes directly.
3. If your worker is async (returns a job ID then you poll), implement
   the polling loop inside `generate_song` using `httpx.AsyncClient`.
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings
from app.models.generation import GenerationRequest
from app.providers.base import BaseMusicGenerator, GenerationError

logger = logging.getLogger(__name__)


class RemoteGpuMusicGenerator(BaseMusicGenerator):
    """
    Sends generation requests to the gpu-worker service and downloads
    the resulting audio file.
    """

    def __init__(self) -> None:
        self._base_url = settings.remote_worker_url.rstrip("/")
        self._timeout = settings.remote_worker_timeout_sec

        # Build auth headers — omit the Authorization header entirely when
        # no token is configured so the worker's auth middleware passes through.
        self._auth_headers: dict[str, str] = {}
        if settings.remote_worker_token:
            self._auth_headers["Authorization"] = f"Bearer {settings.remote_worker_token}"

        self._default_headers = {
            **self._auth_headers,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        logger.info(
            "RemoteGpuMusicGenerator initialised",
            extra={
                "base_url": self._base_url,
                "auth": "enabled" if settings.remote_worker_token else "disabled",
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    async def generate_song(
        self,
        request: GenerationRequest,
        output_dir: Path,
        job_id: str,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}.wav"

        async with httpx.AsyncClient(
            headers=self._default_headers,
            timeout=self._timeout,
        ) as client:
            # ── Step 1: ensure model is loaded ────────────────────────────────
            await self._ensure_model_loaded(client)

            # ── Step 2: trigger generation ────────────────────────────────────
            payload = self._build_payload(request, job_id)
            logger.info(
                "RemoteGpuMusicGenerator: dispatching to worker",
                extra={"job_id": job_id, "worker": self._base_url},
            )

            try:
                resp = await client.post(f"{self._base_url}/generate", json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise GenerationError(
                    f"Worker returned HTTP {exc.response.status_code}: {exc.response.text}",
                    provider=self.name,
                ) from exc
            except httpx.RequestError as exc:
                raise GenerationError(
                    f"Failed to reach worker at {self._base_url}: {exc}",
                    provider=self.name,
                ) from exc

            data = resp.json()

            # ── Step 3: download the audio file ───────────────────────────────
            audio_url = self._parse_audio_url(data)

            logger.info(
                "RemoteGpuMusicGenerator: downloading audio",
                extra={"job_id": job_id, "audio_url": audio_url},
            )

            try:
                audio_resp = await client.get(audio_url)
                audio_resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise GenerationError(
                    f"Failed to download audio from {audio_url}: {exc}",
                    provider=self.name,
                ) from exc

            output_path.write_bytes(audio_resp.content)

        logger.info(
            "RemoteGpuMusicGenerator: audio saved",
            extra={"job_id": job_id, "output": str(output_path)},
        )
        return output_path

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers — adapt these to match your worker's API contract
    # ─────────────────────────────────────────────────────────────────────────

    async def _ensure_model_loaded(self, client: httpx.AsyncClient) -> None:
        """
        Check worker health; if the model is not loaded, call /load-model.

        This is intentionally lightweight — it adds one GET /health round-trip
        per job so the worker always has a model ready before generation starts.

        TODO: If your worker has slow model loading (e.g. 30+ seconds), consider
        calling POST /load-model eagerly when the main app starts rather than
        lazily per job.
        """
        try:
            health_resp = await client.get(f"{self._base_url}/health")
            health_resp.raise_for_status()
            health = health_resp.json()
        except httpx.RequestError as exc:
            raise GenerationError(
                f"Cannot reach worker at {self._base_url}. "
                f"Is the gpu-worker service running? Error: {exc}",
                provider=self.name,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(
                f"Worker health check failed (HTTP {exc.response.status_code}).",
                provider=self.name,
            ) from exc

        if not health.get("model_loaded", False):
            logger.info(
                "RemoteGpuMusicGenerator: model not loaded — calling /load-model"
            )
            try:
                load_resp = await client.post(
                    f"{self._base_url}/load-model",
                    json={},  # Worker uses its DEFAULT_MODEL_NAME
                )
                load_resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise GenerationError(
                    f"Failed to load model on worker: {exc}",
                    provider=self.name,
                ) from exc

    @staticmethod
    def _build_payload(request: GenerationRequest, job_id: str) -> dict:
        """
        Serialise the generation request into the JSON body the worker expects.

        Matches the GenerateRequest schema defined in artifacts/gpu-worker/app/models.py.

        TODO: Adjust field names / structure if you're using a different worker.
        """
        return {
            "job_id": job_id,
            "prompt": request.prompt,
            "lyrics": request.lyrics,
            "style_preset": request.style_preset.value,
            "duration_sec": request.duration_sec,
            "seed": request.seed,
        }

    @staticmethod
    def _parse_audio_url(response_data: dict) -> str:
        """
        Extract the audio download URL from the worker's GenerateResponse.

        Expected: { "audio_url": "http://worker:9000/output/<job_id>.wav", ... }

        TODO: Adjust the key name if your worker uses a different response schema.
        """
        audio_url = response_data.get("audio_url")
        if not audio_url:
            raise GenerationError(
                f"Worker response missing 'audio_url'. Got: {response_data}",
                provider="RemoteGpuMusicGenerator",
            )
        return audio_url
