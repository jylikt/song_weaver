"""
Provider factory.

Returns the appropriate BaseMusicGenerator for a given GenerationMode.
Per-request mode selection is supported so callers can choose a backend
per job rather than relying solely on the server-level env var default.

To register a new provider:
  1. Create a class in app/providers/ that subclasses BaseMusicGenerator
  2. Add it to _REGISTRY below
  3. Add the corresponding enum value to GeneratorProvider in config.py
     and GenerationMode in models/generation.py
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from app.config import GeneratorProvider, settings
from app.models.generation import GenerationMode
from app.providers.base import BaseMusicGenerator

logger = logging.getLogger(__name__)


# ── Cached singletons — model weights are loaded once per class ───────────────

@lru_cache(maxsize=1)
def _local_generator() -> BaseMusicGenerator:
    from app.providers.local_generator import LocalMusicGenerator
    logger.info("Initialising LocalMusicGenerator")
    return LocalMusicGenerator()


@lru_cache(maxsize=1)
def _remote_gpu_generator() -> BaseMusicGenerator:
    from app.providers.remote_gpu_generator import RemoteGpuMusicGenerator
    logger.info("Initialising RemoteGpuMusicGenerator")
    return RemoteGpuMusicGenerator()


# ── Public API ─────────────────────────────────────────────────────────────────

def get_generator(mode: Optional[GenerationMode] = None) -> BaseMusicGenerator:
    """
    Return the generator instance for the given mode.

    Args:
        mode: Explicit backend selection from the request.
              Falls back to the GENERATOR_PROVIDER env var when None.

    Returns:
        Singleton BaseMusicGenerator instance (cached per class).
    """
    if mode is None:
        # Fall back to the server-level default from env
        resolved = settings.generator_provider
    elif mode == GenerationMode.LOCAL:
        resolved = GeneratorProvider.LOCAL
    elif mode == GenerationMode.REMOTE_GPU:
        resolved = GeneratorProvider.REMOTE_GPU
    else:
        raise ValueError(f"Unhandled GenerationMode: {mode!r}")

    if resolved == GeneratorProvider.LOCAL:
        return _local_generator()
    elif resolved == GeneratorProvider.REMOTE_GPU:
        return _remote_gpu_generator()
    else:
        raise ValueError(
            f"Unknown provider '{resolved}'. "
            f"Valid options: {[p.value for p in GeneratorProvider]}"
        )
