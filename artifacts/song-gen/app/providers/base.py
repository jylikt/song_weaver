"""
Abstract base class for music generator providers.

To add a new backend (e.g., a cloud API, a fine-tuned local model, a
third-party service), subclass BaseMusicGenerator and implement
`generate_song`.  Then register it in `providers/factory.py`.
"""

from __future__ import annotations

import abc
from pathlib import Path

from app.models.generation import GenerationRequest


class BaseMusicGenerator(abc.ABC):
    """
    Contract every music generation backend must fulfill.

    The `generate_song` method is the single integration point.
    It receives a validated `GenerationRequest` and the resolved
    output directory, and must return the path to the generated audio
    file (any format — .wav, .mp3, .ogg, etc.).

    Raise `GenerationError` on any failure so the job service can
    capture a clean error message.
    """

    @abc.abstractmethod
    async def generate_song(
        self,
        request: GenerationRequest,
        output_dir: Path,
        job_id: str,
    ) -> Path:
        """
        Generate a song and save it to `output_dir`.

        Args:
            request:    Validated generation parameters (prompt, lyrics,
                        style, duration_sec, seed).
            output_dir: Directory where the output file should be written.
            job_id:     Unique job identifier — use it as the filename stem
                        so outputs are traceable.

        Returns:
            Absolute or relative Path to the generated audio file.

        Raises:
            GenerationError: If generation fails for any reason.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable provider name for logging/metrics."""
        return self.__class__.__name__


class GenerationError(Exception):
    """Raised by providers when song generation fails."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message)
        self.provider = provider
