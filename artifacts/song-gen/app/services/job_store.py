"""
In-memory job store.

Stores active GenerationJob objects keyed by job_id.

Production upgrade path:
  Replace this with a Redis-backed or PostgreSQL-backed store.
  The interface (get / put / list) is intentionally minimal so swapping
  the backend only requires changing this one file.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from app.models.generation import GenerationJob


class InMemoryJobStore:
    """Thread-safe in-memory job store backed by an asyncio Lock."""

    def __init__(self) -> None:
        self._store: dict[str, GenerationJob] = {}
        self._lock = asyncio.Lock()

    async def put(self, job: GenerationJob) -> None:
        async with self._lock:
            self._store[job.job_id] = job

    async def get(self, job_id: str) -> Optional[GenerationJob]:
        async with self._lock:
            return self._store.get(job_id)

    async def list_all(self) -> list[GenerationJob]:
        async with self._lock:
            return list(self._store.values())

    async def delete(self, job_id: str) -> bool:
        async with self._lock:
            return self._store.pop(job_id, None) is not None


# Single shared store instance
job_store = InMemoryJobStore()
