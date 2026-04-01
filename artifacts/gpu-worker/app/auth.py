"""
Optional bearer-token authentication dependency.

If WORKER_TOKEN is set in the environment, every incoming request must
include the header:

    Authorization: Bearer <token>

If WORKER_TOKEN is empty or unset, all requests are allowed through.
This lets you disable auth in local development without code changes.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Header, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)


async def require_auth(authorization: Optional[str] = Header(None)) -> None:
    """
    FastAPI dependency — raise 401 if the request fails token auth.

    Usage:
        from app.auth import require_auth
        router = APIRouter(dependencies=[Depends(require_auth)])
    """
    token = settings.worker_token
    if not token:
        # Auth disabled — allow all requests (development mode)
        return

    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Expected: 'Authorization: Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, provided_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or provided_token != token:
        logger.warning("Rejected request with invalid token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
