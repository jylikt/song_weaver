"""
Entry point for the GPU Worker service.

Run with:
    python main.py
or:
    uvicorn main:app --host 0.0.0.0 --port 9000 --reload

PORT env var is respected so Replit can assign the port.
"""

import os

import uvicorn

from app.main import app  # noqa: F401  (re-exported so `uvicorn main:app` works)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("APP_ENV", "development") == "development",
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
