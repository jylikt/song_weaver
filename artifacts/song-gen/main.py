"""
Entry point — run with:
    python main.py
or:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

The PORT environment variable is respected so Replit can assign the port.
"""

import os

import uvicorn

from app.main import app  # noqa: F401  (re-exported so uvicorn main:app works)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("APP_ENV", "development") == "development",
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
