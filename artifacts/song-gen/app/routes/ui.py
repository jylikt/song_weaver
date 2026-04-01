"""
UI route — serves the server-rendered HTML interface.

GET / → renders templates/index.html via Jinja2
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter(tags=["ui"])

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request) -> HTMLResponse:
    from app.config import settings

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "active_provider": settings.generator_provider.value,
            "app_env": settings.app_env.value,
        },
    )
