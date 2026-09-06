"""FastAPI application serving the Khoroos web UI and its JSON API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from khoroos.pipeline.runner import AnalysisRunner

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from khoroos import __version__
from khoroos.config import Settings, get_settings
from khoroos.pipeline.jobs import JobManager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    settings: Settings | None = None, *, runner: AnalysisRunner | None = None
) -> FastAPI:
    """Build the application. The job manager is attached to app state."""
    settings = settings or (runner.settings if runner is not None else get_settings())
    jobs = JobManager(settings, runner=runner)

    app = FastAPI(
        title="Khoroos",
        description="Poultry behavior analysis from farm video.",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.state.settings = settings
    app.state.jobs = jobs

    from khoroos.web.routes import router

    app.include_router(router, prefix="/api")

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    else:  # pragma: no cover - only if the package was built without assets
        logger.warning("Static assets missing at %s; UI will not be served.", STATIC_DIR)

    return app
