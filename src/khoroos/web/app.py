"""FastAPI application serving the Khoroos web UI and its JSON API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from khoroos.pipeline.runner import AnalysisRunner

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from khoroos import __version__
from khoroos.config import Settings, get_settings
from khoroos.pipeline.jobs import JobManager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    settings: Settings | None = None, *, runner: AnalysisRunner | None = None
) -> FastAPI:
    """Build an app whose lifespan owns its worker; injected runners remain caller-owned.

    Constructing the app does not start a thread or create job storage. ASGI startup
    initializes ``app.state.jobs`` and shutdown asks its worker to finish and release models.
    """
    settings = settings or (runner.settings if runner is not None else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        jobs = JobManager(settings, runner=runner)
        app.state.jobs = jobs
        try:
            yield
        finally:
            # Joining the analysis thread must not block the application's event loop.
            await run_in_threadpool(jobs.close)

    app = FastAPI(
        title="Khoroos",
        description="Poultry behavior analysis from farm video.",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    app.state.settings = settings

    from khoroos.web.routes import router

    app.include_router(router, prefix="/api")

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    else:  # pragma: no cover - only if the package was built without assets
        logger.warning("Static assets missing at %s; UI will not be served.", STATIC_DIR)

    return app
