"""FastAPI application factory."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from reglens import __version__
from reglens.api.routes import ask_router, health_router, observability_router
from reglens.config import get_settings
from reglens.db import close_pool, create_pool
from reglens.db import health as db_health
from reglens.observability.logging import get_logger, safe_extra, setup_logging

logger = get_logger("reglens.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Set up logging and the DB pool; never block startup on a missing database."""
    settings = get_settings()
    setup_logging(settings.log_level)
    app.state.settings = settings
    app.state.pool = None
    app.state.db_health = {"status": "not_configured"}
    try:
        app.state.pool = await create_pool(settings)
        app.state.db_health = await db_health(app.state.pool)
    except Exception as exc:
        app.state.db_health = {"status": "unavailable", "detail": str(exc)}
        logger.warning(
            "database unavailable at startup", extra={"event": "db_unavailable", "detail": str(exc)}
        )
    logger.info(
        "api ready",
        extra=safe_extra(
            event="startup",
            version=__version__,
            env=settings.env,
            corpus_version=settings.corpus_version,
            experiment=settings.experiment_config,
            db=app.state.db_health.get("status"),
        ),
    )
    try:
        yield
    finally:
        await close_pool(app.state.pool)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="RegLens",
        version=__version__,
        summary="Cited, temporally-aware answers over RBI/SEBI regulation and bank filings.",
        description=(
            "Informational only. Not legal or investment advice. "
            "Phase 0 exposes health, config and tracing endpoints; answering arrives in Phase 1."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next: Any) -> Any:
        """Attach a server-side duration header; per-stage timing lives in traces."""
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-reglens-duration-ms"] = str(int((time.perf_counter() - started) * 1000))
        response.headers["x-reglens-disclaimer"] = "informational-only-not-advice"
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        logger.error(
            "unhandled error",
            extra={"event": "unhandled_error", "path": request.url.path, "error": str(exc)},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal error", "path": request.url.path},
        )

    app.include_router(health_router)
    app.include_router(ask_router)
    app.include_router(observability_router)
    return app


app = create_app()
