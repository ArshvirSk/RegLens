"""HTTP routes.

Phase 0 exposes the surface the UI and the eval harness will talk to, but only the
endpoints whose behaviour is already defined are implemented. ``/ask`` returns HTTP 501
with an explicit "arrives in Phase 1" body: an unimplemented endpoint must never look
like a model answering from nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request

from reglens import __version__
from reglens.api.deps import get_app_settings, get_db_health, get_pool
from reglens.api.schemas import (
    DISCLAIMER,
    AskRequest,
    AskResponse,
    ConfigResponse,
    HealthResponse,
    NotImplementedResponse,
    ReadyResponse,
    TracesResponse,
    TraceSummary,
)
from reglens.config import load_experiment
from reglens.observability.cost import describe_pricing
from reglens.observability.tracing import Trace

health_router = APIRouter(tags=["meta"])
ask_router = APIRouter(tags=["ask"])
observability_router = APIRouter(tags=["observability"])


@health_router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness plus the identity of the active configuration."""
    settings = get_app_settings()
    experiment = load_experiment(settings.experiment_config)
    health_state = get_db_health(request)
    return HealthResponse(
        status="ok" if health_state.get("status") == "ok" else "degraded",
        version=__version__,
        env=settings.env,
        corpus_version=settings.corpus_version,
        experiment=experiment.name,
        config_hash=experiment.fingerprint(),
        database=str(health_state.get("status", "unknown")),
    )


@health_router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    """Readiness: can this process actually answer questions end to end?

    Re-probes the database on every call rather than serving the startup snapshot: a
    readiness probe that reports "pgvector missing" minutes after the migration created it
    is worse than useless, because it looks like a live answer.
    """
    pool = get_pool(request)
    health_state = get_db_health(request)
    migrations: dict[str, Any] = {"applied": [], "pending": [], "changed": []}
    if pool is not None:
        from reglens.db import health as probe_db_health
        from reglens.db.runner import migration_status

        health_state = await probe_db_health(pool)
        async with pool.acquire() as connection:
            migrations = await migration_status(connection)
    ready_flag = health_state.get("status") == "ok" and not migrations.get("pending")
    return ReadyResponse(ready=bool(ready_flag), database=health_state, migrations=migrations)


@health_router.get("/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    """The active ablation config, its hash, and the prices used for cost accounting."""
    settings = get_app_settings()
    experiment = load_experiment(settings.experiment_config)
    return ConfigResponse(
        experiment=experiment.name,
        config_hash=experiment.fingerprint(),
        corpus_version=settings.corpus_version,
        summary=experiment.summary_lines(),
        embedding=f"{experiment.indexing.embedding_provider}/{experiment.indexing.embedding_model}",
        generator=experiment.generation.model,
        pricing=describe_pricing(),
        paths={
            "manifest": str(settings.manifest_path),
            "raw_dir": str(settings.raw_dir),
            "golden": str(settings.golden_path),
            "results": str(settings.eval_results_dir),
        },
    )


@ask_router.post(
    "/ask",
    response_model=AskResponse,
    responses={501: {"model": NotImplementedResponse, "description": "Capability not built yet"}},
)
async def ask(request: Request, payload: AskRequest) -> Any:
    """Not implemented until Phase 1.

    Returning 501 (rather than an empty 200 answer) keeps the eval harness honest: a
    question with no pipeline behind it is a hard error, not a wrong answer.
    """
    settings = get_app_settings()
    experiment = load_experiment(payload.experiment or settings.experiment_config)
    trace = Trace(
        "ask.rejected",
        config_hash=experiment.fingerprint(),
        corpus_version=settings.corpus_version,
        route="unknown",
        settings=settings,
        metadata={"question_chars": len(payload.question)},
    )
    with trace.span("validate"):
        pass
    record = await trace.afinish(pool=get_pool(request))
    body = NotImplementedResponse(
        detail=(
            "Retrieval and generation are implemented in Phase 1. This endpoint will answer "
            f"with citations to {settings.manifest_path.name} documents at that point; "
            "until then it refuses rather than guessing."
        ),
        phase=1,
    )
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=501, content={**body.model_dump(), "trace_id": record.trace_id})


@observability_router.get("/traces", response_model=TracesResponse)
async def traces(request: Request, limit: int = 50) -> TracesResponse:
    """Recent request traces with latency percentiles (empty until Postgres is up)."""
    pool = get_pool(request)
    if pool is None:
        return TracesResponse(traces=[], percentiles={})
    from reglens.db.traces import fetch_recent_traces, latency_percentiles

    rows = await fetch_recent_traces(pool, limit=max(1, min(limit, 500)))
    percentiles = await latency_percentiles(pool)
    return TracesResponse(
        traces=[
            TraceSummary(
                trace_id=row["trace_id"],
                name=row["name"],
                route=row.get("route"),
                latency_ms=row.get("latency_ms"),
                input_tokens=row.get("input_tokens") or 0,
                output_tokens=row.get("output_tokens") or 0,
                cost_usd=float(row.get("cost_usd") or 0),
                error=row.get("error"),
                started_at=row["started_at"].isoformat()
                if isinstance(row.get("started_at"), datetime)
                else None,
            )
            for row in rows
        ],
        percentiles=percentiles,
    )


@observability_router.get("/pricing")
async def pricing() -> dict[str, object]:
    """The price table every reported cost is derived from."""
    return {
        "pricing": describe_pricing(),
        "checked_at": datetime.now(UTC).isoformat(),
        "disclaimer": DISCLAIMER,
    }
