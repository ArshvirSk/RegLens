"""HTTP routes.

``/ask`` runs the Phase 1 pipeline (retrieve → grounded generate) and answers with
citations or an explicit refusal; the pipeline is assembled in
:mod:`reglens.routing.pipeline` so the eval harness measures exactly what the API
serves. Every failure mode — missing key, dead database, model error — returns 503 with
a trace id rather than an answer, because an empty 200 would be indistinguishable from
a real refusal.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from reglens import __version__
from reglens.api.deps import get_app_settings, get_db_health, get_pool
from reglens.api.schemas import (
    DISCLAIMER,
    AskRequest,
    AskResponse,
    Citation,
    ConfigResponse,
    HealthResponse,
    ReadyResponse,
    TracesResponse,
    TraceSummary,
)
from reglens.config import load_experiment
from reglens.config.experiment import ExperimentConfig
from reglens.config.settings import MissingApiKeyError
from reglens.generation.answerer import CITATION_RE
from reglens.observability.cost import describe_pricing
from reglens.observability.logging import get_logger
from reglens.observability.tracing import Trace
from reglens.retrieval.base import RetrievedChunk, SearchFilters
from reglens.routing.pipeline import Pipeline, build_pipeline

logger = get_logger("reglens.api")

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


def _get_pipeline(request: Request, experiment: ExperimentConfig) -> Pipeline:
    """Per-process pipeline cache, keyed by experiment name.

    Built lazily so ``/health`` works with no key and no database. Tests monkeypatch
    this function to inject fakes instead of reaching for the network.
    """
    cache = getattr(request.app.state, "pipelines", None)
    if cache is None:
        cache = {}
        request.app.state.pipelines = cache
    if experiment.name not in cache:
        cache[experiment.name] = build_pipeline(get_app_settings(), experiment=experiment)
    return cache[experiment.name]


def _map_citations(labels: list[str], context: list[RetrievedChunk]) -> list[Citation]:
    """Attach each ``[title, p.N, clause]`` label to the retrieved chunk it names.

    Unmatched labels are dropped, not guessed: a citation that maps to nothing in the
    retrieved context is exactly what the Phase 4 verifier will flag, and inventing a
    target here would hide it.
    """
    mapped: list[Citation] = []
    for label in labels:
        match = CITATION_RE.search(label)
        if not match:
            continue
        title, page, clause = match.groups()
        page_number = int(page)
        candidates = [chunk for chunk in context if chunk.doc_title == title.strip()]
        if not candidates:
            logger.warning("citation title not in context", extra={"label": label})
            continue
        chosen = next(
            (
                chunk
                for chunk in candidates
                if chunk.page_start is not None
                and chunk.page_end is not None
                and chunk.page_start <= page_number <= chunk.page_end
            ),
            candidates[0],
        )
        trimmed_clause = clause.strip()
        mapped.append(
            Citation(
                doc_id=chosen.doc_id,
                doc_title=chosen.doc_title,
                page=page_number,
                clause=None if trimmed_clause in {"", "n/a"} else trimmed_clause,
                chunk_id=chosen.chunk_id,
                quote=chosen.text[:200],
            )
        )
    return mapped


@ask_router.post(
    "/ask",
    response_model=AskResponse,
)
async def ask(request: Request, payload: AskRequest) -> Any:
    """Retrieve and answer with citations, or refuse explicitly when evidence is thin."""
    settings = get_app_settings()
    experiment = load_experiment(payload.experiment or settings.experiment_config)
    trace = Trace(
        "ask",
        config_hash=experiment.fingerprint(),
        corpus_version=settings.corpus_version,
        route="text",
        settings=settings,
        metadata={"question_chars": len(payload.question)},
    )

    filters: SearchFilters | None = None
    if payload.filters:
        filters = SearchFilters(
            issuer=payload.filters.issuer,
            doc_type=payload.filters.doc_type,
            fiscal_period=payload.filters.fiscal_period,
            corpus_version=payload.filters.corpus_version,
            as_of_date=payload.as_of_date if experiment.retrieval.temporal_filter else None,
        )
    elif payload.as_of_date and experiment.retrieval.temporal_filter:
        filters = SearchFilters(as_of_date=payload.as_of_date)

    try:
        pipeline = _get_pipeline(request, experiment)
    except MissingApiKeyError as exc:
        record = await trace.afinish(pool=get_pool(request), error=f"missing api key: {exc}")
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "trace_id": record.trace_id},
        )

    try:
        with trace.span("retrieve") as span:
            result = await asyncio.to_thread(
                pipeline.ask, payload.question, top_k=payload.top_k, filters=filters
            )
            span.add_metadata(chunks=len(result.chunks))
            if result.embed_input_tokens:
                span.record_usage(
                    result.embed_model or experiment.indexing.embedding_model,
                    input_tokens=result.embed_input_tokens,
                    output_tokens=0,
                )
        with trace.span("generate") as span:
            answer = result.answer
            span.record_usage(
                answer.model or experiment.generation.model,
                input_tokens=answer.input_tokens,
                output_tokens=answer.output_tokens,
            )
            span.add_metadata(refused=answer.refused, citations_parsed=len(answer.citations))
    except Exception as exc:  # dead DB, provider outage, bad response — never an empty 200
        logger.error("ask pipeline failed", extra={"error": str(exc)})
        record = await trace.afinish(pool=get_pool(request), error=str(exc))
        return JSONResponse(
            status_code=503,
            content={"detail": f"pipeline failed: {exc}", "trace_id": record.trace_id},
        )

    citations = _map_citations(answer.citations, result.chunks)
    trace.add_metadata(citations_mapped=len(citations))
    record = await trace.afinish(pool=get_pool(request))
    return AskResponse(
        answer=answer.text,
        citations=citations,
        refused=answer.refused,
        refusal_reason="no_evidence" if answer.refused else None,
        route="text",
        config_hash=experiment.fingerprint(),
        corpus_version=settings.corpus_version,
        trace_id=record.trace_id,
        latency_ms=record.latency_ms,
        cost_usd=float(record.cost_usd),
        model=answer.model,
    )


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
