"""Tracing skeleton: per-stage timing, token counts and cost per request (FR6).

Three sinks, in increasing order of setup cost:

1. **JSON logs** — always on, zero dependencies.
2. **Postgres** (``request_traces`` + ``request_stages``) — the queryable store behind the
   latency/cost dashboard. Best effort: a tracing failure never fails a request.
3. **OTLP** (OpenTelemetry) or Langfuse — opt-in via ``OTEL_EXPORTER_OTLP_ENDPOINT`` and
   only if the package is installed. Neither is a hard dependency, so a fresh clone runs
   without a collector.

Usage::

    trace = Trace("ask", config_hash=cfg.fingerprint(), route="text")
    with trace.span("retrieve") as span:
        chunks = retrieve(query)
        span.add_metadata(chunks=len(chunks))
    with trace.span("generate") as span:
        answer = llm(...)
        span.record_usage("gpt-4o-mini", input_tokens=1200, output_tokens=180)
    record = await trace.afinish(pool=pool)
"""

from __future__ import annotations

import importlib
import importlib.util
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from reglens.config import Settings, get_settings
from reglens.observability.cost import UnknownModelError, estimate_cost
from reglens.observability.logging import get_logger, safe_extra

logger = get_logger("reglens.trace")


@dataclass
class StageRecord:
    """One pipeline stage within a trace."""

    name: str
    seq: int
    latency_ms: int = 0
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecord:
    """A completed trace, ready to log and persist."""

    trace_id: str
    name: str
    query_id: str | None
    route: str | None
    model: str | None
    corpus_version: str | None
    config_hash: str | None
    started_at: datetime
    finished_at: datetime
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None
    metadata: dict[str, Any]
    stages: list[StageRecord]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["finished_at"] = self.finished_at.isoformat()
        return data


@dataclass
class Span:
    """A timed stage. ``with trace.span(...) as span`` is the intended use."""

    name: str
    seq: int
    trace: Trace
    started: float = field(default_factory=time.perf_counter)
    record: StageRecord = field(init=False)

    def __post_init__(self) -> None:
        self.record = StageRecord(name=self.name, seq=self.seq)

    # ------------------------------------------------------------------ API
    def record_usage(self, model: str, *, input_tokens: int, output_tokens: int) -> None:
        """Attach token usage and its cost to this stage."""
        self.record.model = model
        self.record.input_tokens += input_tokens
        self.record.output_tokens += output_tokens
        try:
            self.record.cost_usd += estimate_cost(model, input_tokens, output_tokens)
        except UnknownModelError:
            # An unpriced model must be loud, not silent: cost would otherwise read 0.
            logger.warning(
                "unknown model price; cost not attributed",
                extra=safe_extra(event="unknown_model_price", model=model, stage=self.name),
            )
        self.trace.record_model(model)

    def add_metadata(self, **fields: Any) -> None:
        self.record.metadata.update(fields)

    def finish(self) -> StageRecord:
        self.record.latency_ms = int((time.perf_counter() - self.started) * 1000)
        return self.record


class Trace:
    """Collects stages for one request and produces a :class:`TraceRecord`."""

    def __init__(
        self,
        name: str,
        *,
        query_id: str | None = None,
        route: str | None = None,
        config_hash: str | None = None,
        corpus_version: str | None = None,
        settings: Settings | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.trace_id = uuid.uuid4().hex
        self.name = name
        self.query_id = query_id
        self.route = route
        self.config_hash = config_hash
        self.corpus_version = corpus_version
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.stages: list[StageRecord] = []
        self.model: str | None = None
        self._started = time.perf_counter()
        self._started_at = datetime.now(UTC)
        self._seq = 0
        self._otel_span: Any | None = _maybe_otel_span(name)

    # ------------------------------------------------------------------ stages
    @contextmanager
    def span(self, name: str) -> Iterator[Span]:
        """Time a stage. Exceptions are recorded and re-raised."""
        self._seq += 1
        span = Span(name=name, seq=self._seq, trace=self)
        try:
            yield span
        except Exception as exc:
            span.add_metadata(error=str(exc))
            span.finish()
            self.stages.append(span.record)
            raise
        else:
            self.stages.append(span.finish())

    def record_model(self, model: str) -> None:
        """Track the model in use; gen-AI traces report the last model seen."""
        self.model = model

    def add_metadata(self, **fields: Any) -> None:
        self.metadata.update(fields)

    def stage(self, name: str) -> StageRecord | None:
        for record in self.stages:
            if record.name == name:
                return record
        return None

    # ------------------------------------------------------------------ totals
    @property
    def input_tokens(self) -> int:
        return sum(stage.input_tokens for stage in self.stages)

    @property
    def output_tokens(self) -> int:
        return sum(stage.output_tokens for stage in self.stages)

    @property
    def cost_usd(self) -> float:
        return round(sum(stage.cost_usd for stage in self.stages), 8)

    def finish(self, error: str | None = None) -> TraceRecord:
        """Close the trace, log it as one JSON line, and return the record."""
        finished_at = datetime.now(UTC)
        record = TraceRecord(
            trace_id=self.trace_id,
            name=self.name,
            query_id=self.query_id,
            route=self.route,
            model=self.model,
            corpus_version=self.corpus_version,
            config_hash=self.config_hash,
            started_at=self._started_at,
            finished_at=finished_at,
            latency_ms=int((time.perf_counter() - self._started) * 1000),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
            error=error,
            metadata=self.metadata,
            stages=self.stages,
        )
        if self._otel_span is not None:  # pragma: no cover - requires an OTLP collector
            try:
                self._otel_span.set_attribute("reglens.latency_ms", record.latency_ms)
                self._otel_span.set_attribute("reglens.cost_usd", record.cost_usd)
                self._otel_span.end()
            except Exception:
                pass
        if self.settings.tracing_enabled:
            logger.info(
                "trace",
                extra=safe_extra(
                    event="trace",
                    trace_id=record.trace_id,
                    query_id=record.query_id,
                    trace_name=record.name,
                    route=record.route,
                    model=record.model,
                    latency_ms=record.latency_ms,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cost_usd=record.cost_usd,
                    config_hash=record.config_hash,
                    corpus_version=record.corpus_version,
                    error=record.error,
                    stages=[
                        {
                            "name": stage.name,
                            "latency_ms": stage.latency_ms,
                            "cost_usd": stage.cost_usd,
                        }
                        for stage in record.stages
                    ],
                ),
            )
        return record

    async def afinish(self, *, pool: Any | None = None, error: str | None = None) -> TraceRecord:
        """``finish()`` plus a best-effort Postgres write when a pool is available."""
        record = self.finish(error=error)
        if pool is not None and self.settings.trace_db_write:
            from reglens.db.traces import insert_trace  # local import: avoids a cycle

            await insert_trace(pool, record)
        return record


def _maybe_otel_span(name: str) -> Any | None:
    """Return an OTel span when a collector is configured *and* the SDK is installed."""
    settings = get_settings()
    if not settings.otel_exporter_otlp_endpoint:
        return None
    if importlib.util.find_spec("opentelemetry") is None:
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry is not installed; "
            "traces go to logs and Postgres only",
            extra={"event": "otel_unavailable"},
        )
        return None
    try:  # pragma: no cover - depends on optional runtime dependency
        trace_api = importlib.import_module("opentelemetry.trace")
        return trace_api.get_tracer("reglens").start_span(name)
    except Exception:
        return None
