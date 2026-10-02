"""Tracing tests.

FR6 requires a trace with per-stage timings and cost for every request. These tests pin
the contract the eval runner and the Phase 4 dashboard will read: stages are ordered, token
totals sum the stages, and a failing stage is recorded rather than swallowed.
"""

from __future__ import annotations

import json
import logging

import pytest

from reglens.observability.logging import JsonFormatter, safe_extra
from reglens.observability.tracing import Trace


def test_stages_are_recorded_in_order() -> None:
    trace = Trace("ask", route="text")
    with trace.span("retrieve"):
        pass
    with trace.span("generate"):
        pass
    record = trace.finish()
    assert [stage.name for stage in record.stages] == ["retrieve", "generate"]
    assert [stage.seq for stage in record.stages] == [1, 2]
    assert record.latency_ms >= 0


def test_token_totals_and_cost_sum_the_stages() -> None:
    trace = Trace("ask")
    with trace.span("retrieve") as span:
        span.record_usage("text-embedding-3-small", input_tokens=1_000, output_tokens=0)
    with trace.span("generate") as span:
        span.record_usage("gpt-4o-mini", input_tokens=2_000, output_tokens=300)
    record = trace.finish()
    assert record.input_tokens == 3_000
    assert record.output_tokens == 300
    # 1000 * 0.02/M + (2000 * 0.15/M + 300 * 0.60/M)
    assert record.cost_usd == pytest.approx(0.00002 + 0.0003 + 0.00018, rel=1e-6)
    assert record.model == "gpt-4o-mini"


def test_stage_metadata_is_captured() -> None:
    trace = Trace("ask")
    with trace.span("retrieve") as span:
        span.add_metadata(chunks=12, mode="dense")
    record = trace.finish()
    assert record.stages[0].metadata == {"chunks": 12, "mode": "dense"}


def test_failing_stage_is_recorded_and_reraised() -> None:
    trace = Trace("ask")
    with pytest.raises(ValueError, match="boom"), trace.span("generate"):
        raise ValueError("boom")
    record = trace.finish(error="boom")
    assert record.stages[0].metadata["error"] == "boom"
    assert record.error == "boom"


def test_trace_identity_is_carried_through() -> None:
    trace = Trace("eval.question", query_id="q0007", config_hash="deadbeef", corpus_version="0.1.0")
    trace.add_metadata(bank="SBI")
    record = trace.finish()
    assert record.query_id == "q0007"
    assert record.config_hash == "deadbeef"
    assert record.corpus_version == "0.1.0"
    assert record.metadata["bank"] == "SBI"
    assert len(record.trace_id) == 32


def test_unknown_model_is_a_loud_warning_not_a_silent_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="reglens.trace")
    trace = Trace("ask")
    with trace.span("generate") as span:
        span.record_usage("gpt-9-imaginary", input_tokens=100, output_tokens=10)
    record = trace.finish()
    assert record.cost_usd == 0.0
    assert any("unknown model price" in message for message in caplog.messages)


def test_trace_is_logged_as_structured_json(caplog: pytest.LogCaptureFixture) -> None:
    """The emitted line must be valid JSON with the fields the dashboard reads.

    caplog captures the record, not the formatted line, so the formatter is applied here
    explicitly: the JSON contract is what downstream tooling depends on.
    """
    caplog.set_level(logging.INFO, logger="reglens.trace")
    trace = Trace("ask", route="text")
    with trace.span("retrieve"):
        pass
    trace.finish()
    payload = json.loads(JsonFormatter().format(caplog.records[-1]))
    assert payload["event"] == "trace"
    assert payload["route"] == "text"
    # `name` is a reserved LogRecord attribute (the logger name), so the trace name is
    # carried as `trace_name` instead of being silently dropped.
    assert payload["trace_name"] == "ask"
    assert payload["logger"] == "reglens.trace"
    assert payload["stages"][0]["name"] == "retrieve"


def test_reserved_log_record_keys_are_renamed_not_lost() -> None:
    """A field named `name` must survive into the JSON payload, not raise or vanish."""
    formatter = JsonFormatter()
    record = logging.LogRecord("reglens.test", logging.INFO, __file__, 1, "x", None, None)
    record.field_name = "my-trace"
    payload = json.loads(formatter.format(record))
    assert payload["name"] == "my-trace"
    assert payload["logger"] == "reglens.test"


def test_safe_extra_renames_reserved_keys() -> None:
    """Passing reserved names to logging raises KeyError; `safe_extra` must prevent it.

    `name` and `filename` each bit this codebase: the trace log and the migration logger
    both crashed with ``Attempt to overwrite ... in LogRecord`` before this helper existed,
    and the migration crash happened *after* the schema change had committed.
    """
    logger = logging.getLogger("reglens.test.safe_extra")
    logger.info("event", extra=safe_extra(name="my-trace", filename="0001_core.sql"))
    assert safe_extra(name="x") == {"field_name": "x"}
    assert safe_extra(filename="f.sql") == {"field_filename": "f.sql"}
    assert safe_extra(chunks=3) == {"chunks": 3}


def test_json_formatter_includes_extra_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord("reglens.test", logging.INFO, __file__, 1, "hello", None, None)
    record.custom = {"a": 1}
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["custom"] == {"a": 1}
    assert "ts" in payload


def test_record_serialises_to_a_dict() -> None:
    trace = Trace("ask")
    with trace.span("retrieve"):
        pass
    payload = trace.finish().as_dict()
    assert isinstance(payload["started_at"], str)
    assert isinstance(payload["stages"], list)
