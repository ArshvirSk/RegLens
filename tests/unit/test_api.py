"""API tests.

The contract that matters: ``/ask`` never invents an answer. Without credentials it
returns 503 with a trace id (fail closed, no empty 200); with an injected pipeline it
answers with citations mapped to real retrieved chunks, refuses explicitly, or reports
a pipeline failure as 503. ``/health`` must keep working with no database.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from reglens.api.main import create_app
from reglens.generation.base import Answer
from reglens.retrieval.base import RetrievedChunk
from reglens.routing.pipeline import AskResult


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A client with no database: the pool is stubbed out to keep tests hermetic."""
    app = create_app()

    async def no_pool(*args: object, **kwargs: object) -> None:
        raise OSError("no database in unit tests")

    import reglens.api.main as main_module

    original = main_module.create_pool
    main_module.create_pool = no_pool  # type: ignore[assignment]
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        main_module.create_pool = original  # type: ignore[assignment]


def test_health_reports_degraded_without_database(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "unavailable"
    assert body["corpus_version"] == "0.1.0"
    assert body["experiment"] == "baseline_naive"
    assert len(body["config_hash"]) == 64
    assert body["disclaimer"] == "Informational only. Not legal or investment advice."


def test_health_carries_the_disclaimer_header(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers["x-reglens-disclaimer"] == "informational-only-not-advice"
    assert int(response.headers["x-reglens-duration-ms"]) >= 0


def test_config_endpoint_exposes_the_ablation_toggles(client: TestClient) -> None:
    body = client.get("/config").json()
    assert body["experiment"] == "baseline_naive"
    assert len(body["summary"]) == 6
    assert body["embedding"] == "gemini/gemini-embedding-001"
    assert body["generator"] == "gemini-3.5-flash-lite"
    assert body["pricing"]["verified_on"] == "2026-10-02"
    assert "gemini-3.5-flash-lite" in body["pricing"]["models"]


def make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="rbi_md_test__fixed__00000",
        doc_id="rbi_md_test",
        text="The liquidity coverage ratio shall be not less than one hundred per cent.",
        doc_title="Master Direction - Test",
        issuer="RBI",
        doc_type="master_direction",
        page_start=4,
        page_end=5,
        clause_path="3.1",
        section_title=None,
        issue_date=None,
        effective_date=None,
        score=0.91,
        dense_score=0.91,
    )


def make_result(*, refused: bool = False) -> AskResult:
    text = (
        "INSUFFICIENT EVIDENCE"
        if refused
        else "The LCR must be at least 100%. [Master Direction - Test, p.4, 3.1]"
    )
    answer = Answer(
        text=text,
        citations=[] if refused else ["[Master Direction - Test, p.4, 3.1]"],
        refused=refused,
        refusal_reason="no_evidence" if refused else None,
        model="gemini-3.5-flash-lite",
        input_tokens=120,
        output_tokens=30,
        latency_ms=5,
    )
    return AskResult(
        answer=answer,
        chunks=[] if refused else [make_chunk()],
        embed_input_tokens=12,
        embed_model="gemini-embedding-001",
    )


def test_ask_fails_closed_without_credentials(client: TestClient) -> None:
    """No key in .env or the environment: 503 with a trace id, never an empty 200."""
    response = client.post("/ask", json={"question": "What is the LCR requirement?"})
    assert response.status_code == 503
    body = response.json()
    assert "GEMINI_API_KEY" in body["detail"]
    assert body["trace_id"]


def test_ask_answers_with_citations_via_the_pipeline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reglens.api.routes as routes_module

    captured: dict[str, object] = {}

    def fake_get_pipeline(request: object, experiment: object):  # type: ignore[no-untyped-def]
        def ask(question: str, *, top_k: int | None = None, filters: object = None):  # type: ignore[no-untyped-def]
            captured.update(question=question, top_k=top_k, filters=filters)
            return make_result()

        return SimpleNamespace(ask=ask)

    monkeypatch.setattr(routes_module, "_get_pipeline", fake_get_pipeline)
    response = client.post(
        "/ask",
        json={
            "question": "What is the LCR requirement?",
            "filters": {"issuer": "RBI"},
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["answer"].startswith("The LCR")
    assert body["disclaimer"] == "Informational only. Not legal or investment advice."
    assert body["route"] == "text"
    assert body["model"] == "gemini-3.5-flash-lite"
    assert body["trace_id"]
    assert len(body["config_hash"]) == 64
    # The citation label maps to the real retrieved chunk (FR1's shape).
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["doc_id"] == "rbi_md_test"
    assert citation["page"] == 4
    assert citation["clause"] == "3.1"
    assert citation["chunk_id"] == "rbi_md_test__fixed__00000"
    assert citation["quote"]
    # The request's top_k and filters reach the pipeline unchanged.
    assert captured["top_k"] == 5
    filters = captured["filters"]
    assert filters.issuer == "RBI"  # type: ignore[union-attr]
    assert (
        filters.as_of_date is None
    )  # temporal filter is off in the baseline  # type: ignore[union-attr]


def test_ask_refusal_is_explicit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import reglens.api.routes as routes_module

    monkeypatch.setattr(
        routes_module,
        "_get_pipeline",
        lambda *args, **kwargs: SimpleNamespace(ask=lambda *a, **k: make_result(refused=True)),
    )
    response = client.post("/ask", json={"question": "Who won the 1998 cricket final?"})
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["refusal_reason"] == "no_evidence"
    assert body["citations"] == []


def test_ask_reports_pipeline_failures_as_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reglens.api.routes as routes_module

    def boom(*args: object, **kwargs: object) -> AskResult:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(routes_module, "_get_pipeline", lambda *a, **k: SimpleNamespace(ask=boom))
    response = client.post("/ask", json={"question": "What is the LCR requirement?"})
    assert response.status_code == 503
    body = response.json()
    assert "connection refused" in body["detail"]
    assert body["trace_id"]


def test_ask_validates_the_request_body(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 5000}).status_code == 422
    assert (
        client.post("/ask", json={"question": "valid question", "unexpected": 1}).status_code == 422
    )


def test_ask_accepts_as_of_date_and_filters(client: TestClient) -> None:
    response = client.post(
        "/ask",
        json={
            "question": "What is the LCR requirement?",
            "as_of_date": "2024-06-30",
            "filters": {"issuer": "RBI", "doc_type": "master_direction"},
            "top_k": 5,
        },
    )
    # No credentials in the hermetic test env: schema is honoured, then fail closed.
    assert response.status_code == 503
    assert response.json()["trace_id"]


def test_traces_endpoint_is_empty_without_a_database(client: TestClient) -> None:
    body = client.get("/traces").json()
    assert body == {"traces": [], "percentiles": {}}


def test_pricing_endpoint_publishes_the_rate_table(client: TestClient) -> None:
    body = client.get("/pricing").json()
    assert body["pricing"]["unit"] == "USD per 1M tokens"
    assert body["disclaimer"].startswith("Informational only")


def test_ready_reports_not_ready_without_a_database(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert body["ready"] is False
    assert body["database"]["status"] == "unavailable"
