"""API tests.

The important assertion is that ``/ask`` fails loudly (501) instead of returning an
unsourced answer, and that ``/health`` still works with no database: a fresh clone must be
able to start the API and see the truth about what is and is not built.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from reglens.api.main import create_app


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
    assert len(body["summary"]) == 5
    assert body["embedding"] == "openai/text-embedding-3-small"
    assert body["generator"] == "gpt-4o-mini"
    assert body["pricing"]["verified_on"] == "2026-10-02"
    assert "gpt-4o-mini" in body["pricing"]["models"]


def test_ask_returns_501_and_refuses_to_guess(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "What is the LCR requirement?"})
    assert response.status_code == 501
    body = response.json()
    assert body["status"] == "not_implemented"
    assert body["phase"] == 1
    assert "Phase 1" in body["detail"]
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
    assert response.status_code == 501  # not implemented, but the schema is honoured


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
