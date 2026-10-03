"""Vector-space API tests: real PCA math, fake database and embedder.

The seams are ``_make_store`` (a database the hermetic env does not have) and
``_get_embedder`` (no API key in tests). Everything else — fitting, caching,
serialisation, ranking order — is the production code path.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import reglens.api.vectors as vectors_module
from reglens.api.main import create_app


def fake_embeddings(n_chunks: int = 60, dim: int = 8) -> list[dict]:
    """Three document clusters, so the PCA fit has real structure to find."""
    rng = np.random.default_rng(11)
    docs = ["rbi_md_a", "rbi_md_b", "rbi_md_c"]
    rows = []
    for index in range(n_chunks):
        doc = docs[index % len(docs)]
        center = np.full(dim, float(index % len(docs)) * 5.0)
        rows.append(
            {
                "chunk_id": f"{doc}__fixed__{index:05d}",
                "doc_id": doc,
                "page_start": index % 40 + 1,
                "doc_title": f"Direction {doc[-1].upper()}",
                "vector": (center + rng.normal(scale=0.2, size=dim)).tolist(),
            }
        )
    return rows


class FakeStore:
    def __init__(self, rows: list[dict] | None = None, *, fail: bool = False) -> None:
        self.rows = rows if rows is not None else fake_embeddings()
        self.fail = fail
        self.all_embeddings_calls = 0

    def count_chunks(self) -> int:
        if self.fail:
            raise RuntimeError("connection refused")
        return len(self.rows)

    def all_embeddings(self) -> list[dict]:
        if self.fail:
            raise RuntimeError("connection refused")
        self.all_embeddings_calls += 1
        # Fresh dicts every call: the production code pops "vector" off them.
        return [dict(row) for row in self.rows]

    def score_all(self, vector: list[float]) -> list[tuple[str, float]]:
        # Distinct descending scores: the endpoint must preserve ranking order.
        return [
            (row["chunk_id"], 1.0 - 0.01 * index) for index, row in enumerate(self.rows)
        ]


class FakeEmbedder:
    name = "fake-embed"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_query_with_usage(self, question: str) -> tuple[list[float], int]:
        return [0.5] * self.dim, 7


def make_client(monkeypatch: pytest.MonkeyPatch, store: FakeStore) -> TestClient:
    monkeypatch.setattr(vectors_module, "_make_store", lambda settings: store)
    monkeypatch.setattr(
        vectors_module, "_get_embedder", lambda request: FakeEmbedder(dim=8)
    )
    return TestClient(create_app())


def test_vectors_endpoint_fits_the_space_and_caches_the_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    client = make_client(monkeypatch, store)

    response = client.get("/vectors")
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "pca"
    assert body["dimensions"] == 3
    assert body["chunks_total"] == 60
    assert len(body["chunks"]) == 60
    assert len(body["variance_explained"]) == 3
    assert sum(body["variance_explained"]) <= 1.0 + 1e-9
    assert body["variance_explained"] == sorted(body["variance_explained"], reverse=True)
    # Documents are deduplicated for the legend.
    assert {doc["doc_id"] for doc in body["documents"]} == {
        "rbi_md_a",
        "rbi_md_b",
        "rbi_md_c",
    }
    assert {"chunk_id", "doc_id", "page", "x", "y", "z"} <= set(body["chunks"][0])

    assert client.get("/vectors").status_code == 200
    assert store.all_embeddings_calls == 1  # second GET reused the fitted frame


def test_vectors_query_projects_and_ranks_in_the_real_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    client = make_client(monkeypatch, store)

    response = client.post("/vectors/query", json={"question": "What is the LCR?"})
    assert response.status_code == 200
    body = response.json()
    assert set(body["query"]) == {"x", "y", "z"}
    assert body["embed_input_tokens"] == 7
    assert body["embed_model"] == "fake-embed"
    assert body["top_k"] >= 1
    assert len(body["ranked"]) == 60
    scores = [entry["score"] for entry in body["ranked"]]
    assert scores == sorted(scores, reverse=True)
    assert [entry["chunk_id"] for entry in body["ranked"]] == [
        row["chunk_id"] for row in store.rows
    ]


def test_vectors_query_validates_the_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(monkeypatch, FakeStore())
    assert client.post("/vectors/query", json={"question": "x"}).status_code == 422


def test_vectors_endpoint_reports_a_dead_index_as_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(monkeypatch, FakeStore(fail=True))
    response = client.get("/vectors")
    assert response.status_code == 503
    assert "connection refused" in response.json()["detail"]
