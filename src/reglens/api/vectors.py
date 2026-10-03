"""Vector-space view: stored embeddings as a 3D map with a live query overlay.

Two rules keep this honest:

* The map is a **view** — PCA of the stored embeddings, with the variance each axis
  captures reported next to it. It accepts the live query as just another vector, so
  the question lands in the same frame as the corpus.
* The decision list is the **real ranking** — cosine similarity computed by the
  database against the full stored dimension, exactly what ``DenseRetriever`` orders
  by. Nothing is ever ranked by the three projected coordinates.

Fitting runs once per corpus version (and re-fits when the chunk count changes) and
is cached on the app state; every database call goes through ``asyncio.to_thread``
because ``PgVectorStore`` is a sync facade.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from reglens.api.deps import get_app_settings
from reglens.config import load_experiment
from reglens.config.settings import MissingApiKeyError
from reglens.indexing.embeddings import build_embedding_model
from reglens.indexing.projection import fit_pca
from reglens.indexing.vector_store import PgVectorStore
from reglens.observability.logging import get_logger

logger = get_logger("reglens.api.vectors")

vectors_router = APIRouter(tags=["vectors"])


class VectorQueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)


def _make_store(settings: Any) -> PgVectorStore:
    """Store factory — the seam the hermetic API tests patch instead of a database."""
    return PgVectorStore(settings=settings)


def _get_embedder(request: Request) -> Any:
    embedder = getattr(request.app.state, "vectors_embedder", None)
    if embedder is None:
        embedder = build_embedding_model(get_app_settings())
        request.app.state.vectors_embedder = embedder
    return embedder


def _get_vector_space(request: Request) -> dict[str, Any]:
    """Fit (or reuse) the PCA frame for the active corpus version.

    Cached by corpus version *and* chunk count: a reindex that keeps the version must
    not serve stale coordinates, and a refit must not happen on every page load.
    """
    settings = get_app_settings()
    store = _make_store(settings)
    count = store.count_chunks()
    cached = getattr(request.app.state, "vector_space", None)
    if (
        cached is not None
        and cached["corpus_version"] == settings.corpus_version
        and cached["n"] == count
    ):
        return cached
    if count == 0:
        raise ValueError("index is empty; ingest documents before opening the vector space")

    rows = store.all_embeddings()
    matrix = np.asarray([row.pop("vector") for row in rows], dtype=np.float64)
    basis = fit_pca(matrix)
    coords = basis.project(matrix)

    documents: dict[str, dict[str, str]] = {}
    for row in rows:
        documents.setdefault(row["doc_id"], {"doc_id": row["doc_id"], "title": row["doc_title"]})

    space: dict[str, Any] = {
        "corpus_version": settings.corpus_version,
        "n": len(rows),
        "variance_explained": [round(value, 6) for value in basis.explained_variance_ratio],
        "documents": list(documents.values()),
        "chunks": [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "page": row["page_start"],
                "x": round(float(point[0]), 4),
                "y": round(float(point[1]), 4),
                "z": round(float(point[2]), 4),
            }
            for row, point in zip(rows, coords, strict=True)
        ],
        "basis": basis,
    }
    request.app.state.vector_space = space
    return space


@vectors_router.get("/vectors")
async def vector_space(request: Request) -> Any:
    """The corpus as a 3D point cloud, with the fit's variance per axis."""
    try:
        space = await asyncio.to_thread(_get_vector_space, request)
    except Exception as exc:  # empty index, dead database — visible, not a fake empty map
        logger.error("vector space build failed", extra={"error": str(exc)})
        return JSONResponse(status_code=503, content={"detail": f"vector space: {exc}"})
    return {
        "method": "pca",
        "dimensions": 3,
        "corpus_version": space["corpus_version"],
        "chunks_total": space["n"],
        "variance_explained": space["variance_explained"],
        "documents": space["documents"],
        "chunks": space["chunks"],
    }


@vectors_router.post("/vectors/query")
async def vector_query(request: Request, payload: VectorQueryRequest) -> Any:
    """Embed the question, place it in the frame, rank the corpus the real way."""
    settings = get_app_settings()
    try:
        embedder = _get_embedder(request)
        space = await asyncio.to_thread(_get_vector_space, request)
        vector, tokens = await asyncio.to_thread(
            embedder.embed_query_with_usage, payload.question
        )
        store = _make_store(settings)
        scores = await asyncio.to_thread(store.score_all, vector)
    except MissingApiKeyError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    except Exception as exc:
        logger.error("vector query failed", extra={"error": str(exc)})
        return JSONResponse(status_code=503, content={"detail": f"vector query: {exc}"})

    point = space["basis"].project(np.asarray([vector], dtype=np.float64))[0]
    top_k = load_experiment(settings.experiment_config).retrieval.top_k
    return {
        "query": {
            "x": round(float(point[0]), 4),
            "y": round(float(point[1]), 4),
            "z": round(float(point[2]), 4),
        },
        "embed_input_tokens": tokens,
        "embed_model": getattr(embedder, "name", None),
        "top_k": top_k,
        # Full-corpus ranking from the database (cosine similarity, full dimension),
        # best first. The UI colours by this and highlights the first `top_k`.
        "ranked": [
            {"chunk_id": chunk_id, "score": round(score, 6)} for chunk_id, score in scores
        ],
    }
