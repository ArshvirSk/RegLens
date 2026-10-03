"""Hosted embedding backends behind one interface.

The PRD wants two embedding models (one hosted, one open-source) comparable behind a
single protocol, so Phase 1 ships the hosted Gemini implementation plus a factory;
Phase 2 adds the open-source backend as a second implementation of the same protocol.
"""

from __future__ import annotations

from typing import Any

from google.genai import types

from reglens.chunking.fixed import TiktokenTokenizer
from reglens.config import Settings, get_settings
from reglens.indexing.base import EmbeddedBatch, EmbeddingModel

#: Google's documented ceiling for ``embed_content`` contents per request.
MAX_CONTENTS_PER_REQUEST = 100
#: Asymmetric retrieval tasks: documents and queries are embedded with different task
#: types because that is what the model was trained for (verified against the SDK's
#: ``EmbedContentConfig.task_type``, 2026-10-02).
DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"


def _reported_tokens(item: Any) -> int:
    """Token usage from the API when it reports one, else 0.

    The field name differs across SDK generations (``token_count`` is current,
    ``total_tokens`` legacy), and for ``gemini-embedding-001`` the embed endpoint
    returns ``statistics=None`` entirely (verified live, 2026-10-03), so callers fall
    back to a local count instead of recording zero cost.
    """
    statistics = getattr(item, "statistics", None)
    if statistics is None:
        return 0
    value = getattr(statistics, "token_count", None)
    if value is None:
        value = getattr(statistics, "total_tokens", 0)
    return int(value or 0)


class GeminiEmbedding:
    """``gemini-embedding-001`` through the official ``google-genai`` SDK.

    ``client`` is injectable so every unit test exercises the request/response mapping
    against a fake — no key, no network, no cost.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-embedding-001",
        dimensions: int = 3072,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.name = model
        self.dimensions = dimensions
        self._api_key = api_key
        self._client = client

    def _sdk(self) -> Any:
        if self._client is None:
            from google import genai

            # An empty/None key falls back to the SDK's own env lookup, so running with
            # GEMINI_API_KEY exported in the shell works exactly like .env does.
            self._client = genai.Client(api_key=self._api_key) if self._api_key else genai.Client()
        return self._client

    def _embed(self, texts: list[str], *, task_type: str) -> EmbeddedBatch:
        if not texts:
            return EmbeddedBatch(
                vectors=[], model=self.name, input_tokens=0, dimensions=self.dimensions
            )
        vectors: list[list[float]] = []
        input_tokens = 0
        # Only used when the API reports no statistics; loads tiktoken lazily.
        proxy = TiktokenTokenizer()
        for start in range(0, len(texts), MAX_CONTENTS_PER_REQUEST):
            batch = texts[start : start + MAX_CONTENTS_PER_REQUEST]
            response = self._sdk().models.embed_content(
                model=self.name,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimensions,
                ),
            )
            if len(response.embeddings) != len(batch):
                raise RuntimeError(
                    f"{self.name} returned {len(response.embeddings)} vectors for "
                    f"{len(batch)} inputs — refusing to index a partial batch"
                )
            for text, item in zip(batch, response.embeddings, strict=True):
                vectors.append(list(item.values or []))
                # No API-reported usage means counting the tokens actually sent, via
                # the same cl100k_base proxy the chunker uses: measured, never zero.
                input_tokens += _reported_tokens(item) or len(proxy.encode(text))

        if not vectors or not vectors[0]:
            raise RuntimeError(f"{self.name} returned no embedding values")
        if len(vectors[0]) != self.dimensions:
            raise RuntimeError(
                f"{self.name} returned {len(vectors[0])}-dim vectors but the experiment "
                f"config expects {self.dimensions}; update the config or the model"
            )
        if any(len(vector) != len(vectors[0]) for vector in vectors):
            raise RuntimeError(f"{self.name} returned vectors of mixed dimensions")
        return EmbeddedBatch(
            vectors=vectors,
            model=self.name,
            input_tokens=input_tokens,
            dimensions=len(vectors[0]),
        )

    def embed_documents(self, texts: list[str]) -> EmbeddedBatch:
        return self._embed(texts, task_type=DOCUMENT_TASK)

    def embed_query_with_usage(self, text: str) -> tuple[list[float], int]:
        """Query vector plus its input tokens, returned by value.

        Cost attribution for the retrieval stage needs the token count; returning it
        (instead of storing it on the client) keeps concurrent /ask requests from
        overwriting each other's usage.
        """
        batch = self._embed([text], task_type=QUERY_TASK)
        return batch.vectors[0], batch.input_tokens

    def embed_query(self, text: str) -> list[float]:
        vector, _ = self.embed_query_with_usage(text)
        return vector


def build_embedding_model(
    settings: Settings | None = None, *, model: str | None = None, dimensions: int | None = None
) -> EmbeddingModel:
    """Embedding backend for the configured provider (Phase 1: gemini only)."""
    active = settings or get_settings()
    if active.embedding_provider != "gemini":
        raise NotImplementedError(
            f"embedding provider {active.embedding_provider!r} ships in Phase 2 as the "
            "open-source ablation; Phase 1 measures gemini only"
        )
    return GeminiEmbedding(
        model=model or active.embedding_model,
        dimensions=dimensions or active.embedding_dim,
        api_key=active.require_gemini_key(),
    )
