"""Gemini client mapping tests: fake SDK clients, no network, no keys, no cost.

What is under test is the *translation* layer — request shaping (batches, task types,
roles, sampling) and response parsing (vectors, token accounting, finish reasons) —
because that is what silently corrupts an index or a cost report when it drifts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reglens.config import Settings
from reglens.config.settings import MissingApiKeyError
from reglens.generation.base import LLMClient, Message
from reglens.generation.gemini import GeminiLLM, build_llm
from reglens.indexing.base import EmbeddingModel
from reglens.indexing.embeddings import (
    MAX_CONTENTS_PER_REQUEST,
    GeminiEmbedding,
    build_embedding_model,
)


class FakeModels:
    """Records every call; returns well-formed responses unless told not to."""

    def __init__(
        self,
        *,
        dims: int = 8,
        tokens_per_content: int = 10,
        text: str | None = "answer",
        drop_last_embedding: bool = False,
    ) -> None:
        self.dims = dims
        self.tokens_per_content = tokens_per_content
        self.text = text
        self.drop_last_embedding = drop_last_embedding
        self.embed_calls: list[dict[str, object]] = []
        self.generate_calls: list[dict[str, object]] = []

    def embed_content(self, *, model: str, contents: list[str], config):  # type: ignore[no-untyped-def]
        self.embed_calls.append({"model": model, "contents": list(contents), "config": config})
        embeddings = [
            SimpleNamespace(
                values=[0.0] * self.dims,
                statistics=SimpleNamespace(total_tokens=self.tokens_per_content),
            )
            for _ in contents
        ]
        if self.drop_last_embedding:
            embeddings = embeddings[:-1]
        return SimpleNamespace(embeddings=embeddings)

    def generate_content(self, *, model: str, contents: list[object], config):  # type: ignore[no-untyped-def]
        self.generate_calls.append({"model": model, "contents": list(contents), "config": config})
        return SimpleNamespace(
            text=self.text,
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=40,
                thoughts_token_count=10,
                total_token_count=150,
            ),
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        )


def fake_client(models: FakeModels) -> SimpleNamespace:
    return SimpleNamespace(models=models)


def test_embed_documents_batches_at_the_sdk_ceiling() -> None:
    models = FakeModels(dims=8)
    embedding = GeminiEmbedding(dimensions=8, client=fake_client(models))
    assert isinstance(embedding, EmbeddingModel)

    texts = [f"chunk {index}" for index in range(250)]
    batch = embedding.embed_documents(texts)

    assert [len(call["contents"]) for call in models.embed_calls] == [  # type: ignore[arg-type]
        MAX_CONTENTS_PER_REQUEST,
        MAX_CONTENTS_PER_REQUEST,
        50,
    ]
    assert len(batch.vectors) == 250
    assert batch.dimensions == 8
    assert batch.input_tokens == 2500
    assert batch.model == "gemini-embedding-001"
    assert all(call["config"].task_type == "RETRIEVAL_DOCUMENT" for call in models.embed_calls)  # type: ignore[attr-defined]
    assert models.embed_calls[0]["config"].output_dimensionality == 8  # type: ignore[attr-defined]


def test_embed_query_uses_the_query_task_and_returns_one_vector() -> None:
    models = FakeModels(dims=4)
    embedding = GeminiEmbedding(dimensions=4, client=fake_client(models))

    vector = embedding.embed_query("what is the LCR requirement")

    assert len(vector) == 4
    call = models.embed_calls[0]
    assert call["contents"] == ["what is the LCR requirement"]
    assert call["config"].task_type == "RETRIEVAL_QUERY"  # type: ignore[attr-defined]


def test_empty_input_returns_an_empty_batch_without_a_request() -> None:
    models = FakeModels()
    embedding = GeminiEmbedding(client=fake_client(models))
    batch = embedding.embed_documents([])
    assert batch.vectors == []
    assert models.embed_calls == []


def test_wrong_dimension_from_the_model_is_rejected() -> None:
    models = FakeModels(dims=4)
    embedding = GeminiEmbedding(dimensions=8, client=fake_client(models))
    with pytest.raises(RuntimeError, match="dim"):
        embedding.embed_documents(["x"])


def test_partial_batch_from_the_model_is_rejected() -> None:
    models = FakeModels(dims=4, drop_last_embedding=True)
    embedding = GeminiEmbedding(dimensions=4, client=fake_client(models))
    with pytest.raises(RuntimeError, match="vectors for"):
        embedding.embed_documents(["x", "y"])


def test_complete_maps_messages_usage_and_finish_reason() -> None:
    models = FakeModels(text="cited answer")
    llm = GeminiLLM(client=fake_client(models))
    assert isinstance(llm, LLMClient)

    response = llm.complete(
        [
            Message(role="system", content="cite everything"),
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
        ],
        model="gemini-3.5-flash-lite",
        temperature=0.2,
        max_output_tokens=500,
    )

    call = models.generate_calls[0]
    assert call["config"].system_instruction == "cite everything"  # type: ignore[attr-defined]
    assert call["config"].temperature == 0.2  # type: ignore[attr-defined]
    assert call["config"].max_output_tokens == 500  # type: ignore[attr-defined]
    assert [content.role for content in call["contents"]] == ["user", "model", "user"]  # type: ignore[attr-defined]
    assert response.text == "cited answer"
    assert response.model == "gemini-3.5-flash-lite"
    assert response.input_tokens == 100
    # 40 candidate tokens + 10 thinking tokens: Gemini bills thoughts as output.
    assert response.output_tokens == 50
    assert response.finish_reason == "STOP"
    assert response.latency_ms >= 0


def test_missing_text_is_an_empty_answer_not_a_crash() -> None:
    models = FakeModels(text=None)
    llm = GeminiLLM(client=fake_client(models))
    response = llm.complete([Message(role="user", content="Q")], model="m")
    assert response.text == ""


def test_factories_read_config_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = Settings(_env_file=None, gemini_api_key="test-key")

    embedding = build_embedding_model(settings)
    assert embedding.name == "gemini-embedding-001"
    assert embedding.dimensions == 3072

    llm = build_llm(settings)
    assert llm.name == "gemini"


def test_missing_key_fails_before_any_paid_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    with pytest.raises(MissingApiKeyError, match="GEMINI_API_KEY"):
        build_embedding_model(settings)
    with pytest.raises(MissingApiKeyError, match="GEMINI_API_KEY"):
        build_llm(settings)


def test_environment_key_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "from-environment")
    settings = Settings(_env_file=None)
    # pydantic-settings reads process env vars as well as .env, so the key lands in
    # settings either way — the factory must accept it without a .env file present.
    assert settings.require_gemini_key() == "from-environment"
    assert build_embedding_model(settings).dimensions == 3072


def test_unimplemented_providers_say_when_they_arrive() -> None:
    settings = Settings(_env_file=None, embedding_provider="sentence_transformers")
    with pytest.raises(NotImplementedError, match="Phase 2"):
        build_embedding_model(settings)
    settings = Settings(_env_file=None, llm_provider="openai")
    with pytest.raises(NotImplementedError, match="Phase 1"):
        build_llm(settings)
