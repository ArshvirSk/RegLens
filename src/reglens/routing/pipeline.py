"""Pipeline assembly: retrieve → generate, wired once for the API and the eval runner.

Phase 1's "router" is this straight line — every question is a text question. Phase 3
replaces the middle with the LangGraph router (text / numeric / hybrid) and a
templated text-to-SQL tool; the assembly point exists from Phase 1 so the eval harness
and ``/ask`` never diverge in what pipeline they measure. A pipeline only the API
knows about is how "the demo works but the eval number is fiction" happens.
"""

from __future__ import annotations

from dataclasses import dataclass

from reglens.config import Settings, get_settings, load_experiment
from reglens.config.experiment import ExperimentConfig
from reglens.generation.answerer import GroundedAnswerer
from reglens.generation.base import Answer
from reglens.generation.gemini import build_llm
from reglens.indexing.base import EmbeddingModel, VectorStore
from reglens.indexing.embeddings import build_embedding_model
from reglens.indexing.vector_store import PgVectorStore
from reglens.retrieval.base import RetrievedChunk, SearchFilters
from reglens.retrieval.dense import DenseRetriever


@dataclass
class AskResult:
    """One ask: the answer, what was retrieved, and the usage for the trace."""

    answer: Answer
    chunks: list[RetrievedChunk]
    embed_input_tokens: int
    embed_model: str | None


class Pipeline:
    """Retrieve and answer, with usage returned by value (never shared state)."""

    def __init__(
        self,
        *,
        experiment: ExperimentConfig,
        embedder: EmbeddingModel,
        retriever: DenseRetriever,
        answerer: GroundedAnswerer,
        store: VectorStore,
    ) -> None:
        self.experiment = experiment
        self.embedder = embedder
        self.retriever = retriever
        self.answerer = answerer
        self.store = store

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> AskResult:
        chunks, embed_tokens, embed_model = self.retriever.retrieve_with_usage(
            question,
            top_k=top_k or self.experiment.retrieval.top_k,
            filters=filters,
        )
        answer = self.answerer.answer(question, chunks)
        return AskResult(
            answer=answer,
            chunks=chunks,
            embed_input_tokens=embed_tokens,
            embed_model=embed_model,
        )


def build_pipeline(
    settings: Settings | None = None, *, experiment: ExperimentConfig | None = None
) -> Pipeline:
    """Wire embedder → store → retriever → answerer from typed settings + config."""
    active = settings or get_settings()
    config = experiment or load_experiment(active.experiment_config)
    embedder = build_embedding_model(active)
    store = PgVectorStore(settings=active)
    retriever = DenseRetriever(embedder=embedder, store=store)
    answerer = GroundedAnswerer(
        llm=build_llm(active),
        model=config.generation.model,
        temperature=config.generation.temperature,
        max_output_tokens=config.generation.max_output_tokens,
    )
    return Pipeline(
        experiment=config,
        embedder=embedder,
        retriever=retriever,
        answerer=answerer,
        store=store,
    )
