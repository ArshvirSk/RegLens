"""Experiment (ablation) configuration.

Every technique the PRD wants evaluated is a field here, so an ablation is a YAML file
plus one ``make eval`` run. Nothing about the pipeline is configured in code: the
active config is hashed and recorded in every eval report and API response, which is
what makes "which configuration produced this number?" answerable later.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from reglens.config import paths

ChunkStrategy = Literal["fixed", "clause_aware", "speaker_aware", "parent_child"]
RetrievalMode = Literal["dense", "bm25", "hybrid"]
FusionMode = Literal["none", "rrf"]


class ParsingConfig(BaseModel):
    """Which PDF parser runs during ingest (compared on real pages in eval/results/)."""

    parser: Literal["pymupdf", "pdfplumber"] = "pymupdf"
    ocr_fallback: bool = False
    #: A page with fewer extracted characters than this is treated as image-only (a scan)
    #: and recorded instead of silently becoming an empty page in the corpus.
    min_chars_per_page: int = 40


class ChunkingConfig(BaseModel):
    """How documents are split. Phase 1 is deliberately naive: fixed windows."""

    strategy: ChunkStrategy = "fixed"
    chunk_size_tokens: int = 512
    overlap_tokens: int = 64
    min_chunk_tokens: int = 48
    # Phase 2 toggles: prepend document/section context, keep small child chunks.
    contextual_headers: bool = False
    parent_child: bool = False
    parent_size_tokens: int = 2048


class IndexingConfig(BaseModel):
    """What gets written into the indexes."""

    embedding_provider: Literal["gemini", "openai", "sentence_transformers"] = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 3072
    normalize_embeddings: bool = True
    batch_size: int = 64
    keyword_index: Literal["none", "postgres_fts"] = "none"
    store: Literal["pgvector"] = "pgvector"


class RetrievalConfig(BaseModel):
    """How chunks are fetched for a query."""

    mode: RetrievalMode = "dense"
    top_k: int = 10
    candidate_k: int = 50
    fusion: FusionMode = "none"
    rrf_k: int = 60
    rerank: bool = False
    rerank_model: str | None = None
    rerank_top_n: int = 5
    # Phase 3 toggles: honour as_of_date and issuer/doc_type filters.
    temporal_filter: bool = False
    metadata_filter: bool = True


class RewritingConfig(BaseModel):
    """Query transformations applied before retrieval (Phase 2)."""

    enabled: bool = False
    acronym_expansion: bool = False
    decomposition: bool = False
    multi_query: bool = False
    hyde: bool = False
    max_sub_queries: int = 3


class GenerationConfig(BaseModel):
    """The answer model and its guardrails."""

    model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    max_output_tokens: int = 900
    require_citations: bool = True
    # Phase 4: answers below this support score are refused instead of guessed.
    min_support_score: float = 0.0
    verify_citations: bool = False


class ExperimentConfig(BaseModel):
    """A named, hashable, fully-specified pipeline configuration."""

    name: str
    phase: int = 0
    description: str = ""
    corpus_version: str | None = None
    parsing: ParsingConfig = ParsingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    indexing: IndexingConfig = IndexingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    rewriting: RewritingConfig = RewritingConfig()
    generation: GenerationConfig = GenerationConfig()
    notes: list[str] = Field(default_factory=list)

    def behavioral_dump(self) -> dict[str, Any]:
        """The subset that can change outputs (prose fields excluded)."""
        return self.model_dump(exclude={"notes", "description"})

    def fingerprint(self) -> str:
        """Stable SHA-256 over behaviour-affecting fields only."""
        canonical = json.dumps(self.behavioral_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def summary_lines(self) -> list[str]:
        """Human-readable one-liners used in eval reports and status reports."""
        p, c, i, r, w, g = (
            self.parsing,
            self.chunking,
            self.indexing,
            self.retrieval,
            self.rewriting,
            self.generation,
        )
        return [
            f"parsing: parser={p.parser} ocr_fallback={p.ocr_fallback}",
            f"chunking: {c.strategy} size={c.chunk_size_tokens} overlap={c.overlap_tokens} "
            f"contextual_headers={c.contextual_headers} parent_child={c.parent_child}",
            f"embeddings: {i.embedding_provider}/{i.embedding_model} dim={i.embedding_dim} "
            f"keyword_index={i.keyword_index}",
            f"retrieval: mode={r.mode} top_k={r.top_k} candidate_k={r.candidate_k} "
            f"fusion={r.fusion} rerank={r.rerank} temporal_filter={r.temporal_filter}",
            f"rewriting: enabled={w.enabled} acronyms={w.acronym_expansion} "
            f"decompose={w.decomposition} hyde={w.hyde}",
            f"generation: {g.model} temperature={g.temperature} "
            f"verify_citations={g.verify_citations} min_support={g.min_support_score}",
        ]


def config_hash(config: ExperimentConfig) -> str:
    """Convenience wrapper: the hash recorded in eval reports and query traces."""
    return config.fingerprint()


def _coerce_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Allow ``yaml`` files that omit ``name`` (fall back to the file stem)."""
    if "name" not in raw:
        raise ValueError("experiment config requires a 'name' field")
    return raw


def load_experiment(name_or_path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config by name (from src/reglens/config/experiments) or path."""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.is_file():
        path = candidate
    else:
        path = paths.resolve(paths.EXPERIMENT_CONFIG_DIR) / f"{name_or_path}.yaml"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in _experiment_files())) or "none"
        raise FileNotFoundError(f"no experiment config at {path} (available: {available})")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return ExperimentConfig.model_validate(_coerce_raw(raw))


def _experiment_files() -> list[Path]:
    directory = paths.resolve(paths.EXPERIMENT_CONFIG_DIR)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.yaml"))


def list_experiments() -> list[str]:
    """Names of the shipped experiment configs."""
    return [p.stem for p in _experiment_files()]
