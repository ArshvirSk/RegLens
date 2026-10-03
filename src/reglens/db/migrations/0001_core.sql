-- RegLens core schema.
--
-- Migrations are immutable once committed: the runner records a checksum per file and
-- refuses to start if an applied migration changed. Placeholders like ${EMBEDDING_DIM}
-- are substituted by the runner from typed settings, because pgvector needs a concrete
-- dimension and the embedding model is a config choice (see docs/architecture.md).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------- corpus versions
CREATE TABLE IF NOT EXISTS corpus_versions (
    corpus_version      TEXT PRIMARY KEY,
    description         TEXT NOT NULL DEFAULT '',
    document_count      INTEGER NOT NULL DEFAULT 0,
    chunk_count         INTEGER NOT NULL DEFAULT 0,
    git_commit          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------- documents
CREATE TABLE IF NOT EXISTS documents (
    doc_id              TEXT PRIMARY KEY,
    corpus_version      TEXT NOT NULL,
    source              TEXT NOT NULL,
    issuer              TEXT NOT NULL,
    doc_type            TEXT NOT NULL,
    title               TEXT NOT NULL,
    issue_date          DATE,
    effective_date      DATE,
    fiscal_period       TEXT,
    url                 TEXT NOT NULL,
    file_hash           TEXT NOT NULL,
    parse_quality       DOUBLE PRECISION,
    parse_reviewed      BOOLEAN NOT NULL DEFAULT false,
    parser              TEXT,
    page_count          INTEGER,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Idempotency (FR5): one row per set of bytes, so re-ingesting a file cannot duplicate it.
    CONSTRAINT documents_file_hash_key UNIQUE (file_hash)
);

CREATE INDEX IF NOT EXISTS documents_issuer_idx ON documents (issuer);
CREATE INDEX IF NOT EXISTS documents_doc_type_idx ON documents (doc_type);
CREATE INDEX IF NOT EXISTS documents_issue_date_idx ON documents (issue_date);
CREATE INDEX IF NOT EXISTS documents_effective_date_idx ON documents (effective_date);
CREATE INDEX IF NOT EXISTS documents_corpus_version_idx ON documents (corpus_version);

-- ---------------------------------------------------------------- amendment graph
CREATE TABLE IF NOT EXISTS document_links (
    from_doc_id         TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    to_doc_id           TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    relation            TEXT NOT NULL CHECK (relation IN ('supersedes', 'amends')),
    note                TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (from_doc_id, to_doc_id, relation)
);

-- ---------------------------------------------------------------- chunks
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    corpus_version      TEXT NOT NULL,
    chunk_index         INTEGER NOT NULL,
    page_start          INTEGER,
    page_end            INTEGER,
    clause_path         TEXT,
    section_title       TEXT,
    speaker             TEXT,
    text                TEXT NOT NULL,
    token_count         INTEGER NOT NULL,
    chunk_strategy      TEXT NOT NULL,
    parent_chunk_id     TEXT REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding           vector(${EMBEDDING_DIM}),
    -- Postgres full-text search backs the keyword arm of hybrid retrieval (Phase 2).
    tsv                 tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Re-running the same chunker over the same document cannot duplicate chunks (FR5).
    CONSTRAINT chunks_identity_key UNIQUE (doc_id, corpus_version, chunk_strategy, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_corpus_version_idx ON chunks (corpus_version);
CREATE INDEX IF NOT EXISTS chunks_text_trgm_idx ON chunks USING gin (text gin_trgm_ops);
-- HNSW is the pragmatic default for pgvector; recall/ef_search defaults are revisited
-- with measured results in Phase 2 rather than guessed at now. Creation is guarded:
-- pgvector cannot build an hnsw index on vector above 2000 dimensions, and the
-- configured embedding dimension (gemini-embedding-001, 3072) is above it. Search then
-- runs as exact kNN, which is correct and cheap at this corpus size. 0002 carries the
-- same guard for a dimension change applied after this file ran.
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
EXCEPTION
    WHEN program_limit_exceeded THEN
        RAISE NOTICE 'chunks_embedding_hnsw_idx skipped: pgvector hnsw limit (2000 dims) for vector(${EMBEDDING_DIM})';
END
$$;

-- ---------------------------------------------------------------- extracted metrics
CREATE TABLE IF NOT EXISTS metrics (
    bank                TEXT NOT NULL,
    fiscal_period       TEXT NOT NULL,
    metric_name         TEXT NOT NULL,
    value               NUMERIC NOT NULL,
    unit                TEXT NOT NULL,
    doc_id              TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    page                INTEGER,
    extraction_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    extractor           TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (bank, fiscal_period, metric_name, doc_id, page)
);

CREATE INDEX IF NOT EXISTS metrics_bank_period_idx ON metrics (bank, fiscal_period);
CREATE INDEX IF NOT EXISTS metrics_metric_name_idx ON metrics (metric_name);

-- ---------------------------------------------------------------- query log
CREATE TABLE IF NOT EXISTS queries (
    query_id            TEXT PRIMARY KEY,
    user_query          TEXT NOT NULL,
    as_of_date          DATE,
    route               TEXT,
    retrieved_chunk_ids TEXT[] NOT NULL DEFAULT '{}',
    answer              TEXT,
    citations           JSONB NOT NULL DEFAULT '[]'::jsonb,
    refusal             BOOLEAN NOT NULL DEFAULT false,
    refusal_reason      TEXT,
    latency_ms          INTEGER,
    cost_usd            NUMERIC(12, 6),
    trace_id            TEXT,
    config_hash         TEXT,
    corpus_version      TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS queries_created_at_idx ON queries (created_at DESC);
CREATE INDEX IF NOT EXISTS queries_config_hash_idx ON queries (config_hash);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id         BIGSERIAL PRIMARY KEY,
    query_id            TEXT NOT NULL REFERENCES queries (query_id) ON DELETE CASCADE,
    label               TEXT NOT NULL CHECK (label IN ('helpful', 'wrong', 'missing_source', 'other')),
    comment             TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------- traces (FR6)
CREATE TABLE IF NOT EXISTS request_traces (
    trace_id            TEXT PRIMARY KEY,
    query_id            TEXT,
    name                TEXT NOT NULL,
    route               TEXT,
    model               TEXT,
    corpus_version      TEXT,
    config_hash         TEXT,
    started_at          TIMESTAMPTZ NOT NULL,
    finished_at         TIMESTAMPTZ,
    latency_ms          INTEGER,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cost_usd            NUMERIC(12, 6) NOT NULL DEFAULT 0,
    error               TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS request_traces_started_at_idx ON request_traces (started_at DESC);
CREATE INDEX IF NOT EXISTS request_traces_name_idx ON request_traces (name);

CREATE TABLE IF NOT EXISTS request_stages (
    stage_id            BIGSERIAL PRIMARY KEY,
    trace_id            TEXT NOT NULL REFERENCES request_traces (trace_id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    seq                 INTEGER NOT NULL,
    latency_ms          INTEGER NOT NULL DEFAULT 0,
    model               TEXT,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cost_usd            NUMERIC(12, 6) NOT NULL DEFAULT 0,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS request_stages_trace_id_idx ON request_stages (trace_id);

-- ---------------------------------------------------------------- helpers
CREATE OR REPLACE VIEW document_timeline AS
SELECT
    d.doc_id,
    d.title,
    d.issuer,
    d.doc_type,
    d.issue_date,
    d.effective_date,
    l.to_doc_id   AS related_doc_id,
    l.relation
FROM documents d
LEFT JOIN document_links l ON l.from_doc_id = d.doc_id;
