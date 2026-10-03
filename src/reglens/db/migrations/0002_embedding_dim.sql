-- Realign chunks.embedding with the configured embedding dimension.
--
-- 0001_core.sql rendered ${EMBEDDING_DIM} once, when it was first applied: under the
-- Phase 0 default of 1536, before gemini-embedding-001 (3072) was configured. Applied
-- migrations are immutable, so the dimension change lands here rather than as an edit
-- to 0001.
--
-- Dropping the HNSW index first is required: an index cannot be rewritten in place for
-- a new typmod. Recreating it is guarded exactly as in 0001, because pgvector cannot
-- build hnsw on vector above 2000 dimensions. At the configured 3072 the index is
-- skipped and retrieval is exact kNN: the honest Phase 1 baseline, with no ANN recall
-- loss folded into the measured numbers. Phase 2 revisits this with measurements
-- (a halfvec expression index reaches 4000 dimensions).
--
-- The table holds no chunks yet, so the type change costs nothing here. On a populated
-- table pgvector rejects the rewrite when stored widths differ, which is the correct
-- outcome: mixed-width vectors in one column must never be papered over.

DROP INDEX IF EXISTS chunks_embedding_hnsw_idx;

ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(${EMBEDDING_DIM});

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
