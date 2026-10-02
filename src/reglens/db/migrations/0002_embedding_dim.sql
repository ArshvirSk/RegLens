-- Realign chunks.embedding with the configured embedding dimension.
--
-- 0001_core.sql rendered ${EMBEDDING_DIM} once, when it was first applied: under the
-- Phase 0 default of 1536, before gemini-embedding-001 (3072) was configured. Applied
-- migrations are immutable, so the dimension change lands here rather than as an edit
-- to 0001. The HNSW index is dropped first because it cannot be rewritten in place for
-- a new typmod; the table holds no chunks yet, so the rebuild costs nothing.
--
-- Safe on a populated table only if every stored vector already has the target width;
-- otherwise pgvector rejects the rewrite, which is the correct outcome (mixed-width
-- vectors in one column must never be papered over).

DROP INDEX IF EXISTS chunks_embedding_hnsw_idx;

ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(${EMBEDDING_DIM});

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
