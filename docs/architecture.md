# RegLens architecture

Status: **Phase 0 complete**. This document records what exists, why each choice was made,
what was rejected, and where each later phase plugs in. It is updated at the end of every
phase, not rewritten.

---

## 1. System overview

```
                    ┌──────────────────────────── Phase 0 (built) ────────────────────────────┐
  public docs ──►   manifest.csv ──► fetcher ──► data/raw (immutable, SHA-256 addressed)
  (RBI/SEBI/IR)     (reviewed)       (robots +     │
                                     rate limit)   │
                                                   ▼
                    ┌──────────── Phase 1 (next) ────────────┐
                    │ parser (+OCR) ──► fixed chunker ──► embeddings ──► pgvector
                    │                                        └──► FastAPI /ask ──► grounded answer
                    └────────────────────────────────────────────────────────────────────────┘
                    ┌──────────── Phase 2 ────────────┐   ┌──────── Phase 3 ────────┐
                    │ clause/speaker chunkers, BM25,  │   │ table extraction → metrics,
                    │ RRF, reranker, rewriting,       │   │ as-of filter, LangGraph
                    │ second embedding model          │   │ router, safe text-to-SQL
                    └─────────────────────────────────┘   └─────────────────────────┘
                    ┌──────────── Phase 4 ────────────┐   ┌──────── Phase 5 ────────┐
                    │ citation verifier, refusal      │   │ held-out eval, failure
                    │ threshold, Next.js UI + PDF     │   │ analysis, CI regression
                    │ source panel, refresh job       │   │ gate, write-up
                    └─────────────────────────────────┘   └─────────────────────────┘

  Every stage ──► Trace (per-stage latency, tokens, cost) ──► Postgres + JSON logs + eval reports
```

---

## 2. Decisions and alternatives

Each entry: **decision** → *why* → *what was rejected*.

### 2.1 Corpus is a reviewed plan, not a crawler

**Decision.** `data/corpus_plan.yaml` is the human-editable source; `make corpus-plan`
regenerates `data/manifest.csv`; `make download` is a dry run unless
`--yes --accept-terms` is passed.

*Why.* Two of the three source families cannot be crawled politely or reliably:

| Source | Finding (verified 2026-10-02) |
|---|---|
| RBI | `robots.txt` answers **HTTP 418** from the WAF for every user agent tried, and document pages render their body from ASP.NET `__VIEWSTATE` (the index page contains ~386 `ViewMasDirections` references but zero parseable anchors). Deep links must come from an index/known ID, and PDFs on `rbidocs.rbi.org.in` return an HTML interstitial unless a `Referer` header is sent. |
| SEBI | `robots.txt` allows crawling (`Disallow:` empty except `/js`, `/css`). Legal listings are server-rendered and parse cleanly into deep links. |
| Bank IR pages | Listing pages, not documents. Annual-report and transcript URLs embed a fiscal year and change annually. |

*Rejected.* A general crawler that walks listings automatically: it would silently collect
the wrong file (a presentation instead of a transcript), or nothing at all from RBI, and
would make the corpus a side effect rather than a decision. Also rejected: scraping RBI's
viewstate with a headless browser in Phase 0 — that is real work, and it belongs in the
Phase 4 refresh job where its output can be diffed and reviewed.

*Consequence.* 95 documents are queued for review and 7 entry-point/listing URLs are marked
`excluded` (they exist so the refresh job knows where to look, not to be ingested).

### 2.2 Raw files are content-addressed and immutable

**Decision.** `data/raw/<source>/<doc_id>/<doc_id>__<sha256[:12]>.<ext>`, hashes recorded in
the manifest, files gitignored, `documents.file_hash` unique in Postgres.

*Why.* Ingestion must be idempotent (FR5) and every citation must be checkable. A
content-addressed store gives both for free: re-running never re-downloads, a changed
document creates a second file instead of overwriting evidence, and `make validate-manifest`
re-hashes everything to detect tampering.

*Rejected.* A database blob store (slower to inspect by hand, worse for the PDF viewer) and
`doc_id`-only filenames (overwrite risk).

### 2.3 Postgres for everything, with a swappable interface

**Decision.** One Postgres instance with pgvector (dense), `tsvector` (keyword), and
ordinary tables (metadata, metrics, traces). Retrieval talks to protocols, not to SQL.

*Why.* One dependency instead of three for a learning project; joins between vectors and
metadata filters are free; Postgres full-text is genuinely adequate for clause numbers and
acronyms at this corpus size. The `KeywordIndex` protocol exists so an OpenSearch swap is a
new implementation, not a refactor.

*Rejected.* A dedicated vector database (extra moving part, no measured benefit at
100–500 documents) and an external BM25 service.

*Open risk.* Postgres FTS has no true BM25 scoring. The Phase 2 ablation will report what it
actually measures rather than assuming parity with a BM25 implementation.

### 2.4 Plain Python now, LangGraph in Phase 3

**Decision.** Phase 1 is a straight-line function: retrieve → build context → generate →
verify. LangGraph arrives with the router.

*Why.* A graph framework in Phase 1 would hide the thing being learned. The PRD asks for
routing and agent loops *from Phase 3*; introducing them earlier would make the baseline
harder to reason about and to compare against.

### 2.5 Migrations are plain SQL with checksums

**Decision.** Numbered `.sql` files, applied once, tracked in `schema_migrations`, checksum
verified, `CREATE TABLE IF NOT EXISTS` so a fresh volume and an existing one converge.
`${EMBEDDING_DIM}` is substituted from typed settings.

*Why.* The interesting parts of this schema *are* the SQL: `vector(1536)`, a generated
`tsvector` column, an HNSW index, unique constraints that enforce idempotency. Alembic would
add indirection over exactly the artifact worth reading. Checksums give the one guarantee
that matters: an applied migration can never silently change.

*Rejected.* Alembic (heavier, and autogenerate would fight the hand-written DDL) and
`psql -f` loops (no record of what ran).

### 2.6 Tracing is local-first and optional at the edges

**Decision.** Every request writes a `Trace` (per-stage latency, tokens, cost). Sinks: JSON
logs (always), Postgres (best effort), OTLP/Langfuse (only if configured *and* installed).

*Why.* FR6 must hold on a laptop with no collector. The cost model has to be honest too: an
unpriced model raises, is logged as a warning, and is reported as unattributed rather than
silently costing $0.00.

*Rejected.* Making Langfuse a hard dependency (a fresh clone would need an account before it
could answer a question).

### 2.7 `POST /ask` returns 501 until it can answer

**Decision.** The endpoint exists, validates its schema, writes a trace, and returns HTTP 501
with "arrives in Phase 1".

*Why.* A stub that returns `200 {"answer": ""}` is indistinguishable from a pipeline that
failed silently. The eval harness must treat "not built" as an error, not as a wrong answer.

### 2.8 Experiment configs are hashed

**Decision.** Every ablation toggle lives in a YAML file under
`src/reglens/config/experiments/`. The hash covers behaviour-affecting fields only, and is
recorded in API responses and eval reports.

*Why.* Without it, "which configuration produced 0.71 recall?" becomes archaeology. Prose
fields are excluded from the hash so documentation edits cannot invalidate recorded results.

*Dependency policy.* A dependency is added in the phase that first uses it. `pyproject.toml`
therefore lists FastAPI, asyncpg, httpx, PyYAML, tenacity and pydantic in Phase 0, and PDF
parsers, embedding clients, torch and langgraph arrive with their phases. The alternative —
pre-installing everything — makes the lock file advertise capability the code does not have.

### 2.9 Golden set: drafted from documents, reviewed by the owner

**Decision.** `eval/golden/questions.jsonl` is empty in Phase 0. The schema, validator, and
per-type quotas exist; the questions do not.

*Why.* The working rule is "draft them from real parsed content, record the gold passage and
answer". A question written before its document is parsed has no verifiable gold passage,
which makes it worthless as ground truth. Phase 1 fills it once parsing works.

---

## 3. Data model

Implemented in [`0001_core.sql`](../src/reglens/db/migrations/0001_core.sql):

| Table | Purpose | Idempotency guard |
|---|---|---|
| `documents` | one row per source document | `UNIQUE (file_hash)` |
| `document_links` | amendment graph (`supersedes`, `amends`) | composite PK |
| `chunks` | retrievable units + `vector(dim)` + generated `tsv` | `UNIQUE (doc_id, corpus_version, chunk_strategy, chunk_index)` |
| `metrics` | extracted financial figures with page refs | composite PK |
| `queries` | one row per answered question (route, citations, cost, trace) | PK `query_id` |
| `feedback` | helpful / wrong / missing_source labels | FK to `queries` |
| `request_traces`, `request_stages` | FR6 tracing | PK `trace_id` |
| `corpus_versions` | what was indexed when | PK `corpus_version` |

Chunk identity includes `corpus_version` and `chunk_strategy` deliberately: re-chunking with
a new strategy must not overwrite the baseline's chunks, because both are needed for the
ablation comparison.

The `chunks` table has **no** `embedding_model` column. Vectors from different models are not
comparable, so a corpus version is indexed with exactly one model, recorded in the experiment
config and `corpus_versions`. Mixing models inside one table is the failure this prevents.

---

## 4. Pipeline interfaces (implemented now, filled in later)

| Stage | Contract | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|---|
| `chunking` | `Chunker.chunk(ParsedDocument) -> list[Chunk]` | — | `fixed` | `clause_aware`, `speaker_aware`, `parent_child` | — |
| `indexing` | `EmbeddingModel`, `VectorStore`, `KeywordIndex` | — | hosted embeddings + pgvector | open-source model + Postgres FTS | — |
| `retrieval` | `Retriever.retrieve(query, top_k, SearchFilters)` | — | dense | BM25, hybrid/RRF, `Reranker` | temporal + metadata filters |
| `routing` | `Router.route(question) -> RouteDecision`, `Tool` | — | — | — | LangGraph router, SQL tool, graph tool |
| `generation` | `Answerer.answer(...) -> Answer`, `LLMClient` | — | grounded answer with citations | model-size ablation | — |

`SearchFilters` carries `as_of_date` centrally (FR2). A filter that only some code paths
honour is exactly how a superseded rule gets cited as current.

---

## 5. Non-functional posture

| Target | Phase 0 status |
|---|---|
| p50 < 4 s, p95 < 10 s (text) | Not measurable: no pipeline yet. Traces are already recorded so Phase 1 can report percentiles immediately. |
| Cost < ~$0.02/query | Not measurable yet. The price table is in `observability/cost.py`, dated and sourced; a typical 6k-in/400-out call on `gpt-4o-mini` prices at ~$0.0011, so the budget is not the binding constraint. |
| Reproducibility | `uv.lock` pins dependencies; experiment configs are hashed; seeds via `REGLENS_SEED`; reports record corpus version, git commit and model names. |
| Security | Read-only DB role created at Postgres init for the Phase 3 SQL tool; secrets only in `.env` (gitignored) with a committed `.env.example`; a test asserts secrets are redacted in dumps. |

---

## 6. Known risks

| Risk | Status / mitigation |
|---|---|
| RBI pages are not machine-parseable | Documented; direct PDF links are queued, refresh job in Phase 4 needs a browser adapter. |
| Listing rows need a human to name the exact file | By design. `make download` is a dry run and `--resolve-listings` prints candidates for review. |
| Titles/dates for a few RBI documents come from third-party citations | Flagged in the manifest `notes` with "confirm in Phase 1"; Phase 1 fills `issue_date` from the PDF itself, which is also what the temporal filter needs. |
| Postgres FTS ≠ BM25 | To be measured, not assumed (Phase 2). |
| Judge reliability | Phase 1 validates the judge against 20 hand-graded answers and reports the agreement. |
