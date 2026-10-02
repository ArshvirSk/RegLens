# PRD: RegLens — Indian Banking Regulatory and Filings Analyst (End-to-End RAG)

**Owner:** Arshvir | **Status:** Draft v1 | **Type:** Learning-first portfolio project
**One-liner:** A cited, temporally-aware question-answering system over RBI/SEBI regulations and bank filings that routes between text search and structured data, and is measured by a real evaluation suite.

---

## 1. Background and Motivation

Compliance analysts and equity researchers spend hours finding which rule applies, when it changed, and what a bank reported against it. The documents are long, clause-structured, frequently amended, and mixed with tables and scanned pages. This makes the domain an ideal testbed: every stage of a RAG system (parsing, chunking, retrieval, reasoning, generation, evaluation, operations) is stressed, and answers can be verified against source PDFs.

**Primary goal of this project is learning.** Every design decision should be justified by a measured result on the eval set, not by intuition.

## 2. Goals and Non-Goals

**Goals**
- G1: Build a complete RAG system covering ingestion, retrieval, routing, generation, evaluation, observability, and refresh.
- G2: Produce an ablation record showing what each technique contributed (chunking, hybrid search, reranking, etc.).
- G3: Answer lookup, numeric, temporal, comparison, and multi-hop questions with page- and clause-level citations.
- G4: Correctly refuse when the corpus does not contain the answer.
- G5: Ship a usable web UI and a public write-up with benchmark numbers and failure analysis.

**Non-Goals**
- Legal or investment advice; the product must display a "not advice" notice.
- Real-time market data or trading signals.
- Supporting every Indian regulator (scope is RBI, SEBI, and selected bank filings).
- Multi-tenant auth, billing, or enterprise features (can be added later as extensions).

## 3. Users and Use Cases

| Persona | Example questions |
|---|---|
| Compliance analyst | "What is the current LCR requirement for scheduled commercial banks?" "What changed in the KYC Master Direction in 2024?" |
| Equity researcher | "How did SBI's gross NPA ratio trend over the last 8 quarters?" "What did management say about NIM guidance on the latest call?" |
| Student / learner (you) | "Which circulars amended the digital lending guidelines, and which entities do they apply to?" |

## 4. Corpus Specification

**Sources (public)**
- RBI: Master Directions, Master Circulars, circulars, notifications (rbi.org.in)
- SEBI: circulars and regulations relevant to banks/NBFCs (sebi.gov.in)
- Companies: annual reports, quarterly results, and earnings-call transcripts for 8-10 banks/NBFCs from their investor-relations pages or BSE/NSE
- Check each site's terms of use before automated collection; start with manual downloads.

**Scale:** 100 documents for MVP, 300-500 for full scope.

**Required document metadata**
`doc_id, source, issuer, doc_type, title, issue_date, effective_date, supersedes[], amends[], fiscal_period, url, file_hash, parse_quality`

**Corpus rules**
- Store raw files immutably; keep a manifest (CSV/JSON) with hashes.
- Keep a held-out set of documents never used for tuning, for final evaluation.

## 5. System Overview

```
Sources → Fetcher → Raw store → Parser/OCR → Structure extractor → Chunker
        → Metadata enricher → Embeddings + BM25 index + SQL tables
Query → Rewriter → Router ─┬→ Text retrieval (hybrid + rerank, date-filtered)
                           ├→ SQL tool (extracted financial tables)
                           └→ Amendment graph lookup (P1)
      → Context builder → LLM generator → Citation verifier → Answer
All stages → Tracing + cost/latency logs → Eval and dashboard
```

## 6. Feature Set

### 6.1 Ingestion and Parsing
**P0**
- Document fetcher with manifest and dedupe by file hash.
- Parser pipeline supporting digital PDFs, with fallback OCR for scanned pages.
- Layout handling: remove headers/footers/page numbers, handle two-column pages.
- Structure extraction: sections, numbered clauses, annexures, tables.
- Parse quality score per document, with a review queue for low scores.

**P1**
- Table extraction into structured form (CSV/JSON) for financial statements.
- Comparison of two or more parsers on a sample, with documented results.

### 6.2 Chunking and Metadata
**P0**
- Fixed-window baseline chunker (for comparison).
- Clause-aware chunker for regulations (chunk = clause plus parent heading path).
- Speaker-aware chunker for earnings-call transcripts (chunk = speaker turn or Q&A pair).
- Every chunk stores `doc_id, page, clause_path, section_title, issue_date, effective_date, doc_type, issuer`.

**P1**
- Contextual chunk headers (prepend document and section context before embedding).
- Parent-child retrieval (retrieve small, return larger context).

### 6.3 Indexing and Retrieval
**P0**
- Dense embeddings in pgvector (or Qdrant).
- BM25/keyword index (Postgres full-text or OpenSearch) for exact terms like clause numbers and acronyms.
- Hybrid fusion (reciprocal rank fusion).
- Metadata filters: issuer, doc_type, date range.
- Cross-encoder reranker over top-k candidates.

**P1**
- Query rewriting (acronym expansion, decomposition of compound questions).
- HyDE or multi-query expansion, evaluated by ablation.
- Embedding model comparison (at least two models).

### 6.4 Temporal Reasoning
**P0**
- "As of date" parameter on every query (default: today).
- Retrieval excludes documents superseded before the as-of date or effective after it.
- Answers show effective date and note when a newer document exists.

**P1**
- Amendment graph (nodes = documents, edges = amends/supersedes) with a lookup tool: "show the change history for this Master Direction."

### 6.5 Structured Data and Routing
**P0**
- SQL store of key metrics per bank per quarter (gross NPA, net NPA, NIM, CASA ratio, CET1, etc.) extracted from filings, with source page references.
- LLM router (built in LangGraph) that classifies each query as text, numeric, or hybrid, and calls the appropriate tool.
- Text-to-SQL restricted to a read-only schema, with query validation.

**P1**
- Multi-step agent: decompose a comparison question into sub-queries, run them, then synthesize.
- Fallback loop: if retrieval confidence is low, rewrite the query and retry once.

### 6.6 Generation, Citations, and Refusals
**P0**
- Answer prompt that requires every claim to cite `[doc title, page, clause]`.
- Citation verifier: check that each cited passage exists in the retrieved context and supports the claim.
- Refusal behavior when evidence is insufficient ("not found in the indexed corpus").
- Standard disclaimer in UI.

**P1**
- Confidence indicator based on retrieval scores and verifier results.
- Side-by-side clause comparison view for two versions of a regulation.

### 6.7 Evaluation (Core Deliverable)
**P0**
- Golden set of at least 100 questions with reference answers and gold source passages, across types: lookup (25), numeric (20), temporal (20), comparison (15), multi-hop (10), unanswerable (10).
- Retrieval metrics: recall@k, MRR, nDCG.
- Generation metrics: correctness (LLM-as-judge validated against 20 hand-graded samples), faithfulness/groundedness, citation precision.
- One-command eval run that outputs a results table and per-question failures.
- Experiment log: every configuration change recorded with metrics.

**P1**
- Regression gate in CI: fail if a key metric drops by more than a set threshold.
- Held-out document set for final reporting.

### 6.8 Observability, Cost, and Caching
**P0**
- Trace per request with stage-level latency, tokens, and cost.
- Dashboard: latency percentiles, cost per query, refusal rate, eval scores over time.

**P1**
- Semantic cache scoped by as-of date and corpus version.
- Model routing (small model for simple lookups, larger for multi-step).

### 6.9 Refresh Pipeline
**P0**
- Scheduled job to check for new RBI/SEBI circulars, download, parse, index, and update supersession links.
- Idempotent ingestion; corpus version number increments on change.

**P1**
- Change report after each refresh (new documents, affected topics).

### 6.10 User Interface
**P0**
- Chat view with streaming answers, inline citations, and a source panel showing the exact PDF page and highlighted passage.
- Filters: issuer, document type, as-of date.
- Feedback buttons (helpful / wrong / missing source) stored for eval expansion.

**P1**
- Document browser with amendment timeline.
- Admin page for parse-quality review and eval results.

## 7. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Every factual statement in an answer carries a citation to a retrievable passage. |
| FR2 | Queries with an as-of date never cite documents outside their validity window. |
| FR3 | Numeric answers come from the SQL store and show the source filing and page. |
| FR4 | If no supporting evidence passes the confidence threshold, the system refuses. |
| FR5 | Ingestion is idempotent; re-running does not duplicate documents or chunks. |
| FR6 | Each request produces a trace with per-stage timings and cost. |
| FR7 | The eval suite runs end to end with a single command and writes a versioned report. |

## 8. Non-Functional Requirements
- **Latency:** p50 under 4 s, p95 under 10 s for text queries (excluding multi-step agent runs).
- **Cost:** target under $0.02 per average query at MVP configuration; tracked and reported.
- **Reproducibility:** pinned model versions, seeds, and corpus version in each eval report.
- **Security:** API keys in env/secret store; read-only DB role for text-to-SQL.
- **Maintainability:** modular pipeline stages with clear interfaces so techniques can be swapped for ablations.

## 9. Technical Architecture (Suggested)

| Layer | Choice |
|---|---|
| API | FastAPI, async |
| Orchestration | LangGraph for routing and agent loops (plain Python for baseline) |
| Storage | PostgreSQL + pgvector, Postgres full-text for BM25 (or OpenSearch), object storage for raw files |
| Embeddings | Compare two options (one hosted, one open-source) |
| Reranker | Open-source cross-encoder |
| LLMs | One small and one larger model for routing experiments |
| Parsing | Compare at least two PDF parsers plus OCR fallback |
| Frontend | Next.js with PDF viewer and highlight support |
| Tracing | OpenTelemetry or a tracing tool such as Langfuse |
| Jobs | Cron or a simple scheduler for refresh |
| Deployment | Docker Compose locally; optional cloud deploy |

## 10. Phased Delivery Plan

| Phase | Weeks | Deliverables | Exit criteria |
|---|---|---|---|
| 0. Foundations | 1 | Corpus manifest (100 docs), repo, tracing skeleton | Documents downloaded and hashed |
| 1. Baseline | 1-2 | Naive parse, fixed chunks, dense retrieval, basic generation, first 40 golden questions | Baseline metrics recorded |
| 2. Retrieval quality | 3-4 | Clause/speaker chunkers, metadata, hybrid search, reranker, query rewriting | Ablation table shows gains; recall@10 target met |
| 3. Structure and reasoning | 5-6 | Table extraction, SQL store, LangGraph router, temporal filters | Numeric and temporal question types pass target accuracy |
| 4. Trust and product | 7-8 | Citation verifier, refusals, UI with source viewer, refresh job, dashboard | Faithfulness and refusal targets met |
| 5. Hardening | 9 | Full 100+ question eval, held-out test, failure analysis, write-up, demo video | Public README and report |

## 11. Success Metrics (Targets, Tune After Baseline)

| Metric | Baseline expected | MVP target |
|---|---|---|
| Recall@10 (gold passage retrieved) | ~55-65% | 85%+ |
| Answer correctness (judge) | ~50% | 75%+ |
| Faithfulness / groundedness | ~70% | 92%+ |
| Citation precision | n/a | 90%+ |
| Temporal question accuracy | ~30% | 80%+ |
| Numeric question accuracy | ~40% | 85%+ |
| Correct refusal on unanswerable | ~20% | 85%+ |
| p95 latency | n/a | under 10 s |
| Cost per query | n/a | under $0.02 |

Baseline figures are guesses; replace them with measured values after Phase 1.

## 12. Evaluation Methodology
- **Golden set construction:** write questions from documents yourself, record the gold passage and answer, and have at least 20 reviewed by a second person if possible.
- **Judge validation:** hand-grade 20 answers, compare with LLM-judge scores, and report agreement.
- **Ablations to run (each isolated):** fixed vs clause-aware chunking; dense vs BM25 vs hybrid; with/without reranker; with/without query rewriting; with/without temporal filter; two embedding models; small vs large generator.
- **Failure analysis:** for the final system, categorize at least 20 failures (parse error, chunk boundary, retrieval miss, wrong routing, generation error, stale document).
- **Leakage control:** keep a held-out set of documents and questions untouched until final reporting.

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Poor PDF parsing corrupts everything downstream | Parse quality score, parser comparison, manual review of a sample |
| Superseded rules cited as current | Effective-date metadata, supersession links, temporal tests |
| LLM judge is biased or unreliable | Validate against hand grades, use reference-based checks where possible |
| Text-to-SQL errors or unsafe queries | Read-only role, schema allow-list, query validation, fixed metric templates first |
| Scope creep | Enforce P0 before P1; freeze the corpus size per phase |
| Legal risk from presenting advice | Disclaimer, no recommendations, cite sources only |
| Source site changes or blocks scraping | Manual download fallback, store raw files locally |

## 14. Future Extensions (Optional Capstones)
- Semantic caching and cost-aware model routing at scale.
- Knowledge graph over amendments and entities for multi-hop queries (GraphRAG).
- Permission-aware retrieval with per-team document access and audit logging.
- Fine-tuning an embedding model on synthetic queries from this corpus.
- Multilingual support for Hindi circulars.

## 15. Deliverables Checklist
- [ ] Corpus manifest and download scripts
- [ ] Ingestion, chunking, indexing pipelines
- [ ] Retrieval, routing, generation services
- [ ] Golden eval set and one-command eval runner
- [ ] Ablation results table and experiment log
- [ ] Web UI with citation source viewer
- [ ] Refresh job and corpus versioning
- [ ] Dashboard for cost, latency, and quality
- [ ] README with architecture diagram, benchmarks, and failure analysis
- [ ] Short demo video

## 16. Open Questions
1. Which 8-10 banks/NBFCs to include, and for which fiscal years?
2. Self-hosted open-source models versus hosted APIs for embeddings and reranking?
3. OpenSearch versus Postgres full-text for keyword search, given deployment simplicity?
4. How many golden questions can realistically be hand-written per week?