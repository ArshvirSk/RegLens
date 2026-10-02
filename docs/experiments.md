# Experiment log

Every technique is a separate, toggleable step, measured before and after. **A row appears
here only after an actual eval run has written a report under `eval/results/`.** No row may
be filled in from intuition, a single example, or a model's opinion of itself.

## How to read this file

* `config` is the experiment YAML in `src/reglens/config/experiments/`; the hash in the
  report identifies the exact behaviour-affecting settings.
* `report` is the file under `eval/results/` that produced the row.
* Metrics are `recall@10`, `MRR@10`, `nDCG@10` (retrieval), `correctness`, `faithfulness`,
  `citation precision`, `refusal accuracy` (generation).
* **—** means "not measured", never "zero". A blank cell is a gap to close, not a result.

## Phase 0 status

No eval run has been performed. There are no metrics in this document, and that is the
correct state: the baseline pipeline does not exist yet, and the golden set is empty because
the working rules require questions to be drafted from parsed documents.

What Phase 0 *does* establish is the machinery that makes a metric trustworthy:

| Requirement | Where it lives | Verified by |
|---|---|---|
| One-command run that writes a versioned report | `make eval` → `reglens.cli eval` (Phase 1 implementation) | CLI test: unimplemented commands exit 2 with the phase number |
| Report carries config, corpus version, git commit, models | `ExperimentConfig.fingerprint()`, `Settings.corpus_version`, `describe_pricing()` | `tests/unit/test_config.py` |
| Questions have a schema, quotas and a review flag | `eval/golden/schema.md`, `eval/runners/golden.py` | `tests/unit/test_golden.py` |
| Held-out questions cannot be tuned on | `held_out` field + validator counts | `tests/unit/test_golden.py::test_review_and_held_out_counts` |
| Judge is validated against humans | Phase 1 deliverable (20 hand-graded answers) | — |

## Phase 1 — baseline (to be filled by the first real run)

| # | Config | Chunking | Embedding | Retrieval | recall@10 | MRR@10 | nDCG@10 | Correctness | Faithfulness | Report |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | `baseline_naive` | fixed 512/64 | `text-embedding-3-small` | dense top-10 | — | — | — | — | — | — |

Baseline is deliberately naive: fixed-size chunks, one hosted embedding model, dense
retrieval only, no reranking, no rewriting, no temporal filtering. It stays runnable for the
rest of the project.

## Phase 2 — ablation queue

Each row is one isolated change against the best config known at the time. Order matters:
cheap, high-leverage changes first.

| # | Change | Config toggle | Hypothesis | recall@10 vs previous | Verdict |
|---|---|---|---|---|---|
| 2.1 | Clause-aware chunking | `chunking.strategy: clause_aware` | Regulation text is clause-structured; fixed windows split requirements mid-condition | — | — |
| 2.2 | Speaker-aware chunking (transcripts) | `chunking.strategy: speaker_aware` | Q&A pairs lose meaning when split; the analyst's question is half the answer | — | — |
| 2.3 | Metadata enrichment | *(chunk metadata fields)* | issuer/doc_type/date context improves filtering and disambiguation | — | — |
| 2.4 | Contextual chunk headers | `chunking.contextual_headers: true` | Prepending document + section context reduces "which document is this from" errors | — | — |
| 2.5 | Postgres FTS keyword arm | `indexing.keyword_index: postgres_fts` | Clause numbers and acronyms are lexical, not semantic | — | — |
| 2.6 | Hybrid fusion (RRF) | `retrieval.fusion: rrf` | Fusion beats either arm alone; RRF needs no score calibration | — | — |
| 2.7 | Cross-encoder reranking | `retrieval.rerank: true` | A cross-encoder fixes ordering errors the bi-encoder cannot see | — | — |
| 2.8 | Query rewriting (acronyms) | `rewriting.acronym_expansion: true` | "LCR" ↔ "Liquidity Coverage Ratio" is a systematic miss | — | — |
| 2.9 | Query decomposition | `rewriting.decomposition: true` | Compound questions need two retrievals, not one | — | — |
| 2.10 | Parent-child retrieval | `chunking.parent_child: true` | Retrieve precisely, return enough context to answer | — | — |
| 2.11 | Second embedding model | `indexing.embedding_model` | Compare hosted vs open-source on the same corpus | — | — |
| 2.12 | Multi-query / HyDE | `rewriting.multi_query` / `hyde` | Might help; expected to cost latency | — | — |

Selection rule: the best Phase 2 config is chosen on the **dev** questions and then
re-checked once on the **held-out** questions. Recall@10 must improve materially over
baseline, and the actual delta is reported — including if it is smaller than hoped.

## Phase 3 — structure, time, routing

| # | Change | Metric | Baseline | Result |
|---|---|---|---|---|
| 3.1 | Metric extraction into `metrics` | numeric accuracy, extraction confidence | — | — |
| 3.2 | As-of date filtering | temporal accuracy, stale-citation rate (must be 0) | — | — |
| 3.3 | LangGraph router (text / numeric / hybrid) | routing accuracy per type | — | — |
| 3.4 | Templated metric queries | numeric accuracy | — | — |
| 3.5 | Free-form text-to-SQL (guarded) | numeric accuracy, unsafe-query rejection | — | — |

## Phase 4 — trust and product

| # | Change | Metric | Result |
|---|---|---|---|
| 4.1 | Citation verifier | citation precision, faithfulness | — |
| 4.2 | Calibrated refusal threshold | refusal accuracy, false-refusal rate | — |
| 4.3 | Confidence indicator | correlation with correctness | — |

## Phase 5 — extras with measured impact

| # | Change | Metric | Result |
|---|---|---|---|
| 5.1 | Semantic cache | p50 latency, cost/query, hit rate | — |
| 5.2 | Model routing (small vs large) | correctness, cost/query | — |

## Non-functional measurements

| Metric | Target | Measured | Where |
|---|---|---|---|
| p50 latency (text) | < 4 s | — | — |
| p95 latency (text) | < 10 s | — | — |
| Cost per query | < ~$0.02 | — | — |
| Judge vs human agreement | report actual | — | — |

## Changelog

| Date | Change |
|---|---|
| 2026-10-02 | Created. Phase 0 complete: no metrics yet, by design. |
