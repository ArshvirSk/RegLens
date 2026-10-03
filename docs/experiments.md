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

No golden-set eval run has been performed. There are no retrieval or generation metrics in
this document, and that is the correct state: the baseline pipeline had not been built, and
the golden set could not be drafted before the documents were parsed.

What Phase 0 *does* establish is the machinery that makes a metric trustworthy:

| Requirement | Where it lives | Verified by |
|---|---|---|
| One-command run that writes a versioned report | `make eval` → `reglens.cli eval` (Phase 1 implementation) | CLI test: unimplemented commands exit 2 with the phase number |
| Report carries config, corpus version, git commit, models | `ExperimentConfig.fingerprint()`, `Settings.corpus_version`, `describe_pricing()` | `tests/unit/test_config.py` |
| Questions have a schema, quotas and a review flag | `eval/golden/schema.md`, `eval/runners/golden.py` | `tests/unit/test_golden.py` |
| Held-out questions cannot be tuned on | `held_out` field + validator counts | `tests/unit/test_golden.py::test_review_and_held_out_counts` |
| Judge is validated against humans | Phase 1 deliverable (20 hand-graded answers) | `eval/grades/phase-1.jsonl` → `eval --agree-with` (Phase 1, agreement section in the report) |

## Parser comparison (measured, Phase 1 preprocessing)

Not an eval metric — this decides which parser feeds the chunker. Full report:
`eval/results/2026-10-02T183304Z-parser-comparison/`.

| Documents | Mean token agreement (pymupdf vs pdfplumber) | Faster | Mean quality (pymupdf) |
|---|---|---|---|
| 20 | 0.9732 | pymupdf 20/20 (8.46 s vs 229.15 s) | 0.998 |


Worst agreement: 0.9048 (`rbi_md_psl_2020`). Pages `[57, 73, 86]` yielded almost no text
and are flagged as OCR candidates. Decision: **pymupdf stays the baseline parser on this
evidence** (`parsing.parser: pymupdf`).

## Phase 1 — baseline (measured)

| # | Config | Chunking | Embedding | Retrieval | recall@10 | MRR@10 | nDCG@10 | Correctness | Faithfulness | Report |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | `baseline_naive` | fixed 512/64 | `gemini-embedding-001` (3072-dim) | dense top-10 | 0.9730 | 0.7820 | 0.7959 | 0.7750 (1.55/2) | 0.9750 (1.95/2) | `eval/results/2026-10-03T121902Z_baseline_naive_e98261b91683` |

Baseline is deliberately naive: fixed-size chunks, one hosted embedding model, dense
retrieval only, no reranking, no rewriting, no temporal filtering. It stays runnable for the
rest of the project.

Generation detail from the same run (all numbers from the report):

| Metric | Value | Denominator |
|---|---|---|
| Citation precision | 1.0000 | 28 answered-with-citations of 37 answerable (9 answers cite nothing → graded undefined, not 0) |
| Refusal rate | 0.2750 | 40 scored |
| Correct refusals on unanswerable | 3/3 | 3 unanswerable |
| False refusals (answerable refused) | 8 | 37 answerable — 6 of them with recall@10 = 1.0, i.e. the evidence *was* in context |
| Judge mean correctness by type | lookup 1.846, numeric 1.556, temporal 1.500, comparison 0.800, multi_hop 1.250, unanswerable 2.000 | judge score 0–2 |
| Judge vs human agreement | exact 0.95 C / 0.80 F; within-one 0.95 C / 0.80 F | paired = 20 hand grades (`eval/grades/phase-1.jsonl`) |
| Cost | $0.345511 per 40-question run; serving path (embed+answer only) **$0.002148/question** | judge scoring is $0.259601 of the run |
| Answer latency (answerer call only) | p50 1284 ms, p95 2271 ms | n=40 |

Caveats that belong to this row: all 40 questions are still `review:draft`, 7 held-out
questions are included in the aggregates, the HNSW index is skipped at 3072 dimensions
(search is exact kNN, not approximate), and this run precedes any tuning — the agreement
numbers were attached post-hoc with `eval --agree-with` and no experiment was changed
after seeing them.

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
| p50 latency (text) | < 4 s | 1.284 s (answerer call; retrieval not in this number) | `eval/results/2026-10-03T121902Z_baseline_naive_e98261b91683` |
| p95 latency (text) | < 10 s | 2.271 s (answerer call; retrieval not in this number) | same report |
| Cost per query | < ~$0.02 | $0.002148 serving path (embed+answer); $0.008638 including judge scoring | report `totals` |
| Judge vs human agreement | report actual | paired=20: exact 0.95 C / 0.80 F, within-one 0.95 C / 0.80 F | report `judge.agreement`, `eval/grades/phase-1.jsonl` |

## Changelog

| Date | Change |
|---|---|
| 2026-10-02 | Created. Phase 0 complete: no metrics yet, by design. |
| 2026-10-02 | Parser comparison measured on all 20 fetched documents; pymupdf retained. |
| 2026-10-03 | Phase 1 baseline run measured (row 1.0 + detail tables). Two measuring-instrument bugs found by reading the first run and fixed before it was recorded: judge output budget (thinking tokens ate the 400-token cap, 21/40 replies truncated) and `CITATION_RE` (rejected the `p.40-41` ranges its own context block emits). First run discarded uncommitted; row 1.0 comes from the clean re-run. |
| 2026-10-03 | 20 hand grades recorded and judge agreement attached post-hoc (`eval --agree-with`): exact 0.95/0.80. All 5 disagreements are refusal-related; judge-prompt conflict documented in `docs/learning/phase-1.md`. |
