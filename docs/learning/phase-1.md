# Phase 1 — Baseline

## 1. Why Phase 1 is a measurement phase

The PRD's exit criterion for Phase 1 is one sentence — "Baseline metrics recorded" — and the
temptation is to smuggle in improvements until the numbers look good. That would poison
everything after it: an ablation can only be read against a baseline that was allowed to be
bad. So Phase 1 shipped the *naive* pipeline end to end and then spent most of its budget on
making the measuring instrument trustworthy, because the first real eval run proved the
instrument itself was broken in two silent ways (§5).

## 2. What was built, and why this way

* **The naive pipeline, one config end to end**: `baseline_naive` — fixed 512/64 chunks,
  `gemini-embedding-001` (3072-dim), dense top-10 retrieval, one answerer prompt
  (`grounded_v0`), `gemini-3.6-flash` judge. No reranker, no rewriting, no temporal filter.
* **40 golden questions** drafted from the parsed corpus (13 lookup / 9 numeric / 6 temporal /
  5 comparison / 4 multi-hop / 3 unanswerable), 7 flagged held-out, all still `review:draft`.
* **Dimension safety before spend**: migration `0002_embedding_dim.sql` realigns
  `chunks.embedding` to the configured dimension, and `ensure_store_dimension` preflights the
  store's actual `atttypmod` *before* any paid embed call — a mismatch is a clean error, not
  1212 chunks of garbage. The HNSW creation is wrapped in a `program_limit_exceeded` guard
  because pgvector's `hnsw` caps at 2000 dimensions and our column is 3072 (verified against
  pgvector 0.8.2): the index is skipped with a NOTICE and search falls back to **exact kNN**,
  which is the honest Phase 1 baseline — no ANN recall loss to explain later.
* **Token accounting that survives a lying API**: `gemini-embedding-001` returns
  `statistics=None`, so embed cost falls back to a tiktoken proxy count; the Gemini chat path
  counts `thoughts_token_count` as output because the API bills it that way.
* **Judge validation harness**: 20 stratified hand grades
  (`eval/grades/phase-1.jsonl`) paired post-hoc with `eval --agree-with`, which re-runs the
  exact `agreement()` the run-time flag uses — no second eval, no regenerated answers
  paired against grades written for different text.

## 3. Alternatives considered and rejected

* **halfvec expression index for 3072-dim HNSW.** Deferred to Phase 2: exact kNN removes a
  confound from the baseline (if recall is wrong, it is retrieval's fault, not the index's),
  and the guard makes the trade visible instead of crashing.
* **Disabling thinking on the judge** (`thinking_budget=0`). Rejected for now: it changes
  judge behaviour, and the bug was the *budget*, not the thinking. Raising
  `max_output_tokens` 400 → 4000 fixes truncation without touching how the judge reasons.
* **Tuning the answerer prompt to stop the false refusals before recording the baseline.**
  Rejected: the refusals *are* the baseline measurement. Fixing them belongs to an
  experiment row with a before/after, not to the first number.
* **Re-running the eval with `--human-grades`** instead of building `--agree-with`.
  Rejected: hand grades are written against saved answers; a re-run generates new ones and
  would pair grades with text nobody graded.

## 4. What the metrics showed

Source of every number: `eval/results/2026-10-03T121902Z_baseline_naive_e98261b91683`
(full table in `docs/experiments.md`, row 1.0). 40/40 judged, 0 failures, 0 missing scores.

| Metric | PRD baseline expectation | Measured |
|---|---|---|
| recall@10 | ~55–65% | **97.30%** (37/37 answerable) |
| MRR@10 / nDCG@10 | — | 0.7820 / 0.7959 |
| Correctness (judge, mean/2) | ~50% | **77.50%** (1.55/2) |
| Faithfulness (judge, mean/2) | ~70% | **97.50%** (1.95/2) |
| Citation precision | n/a | **100%** over 28 citing answers (9 of 37 answers cite nothing → undefined, not 0) |
| Correct refusal on unanswerable | ~20% | **3/3 = 100%** |
| Answer latency | — | p50 1.284 s, p95 2.271 s (answerer call only) |
| Cost per question | < $0.02 | $0.002148 serving path; $0.008638 including judge |

Judge vs the 20 hand grades (`paired=20`): exact **0.95 correctness / 0.80 faithfulness**,
within-one 0.95 / 0.80. The five disagreements are all refusal-related and are the most
useful output of the phase (§5).

By type (judge mean): lookup 1.846, numeric 1.556, temporal 1.500, multi_hop 1.250,
comparison 0.800, unanswerable 2.000. Comparison is the weak spot: 4 of its 5 questions were
refused.

## 5. What surprised me

1. **The first eval run's judge was silently broken.** 21/40 judge replies came back as
   `{"correctness": 2, "faithfulness": 2` — 36 visible characters — because
   `gemini-3.6-flash` spends its thinking tokens inside `max_output_tokens` and the cap was
   400 (recorded `output_tokens=396`). The parse correctly reported "missing, not zero", so
   the report *looked* fine unless you read the error strings. Fix: budget 400 → 4000,
   regression test pins it. The discarded run was never committed.
2. **The citation parser rejected its own context format.** `build_context_block` prints
   multi-page chunks as `p.40-41` and PDF extraction leaves `p. 7`; models cite both
   verbatim. `CITATION_RE` only accepted `p.N`, so 19 of 20 answers that visibly cited were
   reported `has_citations=false` (9 → 28 after the fix). Genuinely mangled output
   (q0019's `| p.18 |` separators) still counts as missing — that is the defect the metric
   exists to surface.
3. **Eight answerable questions were refused — with the evidence in the prompt.** Six of
   the eight had recall@10 = 1.0; I checked the chunk text directly: q0015's context
   literally contains "40 per cent of ANBC … or CEOBE whichever is higher". The small
   answerer refuses anyway, most often on comparison questions whose answer sits in a
   PDF-extracted table (4 of 5 comparison questions refused). This is a *model behaviour*
   baseline, not a bug: retrieval delivered, generation declined. The two refusals with
   recall 0.5 (q0029, q0037) were partly justified — one side of the comparison really was
   missing — and the judge scored those C=2 while scoring the evidence-present ones C=0.
4. **The judge contradicts itself on refusal faithfulness.** For identical refusal behaviour
   with the evidence present, it gave faithfulness 0 (q0015) and 2 (q0023, q0030, q0031,
   q0035) in the same run. My hand grades put those at 0 — "INSUFFICIENT EVIDENCE" when the
   context contains the answer contradicts the context — which is where 4 of the 5
   disagreements come from.
5. **The judge prompt has a real conflict on q0029-type cases.** It says both "refuses a
   question the reference shows is answerable → correctness 0" and "a refusal the context
   supports is fully correct". When retrieval drops one side of a comparison (recall 0.5)
   both rules fire; the judge chose 2, I graded 0 (the correctness axis is defined against
   the reference). This is a prompt-definition issue to resolve *after* the disagreement is
   documented — per `eval/README.md`, the judge is not tuned before that.
6. **Recall@10 is far above the PRD's expectation (97% vs 55–65%).** The guess assumed a
   harder corpus/retrieval problem than 20 structured RBI directions with an
   above-average embedding model pose. The PRD's baseline column is explicitly a guess and
   is now replaced by measured values.

## 6. What to look at first

* `eval/results/2026-10-03T121902Z_baseline_naive_e98261b91683.md` — the run, one screen.
* `eval/grades/phase-1.jsonl` — the 20 hand grades and the reasoning per question.
* The 5 disagreements (q0023, q0029, q0030, q0031, q0035) — every one is a refusal; they
  define Phase 4's refusal-calibration work and the judge-prompt fix.
* `docs/experiments.md` row 1.0 — the numbers every later row is compared against.

## 7. What Phase 2 will do

Re-run the same 40 questions after each isolated change and report deltas against row 1.0,
chosen on dev questions and re-checked once on the 7 held-out: clause-aware chunking,
keyword arm + hybrid fusion, reranking, query rewriting. Specific open threads from this
phase: the 8 false refusals (were the tables the problem? a reranker won't fix a refusal the
evidence was already present for), the halfvec index if we ever want ANN again, and whether
comparison questions need query decomposition rather than a single dense query.
