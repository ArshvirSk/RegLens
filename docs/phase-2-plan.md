# Phase 2 plan — retrieval quality

**Status:** planned, not started. Written before any Phase 2 run, so nothing here can be
retrofitted to a result. §1 decisions need owner sign-off before the first eval run.
**PRD definition of done:** "Ablation table shows gains; recall@10 target met."

---

## 1. Decisions before the first run (recommendations given)

| # | Decision | Recommendation | Why it must happen first |
|---|---|---|---|
| D1 | **Review the golden set**: owner reads all 40 questions and flips `review: draft → reviewed` | Do it before any tuning | We are about to *choose configs* on these questions. Draft questions are exactly the ones whose gold passages might be wrong; tuning against an unreviewed set can bake a bad gold into the winning config. Also removes `--allow-drafts` and makes every number quotable. |
| D2 | **Freeze the corpus at 20 documents** for the whole phase | Freeze | Every chunking ablation re-embeds and reindexes. Adding documents at the same time makes a delta unattributable to the change that produced it. Corpus growth happens as its own re-baselined row (1.2) whenever we do it. |
| D3 | **Fix the judge-prompt conflict** (documented in `docs/learning/phase-1.md` §5.5, q0029-type cases) and **re-run the baseline as row 1.1** | Do it now, before ablation 2.1 | Phase 1 deferred the fix until the disagreement was documented — it is (5 disagreements, all refusal-related). The ambiguity sits on refusal scoring, and Phase 2 will keep producing refusal cases. Rule: the measuring instrument may change only *between* ablations, and any instrument change forces a baseline re-run (~$0.35) so rows stay comparable. Cost of not doing it: every correctness/faithfulness delta carries an asterisk. |
| D4 | **Dev/held-out split plumbing**: reports currently include the 7 held-out questions (the report itself warns about this) | Add `eval --split dev\|held_out\|all` (default `all`, the chosen value recorded in the report header) | The selection rule is "choose on dev, confirm once on held-out". Recomputing dev-only numbers by hand from the JSON each run is how leakage happens. 33 dev / 7 held-out. |

## 2. Success criteria — honestly redefined against row 1.0

Recall@10 is **saturated**: 0.9730 measured vs the PRD's 85% MVP target (the PRD's
"baseline expected" 55–65% was a guess; `docs/experiments.md` replaces guesses with
measured values). There is at most ~1 question-equivalent of recall headroom, and the
known misses are q0029 and q0037 (recall 0.5 — one side of a comparison absent from
context). **A Phase 2 whose only claim is "recall went up" would be noise.** The gains
must show where there is actually room:

| Metric | Row 1.0 measured | Phase 2 target | Why this is the target |
|---|---|---|---|
| recall@10 | 0.9730 | ≥ 0.9730, no regression | Saturated; guard against chunking regressions |
| MRR@10 | 0.7820 | material gain | Ordering quality — what fusion + reranking fix |
| nDCG@10 | 0.7959 | material gain | Same |
| Correctness (mean/2) | 0.7750 | gain driven by weak types | comparison 0.800, multi_hop 1.250, temporal 1.500, numeric 1.556 |
| Faithfulness (mean/2) | 0.9750 | ≥ 0.9750 | Must not regress |
| Citation precision | 1.0000 (28 citing) | ≥ 1.0000 | Must not regress |
| Correct refusal | 3/3 | 3/3 | Must not regress |
| Serving cost/query | $0.002148 | ≤ $0.02 (PRD §8) | Gate, not a goal |
| Answer p95 latency | 2271 ms | ≤ 10 s (PRD §8) | Gate, incl. any rerank cost |

Targets are fixed here, before any run. No target gets moved after seeing a result.

**Phase 1 open threads → where they land in Phase 2:**

* 8 false refusals (6 with recall 1.0): the *generation* side is Phase 4 (calibrated
  refusal threshold, experiments row 4.2) — **not touched here** (generation stays frozen,
  see §4). The *retrieval* side is ours: q0029/q0037 at recall 0.5 are exactly what
  hybrid + rerank should repair; check them per-question after 2.6/2.7.
* Judge self-contradiction on refusals → D3.
* Comparison questions weakest (0.800): if still weak after 2.1–2.8, that is what
  decomposition (2.9) exists for.
* halfvec/HNSW index → deferred, see §5.

## 3. Ablation sequence

Each row = one isolated toggle + one eval run + one `docs/experiments.md` row backed by
a report in `eval/results/`. Build on the **best config so far**, not on baseline, so
each row answers "does this still help on top of what we kept?"

| Step | Row | Change | Build work | Primary metric watched |
|---|---|---|---|---|
| 0 | 1.1 | Judge-prompt fix + baseline re-run (D3) | `eval/runners/judge.py` prompt text + regression test; re-run `baseline_naive` | New reference for correctness/faithfulness |
| 1 | 2.1 | **Clause-aware chunking** (`chunking.strategy: clause_aware`) | New `src/reglens/chunking/clause.py` implementing the existing `Chunker` protocol: chunk = clause + parent heading path; parses numbered clauses out of parsed pages. Then `reglens reindex` (measured ~73 s for 20 docs) | recall, MRR — does structured text beat 512-token windows? |
| 2 | 2.4 | **Contextual chunk headers** (`chunking.contextual_headers: true`) | Prepend document + section title to the embedded text only (stored text unchanged, or a separate embed field — decide at build time, keep citations pointing at original text) | recall on "which document is this from" misses |
| 3 | 2.5 | **Postgres FTS keyword arm** (`indexing.keyword_index: postgres_fts`) | Migration `0003_fts.sql`: `tsvector` generated column + GIN index on chunk text; a `KeywordRetriever` implementing the existing `Retriever` protocol; `websearch_to_tsquery` for the query side. Row is **BM25-only vs dense-only** (the PRD's "dense vs BM25 vs hybrid" comparison) | recall on clause numbers/acronyms (q-numbering, "LCR", "PSL") |
| 4 | 2.6 | **Hybrid fusion, RRF** (`retrieval.mode: hybrid`, `fusion: rrf`, `candidate_k: 50`) | Fusion step over both arms' rankings; no score calibration needed (that is why RRF) | recall ≥ baseline, MRR; fixes q0029/q0037? |
| 5 | 2.7 | **Cross-encoder reranking** (`retrieval.rerank: true`, `BAAI/bge-reranker-base`, top 5 of 50) | Implement the existing `Reranker` protocol. **Dependency decision at build time**: full `sentence-transformers` (torch, heavy image layer) vs minimal ONNX path — choose by measured latency and image size, not preference. Re-measure p95 | MRR/nDCG — the row most likely to move ordering |
| 6 | 2.8 | **Acronym expansion rewriting** (`rewriting.acronym_expansion: true`) | Deterministic acronym map file first (zero cost/latency delta, unit-testable); LLM rewriting only if the map is insufficient and its cost/latency delta is reported | recall on acronym questions |
| 7 | 2.9 | **Query decomposition** (conditional) | Only if comparison/multi-hop remain the weak types after steps 1–6: split compound questions into sub-queries, fuse results | comparison correctness (0.800 → ?) |
| 8 | 2.10–2.12 | Parent-child retrieval, second embedding model, multi-query/HyDE (conditional) | Only if targets in §2 are still missed after 1–6, each as its own row | as labelled in the queue |

**Order rationale:** chunking first (it changes the index, so everything downstream must
be measured against the new chunks); index arms before fusion (fusion needs both arms);
rerank after fusion (it consumes fused candidates); query-side changes last (they need no
reindex and are the cheapest to A/B).

**Per-row working rule:** `ruff check .` + full suite green before each commit; micro
commits; push at end of session; a row appears in `docs/experiments.md` only when its
report exists — including rows whose delta is zero or negative (the log's honesty rule).

## 4. What stays frozen during Phase 2

* **Generation**: same answerer model, prompt, `max_output_tokens` as `baseline_naive`.
  A retrieval change is only attributable if everything after retrieval is byte-identical.
* **The corpus** (D2) and **the golden set** (D1, after review) — no new questions, no
  new documents mid-phase. Adding either invalidates every prior row.
* **The judge** after D3's re-run — disagreements are documented, not tuned away
  (`eval/README.md` rule).
* `baseline_naive` stays runnable and untouched — it is the reference forever.

## 5. Deferred with reasons

| Item | Why not Phase 2 |
|---|---|
| Speaker-aware chunker (2.2, PRD deliverable) | **Blocked on corpus**: 20 RBI regulations, no earnings-call transcripts exist to chunk. Fetching transcripts is a corpus change (D2) and would force a re-baseline; schedule with corpus growth, or accept explicitly as a PRD deliverable deferred with this note. |
| Metadata enrichment (2.3) as its own row | Chunk metadata (`doc_id, page, clause_path, dates, issuer`) is already stored from Phase 1. The remaining work is *using* it — `temporal_filter` / issuer filters — which is Phase 3 territory (as-of dates, stale citations). |
| halfvec expression index / HNSW | 1,212 chunks answer by exact kNN in milliseconds; the guard in migrations 0001/0002 already skips the 3072-dim HNSW with a NOTICE. Revisit only when corpus growth makes exact kNN a measured latency problem. |
| False refusals, refusal thresholds, confidence indicator | Phase 4 (rows 4.1–4.3). Phase 2 only fixes the *retrieval* contribution (missing evidence). |
| Corpus growth beyond 20 docs / 100-doc MVP target | Its own re-baselined row (1.2) with a fresh baseline run, never mixed into an ablation. |

## 6. Measurement protocol (per row)

1. One toggle. If two things change, the row answers nothing.
2. Run on **dev only** (33 questions, after D4); compare against **row 1.1** and against
   the **previous best row** — both deltas reported.
3. Record every §2 metric plus cost and p50/p95; per-question records kept for failure
   analysis (which questions moved, and why).
4. Keep the row in the log even if the delta is ≤ 0; a rejected change is still a result.
5. Gate: reject a change that improves retrieval metrics but violates a §2 gate (cost,
   latency, faithfulness, refusal).
6. Chosen best config gets **exactly one** confirmation run on the held-out 7 at the end
   of the phase. Held-out results are reported, never used to choose.

## 7. Exit criteria — Phase 2 is done when

- [ ] `docs/experiments.md` contains row 1.1 plus every executed row 2.x, each backed by
      a report in `eval/results/` (config hash recorded).
- [ ] At least one material gain over 1.1 on MRR@10 or nDCG@10, with recall@10 not
      regressed — or an honest written finding that the naive baseline is already at the
      ceiling for this corpus and why.
- [ ] The chosen config re-checked once on held-out; dev/held-out numbers reported
      separately.
- [ ] `docs/learning/phase-2.md` retrospective in the Phase 1 format: what shipped, what
      the numbers said, surprises, rejected alternatives.
- [ ] PRD §11 baseline column updated with measured Phase 1 values (the PRD already
      instructs this; it has not been done yet).
- [ ] Status report to the owner, then **STOP** — Phase 3 (tables, SQL, router, temporal
      filters) does not start without confirmation.

## 8. Budget and effort

* Eval run: **$0.345511** per 40-question run (measured, row 1.0). ~9 runs planned
  (1.1 + six mandatory + conditionals) ≈ **$3–4** in judge/serving tokens.
* Reindex after each chunking change: ~73 s, no new embed spend beyond one re-embed of
  615k chunk tokens (measured Phase 1 ingest; exact $ recorded by the next run's cost
  report).
* The only open engineering risk with real cost is step 5's reranker dependency (torch
  image weight + CPU latency); timebox the decision to a measured benchmark, not debate.
