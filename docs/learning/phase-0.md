# Phase 0 — Foundations

Written for a strong developer who has not built a RAG system before. Phase 0 builds no
retrieval and no generation. It builds the things that decide whether the numbers you produce
later can be believed.

**What exists now:** repo scaffold, Docker Compose stack (Postgres+pgvector, FastAPI, Next.js),
typed configuration, hashed experiment configs, the corpus manifest with schema and validator,
a robots-aware fetcher, the Postgres schema, a tracing skeleton, 121 passing tests, CI.

---

## 1. Why Phase 0 is not just setup

A RAG system is easy to build badly and hard to *know* you built badly. Three things in
particular are cheap to do first and painful to retrofit:

**Eval before optimisation.** If you tune retrieval before you can measure it, you cannot
tell a real improvement from a lucky example. So the golden-set schema, the validator, the
per-type quotas and the review/held-out flags all exist before the first pipeline does. The
question file is deliberately *empty*: a question written before its document is parsed has
no verifiable gold passage, and a gold passage you cannot point at is not ground truth.

**Reproducibility.** Every eval report must record the config, the corpus version, the git
commit and the model versions. That only works if the configuration is a first-class,
hashable object rather than a scatter of constants. So `ExperimentConfig` exists in Phase 0
with **every** future toggle declared (chunking, hybrid search, reranking, rewriting,
temporal filtering, generation), and `baseline_naive.yaml` switches them all off.

**Idempotency.** Ingestion runs many times during development. If re-running duplicates
documents or chunks, every later measurement is suspect. So idempotency is enforced by
constraints (`UNIQUE (file_hash)` on documents, `UNIQUE (doc_id, corpus_version,
chunk_strategy, chunk_index)` on chunks) and by content-addressed storage, not by application
logic that a future edit could forget.

---

## 2. What was built, and why this way

### 2.1 Configuration: settings vs experiments

Two kinds of configuration, and mixing them is a common early mistake.

* **Settings** (`config/settings.py`) — environment-driven facts: database URL, API keys,
  directories, which experiment is active. Secrets are `SecretStr`, and `redacted()` exists so
  a settings dump can be logged or served without leaking anything.
* **Experiment configs** (`config/experiment.py` + `experiments/*.yaml`) — the pipeline
  choices under study. These are hashed over *behaviour-affecting fields only*, so editing a
  description cannot invalidate recorded results.

The hash is the small thing that makes the whole ablation story work: every API response and
every eval report carries it, so "which configuration produced this number?" is a lookup, not
an argument.

### 2.2 The corpus manifest

`data/manifest.csv` is one row per document, committed to git; the PDFs it points at are not.
Why CSV: it is diffable in review, editable by hand, and readable by anything. Why a manifest
at all: because a RAG system's answers are only as good as the corpus, and the corpus needs
to be an explicit, reviewable decision rather than "whatever the crawler found".

The validator is the interesting part. It checks identity, enums, date formats, hash formats,
reference validity (`supersedes`/`amends` must point at real `doc_id`s), and — most
importantly — **re-hashes the files on disk**. If a raw file changes after it was fetched,
validation fails. That is what "raw files are immutable" means in practice.

### 2.3 The fetcher

Constraints, in order of how much they change the design:

1. **Only public documents, and only politely.** `robots.txt` is fetched and obeyed per host;
   requests are throttled per host (default 2 s); responses are size-capped; transient errors
   retry with backoff; a non-2xx is recorded, never raised into the middle of a batch.
2. **Nothing is downloaded by accident.** `make download` is a dry run. Fetching requires
   `--yes --accept-terms`, an explicit statement that the operator has checked the site's
   terms of use.
3. **Immutable, content-addressed storage.** Files land at
   `data/raw/<source>/<doc_id>/<doc_id>__<sha256[:12]>.pdf`. Re-fetching the same bytes is a
   no-op; different bytes create a second file and get reported.
4. **Honest failures.** RBI's `robots.txt` returns HTTP 418 and its pages are viewstate-rendered.
   Rather than pretend otherwise, the fetcher treats unreachable robots rules as "no rules
   published" *and logs the reason*, and the RBI listing pages are excluded from the corpus with
   a note explaining that the Phase 4 refresh job needs a browser adapter.

The per-host `Referer` deserves a mention because it is a real-world detail no design document
would have predicted: `rbidocs.rbi.org.in` returns an HTML interstitial (~45 KB, `text/html`)
for a plain GET of a PDF, and the real PDF (2 MB, `application/pdf`) with a `Referer` header.
That was found by testing, not by reading.

### 2.4 The database schema

Plain numbered SQL with checksums, rather than an ORM or a migration framework, because the
interesting parts *are* the SQL:

```sql
embedding  vector(1536),                        -- pgvector, dimension from typed settings
tsv        tsvector GENERATED ALWAYS AS         -- keyword arm, always in sync with `text`
             (to_tsvector('english', text)) STORED,
CONSTRAINT chunks_identity_key UNIQUE (doc_id, corpus_version, chunk_strategy, chunk_index)
```

Two subtleties worth internalising early:

* **A generated `tsvector` column** means the keyword index can never drift from the text.
  A trigger could; a generated column cannot.
* **Chunk identity includes the strategy and corpus version.** When Phase 2 re-chunks with
  `clause_aware`, the baseline's `fixed` chunks must survive, because the ablation compares
  them. Overwriting would destroy the comparison.

One deliberate omission: `chunks` has no `embedding_model` column. Vectors from different
models are not comparable, so a corpus version is indexed with exactly one model, recorded in
the experiment config. Mixing them in one table is the failure this prevents.

### 2.5 Tracing

Every request produces a `Trace` with per-stage timing, token counts and cost. Three sinks, in
increasing order of setup cost: JSON logs (always), Postgres (best effort), OTLP/Langfuse (only
if configured *and* installed). A fresh clone must be able to answer a question without a
collector or an account.

Cost accounting has one rule that matters: an unpriced model **raises**, and the tracing layer
turns that into a loud warning rather than a silent `$0.00`. A cost report that quietly treats
an unknown model as free is worse than no cost report.

### 2.6 The honest stub

`POST /ask` returns **HTTP 501** with a body explaining that retrieval arrives in Phase 1.

This is a deliberate choice worth copying. A stub that returns `200 {"answer": ""}` is
indistinguishable from a pipeline that silently failed. The eval harness must be able to treat
"not built yet" as an error, not as a wrong answer — otherwise the first baseline run would
report a 0% correctness that means nothing.

---

## 3. Alternatives considered and rejected

| Decision | Rejected alternative | Why |
|---|---|---|
| Reviewed corpus plan | General crawler | Would silently collect wrong files, and collect nothing usable from RBI. |
| Content-addressed raw store | `doc_id` filenames | Overwrite risk destroys evidence. |
| Postgres for vectors + FTS + metadata | Separate vector DB + OpenSearch | Extra moving parts with no measured benefit at 100–500 documents. The `KeywordIndex` protocol keeps the swap possible. |
| Plain SQL migrations with checksums | Alembic | Heavier; autogenerate would fight hand-written DDL. Checksums give the guarantee that matters: an applied migration cannot silently change. |
| Plain Python pipeline in Phase 1 | LangGraph from the start | A graph framework would hide the thing being learned. Routing arrives in Phase 3 as the PRD specifies. |
| Local-first tracing | Langfuse as a hard dependency | A fresh clone should not need an account to answer a question. |
| Empty golden set in Phase 0 | Drafting 40 questions now | A question without a verifiable gold passage is not ground truth. |
| `make` delegating to `scripts/tasks.py` | Duplicated command lists | The build host has no GNU make; one source of truth avoids drift. |
| Next.js 15.5.27 | Next.js 15.5.4 | npm warned that 15.5.4 carries CVE-2025-66478. Pinned to a patched release. |

---

## 4. What the metrics showed

**Nothing — and that is the honest answer.** No eval run has been performed, because the
baseline pipeline does not exist and the golden set is empty. `docs/experiments.md` contains
no numbers, and every metric cell is `—` ("not measured"), not `0`.

Two numbers *were* measured during Phase 0, and both are about the code rather than the system:

| Measurement | Value | Source |
|---|---|---|
| Unit tests | 121 passed, 7 deselected (integration, needing Postgres) | `uv run pytest -m "not integration and not eval"` |
| Manifest validation | 102 rows, 0 errors, 0 warnings | `make validate-manifest` |
| Web build | Compiles and type-checks; 3 static routes | `npm run build` |

Cost per query is not measurable yet. For planning: a typical 6,000-token context with a
400-token answer on `gpt-4o-mini` prices at **~$0.0011** using the dated table in
`observability/cost.py`, so the ~$0.02/query budget is not the binding constraint — retrieval
quality is.

---

## 5. What surprised me

1. **The most important Phase 0 code is the boring code.** The validator and the checksum
   runner look like plumbing. They are what make a later number falsifiable. `documents.file_hash
   UNIQUE` is one line and it is the difference between a corpus and a pile of files.
2. **Real sources are messier than any tutorial.** RBI's `robots.txt` returns 418 from a WAF.
   Its index page has 386 document references and zero parseable anchors. Its PDFs need a
   `Referer`. None of that is in the PRD, and all of it changes the design.
3. **Reserved log keys bite.** Passing `name=` to `logging` raises `KeyError: Attempt to
   overwrite 'name' in LogRecord`. It surfaced as a real test failure and led to `safe_extra()`,
   which renames reserved keys rather than dropping the field.
4. **Pydantic settings decode before validators run.** A comma-separated `REGLENS_CORS_ORIGINS`
   string is `json.loads`-ed first and raises. The fix is `Annotated[list[str], NoDecode]` — an
   easy hour to lose, now covered by a test.
5. **A model default is not a default if you pass `""`.** `records_from_plan` filled absent plan
   keys with `""`, which wiped `status="planned"` and produced 239 bogus validation errors.
   Found by running the validator on the real plan, which is the argument for having a validator
   at all.
6. **Cost tables go stale silently.** The official pricing page today lists a `gpt-6-*` family;
   the familiar `gpt-4o`/`gpt-4o-mini` names are legacy but still served. The table therefore
   records `PRICES_VERIFIED_ON` and its source URL, so a reported cost is only as good as a
   dated rate — and re-verifying it is an explicit, visible act.

---

## 6. What to look at first

1. `make status` — one command showing config, manifest and eval-set state.
2. [`docs/corpus-plan.md`](../corpus-plan.md) — the source-verification findings and the honest gaps.
3. [`docs/architecture.md`](../architecture.md) §2 — decisions and rejected alternatives.
4. The schema: [`0001_core.sql`](../../src/reglens/db/migrations/0001_core.sql).

## 7. What Phase 1 will do

Parse the fetched PDFs (two parsers compared, OCR fallback), chunk with fixed windows,
embed with one hosted model, retrieve densely, generate grounded answers with citations, and
run the first 40-question eval — recording recall@k, MRR, nDCG, judge-based correctness and
faithfulness, plus 20 hand-graded answers to validate the judge.
