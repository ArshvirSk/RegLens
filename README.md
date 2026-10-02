# RegLens

Cited, temporally-aware question answering over Indian banking regulation (RBI, SEBI) and
bank filings (annual reports, earnings-call transcripts). Built as a learning-first
project: every technique is a toggle, and every toggle has to beat the baseline on a
golden eval set before it stays.

> **Informational only. Not legal or investment advice.** RegLens cites public documents;
> it does not give advice, recommendations, or trading signals.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Foundations: repo, Compose stack, typed config, experiment configs, corpus manifest + fetcher, tracing skeleton, CI | **done** |
| 1 | Naive baseline (fixed chunks, dense retrieval, grounded answers) + 40-question eval | not started |
| 2 | Retrieval quality ablations (clause/speaker chunking, BM25, RRF, rerank, rewriting, second embedding model) | not started |
| 3 | Metric extraction, as-of temporal filtering, LangGraph routing, safe text-to-SQL | not started |
| 4 | Citation verifier, calibrated refusals, Next.js UI with PDF source view, refresh job, dashboard | not started |
| 5 | Held-out eval, failure analysis, regression gate, write-up | not started |

`POST /ask` currently returns **HTTP 501** with an explicit "arrives in Phase 1" body. That
is deliberate: an endpoint with no pipeline behind it must fail loudly rather than return
an unsourced answer.

## Quickstart

```bash
cp .env.example .env          # fill in OPENAI_API_KEY (or set REGLENS_LLM_PROVIDER)
make setup                    # uv sync --all-groups
make up                       # docker compose up -d --build
make migrate                  # apply the SQL migrations
make status                   # config, manifest, eval-set state (add --with-db to probe Postgres)
```

* API: <http://localhost:8000/health>, docs at `/docs`
* Web: <http://localhost:3000>
* Postgres on host port **5433** (inside the compose network it is `postgres:5432`)

Then review the corpus and fetch it:

```bash
make validate-manifest        # schema + hash checks, writes data/manifest.schema.json
make download                 # dry run: shows exactly what would be fetched
make download -- --yes --accept-terms --limit 5   # only after you have checked terms of use
```

### Windows / no GNU make

`make` is the documented interface, but every target delegates to
`scripts/tasks.py`, so the same commands work without make:

```bash
uv run python scripts/tasks.py            # list targets
uv run python scripts/tasks.py test
```

## Repository layout

```
docs/        PRD, architecture, experiments log, learning notes per phase, corpus plan
data/        manifest.csv (committed) + corpus_plan.yaml; raw/ and parsed/ are gitignored
src/reglens/ config, ingestion, chunking, indexing, retrieval, routing, generation, api,
             observability, db (SQL migrations + asyncpg helpers), cli
eval/        golden/ (questions + schema), runners/, results/ (versioned reports)
web/         Next.js App Router UI
scripts/     tasks.py (make targets) and download/refresh helpers
tests/       unit tests (fast, no Docker) and integration tests (need Postgres)
docker/      postgres init (read-only role for Phase 3 text-to-SQL)
```

## How the pieces fit

```
manifest.csv ──► fetcher (robots-aware, rate-limited, SHA-256 content-addressed)
             └─► raw store (data/raw, immutable) ──► parser/OCR ──► chunker ──► indexes
                                                                   (pgvector + tsvector)
question ──► (rewrite) ──► retrieve (dense / keyword / hybrid, date-filtered)
         ──► rerank ──► generate with citations ──► verify citations ──► answer or refuse
everything ──► traces (per-stage latency, tokens, cost) ──► Postgres + eval reports
```

Full detail, including the reasoning behind each choice and the alternatives rejected, is
in [docs/architecture.md](docs/architecture.md).

## Commands

| Command | What it does |
|---|---|
| `make setup` / `make lock` | Install from `uv.lock` / re-resolve pins |
| `make up` / `make down` / `make logs` | Manage the Compose stack |
| `make migrate` | Apply SQL migrations (checksum-verified, immutable once applied) |
| `make test` / `make test-integration` | Fast unit tests / tests needing Postgres |
| `make lint` / `make fmt` / `make check` | Ruff lint / autofix / the full CI gate |
| `make validate-manifest` | Validate `data/manifest.csv` against its schema |
| `make corpus-plan` | Regenerate the manifest from `data/corpus_plan.yaml` |
| `make download` | Plan downloads (dry run by default) |
| `make status` | Config + manifest + golden-set status |
| `make eval` | Phase 1: run the golden eval set and write a versioned report |

## Reproducibility rules

* `uv.lock` pins every dependency; `pyproject.toml` records the policy of adding a
  dependency in the phase that first uses it.
* Each eval report records the experiment config, its hash, the corpus version, the git
  commit, model names, and the price table used for cost.
* Random seeds come from `REGLENS_SEED`.
* Every number in `docs/` traces back to a report under `eval/results/`. Unknown numbers
  are reported as unknown.

## Data ethics

* Public documents only. `robots.txt` is fetched and obeyed per host; requests are
  throttled (default 2 s per host); nothing is downloaded unless the operator passes
  `--accept-terms` after checking the site's terms of use.
* Raw files are stored immutably and content-addressed by SHA-256, and never committed to
  git. `data/manifest.csv` records the hash of every file.
* RBI does not publish usable crawl rules or machine-readable listings
  (`robots.txt` answers HTTP 418 from its WAF and document pages render from ASP.NET
  viewstate), so RBI documents are collected from direct PDF links with a small number of
  manual checks. See [docs/corpus-plan.md](docs/corpus-plan.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `make: command not found` (Windows) | Use `uv run python scripts/tasks.py <target>`, or install GNU make via `winget install ezwinports.make`. |
| API `/health` says `database: unavailable` | `docker compose ps` — the `postgres` container must be healthy; the API degrades instead of crashing. |
| Port already in use | Override in `.env`: `POSTGRES_PORT`, `REGLENS_API_PORT` (web is fixed at 3000 in `docker-compose.yml`). |
| `manifest invalid: ... hash` errors | A raw file changed after fetching. Restore the file or re-fetch it; raw files are immutable by design. |
