# RegLens task runner.
#
# `make` is the documented interface. Each target delegates to scripts/tasks.py, which is
# the single source of truth for what a target does — so `make test` and
# `uv run python scripts/tasks.py test` are literally the same commands, and a host
# without GNU make (Windows) is not a second-class citizen.
#
# Extra arguments are forwarded:  make download -- --yes --accept-terms
.DEFAULT_GOAL := help
.PHONY: help setup lock up down logs ps build migrate status validate-manifest corpus-plan \
        download ingest reindex eval refresh api web clean reset-db \
        test test-unit test-integration lint fmt check

TASKS := uv run python scripts/tasks.py

help:  ## Show available targets
	@$(TASKS) help

setup:  ## Create the venv and install pinned dependencies
	uv sync --all-groups

lock:  ## Resolve and pin dependency versions into uv.lock
	uv lock

up:  ## Start the whole stack (postgres + api + web)
	@$(TASKS) up
	@echo "api http://localhost:8000/health | web http://localhost:3000"

down:  ## Stop the stack
	@$(TASKS) down

logs:  ## Tail stack logs
	@$(TASKS) logs

ps:  ## Show stack status
	@$(TASKS) ps

build:  ## Build container images
	@$(TASKS) build

migrate:  ## Apply pending SQL migrations
	@$(TASKS) migrate

status:  ## Configuration, manifest and eval-set status
	@$(TASKS) status

test:  ## Run unit tests (no Docker required)
	@$(TASKS) test

test-unit: test

test-integration:  ## Run integration tests against a live Postgres
	@$(TASKS) test-integration

lint:  ## Lint without modifying files
	@$(TASKS) lint

fmt:  ## Auto-format and auto-fix
	@$(TASKS) fmt

check:  ## What CI runs: lint, unit tests, manifest validation
	@$(TASKS) check

validate-manifest:  ## Validate data/manifest.csv and refresh its schema
	@$(TASKS) validate-manifest

corpus-plan:  ## Regenerate data/manifest.csv from data/corpus_plan.yaml
	@$(TASKS) corpus-plan

download:  ## Dry-run the download plan; forward flags after "--"
	@$(TASKS) download $(ARGS)

ingest:  ## Parse + chunk + index the corpus (Phase 1)
	@$(TASKS) ingest $(ARGS)

reindex:  ## Rebuild indexes for the active corpus version (Phase 1)
	@$(TASKS) reindex

eval:  ## Run the golden eval set and write a versioned report (Phase 1)
	@$(TASKS) eval $(ARGS)

refresh:  ## Check for new circulars and ingest idempotently (Phase 4)
	@$(TASKS) refresh

api:  ## Run the API locally with autoreload
	@$(TASKS) api

web:  ## Run the Next.js dev server locally
	@$(TASKS) web

clean:  ## Remove caches and build artifacts
	@$(TASKS) clean

reset-db:  ## Drop and recreate the local database volume (destructive, local only)
	@$(TASKS) reset-db
