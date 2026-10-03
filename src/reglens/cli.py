"""RegLens command line.

Every ``make`` target maps to one subcommand here, so the same code runs on a host without
GNU make (Windows) and inside a container. Subcommands that describe a capability the
project does not have yet exit with code 2 and say which phase implements them, instead
of pretending to succeed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reglens import __version__
from reglens.config import get_settings, load_experiment
from reglens.ingestion.fetch import FetchPolicy, download_manifest, resolve_listing
from reglens.ingestion.manifest import (
    DocumentRecord,
    load_corpus_plan,
    load_manifest,
    payload_matches_url,
    plan_summary,
    raw_path_for,
    records_from_plan,
    validate_records,
    write_manifest,
    write_manifest_schema,
)
from reglens.observability.cost import describe_pricing
from reglens.observability.logging import setup_logging
from reglens.observability.repro import git_commit

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_NOT_IMPLEMENTED = 2
EXIT_ERROR = 3

NOT_IMPLEMENTED = {
    "refresh": (4, "poll for new circulars, ingest idempotently, bump corpus_version"),
}


# ------------------------------------------------------------------ helpers
def _emit(payload: dict[str, Any], *, as_json: bool, text: str = "") -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    elif text:
        print(text)


def _question_counts() -> dict[str, Any]:
    """Golden-set counts, or an honest 'not drafted yet' marker."""
    try:
        from eval.runners.golden import load_golden, validate_golden
    except ImportError:  # pragma: no cover - eval/ not shipped in a runtime image
        return {"available": False}
    try:
        questions = load_golden()
    except FileNotFoundError:
        return {"available": False, "reason": "eval/golden/questions.jsonl missing"}
    return {"available": True, **validate_golden(questions).as_dict()}


# ------------------------------------------------------------------ commands
def cmd_status(args: argparse.Namespace) -> int:
    settings = get_settings()
    experiment = load_experiment(settings.experiment_config)
    payload: dict[str, Any] = {
        "version": __version__,
        "env": settings.env,
        "corpus_version": settings.corpus_version,
        "experiment": {"name": experiment.name, "config_hash": experiment.fingerprint()},
        "paths": {
            "manifest": str(settings.manifest_path),
            "raw_dir": str(settings.raw_dir),
            "golden": str(settings.golden_path),
            "results": str(settings.eval_results_dir),
        },
        "pricing": describe_pricing(),
    }

    try:
        records = load_manifest()
    except (FileNotFoundError, ValueError) as exc:
        payload["manifest"] = {"error": str(exc)}
    else:
        report = validate_records(records)
        payload["manifest"] = {"summary": plan_summary(records), **report.as_dict()}

    payload["golden"] = _question_counts()

    if args.with_db:
        payload["database"] = asyncio.run(_db_status())

    lines = [
        f"RegLens {payload['version']} ({payload['env']})",
        f"experiment: {experiment.name}  config_hash={experiment.fingerprint()[:12]}",
        *[f"  {line}" for line in experiment.summary_lines()],
    ]
    manifest = payload.get("manifest", {})
    if "summary" in manifest:
        summary = manifest["summary"]
        lines.append(
            f"manifest: {summary['total']} documents {summary['by_status']} "
            f"errors={manifest['error_count']} warnings={manifest['warning_count']}"
        )
    golden = payload["golden"]
    lines.append(
        f"golden: {golden.get('total', 0)} questions"
        if golden.get("available")
        else f"golden: not drafted yet ({golden.get('reason', 'unavailable')})"
    )
    if "database" in payload:
        lines.append(f"database: {payload['database']}")
    _emit(payload, as_json=args.json, text="\n".join(lines))
    return EXIT_OK


async def _db_status() -> dict[str, Any]:
    from reglens.db.pool import connect
    from reglens.db.runner import migration_status

    try:
        connection = await connect()
    except Exception as exc:
        return {"status": "unavailable", "detail": str(exc)}
    try:
        version = await connection.fetchval("SHOW server_version")
        return {
            "status": "ok",
            "server_version": version,
            "migrations": await migration_status(connection),
        }
    finally:
        await connection.close()


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        records = load_manifest()
    except (FileNotFoundError, ValueError) as exc:
        print(f"manifest invalid: {exc}", file=sys.stderr)
        return EXIT_INVALID
    report = validate_records(records)
    schema_path = (
        write_manifest_schema(settings.manifest_schema_path) if args.write_schema else None
    )
    payload = {
        "manifest": str(settings.manifest_path),
        "schema": str(schema_path) if schema_path else None,
    }
    payload.update(report.as_dict())

    lines = [f"{len(records)} rows in {settings.manifest_path}"]
    for issue in report.errors:
        lines.append(f"  ERROR   [{issue.doc_id}/{issue.column}] {issue.message}")
    for issue in report.warnings:
        lines.append(f"  warning [{issue.doc_id}/{issue.column}] {issue.message}")
    lines.append(f"=> {len(report.errors)} errors, {len(report.warnings)} warnings")
    if args.write_schema and schema_path:
        lines.append(f"schema written to {schema_path}")
    _emit(payload, as_json=args.json, text="\n".join(lines))
    return EXIT_OK if report.ok else EXIT_INVALID


def cmd_golden_validate(args: argparse.Namespace) -> int:
    from eval.runners.golden import load_golden, validate_golden

    try:
        questions = load_golden()
    except (FileNotFoundError, ValueError) as exc:
        print(f"golden set invalid: {exc}", file=sys.stderr)
        return EXIT_INVALID
    manifest_ids: set[str] = set()
    try:
        manifest_ids = {record.doc_id for record in load_manifest()}
    except (FileNotFoundError, ValueError):
        pass
    report = validate_golden(
        questions, manifest_ids=manifest_ids or None, require_full_minimum=args.require_full_minimum
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"{report.total} questions {report.counts_by_type}")
        print(f"held-out={report.held_out_count} reviewed={report.reviewed_count}")
        for issue in report.errors:
            print(f"  ERROR   [{issue.question_id}] {issue.message}")
        for issue in report.warnings[:20]:
            print(f"  warning [{issue.question_id}] {issue.message}")
        if len(report.warnings) > 20:
            print(f"  … {len(report.warnings) - 20} more warnings")
    return EXIT_OK if report.ok else EXIT_INVALID


def cmd_corpus_plan(args: argparse.Namespace) -> int:
    settings = get_settings()
    plan = load_corpus_plan()
    try:
        existing = load_manifest()
    except (FileNotFoundError, ValueError):
        existing = []
    records = records_from_plan(plan, existing)
    report = validate_records(records) if existing else None
    summary = plan_summary(records)

    if args.write:
        write_manifest(records, settings.manifest_path)
        write_manifest_schema(settings.manifest_schema_path)
        payload = {"written": str(settings.manifest_path), "summary": summary}
        payload["validation"] = report.as_dict() if report else None
        _emit(
            payload,
            as_json=True,
            text=f"wrote {settings.manifest_path}: {summary}",
        )
        return EXIT_OK

    print(f"plan would produce {summary['total']} rows: {summary['by_source']}")
    print("dry run (pass --write to apply)")
    return EXIT_OK


def cmd_download(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        records: list[DocumentRecord] = load_manifest()
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot download: {exc}", file=sys.stderr)
        return EXIT_ERROR

    policy = FetchPolicy.from_settings(settings)

    if args.resolve_listings:
        listings = [record for record in records if record.discovery == "listing"]
        if args.only:
            listings = [record for record in listings if record.doc_id in args.only]
        print(
            f"{len(listings)} listing rows to inspect ({settings.fetch_delay_seconds}s between requests)"
        )
        for record in listings:
            candidates, error = resolve_listing(record.url, policy=policy, patterns=(".pdf",))
            if error:
                print(f"  {record.doc_id}: ERROR {error}")
                continue
            print(f"  {record.doc_id}: {len(candidates)} candidate PDFs from {record.url}")
            for candidate in candidates[:20]:
                print(f"      {candidate}")
        return EXIT_OK

    report = download_manifest(
        records,
        policy=policy,
        settings=settings,
        dry_run=not args.yes,
        accept_terms=args.accept_terms,
        only=set(args.only) if args.only else None,
        limit=args.limit,
    )
    counts = report.counts()
    header = (
        "DRY RUN (pass --yes --accept-terms to fetch):" if report.dry_run else "download complete:"
    )
    rendered = " ".join(f"{key}={value}" for key, value in counts.items() if value)
    print(f"{header} {rendered}")
    if report.dry_run:
        for action in report.actions[:25]:
            print(f"  {action.kind:14s} {action.doc_id:42s} {action.reason}")
        if len(report.actions) > 25:
            print(f"  … {len(report.actions) - 25} more")
    for doc_id, error in report.failures:
        print(f"  FAILED {doc_id}: {error}", file=sys.stderr)
    if report.downloaded:
        print(f"  downloaded: {', '.join(report.downloaded[:20])}")
    return EXIT_OK if not report.failures else EXIT_ERROR


def cmd_ingest(args: argparse.Namespace) -> int:
    """Parse, chunk, embed and store the fetched corpus (dry run unless --yes)."""
    from reglens.ingestion.ingest import EmbeddingDimensionError, ingest_corpus

    settings = get_settings()
    try:
        records = load_manifest()
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot ingest: {exc}", file=sys.stderr)
        return EXIT_ERROR
    selected = [
        record
        for record in records
        if record.status == "fetched"
        and record.discovery == "direct"
        and record.file_hash
        and (not args.only or record.doc_id in args.only)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    if not args.yes:
        print(f"DRY RUN (pass --yes to parse and embed): {len(selected)} document(s)")
        for record in selected:
            print(f"  ingest {record.doc_id:44s} {record.title[:60]}")
        return EXIT_OK

    try:
        report = ingest_corpus(
            settings=settings,
            only=set(args.only) if args.only else None,
            limit=args.limit,
        )
    except EmbeddingDimensionError as exc:
        print(f"cannot ingest: {exc}", file=sys.stderr)
        return EXIT_ERROR
    payload = report.as_dict()
    lines = [
        f"ingested {payload['documents']} document(s): {payload['chunks']} chunks, "
        f"{payload['embed_input_tokens']} embed tokens, "
        f"${payload['embed_cost_usd']:.6f}, {payload['seconds']}s "
        f"(skipped {payload['skipped_count']} already indexed, {payload['failed_count']} failed)"
    ]
    for doc_id, error in payload["failed"]:
        lines.append(f"  FAILED {doc_id}: {error}")
    _emit(payload, as_json=args.json, text="\n".join(lines))
    return EXIT_ERROR if payload["failed"] else EXIT_OK


def cmd_reindex(args: argparse.Namespace) -> int:
    """Rebuild the vector index for the active corpus version (Phase 1).

    Purges the version's chunks (documents and their file hashes stay, so nothing is
    re-downloaded) and re-runs the ingest pipeline. Needed after a chunking or
    embedding config change, when ``ingest`` alone would skip everything as already
    indexed. Dry run unless ``--yes``: rebuilding costs a full re-embed.
    """
    from reglens.config.settings import MissingApiKeyError
    from reglens.indexing.vector_store import PgVectorStore
    from reglens.ingestion.ingest import EmbeddingDimensionError, ensure_store_dimension

    settings = get_settings()
    store = PgVectorStore(settings=settings)
    try:
        existing = store.count_chunks()
    except Exception as exc:  # database unreachable is a preflight failure, not a crash
        print(f"cannot reindex: index unreachable: {exc}", file=sys.stderr)
        return EXIT_ERROR

    # Preflight before the purge: a mismatched column would fail the re-ingest anyway,
    # but only after the existing index had already been destroyed.
    try:
        ensure_store_dimension(store, settings)
    except EmbeddingDimensionError as exc:
        print(f"cannot reindex: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if not args.yes:
        print(
            f"DRY RUN (pass --yes to purge and re-embed): corpus_version="
            f"{settings.corpus_version!r} has {existing} chunk(s) that would be "
            "deleted and rebuilt (documents kept; no downloads)"
        )
        return EXIT_OK

    try:
        settings.require_gemini_key()
    except MissingApiKeyError as exc:
        print(f"cannot reindex: {exc}", file=sys.stderr)
        return EXIT_ERROR

    purged = store.purge_chunks()
    from reglens.ingestion.ingest import ingest_corpus

    report = ingest_corpus(settings=settings, only=set(args.only) if args.only else None)
    payload = report.as_dict()
    lines = [
        f"reindexed: purged {purged} chunk(s), rebuilt {payload['chunks']} chunk(s) "
        f"across {payload['documents']} document(s)",
        f"  embed tokens: {payload['embed_input_tokens']} "
        f"(${payload['embed_cost_usd']:.6f}), {payload['seconds']}s "
        f"(failed {payload['failed_count']})",
    ]
    for doc_id, error in payload["failed"]:
        lines.append(f"  FAILED {doc_id}: {error}")
    _emit(payload, as_json=args.json, text="\n".join(lines))
    return EXIT_ERROR if payload["failed"] else EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the golden set through the pipeline and write a versioned report (Phase 1).

    Dry run unless ``--yes``: a live eval spends money (embedding + generation + judge),
    so the default only says what *would* run. The preflight refuses to start when a
    prerequisite is missing — no questions, an unreviewed draft set without
    ``--allow-drafts``, an empty index — because a run that fails halfway still writes a
    report, and a report with 6 failed questions is worse than no report.
    """
    from eval.runners.golden import load_golden, validate_golden
    from eval.runners.judge import GeminiJudge, load_human_grades
    from eval.runners.run_eval import make_ask_fn, run_eval

    from reglens.config.settings import MissingApiKeyError
    from reglens.generation.gemini import build_llm
    from reglens.indexing.vector_store import PgVectorStore
    from reglens.routing.pipeline import build_pipeline

    settings = get_settings()
    try:
        questions = load_golden()
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot eval: {exc}", file=sys.stderr)
        return EXIT_INVALID

    selected = [question for question in questions if question.review != "rejected"]
    if args.only:
        only = set(args.only)
        selected = [question for question in selected if question.id in only]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        print("no questions to run (check eval/golden/questions.jsonl)", file=sys.stderr)
        return EXIT_INVALID

    manifest_ids: set[str] = set()
    try:
        manifest_ids = {record.doc_id for record in load_manifest()}
    except (FileNotFoundError, ValueError):
        pass
    golden_report = validate_golden(selected, manifest_ids=manifest_ids or None)
    if not golden_report.ok:
        for issue in golden_report.errors:
            print(f"  ERROR [{issue.question_id}] {issue.message}", file=sys.stderr)
        return EXIT_INVALID

    drafts = sum(1 for question in selected if question.review == "draft")
    if drafts and not args.allow_drafts:
        print(
            f"{drafts} of {len(selected)} questions are drafts (owner review pending).\n"
            "Metrics from unreviewed questions are not quotable (eval/golden/schema.md): "
            "review them first, or re-run with --allow-drafts for a non-quotable smoke run.",
            file=sys.stderr,
        )
        return EXIT_INVALID

    experiment = load_experiment(settings.experiment_config)
    human_grades = None
    if args.human_grades:
        try:
            human_grades = load_human_grades(args.human_grades)
        except (FileNotFoundError, ValueError) as exc:
            print(f"cannot load human grades: {exc}", file=sys.stderr)
            return EXIT_INVALID

    if not args.yes:
        print(f"DRY RUN (pass --yes to spend tokens): {len(selected)} question(s)")
        print(f"  experiment: {experiment.name}  config_hash={experiment.fingerprint()[:12]}")
        print(
            f"  embed={experiment.indexing.embedding_model} "
            f"answer={experiment.generation.model} "
            f"judge={'off' if args.no_judge else settings.llm_model_large}"
        )
        print(
            f"  reviewed={len(selected) - drafts} draft={drafts} "
            f"held_out={sum(1 for q in selected if q.held_out)}"
        )
        if human_grades:
            print(f"  human grades: {len(human_grades)} (agreement will be reported)")
        return EXIT_OK

    try:
        settings.require_gemini_key()
        store = PgVectorStore(settings=settings)
        chunk_count = store.count_chunks()
    except MissingApiKeyError as exc:
        print(f"cannot eval: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # database unreachable is a preflight failure, not a crash
        print(f"cannot eval: index unreachable: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if chunk_count == 0:
        print(
            f"cannot eval: index is empty for corpus_version={settings.corpus_version!r}; "
            "run 'make migrate' and 'make ingest -- --yes' first",
            file=sys.stderr,
        )
        return EXIT_ERROR

    judge = None
    if not args.no_judge:
        judge = GeminiJudge(build_llm(settings), model=settings.llm_model_large)

    try:
        pipeline = build_pipeline(settings, experiment=experiment)
    except MissingApiKeyError as exc:
        print(f"cannot eval: {exc}", file=sys.stderr)
        return EXIT_ERROR

    report, json_path, md_path = run_eval(
        questions=selected,
        ask=make_ask_fn(pipeline, experiment),
        experiment=experiment,
        settings=settings,
        judge=judge,
        human_grades=human_grades,
    )
    agg = report["aggregates"]
    retrieval = agg["retrieval"]
    lines = [
        f"eval complete: {len(report['questions'])} scored, {len(report['failures'])} failed",
        f"  recall@{retrieval['k']}={retrieval['recall_at_k']} "
        f"MRR={retrieval['mrr']} nDCG={retrieval['ndcg_at_k']}",
        f"  report: {json_path}",
        f"          {md_path}",
    ]
    for caveat in report["caveats"]:
        lines.append(f"  CAVEAT: {caveat}")
    _emit(
        report if args.json else {"report": str(json_path)},
        as_json=args.json,
        text="\n".join(lines),
    )
    return EXIT_OK if report["questions"] else EXIT_ERROR


def cmd_repair_raw(args: argparse.Namespace) -> int:
    """Quarantine stored payloads that are not the document they claim to be.

    The first real fetch recorded 18 HTML interstitials as PDFs because nothing checked
    the bytes. This command is the standing fix: it is idempotent, moves rejected payloads
    to ``data/derived/rejected/`` (out of the immutable raw store), demotes the rows, and
    leaves a note saying what was found. A dry run unless ``--apply`` is passed.
    """
    settings = get_settings()
    records = load_manifest()
    demoted: list[tuple[str, str]] = []

    for record in records:
        if not record.local_path:
            continue
        path = settings.raw_dir / record.local_path
        if not path.is_file():
            continue
        ok, detail = payload_matches_url(record.url, path.read_bytes())
        if ok:
            continue
        if args.apply:
            rejected_dir = settings.derived_dir / "rejected"
            rejected_dir.mkdir(parents=True, exist_ok=True)
            rejected = rejected_dir / path.name
            suffix_index = 2
            while rejected.exists():
                rejected = rejected_dir / f"{path.stem}__{suffix_index}{path.suffix}"
                suffix_index += 1
            path.replace(rejected)
            note = f"payload rejected on repair: {detail}"
            record.notes = f"{record.notes} | {note}" if record.notes else note
            record.status = "failed"
            record.file_hash = ""
            record.local_path = ""
        demoted.append((record.doc_id, detail))

    if args.apply:
        write_manifest(records, settings.manifest_path)

    payload = {
        "applied": args.apply,
        "demoted": len(demoted),
        "details": dict(demoted),
    }
    verb = "quarantined" if args.apply else "would quarantine"
    lines = [f"{verb} {len(demoted)} row(s) whose stored payload is not the document"]
    for doc_id, detail in demoted:
        lines.append(f"  {doc_id}: {detail}")
    if not demoted:
        lines.append("  nothing to do - raw store matches the manifest")
    elif not args.apply:
        lines.append("  dry run: pass --apply to write the demoted rows")
    _emit(payload, as_json=args.json, text="\n".join(lines))
    return EXIT_OK


def _render_parser_report(report: dict[str, Any]) -> str:
    """Markdown twin of report.json so the comparison is readable in a review."""
    aggregate = report["aggregate"]
    lines = [
        "# Parser comparison report",
        "",
        f"- generated: {report['generated_at']}",
        f"- git commit: {report['git_commit']}",
        f"- experiment config_hash: {report['config_hash']}",
        f"- corpus_version: {report['corpus_version']}",
        f"- documents: {aggregate['documents']}",
        f"- mean agreement (token Jaccard): {aggregate['mean_agreement']}",
        f"- faster count: {aggregate['faster_count']}",
        "",
        "| doc_id | pages | pymupdf s | pdfplumber s | agreement | quality pymupdf/plumber |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["documents"]:
        lines.append(
            f"| {row['doc_id']} | {row['pages']} | {row['pymupdf']['seconds']} "
            f"| {row['pdfplumber']['seconds']} | {row['agreement_token_jaccard']} "
            f"| {row['pymupdf']['quality']}/{row['pdfplumber']['quality']} |"
        )
    lines += ["", "## Totals", ""]
    for parser, stats in aggregate["per_parser"].items():
        lines.append(f"- **{parser}**: {stats}")
    lines += [
        "",
        "Timings are single wall-clock runs: indicative, not microbenchmarks. Pages listed "
        "in `scan_pages` yielded almost no text and are the OCR candidates.",
        "",
    ]
    return "\n".join(lines)


def cmd_parse_compare(args: argparse.Namespace) -> int:
    """Parse every fetched PDF with both parsers; write a versioned report."""
    from reglens.ingestion.parse import compare_parsers

    settings = get_settings()
    try:
        records = load_manifest()
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot compare parsers: {exc}", file=sys.stderr)
        return EXIT_ERROR

    selected = [
        record
        for record in records
        if record.status == "fetched" and (not args.only or record.doc_id in args.only)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    documents: list[tuple[str, Path]] = []
    for record in selected:
        path = raw_path_for(record)
        if path is not None and path.is_file():
            documents.append((record.doc_id, path))
    if not documents:
        print("no fetched documents with local files to compare", file=sys.stderr)
        return EXIT_ERROR

    experiment = load_experiment(settings.experiment_config)
    report = compare_parsers(documents, min_chars_per_page=experiment.parsing.min_chars_per_page)
    generated_at = datetime.now(UTC)
    envelope: dict[str, Any] = {
        "kind": "parser-comparison",
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config_hash": experiment.fingerprint(),
        "corpus_version": settings.corpus_version,
        "seed": settings.seed,
        **report,
    }
    stamp = generated_at.strftime("%Y-%m-%dT%H%M%SZ")
    out_dir = settings.eval_results_dir / f"{stamp}-parser-comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(envelope, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_render_parser_report(envelope), encoding="utf-8")

    if args.json:
        print(json.dumps(envelope, indent=2, default=str))
    else:
        aggregate = report["aggregate"]
        print(f"compared {aggregate['documents']} document(s) with {report['parsers']}")
        print(f"  mean agreement (token Jaccard): {aggregate['mean_agreement']}")
        print(f"  faster: {aggregate['faster_count']}")
        for parser, stats in aggregate["per_parser"].items():
            print(
                f"  {parser}: {stats['total_seconds']}s total, {stats['total_chars']} chars, "
                f"mean quality {stats['mean_quality']}"
            )
        print(f"  report: {out_dir}")
    return EXIT_OK


def cmd_migrate(args: argparse.Namespace) -> int:
    import asyncpg

    from reglens.db.pool import connect
    from reglens.db.runner import MigrationError, apply_migrations, migration_status

    async def run() -> dict[str, Any]:
        connection = await connect()
        try:
            if args.status:
                return await migration_status(connection)
            applied = await apply_migrations(connection, verbose=not args.json)
            return {"applied": applied, **await migration_status(connection)}
        finally:
            await connection.close()

    try:
        result = asyncio.run(run())
    except (MigrationError, OSError, asyncpg.PostgresError) as exc:
        print(f"migration failed: {exc}", file=sys.stderr)
        return EXIT_ERROR
    _emit(result, as_json=args.json, text=f"migrations: {result}")
    return EXIT_OK


def _not_implemented(command: str) -> int:
    phase, description = NOT_IMPLEMENTED[command]
    print(
        f"'{command}' ({description}) is not implemented yet: it arrives in Phase {phase}.\n"
        f"See docs/architecture.md for the phase plan and docs/learning/phase-0.md for the "
        f"current state. Nothing was written.",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reglens", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"reglens {__version__}")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="configuration, manifest and eval-set status")
    status.add_argument("--with-db", action="store_true", help="also probe Postgres")
    status.set_defaults(func=cmd_status)

    validate = sub.add_parser("validate-manifest", help="validate data/manifest.csv")
    validate.add_argument(
        "--write-schema", action="store_true", help="write data/manifest.schema.json"
    )
    validate.set_defaults(func=cmd_validate_manifest)

    golden = sub.add_parser("golden-validate", help="validate the golden question set")
    golden.add_argument(
        "--require-full-minimum", action="store_true", help="enforce the 100+ question quotas"
    )
    golden.set_defaults(func=cmd_golden_validate)

    plan = sub.add_parser("corpus-plan", help="regenerate the manifest from data/corpus_plan.yaml")
    plan.add_argument("--write", action="store_true", help="apply the change (default: dry run)")
    plan.set_defaults(func=cmd_corpus_plan)

    download = sub.add_parser("download", help="plan or fetch manifest documents")
    download.add_argument("--yes", action="store_true", help="actually fetch (default: dry run)")
    download.add_argument(
        "--accept-terms",
        action="store_true",
        help="confirm you have checked the source site's terms of use and robots.txt",
    )
    download.add_argument("--only", nargs="*", help="restrict to these doc_ids")
    download.add_argument("--limit", type=int, help="fetch at most N documents this run")
    download.add_argument(
        "--resolve-listings", action="store_true", help="print candidate links for listing pages"
    )
    download.set_defaults(func=cmd_download)

    repair = sub.add_parser(
        "repair-raw", help="quarantine stored payloads that are not the document they claim to be"
    )
    repair.add_argument(
        "--apply", action="store_true", help="write the demoted rows back to the manifest"
    )
    repair.set_defaults(func=cmd_repair_raw)

    ingest = sub.add_parser("ingest", help="parse, chunk and index the fetched corpus")
    ingest.add_argument(
        "--yes", action="store_true", help="actually parse and embed (default: dry run)"
    )
    ingest.add_argument("--only", nargs="*", help="restrict to these doc_ids")
    ingest.add_argument("--limit", type=int, help="ingest at most N documents")
    ingest.set_defaults(func=cmd_ingest)

    eval_parser = sub.add_parser(
        "eval", help="run the golden eval set and write a versioned report"
    )
    eval_parser.add_argument("--yes", action="store_true", help="actually run (default: dry run)")
    eval_parser.add_argument("--only", nargs="*", help="restrict to these question ids")
    eval_parser.add_argument("--limit", type=int, help="run at most N questions")
    eval_parser.add_argument(
        "--allow-drafts", action="store_true", help="run unreviewed questions (non-quotable)"
    )
    eval_parser.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge scoring")
    eval_parser.add_argument(
        "--human-grades", type=Path, help="JSONL of hand grades for judge agreement"
    )
    eval_parser.set_defaults(func=cmd_eval)

    parse_compare = sub.add_parser(
        "parse-compare",
        help="parse every fetched PDF with both parsers and write a versioned report",
    )
    parse_compare.add_argument("--only", nargs="*", help="restrict to these doc_ids")
    parse_compare.add_argument("--limit", type=int, help="compare at most N documents")
    parse_compare.set_defaults(func=cmd_parse_compare)

    reindex = sub.add_parser(
        "reindex", help="rebuild the vector and keyword indexes for the active corpus version"
    )
    reindex.add_argument(
        "--yes", action="store_true", help="actually purge and re-embed (default: dry run)"
    )
    reindex.add_argument("--only", nargs="*", help="restrict to these doc_ids")
    reindex.set_defaults(func=cmd_reindex)

    migrate = sub.add_parser("migrate", help="apply SQL migrations")
    migrate.add_argument(
        "--status", action="store_true", help="report pending/applied instead of applying"
    )
    migrate.set_defaults(func=cmd_migrate)

    for name in NOT_IMPLEMENTED:
        stub = sub.add_parser(
            name, help=f"(Phase {NOT_IMPLEMENTED[name][0]}) {NOT_IMPLEMENTED[name][1]}"
        )
        stub.set_defaults(func=lambda _args, _name=name: _not_implemented(_name))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(get_settings().log_level)
    try:
        return int(args.func(args))
    except BrokenPipeError:  # pragma: no cover - piping into head
        return EXIT_OK
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
