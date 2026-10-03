"""One-command golden eval: predict → score → write a versioned report.

This is the only place numbers in ``docs/`` may come from (eval/README.md). The contract:

* Load the golden set, run each question through the **same** ``Pipeline`` the API uses
  (``reglens.routing.pipeline``), score retrieval against ``gold_passages`` with
  ``metrics.py``, optionally judge answers with the versioned judge prompt, and write an
  immutable ``<stamp>_<experiment>_<config_hash12>.json`` + ``.md`` into
  ``eval/results/``.
* Failures are per-question: one dead question records an error and the run continues.
  A failed question contributes *nothing* to aggregates — a missing number, never a 0.
* ``rejected`` questions are skipped; ``draft`` questions require ``--allow-drafts``
  because the schema says metrics are only reported for reviewed questions — and when
  drafts do run, the report carries a caveat saying so.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.runners.golden import GoldenQuestion
from eval.runners.judge import (
    JUDGE_PROMPT_VERSION,
    AgreementReport,
    GeminiJudge,
    HumanGrade,
    JudgeScore,
    agreement,
)
from eval.runners.metrics import ndcg_at_k, recall_at_k, reciprocal_rank, score_citations, summarize
from reglens.config import Settings, get_settings
from reglens.config.experiment import ExperimentConfig
from reglens.observability.cost import try_estimate_cost
from reglens.observability.repro import git_commit
from reglens.retrieval.base import SearchFilters

#: ``ask`` receives a question and returns an object shaped like ``routing.Pipeline``'s
#: AskResult (``.answer``, ``.chunks``, ``.embed_input_tokens``, ``.embed_model``).
AskFn = Callable[[GoldenQuestion], Any]


def make_ask_fn(pipeline: Any, experiment: ExperimentConfig) -> AskFn:
    """Adapt ``Pipeline.ask`` (text in) to the runner's AskFn (question in).

    Keeping this here means the CLI, the tests, and any future batch runner all apply
    the same per-question filters — an eval that wires filters differently from the API
    is measuring a pipeline nobody can talk to.
    """

    def ask(question: GoldenQuestion) -> Any:
        return pipeline.ask(
            question.question,
            top_k=experiment.retrieval.top_k,
            filters=_filters_for(question, experiment),
        )

    return ask


def _filters_for(question: GoldenQuestion, experiment: ExperimentConfig) -> SearchFilters | None:
    """Mirror the API: as_of_date only binds when the experiment turns temporal on."""
    if question.as_of_date is not None and experiment.retrieval.temporal_filter:
        return SearchFilters(as_of_date=question.as_of_date)
    return None


def evaluate(
    questions: list[GoldenQuestion],
    ask: AskFn,
    *,
    k: int,
    experiment: ExperimentConfig,
    judge: GeminiJudge | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every question; return (successful records, per-question failures)."""
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for question in questions:
        try:
            result = ask(question)
        except Exception as exc:  # one bad question must not sink the run
            failures.append({"id": question.id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        answer = result.answer
        chunks = result.chunks
        golds = [(passage.doc_id, passage.page) for passage in question.gold_passages]
        citations = score_citations(answer.citations, chunks)
        record: dict[str, Any] = {
            "id": question.id,
            "type": question.type,
            "question": question.question,
            "as_of_date": question.as_of_date.isoformat() if question.as_of_date else None,
            "review": question.review,
            "held_out": question.held_out,
            "gold_count": len(golds),
            "recall": recall_at_k(golds, chunks, k),
            "rr": reciprocal_rank(golds, chunks),
            "ndcg": ndcg_at_k(golds, chunks, k),
            "retrieved": [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "score": round(chunk.score, 6),
                }
                for chunk in chunks
            ],
            "answer": {
                "text": answer.text,
                "refused": answer.refused,
                "citations": list(answer.citations),
                "citation_total": citations.total,
                "citation_resolved": citations.resolved,
                "citation_precision": citations.precision,
                "has_citations": citations.total > 0,
                "model": answer.model,
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
                "latency_ms": answer.latency_ms,
            },
            "refused": answer.refused,
            "citation_precision": citations.precision,
            "has_citations": citations.total > 0,
            "embed_input_tokens": result.embed_input_tokens,
            "embed_model": result.embed_model,
            "judge": None,
        }
        if judge is not None:
            score = judge.judge(
                question=question.question,
                reference_answer=question.reference_answer,
                context=chunks,
                candidate=answer.text,
            )
            record["judge"] = {
                "correctness": score.correctness,
                "faithfulness": score.faithfulness,
                "notes": score.notes,
                "error": score.error,
                "prompt_version": score.prompt_version,
                "model": score.model,
                "input_tokens": score.input_tokens,
                "output_tokens": score.output_tokens,
            }
        records.append(record)
    return records, failures


def _totals(
    records: list[dict[str, Any]], *, embed_model: str, judge_enabled: bool
) -> dict[str, Any]:
    """Token totals + estimated cost; cost is None when nothing is priced, never 0."""
    embed_in = sum(record.get("embed_input_tokens", 0) for record in records)
    answer_in = sum(record["answer"]["input_tokens"] for record in records)
    answer_out = sum(record["answer"]["output_tokens"] for record in records)
    judge_in = sum((record.get("judge") or {}).get("input_tokens", 0) for record in records)
    judge_out = sum((record.get("judge") or {}).get("output_tokens", 0) for record in records)

    answer_model = next(
        (record["answer"]["model"] for record in records if record["answer"]["model"]), None
    )
    judge_model = next(
        (
            record["judge"]["model"]
            for record in records
            if record.get("judge") and record["judge"]["model"]
        ),
        None,
    )
    costs = [
        cost
        for cost in (
            try_estimate_cost(embed_model, embed_in, 0),
            try_estimate_cost(answer_model, answer_in, answer_out) if answer_model else None,
            try_estimate_cost(judge_model, judge_in, judge_out)
            if judge_enabled and judge_model
            else None,
        )
        if cost is not None
    ]
    return {
        "embed_input_tokens": embed_in,
        "answer_input_tokens": answer_in,
        "answer_output_tokens": answer_out,
        "judge_input_tokens": judge_in if judge_enabled else None,
        "judge_output_tokens": judge_out if judge_enabled else None,
        "estimated_cost_usd": sum(costs) if costs else None,
    }


def _judge_scores(records: list[dict[str, Any]]) -> dict[str, JudgeScore]:
    """Judge scores by question id, from saved records (errored entries kept —
    ``agreement()`` drops them and counts them out rather than pairing nothing)."""
    scores: dict[str, JudgeScore] = {}
    for record in records:
        judged = record.get("judge")
        if judged is not None:
            scores[record["id"]] = JudgeScore(
                correctness=judged["correctness"],
                faithfulness=judged["faithfulness"],
                notes=judged["notes"],
                error=judged["error"],
            )
    return scores


def build_report(
    *,
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    questions: list[GoldenQuestion],
    experiment: ExperimentConfig,
    settings: Settings,
    judge_enabled: bool,
    generated_at: datetime | None = None,
    human_grades: list[HumanGrade] | None = None,
) -> dict[str, Any]:
    """Assemble the report envelope (every field the PRD requires to reproduce a run)."""
    stamp_dt = generated_at or datetime.now(UTC)
    from eval.runners.golden import validate_golden

    golden_report = validate_golden(questions)

    judge_section: dict[str, Any] = {
        "enabled": judge_enabled,
        "prompt_version": JUDGE_PROMPT_VERSION if judge_enabled else None,
        "model": settings.llm_model_large if judge_enabled else None,
        "agreement": None,
    }
    if judge_enabled and human_grades:
        result: AgreementReport = agreement(_judge_scores(records), human_grades)
        judge_section["agreement"] = result.as_dict()

    counts_by_type: dict[str, int] = {}
    for question in questions:
        counts_by_type[question.type] = counts_by_type.get(question.type, 0) + 1

    caveats: list[str] = []
    draft_count = sum(1 for question in questions if question.review == "draft")
    if draft_count:
        caveats.append(
            f"{draft_count} of {len(questions)} questions are drafts (unreviewed): "
            "these numbers are not quotable until the owner signs off (eval/golden/schema.md)."
        )
    if failures:
        caveats.append(f"{len(failures)} question(s) failed and are excluded from every aggregate.")
    held_out = sum(1 for question in questions if question.held_out)
    if held_out:
        caveats.append(
            f"{held_out} held-out question(s) are included in aggregates: use the "
            "per-question records to recompute dev-only numbers before tuning a config."
        )

    return {
        "kind": "golden-eval",
        "schema_version": 1,
        "generated_at": stamp_dt.isoformat(timespec="seconds"),
        "report_id": f"{stamp_dt.strftime('%Y-%m-%dT%H%M%SZ')}_{experiment.name}_{experiment.fingerprint()[:12]}",
        "git_commit": git_commit(),
        "config_hash": experiment.fingerprint(),
        "experiment": experiment.name,
        "experiment_summary": experiment.summary_lines(),
        "corpus_version": settings.corpus_version,
        "seed": settings.seed,
        "models": {
            "embedding": experiment.indexing.embedding_model,
            "generation": experiment.generation.model,
            "judge": settings.llm_model_large if judge_enabled else None,
        },
        "golden": {
            "total": len(questions),
            "counts_by_type": counts_by_type,
            "reviewed": golden_report.reviewed_count,
            "draft": len(questions) - golden_report.reviewed_count,
            "held_out": golden_report.held_out_count,
            "validation_ok": golden_report.ok,
        },
        "judge": judge_section,
        "aggregates": summarize(records, k=experiment.retrieval.top_k),
        "totals": _totals(
            records,
            embed_model=experiment.indexing.embedding_model,
            judge_enabled=judge_enabled,
        ),
        "failures": failures,
        "caveats": caveats,
        "questions": records,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable twin of report.json — same numbers, no extra ones."""
    agg = report["aggregates"]
    retrieval = agg["retrieval"]
    generation = agg["generation"]

    def fmt(value: float | None) -> str:
        return "missing" if value is None else f"{value:.4f}"

    lines = [
        f"# Golden eval — {report['experiment']} — {report['report_id']}",
        "",
        f"- generated: {report['generated_at']}  git: `{report['git_commit'][:12]}`",
        f"- config_hash: `{report['config_hash'][:12]}`  corpus_version: {report['corpus_version']}",
        f"- models: embed={report['models']['embedding']} "
        f"answer={report['models']['generation']} judge={report['models']['judge']}",
        f"- golden: {report['golden']['total']} questions "
        f"({', '.join(f'{k}={v}' for k, v in report['golden']['counts_by_type'].items())}) — "
        f"reviewed={report['golden']['reviewed']} draft={report['golden']['draft']} "
        f"held_out={report['golden']['held_out']}",
        "",
        "## Retrieval",
        "",
        "| metric | value | defined/answerable |",
        "|---|---|---|",
        f"| recall@{retrieval['k']} | {fmt(retrieval['recall_at_k'])} "
        f"| {retrieval['recall_at_k_defined']}/{agg['answerable']} |",
        f"| MRR | {fmt(retrieval['mrr'])} | {retrieval['mrr_defined']}/{agg['answerable']} |",
        f"| nDCG@{retrieval['k']} | {fmt(retrieval['ndcg_at_k'])} "
        f"| {retrieval['ndcg_at_k_defined']}/{agg['answerable']} |",
        "",
        "## Generation",
        "",
        "| metric | value | defined/answered |",
        "|---|---|---|",
        f"| citation precision | {fmt(generation['citation_precision'])} "
        f"| {generation['citation_precision_defined']} answered with citations graded |",
        f"| refusal rate | {fmt(generation['refusal_rate'])} | {agg['questions']} scored |",
        f"| answers with citations | {generation['answers_with_citations']} "
        f"| {agg['questions'] - agg['unanswerable']} answerable scored |",
        f"| correct refusals (of {agg['unanswerable']} unanswerable) "
        f"| {generation['correct_refusals']} | |",
        "",
    ]

    judge_section = report["judge"]
    lines += ["## Judge", ""]
    if not judge_section["enabled"]:
        lines += ["Judge disabled for this run (`--no-judge`).", ""]
    else:
        judged = [record for record in report["questions"] if record.get("judge")]
        ok = [record for record in judged if record["judge"]["error"] is None]
        correct = [record["judge"]["correctness"] for record in ok]
        faithful = [record["judge"]["faithfulness"] for record in ok]
        lines += [
            f"- prompt: `{judge_section['prompt_version']}`  model: {judge_section['model']}",
            f"- scored: {len(ok)}/{len(report['questions'])} "
            f"(missing {len(report['questions']) - len(ok)})",
            f"- mean correctness: {fmt(sum(correct) / len(correct) if correct else None)} "
            f"/ 2  mean faithfulness: {fmt(sum(faithful) / len(faithful) if faithful else None)} / 2",
        ]
        agreement_section = judge_section.get("agreement")
        if agreement_section:
            exact = agreement_section["exact_agreement"]
            lines += [
                f"- human agreement (paired={agreement_section['paired']}): "
                f"exact correctness={fmt(exact['correctness'])} "
                f"faithfulness={fmt(exact['faithfulness'])}",
            ]
        lines.append("")

    totals = report["totals"]
    lines += [
        "## Cost",
        "",
        f"- embed tokens: {totals['embed_input_tokens']}  "
        f"answer tokens: {totals['answer_input_tokens']}+{totals['answer_output_tokens']}  "
        f"judge tokens: {totals['judge_input_tokens']}+{totals['judge_output_tokens']}",
        "- estimated cost: "
        + (
            f"${totals['estimated_cost_usd']:.6f}"
            if totals["estimated_cost_usd"] is not None
            else "missing (unpriced model)"
        ),
        "",
    ]

    if report["failures"]:
        lines += ["## Failures", ""]
        for failure in report["failures"]:
            lines.append(f"- `{failure['id']}`: {failure['error']}")
        lines.append("")

    if report["caveats"]:
        lines += ["## Caveats", ""]
        lines += [f"- {caveat}" for caveat in report["caveats"]]
        lines.append("")

    lines += [
        "## Per-question",
        "",
        "| id | type | recall@k | MRR | nDCG | refused | cit.prec | judge C/F |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in report["questions"]:
        judge = record.get("judge")
        if judge and judge["error"] is None:
            judge_cell = f"{judge['correctness']}/{judge['faithfulness']}"
        elif judge:
            judge_cell = "error"
        else:
            judge_cell = "not judged"
        lines.append(
            f"| {record['id']} | {record['type']} | {fmt(record['recall'])} "
            f"| {fmt(record['rr'])} | {fmt(record['ndcg'])} "
            f"| {'yes' if record['refused'] else 'no'} "
            f"| {fmt(record['citation_precision'])} | {judge_cell} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], results_dir: Path) -> tuple[Path, Path]:
    """Write the immutable json + md pair; report_id names both files."""
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{report['report_id']}.json"
    md_path = results_dir / f"{report['report_id']}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def attach_agreement(report: dict[str, Any], human_grades: list[HumanGrade]) -> dict[str, Any]:
    """Post-hoc twin of the ``human_grades`` branch in :func:`build_report`.

    Hand grades are written *after* reading a saved run, so they must pair with that
    report's stored judge scores — re-running the eval would pair them with newly
    generated answers. Same ``agreement()`` and same record mapping as run time, so
    the embedded section is identical whichever route produced it.
    """
    if not report.get("judge", {}).get("enabled"):
        raise ValueError("report has no judge scores to agree with (judge disabled)")
    result = agreement(_judge_scores(report.get("questions", [])), human_grades)
    if result.paired == 0:
        raise ValueError(
            f"none of the {len(human_grades)} hand grade(s) match a judged question id"
        )
    report["judge"]["agreement"] = result.as_dict()
    return report


def run_eval(
    *,
    questions: list[GoldenQuestion],
    ask: AskFn,
    experiment: ExperimentConfig,
    settings: Settings | None = None,
    judge: GeminiJudge | None = None,
    human_grades: list[HumanGrade] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Evaluate, build the envelope, and write both report files."""
    active = settings or get_settings()
    records, failures = evaluate(
        questions,
        ask,
        k=experiment.retrieval.top_k,
        experiment=experiment,
        judge=judge,
    )
    report = build_report(
        records=records,
        failures=failures,
        questions=questions,
        experiment=experiment,
        settings=active,
        judge_enabled=judge is not None,
        human_grades=human_grades,
    )
    json_path, md_path = write_report(report, active.eval_results_dir)
    return report, json_path, md_path
