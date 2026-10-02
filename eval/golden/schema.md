# Golden set schema

One JSON object per line in `questions.jsonl` (UTF-8, LF, trailing newline optional).
Validated by `python -m reglens.cli golden-validate` and by `tests/unit/test_golden.py`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | string | yes | `q0001` format, unique, stable forever. Never renumber: results reference these ids. |
| `type` | enum | yes | `lookup`, `numeric`, `temporal`, `comparison`, `multi_hop`, `unanswerable`. |
| `question` | string | yes | Asked exactly as a user would type it. |
| `as_of_date` | `YYYY-MM-DD` or null | no | Required in practice for `temporal` questions; the system must not cite documents outside the window. |
| `reference_answer` | string | yes | What a correct answer contains. For `unanswerable`, what a correct *refusal* should say. |
| `gold_passages` | array | yes for answerable types | Each: `doc_id` (must exist in `data/manifest.csv`), `page` (1-indexed), `clause` (e.g. `4.2.1`). At least one of page/clause. |
| `notes` | string | no | Grading guidance: exact figures, tolerance, what does *not* count as correct. |
| `review` | enum | no | `draft` (default), `reviewed`, `rejected`. Owner signs off; metrics are only reported for reviewed questions. |
| `held_out` | bool | no | Leakage control. Held-out questions are never used to choose a configuration, only to report final results. |
| `source_checked` | bool | no | Set once the gold passage has been read back against the source page. |

## Quotas (PRD §12)

`lookup` 25 · `numeric` 20 · `temporal` 20 · `comparison` 15 · `multi_hop` 10 · `unanswerable` 10

Phase 1 requires 40 questions minimum; the full set requires 100+. The validator warns on
quota gaps when run with `--require-full-minimum`.

## Rules

1. Questions are drafted **from parsed documents**, with the gold passage recorded while
   reading it. A question whose gold passage cannot be pointed at is not a question yet.
2. Every question is flagged for owner review (`review: draft` until then).
3. Held-out questions stay untouched until final reporting (PRD §12, leakage control).
4. Judge validation is reported honestly: hand-grade 20 answers, then report
   judge-vs-human agreement. Do not tune the judge until disagreement is understood.
