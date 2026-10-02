# Eval

| Path | Status | Contents |
|---|---|---|
| `golden/schema.md` | Phase 0 | Record schema, quotas, and the review/held-out rules. |
| `golden/questions.jsonl` | Phase 1 | 40 questions minimum, drafted from parsed documents and flagged for owner review. Empty in Phase 0 on purpose: a question written before its document is parsed is a guess, not a gold standard. |
| `runners/golden.py` | Phase 0 | Schema plus validator (`reglens golden-validate`). |
| `runners/run_eval.py` | Phase 1 | One-command run: predicts, scores retrieval and generation metrics, writes a versioned report. |
| `runners/metrics.py` | Phase 1 | recall@k, MRR, nDCG, citation precision, correctness, faithfulness. |
| `runners/judge.py` | Phase 1 | LLM judge with a versioned prompt, plus the human-agreement harness. |
| `results/` | Phase 1 | One immutable report per run: `<date>_<experiment>_<config_hash12>.json` + `.md`, containing config, corpus version, git commit, model versions and per-question failures. |

Rules that keep results trustworthy:

* Every number in `docs/` traces to a saved report in `results/`. Nothing is hand-entered.
* Retrieval metrics are computed against `gold_passages`, not against a model's opinion.
* The judge is validated against 20 hand-graded answers; the agreement number is reported
  even when it is unflattering.
* A missing number is reported as missing, never as zero.
