"""LLM judge with a versioned prompt, plus the human-agreement harness.

The judge scores two things per answer (PRD §metrics): **correctness** against the
reference answer and **faithfulness** against the retrieved context it was given.
Both are 0-2 integers so a hand grader can replicate the scale exactly:

* correctness: 0 wrong, 1 partially right, 2 fully right
* faithfulness: 0 unsupported/contradicted, 1 partly supported, 2 fully supported

Two rules keep this honest:

1. The prompt carries a version id (``judge-v1``). Reports record it, so a score from
   two different prompts can never be averaged into one number.
2. ``agreement()`` compares judge scores against hand grades and reports exact-match
   and within-one rates *per dimension*, even when they are unflattering
   (eval/README.md). The judge is never tuned before the disagreement is understood.

A judge failure returns ``None`` scores with ``error`` set — a missing number, never a
zero.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reglens.generation.base import LLMClient, Message
from reglens.retrieval.base import RetrievedChunk

JUDGE_PROMPT_VERSION = "judge-v1"
#: Scale max: scores are integers in [0, JUDGE_MAX_SCORE].
JUDGE_MAX_SCORE = 2

SYSTEM_PROMPT = f"""\
You are a strict evaluator of answers about Indian banking regulation.

You will be given: a QUESTION, a REFERENCE ANSWER (what a correct answer contains),
the CONTEXT passages the answerer was allowed to use, and the CANDIDATE ANSWER.

Score the candidate on two axes, each an integer from 0 to {JUDGE_MAX_SCORE}:

correctness — against the REFERENCE ANSWER only:
  2 = fully correct: contains every fact the reference requires, no wrong facts
  1 = partially correct: right on the main point but missing or wrong on details
  0 = wrong, or refuses a question the reference shows is answerable

faithfulness — against the CONTEXT only (ignore the reference here):
  2 = every claim in the candidate is supported by the context
  1 = mostly supported, but some detail is unsupported or loosely paraphrased
  0 = contradicts the context, or asserts facts not present in it
  A correct refusal of an unanswerable question scores 2 for faithfulness.

Rules:
- Judge only what is written. Do not reward answers for being true outside the context.
- A candidate that refuses despite the context containing the answer loses correctness
  points; a refusal the context supports is fully correct.
- If the REFERENCE ANSWER or CONTEXT is empty, score 0 and say why in "notes".

Reply with ONLY a JSON object, no prose, no code fences:
{{"correctness": <int>, "faithfulness": <int>, "notes": "<one sentence>"}}
"""

QUESTION_TEMPLATE = """\
QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

CONTEXT PASSAGES:
{context}

CANDIDATE ANSWER:
{candidate}
"""


@dataclass(frozen=True)
class JudgeScore:
    """One answer's judged scores. ``None`` scores + ``error`` mean 'missing', not 0."""

    correctness: int | None
    faithfulness: int | None
    notes: str = ""
    error: str | None = None
    prompt_version: str = JUDGE_PROMPT_VERSION
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    raw: str = field(default="", repr=False)

    @property
    def ok(self) -> bool:
        return self.error is None and self.correctness is not None


def build_context_block(chunks: list[RetrievedChunk], *, max_chars: int = 24000) -> str:
    """Numbered context lines for the judge prompt (same order the answerer saw)."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        pages = (
            f"p.{chunk.page_start}"
            if chunk.page_end is None or chunk.page_start == chunk.page_end
            else f"p.{chunk.page_start}-{chunk.page_end}"
        )
        parts.append(f"[{index}] {chunk.doc_title} | {pages}\n{chunk.text}")
    block = "\n\n".join(parts)
    return block[:max_chars]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(text: str) -> tuple[int | None, int | None, str]:
    """Extract (correctness, faithfulness, notes) from the judge's reply.

    Tolerates code fences and surrounding prose; anything unparseable or out of range
    yields (None, None, reason) so the report shows a missing score.
    """
    match = _JSON_RE.search(text)
    if not match:
        return None, None, f"no JSON object in reply: {text[:120]!r}"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, None, f"invalid JSON: {exc}"
    correctness = _score(payload.get("correctness"))
    faithfulness = _score(payload.get("faithfulness"))
    if correctness is None or faithfulness is None:
        return None, None, f"missing or out-of-range score in {payload!r}"
    notes = str(payload.get("notes", ""))
    return correctness, faithfulness, notes


def _score(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    integer = int(value)
    if integer != value or not 0 <= integer <= JUDGE_MAX_SCORE:
        return None
    return integer


class GeminiJudge:
    """Scores one answer at a time through the shared ``LLMClient`` protocol."""

    def __init__(self, llm: LLMClient, *, model: str) -> None:
        self.llm = llm
        self.model = model

    def judge(
        self,
        *,
        question: str,
        reference_answer: str,
        context: list[RetrievedChunk],
        candidate: str,
    ) -> JudgeScore:
        if not candidate.strip():
            return JudgeScore(
                correctness=0,
                faithfulness=0,
                notes="empty candidate answer",
                error=None,
                model=self.model,
            )
        prompt = QUESTION_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer or "(none provided)",
            context=build_context_block(context) or "(no context passages)",
            candidate=candidate,
        )
        try:
            response = self.llm.complete(
                [
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=prompt),
                ],
                model=self.model,
                temperature=0.0,
                # gemini-3.6-flash thinks *inside* the output budget (thoughts_tokens are
                # billed as output), and 400 left ~36 chars of visible JSON — 21/40
                # replies were truncated mid-object in the first eval run.
                max_output_tokens=4000,
            )
        except Exception as exc:  # provider outage must not zero the metric
            return JudgeScore(None, None, error=f"judge call failed: {exc}", model=self.model)
        correctness, faithfulness, notes = parse_judge_output(response.text)
        if correctness is None:
            return JudgeScore(
                None,
                None,
                notes=notes,
                error=notes,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                raw=response.text,
            )
        return JudgeScore(
            correctness=correctness,
            faithfulness=faithfulness,
            notes=notes,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            raw=response.text,
        )


# ------------------------------------------------------------------ agreement
def load_human_grades(path: Any) -> list[HumanGrade]:
    """Read hand grades from JSONL: one ``{question_id, correctness, faithfulness}`` per line.

    Scores are validated against the same 0-2 scale the judge uses — a hand grade of 3
    is a mistake in the grading sheet, and silently clamping it would put a number in
    the report that no one actually graded.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"human grades not found: {target}")
    grades: list[HumanGrade] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{line_number}: {exc}") from exc
        try:
            question_id = str(payload["question_id"])
            correctness = _grade(payload["correctness"])
            faithfulness = _grade(payload["faithfulness"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{target}:{line_number}: {exc}") from exc
        grades.append(
            HumanGrade(
                question_id=question_id,
                correctness=correctness,
                faithfulness=faithfulness,
                notes=str(payload.get("notes", "")),
            )
        )
    return grades


def _grade(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"grade must be an integer 0-{JUDGE_MAX_SCORE}, got {value!r}")
    integer = int(value)
    if not 0 <= integer <= JUDGE_MAX_SCORE:
        raise ValueError(f"grade must be 0-{JUDGE_MAX_SCORE}, got {value!r}")
    return integer


@dataclass(frozen=True)
class HumanGrade:
    """One hand-graded answer: the same 0-2 axes, graded by a human."""

    question_id: str
    correctness: int
    faithfulness: int
    notes: str = ""


@dataclass(frozen=True)
class AgreementReport:
    """Judge-vs-human agreement, per dimension and overall."""

    paired: int
    exact: dict[str, float | None]
    within_one: dict[str, float | None]
    mean_abs_error: dict[str, float | None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "paired": self.paired,
            "exact_agreement": self.exact,
            "within_one_agreement": self.within_one,
            "mean_abs_error": self.mean_abs_error,
        }


def agreement(
    judge_scores: dict[str, JudgeScore], human_grades: list[HumanGrade]
) -> AgreementReport:
    """Compare judge scores to hand grades on their intersection.

    Missing judge scores (errors) are dropped from the pairing and counted out — the
    report's ``paired`` is the denominator every rate here uses.
    """
    pairs = [
        (judge_scores[grade.question_id], grade)
        for grade in human_grades
        if grade.question_id in judge_scores and judge_scores[grade.question_id].ok
    ]

    def rate(dimension: str) -> float | None:
        if not pairs:
            return None
        matches = sum(
            1 for score, grade in pairs if getattr(score, dimension) == getattr(grade, dimension)
        )
        return matches / len(pairs)

    def within_one(dimension: str) -> float | None:
        if not pairs:
            return None
        close = sum(
            1
            for score, grade in pairs
            if abs(int(getattr(score, dimension)) - getattr(grade, dimension)) <= 1
        )
        return close / len(pairs)

    def mae(dimension: str) -> float | None:
        if not pairs:
            return None
        total = sum(
            abs(int(getattr(score, dimension)) - getattr(grade, dimension))
            for score, grade in pairs
        )
        return total / len(pairs)

    return AgreementReport(
        paired=len(pairs),
        exact={dim: rate(dim) for dim in ("correctness", "faithfulness")},
        within_one={dim: within_one(dim) for dim in ("correctness", "faithfulness")},
        mean_abs_error={dim: mae(dim) for dim in ("correctness", "faithfulness")},
    )
