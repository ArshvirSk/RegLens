"""Golden eval set: schema and validation.

The golden set is the measuring instrument, so it gets the same treatment as code: a
declared schema, a validator, per-type quotas from the PRD, and two flags that matter
more than people expect:

* ``reviewed`` — the owner signs off on every question before it is used. Until then it is
  a draft, and drafts must not be quoted in reported metrics.
* ``held_out`` — leakage control. Held-out questions are never used to choose a
  configuration; they are read once, at final reporting. Tuning on them inflates results
  and destroys the only honest number the project has.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reglens.config import get_settings

QuestionType = Literal["lookup", "numeric", "temporal", "comparison", "multi_hop", "unanswerable"]
ReviewStatus = Literal["draft", "reviewed", "rejected"]

# PRD phase-1 target is 40 questions; the full set is 100+. Quotas are checked per type.
QUESTION_TYPE_QUOTAS: dict[str, int] = {
    "lookup": 25,
    "numeric": 20,
    "temporal": 20,
    "comparison": 15,
    "multi_hop": 10,
    "unanswerable": 10,
}
PHASE1_MIN_QUESTIONS = 40
FULL_MIN_QUESTIONS = 100
ID_RE = re.compile(r"^q\d{4}$")


class GoldPassage(BaseModel):
    """The passage a correct answer must be based on. Page and clause are both optional
    only because some filings have no clause structure; at least one must be present."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=3)
    page: int | None = Field(default=None, ge=1)
    clause: str | None = None

    @field_validator("clause")
    @classmethod
    def _non_empty_clause(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("clause must be null or a non-empty string")
        return value


class GoldenQuestion(BaseModel):
    """One eval record."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: QuestionType
    question: str = Field(min_length=8)
    as_of_date: date | None = None
    reference_answer: str = Field(min_length=1)
    gold_passages: list[GoldPassage] = Field(default_factory=list)
    #: Free-text grading guidance for the judge and for human graders.
    notes: str = ""
    review: ReviewStatus = "draft"
    held_out: bool = False
    #: Set once the owner has checked the question against the source PDF.
    source_checked: bool = False

    @field_validator("id")
    @classmethod
    def _id_format(cls, value: str) -> str:
        if not ID_RE.match(value):
            raise ValueError("id must look like q0001")
        return value

    @property
    def is_answerable(self) -> bool:
        return self.type != "unanswerable"


@dataclass(frozen=True)
class GoldenIssue:
    level: Literal["error", "warning"]
    question_id: str
    message: str


@dataclass(frozen=True)
class GoldenReport:
    issues: list[GoldenIssue]
    counts_by_type: dict[str, int]
    held_out_count: int
    reviewed_count: int

    @property
    def errors(self) -> list[GoldenIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[GoldenIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def total(self) -> int:
        return sum(self.counts_by_type.values())

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "total": self.total,
            "counts_by_type": self.counts_by_type,
            "quota_targets": QUESTION_TYPE_QUOTAS,
            "held_out": self.held_out_count,
            "reviewed": self.reviewed_count,
            "errors": [issue.__dict__ for issue in self.errors],
            "warnings": [issue.__dict__ for issue in self.warnings],
        }


def load_golden(path: Path | None = None) -> list[GoldenQuestion]:
    """Read the JSONL golden set. Blank lines are ignored so trailing newlines are safe."""
    target = path or get_settings().golden_path
    if not target.is_file():
        raise FileNotFoundError(f"golden set not found: {target}")
    questions: list[GoldenQuestion] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            questions.append(GoldenQuestion.model_validate(json.loads(stripped)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{target}:{line_number}: {exc}") from exc
    return questions


def write_golden(questions: list[GoldenQuestion], path: Path | None = None) -> Path:
    """Write the golden set as JSONL, sorted by id, one record per line."""
    target = path or get_settings().golden_path
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        json.dumps(question.model_dump(mode="json"), ensure_ascii=False)
        for question in sorted(questions, key=lambda q: q.id)
    )
    target.write_text(body + ("\n" if body else ""), encoding="utf-8")
    return target


def validate_golden(
    questions: list[GoldenQuestion],
    *,
    manifest_ids: set[str] | None = None,
    require_full_minimum: bool = False,
) -> GoldenReport:
    """Validate ids, gold passages, manifest references and quota coverage."""
    issues: list[GoldenIssue] = []
    seen: Counter[str] = Counter(question.id for question in questions)

    for question_id, times in seen.items():
        if times > 1:
            issues.append(GoldenIssue("error", question_id, f"duplicate id ({times} occurrences)"))

    for question in questions:
        if question.is_answerable and not question.gold_passages:
            issues.append(
                GoldenIssue(
                    "error",
                    question.id,
                    f"{question.type} question needs at least one gold passage",
                )
            )
        if not question.is_answerable and question.gold_passages:
            issues.append(
                GoldenIssue("warning", question.id, "unanswerable question lists gold passages")
            )
        for passage in question.gold_passages:
            if passage.page is None and passage.clause is None:
                issues.append(
                    GoldenIssue(
                        "error",
                        question.id,
                        f"gold passage {passage.doc_id} has neither page nor clause",
                    )
                )
            if manifest_ids is not None and passage.doc_id not in manifest_ids:
                issues.append(
                    GoldenIssue(
                        "error",
                        question.id,
                        f"gold passage doc_id {passage.doc_id!r} is not in the manifest",
                    )
                )
        if question.type == "temporal" and question.as_of_date is None:
            issues.append(
                GoldenIssue("warning", question.id, "temporal question has no as_of_date")
            )
        if question.type in {"numeric", "comparison"} and not question.notes:
            issues.append(
                GoldenIssue(
                    "warning",
                    question.id,
                    "numeric/comparison questions should record grading notes",
                )
            )
        if not question.source_checked:
            issues.append(
                GoldenIssue(
                    "warning", question.id, "gold passage not yet checked against the source PDF"
                )
            )
        if question.review == "draft":
            issues.append(GoldenIssue("warning", question.id, "awaiting owner review"))

    counts = Counter(question.type for question in questions)
    minimum = FULL_MIN_QUESTIONS if require_full_minimum else PHASE1_MIN_QUESTIONS
    if len(questions) < minimum:
        issues.append(
            GoldenIssue(
                "warning",
                "-",
                f"{len(questions)} questions; {minimum} required for this phase's acceptance check",
            )
        )
    for question_type, quota in QUESTION_TYPE_QUOTAS.items():
        if require_full_minimum and counts.get(question_type, 0) < quota:
            issues.append(
                GoldenIssue(
                    "warning",
                    "-",
                    f"type {question_type}: {counts.get(question_type, 0)} < quota {quota}",
                )
            )

    return GoldenReport(
        issues=issues,
        counts_by_type=dict(sorted(counts.items())),
        held_out_count=sum(1 for question in questions if question.held_out),
        reviewed_count=sum(1 for question in questions if question.review == "reviewed"),
    )
