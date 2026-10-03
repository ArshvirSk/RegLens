"""Judge tests: prompt parsing, score bounds, failure-as-missing, human agreement.

No live model calls — the judge's LLM is injected through the protocol like every
other client in this codebase.
"""

from __future__ import annotations

from eval.runners.judge import (
    JUDGE_MAX_SCORE,
    JUDGE_PROMPT_VERSION,
    GeminiJudge,
    HumanGrade,
    JudgeScore,
    agreement,
    parse_judge_output,
)

from reglens.generation.base import LLMResponse, Message


def test_parse_clean_json() -> None:
    correctness, faithfulness, notes = parse_judge_output(
        '{"correctness": 2, "faithfulness": 1, "notes": "solid"}'
    )
    assert (correctness, faithfulness) == (2, 1)
    assert notes == "solid"


def test_parse_tolerates_code_fences_and_prose() -> None:
    text = 'Sure! Here is my evaluation:\n```json\n{"correctness": 1, "faithfulness": 2, "notes": "ok"}\n```'
    correctness, faithfulness, _ = parse_judge_output(text)
    assert (correctness, faithfulness) == (1, 2)


def test_parse_rejects_out_of_range_and_non_integer_scores() -> None:
    for payload in (
        '{"correctness": 3, "faithfulness": 1, "notes": ""}',
        '{"correctness": 1.5, "faithfulness": 1, "notes": ""}',
        '{"correctness": 1, "notes": "no faithfulness"}',
        '{"correctness": true, "faithfulness": 1, "notes": ""}',
    ):
        correctness, faithfulness, reason = parse_judge_output(payload)
        assert correctness is None and faithfulness is None
        assert reason


def test_parse_rejects_non_json() -> None:
    correctness, _faithfulness, reason = parse_judge_output("the answer looks correct to me")
    assert correctness is None
    assert "no JSON object" in reason


class FakeLLM:
    """Protocol-compatible stub: canned reply or an exception."""

    name = "fake"

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[list[Message]] = []
        self.budgets: list[int] = []

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 900,
    ) -> LLMResponse:
        self.calls.append(messages)
        self.budgets.append(max_output_tokens)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            text=self.reply, model=model, input_tokens=100, output_tokens=20, finish_reason="STOP"
        )


def test_judge_returns_scores_and_records_prompt_version() -> None:
    llm = FakeLLM('{"correctness": 2, "faithfulness": 2, "notes": "grounded"}')
    judge = GeminiJudge(llm, model="judge-model")
    score = judge.judge(
        question="What is X?",
        reference_answer="X is 100%.",
        context=[],
        candidate="X is 100% [Doc, p.1, 4.2].",
    )
    assert score.ok
    assert (score.correctness, score.faithfulness) == (2, 2)
    assert score.prompt_version == JUDGE_PROMPT_VERSION
    assert score.input_tokens == 100
    # the reference answer must actually reach the prompt
    assert "X is 100%." in llm.calls[0][1].content
    # the system prompt carries the scale the report documents
    assert str(JUDGE_MAX_SCORE) in llm.calls[0][0].content


def test_judge_output_budget_leaves_room_for_thinking() -> None:
    """gemini-3.6-flash spends max_output_tokens on thinking; 400 truncated 21/40
    replies mid-JSON in the first eval run, so the budget must dwarf a JSON answer."""
    llm = FakeLLM('{"correctness": 1, "faithfulness": 1, "notes": "thin"}')
    judge = GeminiJudge(llm, model="judge-model")
    score = judge.judge(question="Q?", reference_answer="A", context=[], candidate="ans")
    assert score.ok
    assert llm.budgets and llm.budgets[0] >= 2000


def test_judge_provider_failure_is_missing_not_zero() -> None:
    llm = FakeLLM(error=RuntimeError("quota exceeded"))
    judge = GeminiJudge(llm, model="judge-model")
    score = judge.judge(
        question="Q?",
        reference_answer="A",
        context=[],
        candidate="some answer",
    )
    assert not score.ok
    assert score.correctness is None and score.faithfulness is None
    assert "quota exceeded" in (score.error or "")


def test_judge_unparseable_reply_is_missing_not_zero() -> None:
    judge = GeminiJudge(FakeLLM("I refuse to score this."), model="judge-model")
    score = judge.judge(question="Q?", reference_answer="A", context=[], candidate="ans")
    assert not score.ok
    assert score.correctness is None


def test_judge_empty_candidate_scores_zero_without_a_model_call() -> None:
    llm = FakeLLM()
    judge = GeminiJudge(llm, model="judge-model")
    score = judge.judge(question="Q?", reference_answer="A", context=[], candidate="   ")
    assert score.ok
    assert (score.correctness, score.faithfulness) == (0, 0)
    assert llm.calls == []


def test_agreement_exact_within_one_and_missing_pairs() -> None:
    judge_scores = {
        "q0001": JudgeScore(correctness=2, faithfulness=1),
        "q0002": JudgeScore(correctness=0, faithfulness=2),
        "q0003": JudgeScore(correctness=None, faithfulness=None, error="boom"),  # dropped
        "q0004": JudgeScore(correctness=1, faithfulness=2),  # no human grade: dropped
    }
    human = [
        HumanGrade("q0001", correctness=2, faithfulness=2),  # exact C, off-by-one F
        HumanGrade("q0002", correctness=1, faithfulness=2),  # off-by-one C, exact F
    ]
    report = agreement(judge_scores, human)
    assert report.paired == 2
    assert report.exact["correctness"] == 0.5
    assert report.exact["faithfulness"] == 0.5
    assert report.within_one["correctness"] == 1.0
    assert report.within_one["faithfulness"] == 1.0
    assert report.mean_abs_error["correctness"] == 0.5
    payload = report.as_dict()
    assert payload["paired"] == 2


def test_agreement_with_no_pairs_reports_missing_rates() -> None:
    report = agreement({}, [HumanGrade("q0001", correctness=1, faithfulness=1)])
    assert report.paired == 0
    assert report.exact["correctness"] is None
    assert report.within_one["faithfulness"] is None
    assert report.mean_abs_error["correctness"] is None
