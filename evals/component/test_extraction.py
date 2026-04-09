# Run: uv run pytest evals/component/ -v  (requires OPENAI_API_KEY)
from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_eval.eval.dataset import load_golden_set
from llm_eval.eval.runner import run_eval
from llm_eval.schemas import EvalReport

GOLDEN_SET_PATH = Path(__file__).parents[2] / "data" / "golden_set.jsonl"
MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")
PASS_RATE_THRESHOLD = 0.7
FIELD_SCORE_THRESHOLD = 0.5


@pytest.fixture(scope="session")
async def eval_report() -> EvalReport:
    examples = load_golden_set(GOLDEN_SET_PATH)
    return await run_eval(examples, model=MODEL)


async def test_overall_pass_rate(eval_report: EvalReport) -> None:
    assert eval_report.pass_rate >= PASS_RATE_THRESHOLD, (
        f"Pass rate {eval_report.pass_rate:.1%} is below threshold {PASS_RATE_THRESHOLD:.1%}\n"
        f"Mean score: {eval_report.mean_score:.2f}"
    )


async def test_per_field_mean_score(eval_report: EvalReport) -> None:
    field_scores: dict[str, list[float]] = {}
    for ex_report in eval_report.examples:
        for fr in ex_report.field_results:
            field_scores.setdefault(fr.field, []).append(fr.result.score)

    failures = [
        f"  {field}: mean={sum(scores)/len(scores):.2f} (threshold={FIELD_SCORE_THRESHOLD})"
        for field, scores in field_scores.items()
        if sum(scores) / len(scores) < FIELD_SCORE_THRESHOLD
    ]
    assert not failures, "Fields below threshold:\n" + "\n".join(failures)
