"""Component eval: field-level accuracy on the full golden set.

Run with:
    uv run pytest evals/component/ -v
Requires OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_eval.eval.dataset import load_golden_set
from llm_eval.eval.runner import run_eval

GOLDEN_SET_PATH = Path(__file__).parents[2] / "data" / "golden_set.jsonl"
MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")
PASS_RATE_THRESHOLD = 0.7
FIELD_SCORE_THRESHOLD = 0.5


@pytest.mark.asyncio
async def test_overall_pass_rate() -> None:
    """Overall pass rate must be >= 70% across all golden examples."""
    examples = load_golden_set(GOLDEN_SET_PATH)
    report = await run_eval(examples, model=MODEL)
    assert report.pass_rate >= PASS_RATE_THRESHOLD, (
        f"Pass rate {report.pass_rate:.1%} is below threshold {PASS_RATE_THRESHOLD:.1%}\n"
        f"Mean score: {report.mean_score:.2f}"
    )


@pytest.mark.asyncio
async def test_per_field_mean_score() -> None:
    """Every field must achieve a mean score >= 0.5 across all examples."""
    examples = load_golden_set(GOLDEN_SET_PATH)
    report = await run_eval(examples, model=MODEL)

    field_scores: dict[str, list[float]] = {}
    for ex_report in report.examples:
        for fr in ex_report.field_results:
            field_scores.setdefault(fr.field, []).append(fr.result.score)

    failures: list[str] = []
    for field, scores in field_scores.items():
        mean = sum(scores) / len(scores)
        if mean < FIELD_SCORE_THRESHOLD:
            failures.append(f"  {field}: mean={mean:.2f} (threshold={FIELD_SCORE_THRESHOLD})")

    assert not failures, "Fields below threshold:\n" + "\n".join(failures)
