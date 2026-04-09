# Run: uv run pytest evals/judge/ -v  (requires OPENAI_API_KEY)
# Asserts judge variance < 0.2 across 3 independent runs on identical pairs.
from __future__ import annotations

import statistics

import pytest

from llm_eval.scoring.llm_judge import LLMJudgeScorer

PAIRS = [
    ("Senior Data Engineer", "Data Engineer – Senior Level"),
    ("Junior Software Engineer", "Backend Developer"),
    ("Machine Learning Engineer", "Data Scientist"),
]

VARIANCE_THRESHOLD = 0.2
N_RUNS = 3


@pytest.mark.asyncio
@pytest.mark.parametrize("expected,actual", PAIRS)
async def test_judge_consistency(expected: str, actual: str) -> None:
    # Fresh scorer each run so there's no shared cache between runs.
    scores = [await LLMJudgeScorer().score(expected, actual) for _ in range(N_RUNS)]
    variance = statistics.variance(s.score for s in scores) if N_RUNS > 1 else 0.0
    assert variance < VARIANCE_THRESHOLD, (
        f"Judge variance {variance:.3f} >= {VARIANCE_THRESHOLD} "
        f"for ({expected!r}, {actual!r}): scores={[s.score for s in scores]}"
    )


@pytest.mark.asyncio
async def test_judge_cache_is_deterministic() -> None:
    scorer = LLMJudgeScorer()
    first = await scorer.score("Senior Data Engineer", "Senior Data Engineer")
    second = await scorer.score("Senior Data Engineer", "Senior Data Engineer")
    assert first.score == second.score
    assert first.reason == second.reason


@pytest.mark.asyncio
async def test_judge_exact_match_scores_high() -> None:
    result = await LLMJudgeScorer().score("Senior Data Engineer", "Senior Data Engineer")
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_judge_unrelated_scores_low() -> None:
    result = await LLMJudgeScorer().score("Senior Data Engineer", "Junior Graphic Designer")
    assert result.score <= 0.5
