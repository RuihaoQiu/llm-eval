"""Meta-eval: judge consistency across repeated runs.

Asserts that the LLM judge returns the same score (variance < 0.2) when run
3 times on identical (expected, actual) pairs.

Run with:
    uv run pytest evals/judge/ -v
Requires OPENAI_API_KEY in the environment.
"""

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
    """Judge must produce low-variance scores across 3 independent runs.

    Each run uses a fresh scorer instance (no shared cache) so results are
    independent. Variance must be below 0.2.
    """
    scores: list[float] = []
    for _ in range(N_RUNS):
        scorer = LLMJudgeScorer()  # fresh cache each run
        result = await scorer.score(expected, actual)
        scores.append(result.score)

    variance = statistics.variance(scores) if len(scores) > 1 else 0.0
    assert variance < VARIANCE_THRESHOLD, (
        f"Judge variance {variance:.3f} >= {VARIANCE_THRESHOLD} "
        f"for ({expected!r}, {actual!r}): scores={scores}"
    )


@pytest.mark.asyncio
async def test_judge_cache_is_deterministic() -> None:
    """Cached result must be identical to the original result."""
    scorer = LLMJudgeScorer()
    expected, actual = "Senior Data Engineer", "Senior Data Engineer"
    first = await scorer.score(expected, actual)
    second = await scorer.score(expected, actual)  # should hit cache
    assert first.score == second.score
    assert first.reason == second.reason


@pytest.mark.asyncio
async def test_judge_exact_match_scores_high() -> None:
    """Identical strings must score 1.0 (score=2 from judge)."""
    scorer = LLMJudgeScorer()
    result = await scorer.score("Senior Data Engineer", "Senior Data Engineer")
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_judge_unrelated_scores_low() -> None:
    """Completely unrelated titles must score <= 0.5."""
    scorer = LLMJudgeScorer()
    result = await scorer.score("Senior Data Engineer", "Junior Graphic Designer")
    assert result.score <= 0.5
