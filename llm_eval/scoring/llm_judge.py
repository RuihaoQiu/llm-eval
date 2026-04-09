"""LLM-based judge scorer with structured output and result caching."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

from llm_eval.schemas import ScorerResult

logger = logging.getLogger(__name__)

_client = AsyncOpenAI()

_SCORE_MAP: dict[int, float] = {0: 0.0, 1: 0.5, 2: 1.0}

_DEFAULT_RUBRIC = """You are evaluating whether an extracted job title matches the expected job title.

Score the match on a 0–2 scale:
  2: The titles are semantically equivalent or one is a minor reformulation of the other
     (e.g. "Senior Data Engineer" vs "Data Engineer – Senior")
  1: The titles are related but meaningfully different
     (e.g. "Data Engineer" vs "Data Analyst")
  0: The titles are unrelated or one is clearly wrong

Return only the score (0, 1, or 2) and a one-sentence reasoning."""


class JudgeVerdict(BaseModel):
    """Structured output from the LLM judge."""

    score: Literal[0, 1, 2]
    reasoning: str


@dataclass
class LLMJudgeScorer:
    """Judge scorer using structured OpenAI output.

    Results are cached by a hash of (rubric, expected, actual) to avoid
    re-spending tokens on identical comparisons across runs.

    Attributes:
        name: Scorer identifier used in reports.
        rubric: Evaluation rubric passed to the judge.
        model: OpenAI model to use for judging.
        threshold: Minimum score (0–1 continuous) to count as passed.
    """

    name: str = "llm_judge"
    rubric: str = _DEFAULT_RUBRIC
    model: str = "gpt-4o-mini"
    threshold: float = 0.5
    _cache: dict[str, ScorerResult] = field(default_factory=dict, repr=False)

    def _cache_key(self, expected: str, actual: str) -> str:
        payload = f"{self.rubric}||{expected}||{actual}"
        return hashlib.sha256(payload.encode()).hexdigest()

    async def score(self, expected: str | None, actual: str | None) -> ScorerResult:
        """Judge whether actual matches expected according to the rubric.

        Args:
            expected: Ground-truth value (or None).
            actual: Agent-produced value (or None).

        Returns:
            ScorerResult with continuous score from the 0→0.0, 1→0.5, 2→1.0 mapping.
        """
        if expected is None and actual is None:
            return ScorerResult(score=1.0, passed=True)
        if expected is None or actual is None:
            return ScorerResult(
                score=0.0,
                passed=False,
                reason=f"expected={expected!r}, actual={actual!r}",
            )

        key = self._cache_key(expected, actual)
        if key in self._cache:
            logger.debug("LLMJudge cache hit for expected=%r", expected[:30])
            return self._cache[key]

        user_msg = f"Expected: {expected}\nActual: {actual}"
        response = await _client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.rubric},
                {"role": "user", "content": user_msg},
            ],
            response_format=JudgeVerdict,
        )
        verdict = response.choices[0].message.parsed
        score = _SCORE_MAP[verdict.score]
        result = ScorerResult(score=score, passed=score >= self.threshold, reason=verdict.reasoning)
        self._cache[key] = result
        logger.debug("LLMJudge: expected=%r actual=%r score=%d", expected, actual, verdict.score)
        return result
