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

TITLE_RUBRIC = """You are evaluating whether an extracted job title matches the expected job title.

Score the match on a 0–2 scale:
  2: The titles are semantically equivalent or one is a minor reformulation of the other
     (e.g. "Senior Data Engineer" vs "Data Engineer – Senior")
  1: The titles are related but meaningfully different
     (e.g. "Data Engineer" vs "Data Analyst")
  0: The titles are unrelated or one is clearly wrong

Return only the score (0, 1, or 2) and a one-sentence reasoning."""

SKILLS_RUBRIC = """You are evaluating whether an extracted skills list matches the expected skills list.

Score the match on a 0–2 scale:
  2: The lists cover the same skills (order-invariant), allowing for abbreviation and synonym
     differences (e.g. "k8s" = "Kubernetes", "sklearn" = "scikit-learn", "PyTorch" = "torch").
  1: Partial overlap — some expected skills are missing or some extra skills were hallucinated,
     but the core skills are present.
  0: Little or no overlap — the lists are fundamentally different.

Return only the score (0, 1, or 2) and a one-sentence reasoning."""


class JudgeVerdict(BaseModel):
    score: Literal[0, 1, 2]
    reasoning: str


@dataclass
class LLMJudgeScorer:
    """Judge scorer using structured OpenAI output, cached by hash(rubric + expected + actual).

    Attributes:
        rubric: Evaluation rubric passed to the judge.
        model: OpenAI model to use for judging.
        threshold: Minimum score (0–1 continuous) to count as passed.
    """

    name: str = "llm_judge"
    rubric: str = TITLE_RUBRIC
    model: str = "gpt-4o-mini"
    threshold: float = 0.5
    _cache: dict[str, ScorerResult] = field(default_factory=dict, repr=False)

    def _cache_key(self, expected: str, actual: str) -> str:
        return hashlib.sha256(f"{self.rubric}||{expected}||{actual}".encode()).hexdigest()

    async def _call_judge(self, expected: str, actual: str) -> ScorerResult:
        response = await _client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.rubric},
                {"role": "user", "content": f"Expected: {expected}\nActual: {actual}"},
            ],
            response_format=JudgeVerdict,
        )
        verdict = response.choices[0].message.parsed
        score = _SCORE_MAP[verdict.score]
        logger.debug("LLMJudge: expected=%r actual=%r score=%d", expected, actual, verdict.score)
        return ScorerResult(score=score, passed=score >= self.threshold, reason=verdict.reasoning)

    async def score(
        self,
        expected: str | list[str] | None,
        actual: str | list[str] | None,
    ) -> ScorerResult:
        if expected is None and actual is None:
            return ScorerResult(score=1.0, passed=True)
        if (not expected) and (not actual):
            return ScorerResult(score=1.0, passed=True)
        if not expected or not actual:
            return ScorerResult(score=0.0, passed=False, reason=f"expected={expected!r}, actual={actual!r}")

        exp_str = ", ".join(expected) if isinstance(expected, list) else expected
        act_str = ", ".join(actual) if isinstance(actual, list) else actual

        key = self._cache_key(exp_str, act_str)
        if key in self._cache:
            logger.debug("LLMJudge cache hit for expected=%r", exp_str[:30])
            return self._cache[key]

        result = await self._call_judge(exp_str, act_str)
        self._cache[key] = result
        return result
