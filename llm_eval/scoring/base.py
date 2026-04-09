"""Scorer protocols — every scorer must satisfy one of these interfaces."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from llm_eval.schemas import ScorerResult


@runtime_checkable
class Scorer(Protocol):
    """Protocol for synchronous scorers (deterministic, no I/O).

    Each scorer compares an expected value against an actual value and returns
    a continuous ``ScorerResult`` with a score in [0, 1].
    """

    name: str

    def score(self, expected: Any, actual: Any) -> ScorerResult:
        """Score an (expected, actual) pair.

        Args:
            expected: Ground-truth value from the golden set.
            actual: Value produced by the agent under evaluation.

        Returns:
            A ``ScorerResult`` with a score in [0, 1].
        """
        ...


@runtime_checkable
class AsyncScorer(Protocol):
    """Protocol for async scorers that make I/O calls (embeddings, LLM judge)."""

    name: str

    async def score(self, expected: Any, actual: Any) -> ScorerResult:
        """Async version of score.

        Args:
            expected: Ground-truth value from the golden set.
            actual: Value produced by the agent under evaluation.

        Returns:
            A ``ScorerResult`` with a score in [0, 1].
        """
        ...
