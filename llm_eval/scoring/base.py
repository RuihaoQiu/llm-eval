from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from llm_eval.schemas import ScorerResult


@runtime_checkable
class Scorer(Protocol):
    """Synchronous scorer protocol for deterministic, I/O-free scorers."""

    name: str

    def score(self, expected: Any, actual: Any) -> ScorerResult: ...


@runtime_checkable
class AsyncScorer(Protocol):
    """Async scorer protocol for scorers that make API calls (embeddings, LLM judge)."""

    name: str

    async def score(self, expected: Any, actual: Any) -> ScorerResult: ...
