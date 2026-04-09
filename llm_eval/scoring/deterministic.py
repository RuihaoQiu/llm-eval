"""Deterministic scorers: exact match and fuzzy match."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from llm_eval.schemas import ScorerResult

logger = logging.getLogger(__name__)

_NULL_MATCH = ScorerResult(score=1.0, passed=True)


def _null_mismatch(expected: Any, actual: Any) -> ScorerResult:
    return ScorerResult(score=0.0, passed=False, reason=f"expected={expected!r}, actual={actual!r}")


@dataclass
class ExactMatchScorer:
    """Case-insensitive exact match for enum fields (seniority, work_mode)."""

    name: str = "exact_match"

    def score(self, expected: Any, actual: Any) -> ScorerResult:
        if expected is None and actual is None:
            return _NULL_MATCH
        if expected is None or actual is None:
            return _null_mismatch(expected, actual)
        matched = str(expected).lower().strip() == str(actual).lower().strip()
        logger.debug("ExactMatch: expected=%r actual=%r match=%s", expected, actual, matched)
        return ScorerResult(score=1.0 if matched else 0.0, passed=matched)


@dataclass
class FuzzyMatchScorer:
    """Normalised Levenshtein ratio scorer for text fields (location).

    Attributes:
        threshold: Minimum ratio (0–1) to count as passed.
    """

    name: str = "fuzzy_match"
    threshold: float = 0.8

    def score(self, expected: Any, actual: Any) -> ScorerResult:
        if expected is None and actual is None:
            return _NULL_MATCH
        if expected is None or actual is None:
            return _null_mismatch(expected, actual)
        ratio = fuzz.ratio(str(expected).lower(), str(actual).lower()) / 100.0
        logger.debug("FuzzyMatch: expected=%r actual=%r ratio=%.2f", expected, actual, ratio)
        return ScorerResult(score=ratio, passed=ratio >= self.threshold, reason=f"ratio={ratio:.2f}")
