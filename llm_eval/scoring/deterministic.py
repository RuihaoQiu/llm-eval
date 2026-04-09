"""Deterministic scorers: exact match and fuzzy match."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from llm_eval.schemas import ScorerResult

logger = logging.getLogger(__name__)


@dataclass
class ExactMatchScorer:
    """Case-insensitive exact match. Used for enum fields (seniority, work_mode).

    Attributes:
        name: Scorer identifier used in reports.
        threshold: Pass threshold (always 1.0 for exact match).
    """

    name: str = "exact_match"
    threshold: float = 1.0

    def score(self, expected: Any, actual: Any) -> ScorerResult:
        """Score by case-insensitive string equality.

        Args:
            expected: Ground-truth value (str, enum value, or None).
            actual: Agent-produced value.

        Returns:
            ScorerResult with score 1.0 if match, 0.0 otherwise.
        """
        if expected is None and actual is None:
            return ScorerResult(score=1.0, passed=True)
        if expected is None or actual is None:
            return ScorerResult(
                score=0.0,
                passed=False,
                reason=f"expected={expected!r}, actual={actual!r}",
            )
        match = str(expected).lower().strip() == str(actual).lower().strip()
        logger.debug("ExactMatch: expected=%r actual=%r match=%s", expected, actual, match)
        return ScorerResult(score=1.0 if match else 0.0, passed=match)


@dataclass
class FuzzyMatchScorer:
    """Character-level fuzzy match via RapidFuzz. Used for text fields (title, location).

    Attributes:
        name: Scorer identifier used in reports.
        threshold: Minimum ratio (0–1) to count as passed.
    """

    name: str = "fuzzy_match"
    threshold: float = 0.8

    def score(self, expected: Any, actual: Any) -> ScorerResult:
        """Score by normalised Levenshtein ratio.

        Args:
            expected: Ground-truth string (or None).
            actual: Agent-produced string (or None).

        Returns:
            ScorerResult with score equal to the fuzzy ratio.
        """
        if expected is None and actual is None:
            return ScorerResult(score=1.0, passed=True)
        if expected is None or actual is None:
            return ScorerResult(
                score=0.0,
                passed=False,
                reason=f"expected={expected!r}, actual={actual!r}",
            )
        ratio = fuzz.ratio(str(expected).lower(), str(actual).lower()) / 100.0
        passed = ratio >= self.threshold
        logger.debug("FuzzyMatch: expected=%r actual=%r ratio=%.2f", expected, actual, ratio)
        return ScorerResult(score=ratio, passed=passed, reason=f"ratio={ratio:.2f}")
