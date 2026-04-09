"""Unit tests for deterministic scorers — no LLM calls."""

from __future__ import annotations

import pytest

from llm_eval.scoring.deterministic import ExactMatchScorer, FuzzyMatchScorer


class TestExactMatchScorer:
    scorer = ExactMatchScorer()

    def test_identical_strings(self) -> None:
        result = self.scorer.score("senior", "senior")
        assert result.score == 1.0
        assert result.passed

    def test_case_insensitive(self) -> None:
        result = self.scorer.score("Senior", "senior")
        assert result.score == 1.0
        assert result.passed

    def test_mismatch(self) -> None:
        result = self.scorer.score("senior", "mid")
        assert result.score == 0.0
        assert not result.passed

    def test_both_none(self) -> None:
        result = self.scorer.score(None, None)
        assert result.score == 1.0
        assert result.passed

    def test_one_none(self) -> None:
        result = self.scorer.score("senior", None)
        assert result.score == 0.0
        assert not result.passed


class TestFuzzyMatchScorer:
    scorer = FuzzyMatchScorer(threshold=0.8)

    def test_exact(self) -> None:
        result = self.scorer.score("Berlin, Germany", "Berlin, Germany")
        assert result.score == 1.0
        assert result.passed

    def test_close_match(self) -> None:
        result = self.scorer.score("New York, USA", "New York, US")
        assert result.score >= 0.8
        assert result.passed

    def test_poor_match(self) -> None:
        result = self.scorer.score("Berlin, Germany", "Tokyo, Japan")
        assert result.score < 0.8
        assert not result.passed

    def test_both_none(self) -> None:
        result = self.scorer.score(None, None)
        assert result.score == 1.0
        assert result.passed

    def test_one_none(self) -> None:
        result = self.scorer.score("Berlin", None)
        assert result.score == 0.0
        assert not result.passed

    def test_custom_threshold(self) -> None:
        scorer_low = FuzzyMatchScorer(threshold=0.5)
        result = scorer_low.score("London", "Lndn")
        assert result.passed  # low threshold, partial match passes
