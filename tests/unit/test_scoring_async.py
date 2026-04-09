"""Unit tests for async scorers — OpenAI calls are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_eval.scoring.embedding import EmbeddingF1Scorer, EmbeddingScorer, _cosine
from llm_eval.scoring.llm_judge import JudgeVerdict, LLMJudgeScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_emb(values: list[float]) -> list[float]:
    """Return a mock embedding vector."""
    return values


def _make_embedding_response(vector: list[float]) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=vector)]
    return resp


def _make_judge_response(score: int, reasoning: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(parsed=JudgeVerdict(score=score, reasoning=reasoning)))]
    return resp


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal(self) -> None:
        assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_zero_vector(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# EmbeddingScorer
# ---------------------------------------------------------------------------

class TestEmbeddingScorer:
    @pytest.mark.asyncio
    async def test_identical_strings(self) -> None:
        vec = [1.0, 0.0, 0.0]
        with patch(
            "llm_eval.scoring.embedding._client.embeddings.create",
            new=AsyncMock(return_value=_make_embedding_response(vec)),
        ):
            scorer = EmbeddingScorer(threshold=0.8)
            result = await scorer.score("senior", "senior")
        assert result.score >= 0.99
        assert result.passed

    @pytest.mark.asyncio
    async def test_both_none(self) -> None:
        scorer = EmbeddingScorer()
        result = await scorer.score(None, None)
        assert result.score == 1.0
        assert result.passed

    @pytest.mark.asyncio
    async def test_one_none(self) -> None:
        scorer = EmbeddingScorer()
        result = await scorer.score("senior", None)
        assert result.score == 0.0
        assert not result.passed

    @pytest.mark.asyncio
    async def test_dissimilar_scores_low(self) -> None:
        with patch(
            "llm_eval.scoring.embedding._client.embeddings.create",
            new=AsyncMock(side_effect=[
                _make_embedding_response([1.0, 0.0]),
                _make_embedding_response([0.0, 1.0]),
            ]),
        ):
            scorer = EmbeddingScorer(threshold=0.8)
            # Clear cache between tests
            import llm_eval.scoring.embedding as emb_mod
            emb_mod._embedding_cache.clear()
            result = await scorer.score("Berlin", "Tokyo")
        assert result.score < 0.8
        assert not result.passed


# ---------------------------------------------------------------------------
# EmbeddingF1Scorer
# ---------------------------------------------------------------------------

class TestEmbeddingF1Scorer:
    @pytest.mark.asyncio
    async def test_empty_both(self) -> None:
        scorer = EmbeddingF1Scorer()
        result = await scorer.score([], [])
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_empty_actual(self) -> None:
        scorer = EmbeddingF1Scorer()
        result = await scorer.score(["Python"], [])
        assert result.score == 0.0
        assert not result.passed

    @pytest.mark.asyncio
    async def test_perfect_match(self) -> None:
        vec = [1.0, 0.0]
        import llm_eval.scoring.embedding as emb_mod
        emb_mod._embedding_cache.clear()
        with patch(
            "llm_eval.scoring.embedding._client.embeddings.create",
            new=AsyncMock(return_value=_make_embedding_response(vec)),
        ):
            scorer = EmbeddingF1Scorer(threshold=0.5)
            result = await scorer.score(["Python"], ["Python"])
        assert result.score >= 0.99
        assert result.passed


# ---------------------------------------------------------------------------
# LLMJudgeScorer
# ---------------------------------------------------------------------------

class TestLLMJudgeScorer:
    @pytest.mark.asyncio
    async def test_score_2_maps_to_1(self) -> None:
        with patch(
            "llm_eval.scoring.llm_judge._client.beta.chat.completions.parse",
            new=AsyncMock(return_value=_make_judge_response(2, "exact match")),
        ):
            scorer = LLMJudgeScorer()
            result = await scorer.score("Senior Data Engineer", "Data Engineer – Senior")
        assert result.score == 1.0
        assert result.passed

    @pytest.mark.asyncio
    async def test_score_1_maps_to_half(self) -> None:
        with patch(
            "llm_eval.scoring.llm_judge._client.beta.chat.completions.parse",
            new=AsyncMock(return_value=_make_judge_response(1, "related but different")),
        ):
            scorer = LLMJudgeScorer()
            result = await scorer.score("Data Engineer", "Data Analyst")
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_score_0_maps_to_zero(self) -> None:
        with patch(
            "llm_eval.scoring.llm_judge._client.beta.chat.completions.parse",
            new=AsyncMock(return_value=_make_judge_response(0, "unrelated")),
        ):
            scorer = LLMJudgeScorer()
            result = await scorer.score("Data Engineer", "Graphic Designer")
        assert result.score == 0.0
        assert not result.passed

    @pytest.mark.asyncio
    async def test_cache_avoids_second_call(self) -> None:
        mock_parse = AsyncMock(return_value=_make_judge_response(2, "cached"))
        with patch("llm_eval.scoring.llm_judge._client.beta.chat.completions.parse", new=mock_parse):
            scorer = LLMJudgeScorer()
            await scorer.score("A", "B")
            await scorer.score("A", "B")  # should hit cache
        assert mock_parse.call_count == 1

    @pytest.mark.asyncio
    async def test_both_none(self) -> None:
        scorer = LLMJudgeScorer()
        result = await scorer.score(None, None)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_one_none(self) -> None:
        scorer = LLMJudgeScorer()
        result = await scorer.score("Senior Data Engineer", None)
        assert result.score == 0.0
