from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_eval.scoring.llm_judge import JudgeVerdict, LLMJudgeScorer


def _make_judge_response(score: int, reasoning: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(parsed=JudgeVerdict(score=score, reasoning=reasoning)))]
    return resp


class TestLLMJudgeScorer:
    @pytest.mark.asyncio
    async def test_score_2_maps_to_1(self) -> None:
        with patch(
            "llm_eval.scoring.llm_judge._client.beta.chat.completions.parse",
            new=AsyncMock(return_value=_make_judge_response(2, "exact match")),
        ):
            result = await LLMJudgeScorer().score("Senior Data Engineer", "Data Engineer – Senior")
        assert result.score == 1.0
        assert result.passed

    @pytest.mark.asyncio
    async def test_score_1_maps_to_half(self) -> None:
        with patch(
            "llm_eval.scoring.llm_judge._client.beta.chat.completions.parse",
            new=AsyncMock(return_value=_make_judge_response(1, "related but different")),
        ):
            result = await LLMJudgeScorer().score("Data Engineer", "Data Analyst")
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_score_0_maps_to_zero(self) -> None:
        with patch(
            "llm_eval.scoring.llm_judge._client.beta.chat.completions.parse",
            new=AsyncMock(return_value=_make_judge_response(0, "unrelated")),
        ):
            result = await LLMJudgeScorer().score("Data Engineer", "Graphic Designer")
        assert result.score == 0.0
        assert not result.passed

    @pytest.mark.asyncio
    async def test_cache_avoids_second_call(self) -> None:
        mock_parse = AsyncMock(return_value=_make_judge_response(2, "cached"))
        with patch("llm_eval.scoring.llm_judge._client.beta.chat.completions.parse", new=mock_parse):
            scorer = LLMJudgeScorer()
            await scorer.score("A", "B")
            await scorer.score("A", "B")
        assert mock_parse.call_count == 1

    @pytest.mark.asyncio
    async def test_both_none(self) -> None:
        result = await LLMJudgeScorer().score(None, None)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_one_none(self) -> None:
        result = await LLMJudgeScorer().score("Senior Data Engineer", None)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_list_inputs(self) -> None:
        with patch(
            "llm_eval.scoring.llm_judge._client.beta.chat.completions.parse",
            new=AsyncMock(return_value=_make_judge_response(2, "same skills")),
        ):
            result = await LLMJudgeScorer().score(["Python", "Spark"], ["Python", "Apache Spark"])
        assert result.score == 1.0
        assert result.passed

    @pytest.mark.asyncio
    async def test_empty_lists_both(self) -> None:
        result = await LLMJudgeScorer().score([], [])
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_empty_list_one_side(self) -> None:
        result = await LLMJudgeScorer().score(["Python"], [])
        assert result.score == 0.0
        assert not result.passed
