"""Async eval loop: runs examples through the agent and scores them."""

from __future__ import annotations

import asyncio
import logging

import logfire

from llm_eval.agent import extract_job_info
from llm_eval.schemas import (
    EvalReport,
    ExampleReport,
    FieldResult,
    GoldenExample,
)
from llm_eval.scoring.deterministic import ExactMatchScorer, FuzzyMatchScorer
from llm_eval.scoring.embedding import EmbeddingF1Scorer, EmbeddingScorer
from llm_eval.scoring.llm_judge import LLMJudgeScorer

logger = logging.getLogger(__name__)

# Sync scorers for deterministic fields.
_SYNC_SCORERS = {
    "seniority": ExactMatchScorer(),
    "work_mode": ExactMatchScorer(),
    "location": FuzzyMatchScorer(threshold=0.8),
}

# Async scorers for semantic fields.
_title_embedding = EmbeddingScorer(threshold=0.8)
_title_judge = LLMJudgeScorer()
_skills_scorer = EmbeddingF1Scorer(threshold=0.5)


async def _score_title(expected: str | None, actual: str | None) -> list[FieldResult]:
    """Score the title field: embedding similarity, with LLM judge on failures.

    Args:
        expected: Ground-truth title.
        actual: Agent-produced title.

    Returns:
        One or two FieldResult entries (embedding always; judge only on failure).
    """
    emb_result = await _title_embedding.score(expected, actual)
    results = [FieldResult(field="title", scorer=_title_embedding.name, result=emb_result)]

    if not emb_result.passed:
        judge_result = await _title_judge.score(expected, actual)
        results.append(FieldResult(field="title", scorer=_title_judge.name, result=judge_result))

    return results


async def _run_example(example: GoldenExample, model: str) -> ExampleReport:
    """Run the agent on one example and score all fields.

    Args:
        example: The golden example to evaluate.
        model: OpenAI model identifier.

    Returns:
        ExampleReport with per-field scores.
    """
    with logfire.span("score_example", example_id=example.id):
        actual = await extract_job_info(
            raw_description=example.input["raw_description"],
            raw_title=example.input["raw_title"],
            model=model,
            job_id=example.id,
        )

        field_results: list[FieldResult] = []

        # Deterministic fields (sync).
        for field_name, scorer in _SYNC_SCORERS.items():
            result = scorer.score(getattr(example.expected, field_name), getattr(actual, field_name))
            field_results.append(FieldResult(field=field_name, scorer=scorer.name, result=result))

        # Title: embedding + optional LLM judge.
        field_results.extend(await _score_title(example.expected.title, actual.title))

        # Skills: soft embedding F1.
        skills_result = await _skills_scorer.score(example.expected.skills, actual.skills)
        field_results.append(FieldResult(field="skills", scorer=_skills_scorer.name, result=skills_result))

        # Overall: average the primary score per field (one per field name).
        primary: dict[str, float] = {}
        for fr in field_results:
            primary.setdefault(fr.field, fr.result.score)
        overall = sum(primary.values()) / len(primary)
        passed = overall >= 0.7

        logfire.info("example scored", example_id=example.id, overall=round(overall, 3), passed=passed)
        logger.info("Example %s overall=%.2f passed=%s", example.id, overall, passed)
        return ExampleReport(
            example_id=example.id,
            field_results=field_results,
            overall_score=overall,
            passed=passed,
        )


async def run_eval(
    examples: list[GoldenExample],
    model: str = "gpt-4o-mini",
    concurrency: int = 5,
) -> EvalReport:
    """Run the full eval loop over a list of golden examples.

    Args:
        examples: Golden examples to evaluate.
        model: OpenAI model identifier.
        concurrency: Max parallel API calls.

    Returns:
        EvalReport with per-example and aggregate scores.
    """
    sem = asyncio.Semaphore(concurrency)

    async def bounded(ex: GoldenExample) -> ExampleReport:
        async with sem:
            return await _run_example(ex, model)

    with logfire.span("eval_run", model=model, n_examples=len(examples)):
        reports = await asyncio.gather(*[bounded(ex) for ex in examples])

    mean_score = sum(r.overall_score for r in reports) / len(reports)
    pass_rate = sum(1 for r in reports if r.passed) / len(reports)
    logger.info("Eval complete model=%s mean=%.2f pass_rate=%.2f", model, mean_score, pass_rate)
    return EvalReport(
        model=model,
        examples=list(reports),
        mean_score=mean_score,
        pass_rate=pass_rate,
    )
