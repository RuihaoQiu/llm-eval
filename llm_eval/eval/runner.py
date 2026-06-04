"""Async eval loop: runs examples through the agent and scores them."""

from __future__ import annotations

import asyncio
import logging

import logfire

from llm_eval.agent import extract_job_info
from llm_eval.schemas import EvalReport, ExampleReport, FieldResult, GoldenExample, JobInfo
from llm_eval.scoring.deterministic import ExactMatchScorer, FuzzyMatchScorer
from llm_eval.scoring.llm_judge import SKILLS_RUBRIC, LLMJudgeScorer

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.7
DEFAULT_JUDGE_MODEL = "gpt-4o"

_SYNC_SCORERS = {
    "seniority": ExactMatchScorer(),
    "work_mode": ExactMatchScorer(),
    "location": FuzzyMatchScorer(threshold=0.8),
}


def _make_judges(model: str) -> tuple[LLMJudgeScorer, LLMJudgeScorer]:
    return (
        LLMJudgeScorer(model=model),
        LLMJudgeScorer(name="llm_judge_skills", rubric=SKILLS_RUBRIC, model=model),
    )


async def _score_all_fields(
    expected: JobInfo,
    actual: JobInfo,
    title_judge: LLMJudgeScorer,
    skills_judge: LLMJudgeScorer,
) -> list[FieldResult]:
    results: list[FieldResult] = []
    for field_name, scorer in _SYNC_SCORERS.items():
        result = scorer.score(getattr(expected, field_name), getattr(actual, field_name))
        results.append(FieldResult(field=field_name, scorer=scorer.name, result=result))
    title_result = await title_judge.score(expected.title, actual.title)
    results.append(FieldResult(field="title", scorer=title_judge.name, result=title_result))
    skills_result = await skills_judge.score(expected.skills, actual.skills)
    results.append(FieldResult(field="skills", scorer=skills_judge.name, result=skills_result))
    return results


def _compute_overall_score(field_results: list[FieldResult]) -> float:
    """First result wins when a field has two entries (e.g. title: embedding + judge)."""
    primary: dict[str, float] = {}
    for fr in field_results:
        primary.setdefault(fr.field, fr.result.score)
    return sum(primary.values()) / len(primary)


async def _run_example(
    example: GoldenExample,
    model: str,
    title_judge: LLMJudgeScorer,
    skills_judge: LLMJudgeScorer,
) -> ExampleReport:
    with logfire.span("score_example", example_id=example.id):
        extraction = await extract_job_info(
            raw_description=example.input["raw_description"],
            raw_title=example.input["raw_title"],
            model=model,
            job_id=example.id,
        )
        field_results = await _score_all_fields(
            example.expected, extraction.job_info, title_judge, skills_judge,
        )
        overall = _compute_overall_score(field_results)
        passed = overall >= PASS_THRESHOLD
        logfire.info("example scored", example_id=example.id, overall=round(overall, 3), passed=passed)
        logger.info("Example %s overall=%.2f passed=%s", example.id, overall, passed)
        return ExampleReport(
            example_id=example.id,
            field_results=field_results,
            overall_score=overall,
            passed=passed,
            latency_ms=extraction.latency_ms,
            total_tokens=extraction.total_tokens,
        )


async def run_eval(
    examples: list[GoldenExample],
    model: str = "gpt-4o-mini",
    judge_model: str = DEFAULT_JUDGE_MODEL,
    concurrency: int = 5,
) -> EvalReport:
    title_judge, skills_judge = _make_judges(judge_model)
    sem = asyncio.Semaphore(concurrency)

    async def bounded(ex: GoldenExample) -> ExampleReport:
        async with sem:
            return await _run_example(ex, model, title_judge, skills_judge)

    with logfire.span("eval_run", model=model, n_examples=len(examples)):
        reports = await asyncio.gather(*[bounded(ex) for ex in examples])

    mean_score = sum(r.overall_score for r in reports) / len(reports)
    pass_rate = sum(1 for r in reports if r.passed) / len(reports)
    mean_latency = sum(r.latency_ms for r in reports) / len(reports)
    total_tokens = sum(r.total_tokens for r in reports)
    logger.info("Eval complete model=%s mean=%.2f pass_rate=%.2f", model, mean_score, pass_rate)
    return EvalReport(
        model=model,
        examples=list(reports),
        mean_score=mean_score,
        pass_rate=pass_rate,
        mean_latency_ms=mean_latency,
        total_tokens=total_tokens,
    )
