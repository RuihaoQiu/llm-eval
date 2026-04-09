# Run: uv run pytest evals/component/test_router.py -v  (requires OPENAI_API_KEY)
#
# Abstention/calibration eval: on sparse or vague descriptions the model should
# return null/empty rather than hallucinate plausible-sounding values.
from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_eval.eval.dataset import load_golden_set
from llm_eval.eval.runner import run_eval
from llm_eval.schemas import EvalReport

GOLDEN_SET_PATH = Path(__file__).parents[2] / "data" / "golden_set.jsonl"
MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")

# Sparse examples deliberately have null/empty expected values to test that the
# model returns null rather than hallucinating a plausible answer.
SPARSE_CATEGORY = "sparse_description"

# Threshold for how many sparse examples must pass — we allow one miss because
# some descriptions (e.g. ex_046) are genuinely ambiguous about work_mode.
SPARSE_PASS_THRESHOLD = 0.7


@pytest.fixture(scope="session")
async def router_report() -> EvalReport:
    all_examples = load_golden_set(GOLDEN_SET_PATH)
    sparse = [ex for ex in all_examples if ex.category == SPARSE_CATEGORY]
    return await run_eval(sparse, model=MODEL)


async def test_sparse_overall_pass_rate(router_report: EvalReport) -> None:
    assert router_report.pass_rate >= SPARSE_PASS_THRESHOLD, (
        f"Sparse pass rate {router_report.pass_rate:.1%} is below {SPARSE_PASS_THRESHOLD:.1%} — "
        f"model may be hallucinating on vague descriptions"
    )


async def test_no_hallucinated_location(router_report: EvalReport) -> None:
    # Examples with expected location=null should never return a non-null location.
    # A non-null prediction on a null-expected location gets score 0 (null mismatch).
    hallucinations = [
        ex.example_id
        for ex in router_report.examples
        for fr in ex.field_results
        if fr.field == "location" and fr.result.score == 0.0
    ]
    assert not hallucinations, (
        f"Possible location hallucination in: {hallucinations}"
    )


async def test_no_hallucinated_skills(router_report: EvalReport) -> None:
    # Examples with expected skills=[] should score above 0 on skills.
    # Score of 0 means the model predicted skills where none were expected (false positives).
    hallucinations = [
        ex.example_id
        for ex in router_report.examples
        for fr in ex.field_results
        if fr.field == "skills" and fr.result.score == 0.0
    ]
    assert not hallucinations, (
        f"Possible skills hallucination in: {hallucinations}"
    )
