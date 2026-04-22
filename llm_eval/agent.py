"""Single-step job-info extractor via OpenAI structured output."""

from __future__ import annotations

import logging
import time

import logfire
from openai import AsyncOpenAI

from llm_eval.schemas import ExtractionResult, JobInfo

logger = logging.getLogger(__name__)

_client = AsyncOpenAI()

_SYSTEM_PROMPT = (
    "You are a job-posting parser. Extract structured information from the posting. "
    "Set any field to null if it cannot be reliably determined from the text. "
    "For skills, list only concrete technical skills or tools — not soft skills."
)


async def extract_job_info(
    raw_description: str,
    raw_title: str,
    model: str = "gpt-4o-mini",
    job_id: str = "unknown",
) -> ExtractionResult:
    with logfire.span("extract_job_info", job_id=job_id, model=model):
        logger.debug("Extracting job info model=%s job_id=%s title=%.50s", model, job_id, raw_title)
        t0 = time.monotonic()
        response = await _client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Title: {raw_title}\n\nDescription:\n{raw_description}"},
            ],
            response_format=JobInfo,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        result = response.choices[0].message.parsed
        usage = response.usage
        total_tokens = usage.total_tokens if usage else 0
        logfire.info("extraction complete", job_id=job_id, result=result.model_dump())
        return ExtractionResult(
            job_info=result,
            latency_ms=latency_ms,
            total_tokens=total_tokens,
        )
