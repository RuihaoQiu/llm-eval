"""Logfire tracing configuration and helpers."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

import logfire

logger = logging.getLogger(__name__)


def configure_logfire(send_to_logfire: bool | None = None) -> None:
    """Configure Logfire for the eval framework.

    By default, telemetry is sent to Logfire only when the ``LOGFIRE_TOKEN``
    environment variable is present. Set ``send_to_logfire=False`` to force
    local-only mode regardless of the token.

    Args:
        send_to_logfire: Override the auto-detection of whether to send spans
            to the Logfire cloud backend. ``None`` (default) auto-detects.
    """
    if send_to_logfire is None:
        send_to_logfire = bool(os.getenv("LOGFIRE_TOKEN"))

    logfire.configure(
        service_name="llm-eval",
        send_to_logfire=send_to_logfire,
    )
    logfire.instrument_openai()
    logger.info("Logfire configured (send_to_logfire=%s)", send_to_logfire)


@contextmanager
def agent_span(job_id: str, model: str) -> Generator[None, None, None]:
    """Context manager that wraps a single agent extraction call in a Logfire span.

    Args:
        job_id: Golden example ID (used as a span attribute for filtering).
        model: OpenAI model identifier.

    Yields:
        Nothing; the span is closed on exit.
    """
    with logfire.span("extract_job_info", job_id=job_id, model=model):
        yield


@contextmanager
def eval_span(model: str, n_examples: int) -> Generator[None, None, None]:
    """Context manager that wraps a full eval run in a Logfire span.

    Args:
        model: OpenAI model identifier.
        n_examples: Number of examples being evaluated.

    Yields:
        Nothing; the span is closed on exit.
    """
    with logfire.span("eval_run", model=model, n_examples=n_examples):
        yield
