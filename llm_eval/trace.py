from __future__ import annotations

import logging

import logfire

logger = logging.getLogger(__name__)


def configure_logfire() -> None:
    logfire.configure(service_name="llm-eval", console=False)
    logfire.instrument_openai()
    logger.info("Logfire configured")
