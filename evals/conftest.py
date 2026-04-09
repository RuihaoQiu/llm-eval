import logfire
import pytest

from llm_eval.trace import configure_logfire

configure_logfire()


@pytest.fixture(scope="session", autouse=True)
def flush_logfire():
    yield
    logfire.force_flush()
