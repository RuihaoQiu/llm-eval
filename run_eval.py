import asyncio
import logging
from pathlib import Path

import logfire

from llm_eval.eval.dataset import load_golden_set
from llm_eval.eval.runner import run_eval
from llm_eval.trace import configure_logfire

logging.basicConfig(level=logging.INFO)

configure_logfire()

async def main() -> None:
    examples = load_golden_set(Path("data/golden_set.jsonl"))
    report = await run_eval(examples)
    print(f"mean={report.mean_score:.2f} pass_rate={report.pass_rate:.0%}")

asyncio.run(main())
logfire.force_flush()
