"""Load and validate the golden dataset from JSONL."""

from __future__ import annotations

import logging
from pathlib import Path

from llm_eval.schemas import GoldenExample

logger = logging.getLogger(__name__)


def load_golden_set(path: Path) -> list[GoldenExample]:
    examples: list[GoldenExample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(GoldenExample.model_validate_json(line))
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples
