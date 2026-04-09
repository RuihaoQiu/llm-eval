"""Load and validate the golden dataset from JSONL."""

from __future__ import annotations

import logging
from pathlib import Path

from llm_eval.schemas import GoldenExample

logger = logging.getLogger(__name__)


def load_golden_set(path: Path) -> list[GoldenExample]:
    """Load golden examples from a JSONL file.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of validated ``GoldenExample`` instances.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValidationError: If any line fails Pydantic validation.
    """
    examples: list[GoldenExample] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            example = GoldenExample.model_validate_json(line)
            examples.append(example)
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples
