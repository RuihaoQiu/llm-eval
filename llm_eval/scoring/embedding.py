"""Embedding-based scorers using OpenAI embeddings API."""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from openai import AsyncOpenAI

from llm_eval.schemas import ScorerResult

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"
_client = AsyncOpenAI()

# Module-level cache: text → embedding vector.
_embedding_cache: dict[str, list[float]] = {}


async def _get_embedding(text: str) -> list[float]:
    """Fetch an embedding, using an in-process cache to avoid duplicate calls.

    Args:
        text: Input text to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    key = hashlib.md5(text.encode()).hexdigest()
    if key not in _embedding_cache:
        resp = await _client.embeddings.create(input=text, model=_EMBEDDING_MODEL)
        _embedding_cache[key] = resp.data[0].embedding
        logger.debug("Fetched embedding for %r (cache size=%d)", text[:40], len(_embedding_cache))
    return _embedding_cache[key]


def _cosine(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1]; 0.0 if either vector is zero.
    """
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


@dataclass
class EmbeddingF1Scorer:
    """Soft F1 over set-valued fields using embedding cosine similarity.

    For each item in the expected set, the best cosine similarity against any
    item in the actual set is used as the recall contribution, and vice versa
    for precision. F1 is the harmonic mean of soft precision and recall.

    Attributes:
        name: Scorer identifier used in reports.
        threshold: Minimum F1 score to count as passed.
        sim_threshold: Minimum cosine similarity to count a pair as a match for
            the binary pass/fail breakdown (not used in the continuous score).
    """

    name: str = "embedding_f1"
    threshold: float = 0.5
    sim_threshold: float = 0.7

    async def score(self, expected: list[str], actual: list[str]) -> ScorerResult:
        """Compute soft F1 between two skill lists.

        Args:
            expected: Ground-truth list of skill strings.
            actual: Agent-produced list of skill strings.

        Returns:
            ScorerResult with score equal to the soft F1 value.
        """
        if not expected and not actual:
            return ScorerResult(score=1.0, passed=True)
        if not expected:
            return ScorerResult(score=0.0, passed=False, reason="expected empty, actual non-empty")
        if not actual:
            return ScorerResult(score=0.0, passed=False, reason="actual empty, expected non-empty")

        exp_embs = [await _get_embedding(s) for s in expected]
        act_embs = [await _get_embedding(s) for s in actual]

        # Recall: for each expected item, max similarity to any actual item.
        recall_scores = [max(_cosine(e, a) for a in act_embs) for e in exp_embs]
        recall = sum(recall_scores) / len(recall_scores)

        # Precision: for each actual item, max similarity to any expected item.
        precision_scores = [max(_cosine(a, e) for e in exp_embs) for a in act_embs]
        precision = sum(precision_scores) / len(precision_scores)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        passed = f1 >= self.threshold
        logger.debug(
            "EmbeddingF1: precision=%.2f recall=%.2f f1=%.2f", precision, recall, f1
        )
        return ScorerResult(
            score=f1,
            passed=passed,
            reason=f"precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}",
        )


@dataclass
class EmbeddingScorer:
    """Single-value embedding cosine similarity scorer. Used for open-ended text fields.

    Attributes:
        name: Scorer identifier used in reports.
        threshold: Minimum cosine similarity to count as passed.
    """

    name: str = "embedding_similarity"
    threshold: float = 0.8

    async def score(self, expected: str | None, actual: str | None) -> ScorerResult:
        """Compute cosine similarity between two strings.

        Args:
            expected: Ground-truth string (or None).
            actual: Agent-produced string (or None).

        Returns:
            ScorerResult with score equal to cosine similarity.
        """
        if expected is None and actual is None:
            return ScorerResult(score=1.0, passed=True)
        if expected is None or actual is None:
            return ScorerResult(
                score=0.0,
                passed=False,
                reason=f"expected={expected!r}, actual={actual!r}",
            )
        exp_emb = await _get_embedding(expected)
        act_emb = await _get_embedding(actual)
        sim = _cosine(exp_emb, act_emb)
        # Clamp to [0, 1] — cosine can be slightly negative for unrelated texts.
        score = max(0.0, sim)
        passed = score >= self.threshold
        logger.debug("EmbeddingSim: expected=%r actual=%r sim=%.3f", expected, actual, sim)
        return ScorerResult(score=score, passed=passed, reason=f"cosine={sim:.3f}")
