"""Embedding-based scorers using OpenAI embeddings API."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
from openai import AsyncOpenAI

from llm_eval.schemas import ScorerResult

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"
_client = AsyncOpenAI()
_embedding_cache: dict[str, list[float]] = {}


async def _get_embedding(text: str) -> list[float]:
    key = hashlib.md5(text.encode()).hexdigest()
    if key not in _embedding_cache:
        resp = await _client.embeddings.create(input=text, model=_EMBEDDING_MODEL)
        _embedding_cache[key] = resp.data[0].embedding
        logger.debug("Fetched embedding for %r (cache size=%d)", text[:40], len(_embedding_cache))
    return _embedding_cache[key]


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _soft_recall(source_embs: list[list[float]], target_embs: list[list[float]]) -> float:
    scores = [max(_cosine(s, t) for t in target_embs) for s in source_embs]
    return sum(scores) / len(scores)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class EmbeddingF1Scorer:
    """Soft F1 for set-valued fields; precision and recall via mean best-match cosine similarity.

    Attributes:
        threshold: Minimum F1 score to count as passed.
    """

    name: str = "embedding_f1"
    threshold: float = 0.5

    async def score(self, expected: list[str], actual: list[str]) -> ScorerResult:
        if not expected and not actual:
            return ScorerResult(score=1.0, passed=True)
        if not expected:
            return ScorerResult(score=0.0, passed=False, reason="expected empty, actual non-empty")
        if not actual:
            return ScorerResult(score=0.0, passed=False, reason="actual empty, expected non-empty")

        exp_embs = [await _get_embedding(s) for s in expected]
        act_embs = [await _get_embedding(s) for s in actual]

        recall = _soft_recall(exp_embs, act_embs)
        precision = _soft_recall(act_embs, exp_embs)
        f1 = _f1(precision, recall)

        logger.debug("EmbeddingF1: precision=%.2f recall=%.2f f1=%.2f", precision, recall, f1)
        return ScorerResult(
            score=f1,
            passed=f1 >= self.threshold,
            reason=f"precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}",
        )


@dataclass
class EmbeddingScorer:
    """Cosine similarity scorer for single-value text fields.

    Attributes:
        threshold: Minimum cosine similarity to count as passed.
    """

    name: str = "embedding_similarity"
    threshold: float = 0.8

    async def score(self, expected: str | None, actual: str | None) -> ScorerResult:
        if expected is None and actual is None:
            return ScorerResult(score=1.0, passed=True)
        if expected is None or actual is None:
            return ScorerResult(score=0.0, passed=False, reason=f"expected={expected!r}, actual={actual!r}")
        exp_emb = await _get_embedding(expected)
        act_emb = await _get_embedding(actual)
        # Clamp to [0, 1] — cosine can be slightly negative for unrelated texts.
        score = max(0.0, _cosine(exp_emb, act_emb))
        logger.debug("EmbeddingSim: expected=%r actual=%r score=%.3f", expected, actual, score)
        return ScorerResult(score=score, passed=score >= self.threshold, reason=f"cosine={score:.3f}")
