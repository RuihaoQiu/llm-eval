"""Pydantic models shared across the eval framework."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class JobInfo(BaseModel):
    """Structured output extracted from a raw job posting."""

    title: str | None = None
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "executive"] | None = None
    work_mode: Literal["remote", "hybrid", "onsite"] | None = None
    location: str | None = None
    skills: list[str] = []


class GoldenExample(BaseModel):
    """One labelled example in the golden dataset."""

    id: str
    category: str
    input: dict[str, str]
    expected: JobInfo
    notes: str | None = None


class ScorerResult(BaseModel):
    """Continuous result from a single scorer."""

    score: float  # 0.0–1.0
    passed: bool
    reason: str | None = None


class FieldResult(BaseModel):
    """Score for one field on one example."""

    field: str
    scorer: str
    result: ScorerResult


class ExampleReport(BaseModel):
    """Aggregated scores for one golden example."""

    example_id: str
    field_results: list[FieldResult]
    overall_score: float
    passed: bool


class EvalReport(BaseModel):
    """Top-level report for a full eval run."""

    model: str
    examples: list[ExampleReport]
    mean_score: float
    pass_rate: float
