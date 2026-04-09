from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class JobInfo(BaseModel):
    title: str | None = None
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "executive"] | None = None
    work_mode: Literal["remote", "hybrid", "onsite"] | None = None
    location: str | None = None
    skills: list[str] = []


class GoldenExample(BaseModel):
    id: str
    category: str
    input: dict[str, str]
    expected: JobInfo
    notes: str | None = None


class ScorerResult(BaseModel):
    score: float  # 0.0–1.0
    passed: bool
    reason: str | None = None


class FieldResult(BaseModel):
    field: str
    scorer: str
    result: ScorerResult


class ExampleReport(BaseModel):
    example_id: str
    field_results: list[FieldResult]
    overall_score: float
    passed: bool


class EvalReport(BaseModel):
    model: str
    examples: list[ExampleReport]
    mean_score: float
    pass_rate: float
