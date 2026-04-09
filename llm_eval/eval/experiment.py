from __future__ import annotations

import json
from pathlib import Path

from llm_eval.schemas import EvalReport


FIELDS = ["title", "seniority", "work_mode", "location", "skills"]


def load_experiment(path: Path) -> EvalReport:
    return EvalReport.model_validate_json(path.read_text())


def _field_mean(report: EvalReport, field: str) -> float:
    scores: list[float] = []
    for ex in report.examples:
        seen: set[str] = set()
        for fr in ex.field_results:
            if fr.field == field and fr.field not in seen:
                scores.append(fr.result.score)
                seen.add(fr.field)
    return sum(scores) / len(scores) if scores else 0.0


def _field_pass_rate(report: EvalReport, field: str) -> float:
    passed = 0
    total = 0
    for ex in report.examples:
        seen: set[str] = set()
        for fr in ex.field_results:
            if fr.field == field and fr.field not in seen:
                total += 1
                passed += 1 if fr.result.score >= 0.5 else 0
                seen.add(fr.field)
    return passed / total if total else 0.0


class FieldStats(dict):
    """Per-field stats: mean_score, pass_rate, delta_mean, delta_pass_rate."""


def compare_experiments(
    baseline: EvalReport, candidate: EvalReport
) -> list[dict[str, object]]:
    rows = []
    for field in FIELDS:
        base_mean = _field_mean(baseline, field)
        cand_mean = _field_mean(candidate, field)
        base_pr = _field_pass_rate(baseline, field)
        cand_pr = _field_pass_rate(candidate, field)
        rows.append(
            {
                "field": field,
                "baseline_mean": base_mean,
                "candidate_mean": cand_mean,
                "delta_mean": cand_mean - base_mean,
                "baseline_pr": base_pr,
                "candidate_pr": cand_pr,
                "delta_pr": cand_pr - base_pr,
            }
        )
    rows.append(
        {
            "field": "overall",
            "baseline_mean": baseline.mean_score,
            "candidate_mean": candidate.mean_score,
            "delta_mean": candidate.mean_score - baseline.mean_score,
            "baseline_pr": baseline.pass_rate,
            "candidate_pr": candidate.pass_rate,
            "delta_pr": candidate.pass_rate - baseline.pass_rate,
        }
    )
    return rows
