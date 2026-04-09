# LLM Eval Framework — Implementation Plan

## Goal

A clean, public portfolio piece demonstrating rigorous LLM pipeline evaluation on the job-extraction use case. The eval framework is the point — the job-extraction task is the domain.

One framing sentence: "Rigorous LLM evaluation requires choosing the right measurement for each output type, questioning your judge, and treating eval results as versioned artifacts — not one-time metrics."

---

## Agent Scope

Simplified single-step extractor: raw job description → structured `JobInfo`. Five fields, chosen to cover all scorer types:

| Field | Type | Scorer |
|---|---|---|
| `title` | str | Embedding similarity + LLM judge |
| `seniority` | enum | Exact match |
| `work_mode` | enum | Exact match |
| `location` | str | Fuzzy match |
| `skills` | list[str] | Embedding F1 |

---

## Repo Structure

```
llm-eval/
├── llm_eval/
│   ├── agent.py               # Single-step extractor (openai structured output)
│   ├── schemas.py             # Pydantic models: JobInfo, GoldenExample, EvalReport
│   ├── trace.py               # Logfire tracing wrapper
│   ├── scoring/
│   │   ├── base.py            # Scorer protocol + ScorerResult (continuous 0–1)
│   │   ├── deterministic.py   # ExactMatchScorer, FuzzyMatchScorer, NumericScorer
│   │   ├── embedding.py       # EmbeddingF1Scorer for set-valued fields (skills)
│   │   └── llm_judge.py       # LLMJudgeScorer with structured output + result caching
│   └── eval/
│       ├── runner.py          # Async eval loop (semaphore, progress, tracing)
│       ├── dataset.py         # Load/validate golden set from JSONL
│       └── experiment.py      # ExperimentConfig, ExperimentResult, compare_experiments()
├── evals/
│   ├── component/
│   │   ├── test_extraction.py # Field-level accuracy per component (skill eval)
│   │   └── test_router.py     # Abstention / calibration: does the model return null vs hallucinate?
│   ├── trajectory/
│   │   └── test_trajectory.py # Multi-step trace scoring (retry-on-failure loop)
│   └── judge/
│       └── test_judge.py      # Meta-eval: judge consistency (variance < 0.2 across 3 runs)
├── data/
│   ├── golden_set.jsonl       # 50 hand-curated synthetic examples
│   ├── golden_set_schema.json # JSON Schema for the golden set format
│   └── experiments/           # Pre-run results JSON (version-controlled artifacts)
├── notebooks/
│   ├── 01_trace_explorer.py   # Marimo: visualize a single trace span-by-span
│   ├── 02_eval_report.py      # Marimo: per-field accuracy, failure breakdown
│   └── 03_model_comparison.py # Marimo: side-by-side experiment comparison
├── docs/
│   ├── plan.md                # This file
│   ├── design.md              # Why each scoring strategy was chosen per field
│   └── scoring.md             # Field-level scoring table
├── tests/
│   ├── unit/
│   │   ├── test_scoring.py    # Pure unit tests — no LLM calls
│   │   └── test_dataset.py
│   └── integration/
│       └── test_runner.py     # 3-example subset, requires API key
├── pyproject.toml
├── .python-version
├── .env.example
└── README.md
```

---

## Scoring Design

### Key principle
Every scorer returns a continuous `ScorerResult(score: float, passed: bool, reason: str | None)` — not just pass/fail. This enables partial credit, delta tables across experiments, and richer failure analysis.

### Scorer per field type

| Pattern | Scorer | Reason |
|---|---|---|
| Enum fields (seniority, work_mode) | `ExactMatchScorer` | No partial credit for wrong enum values |
| Text fields (location) | `FuzzyMatchScorer` | Handles transliteration, punctuation, abbreviations |
| Numeric fields (salary) | `NumericScorer` | Allows small tolerance band |
| Set-valued fields (skills) | `EmbeddingF1Scorer` | Order-invariant, semantic, computes precision/recall/F1 |
| Open-ended text (title) | Embedding similarity + `LLMJudgeScorer` | String similarity first, LLM judge on failures |

### LLM Judge
- Structured output: `JudgeVerdict(score: Literal[0, 1, 2], reasoning: str)`
- Score mapping: 0→0.0, 1→0.5, 2→1.0
- Result caching by hash(rubric + expected + actual) to avoid re-spending tokens
- Consistency meta-eval in `evals/judge/test_judge.py`: run 3x on same pair, assert variance < 0.2

---

## Golden Set Design

50 synthetic examples, public-safe (no real job IDs, no scraped URLs, paraphrased descriptions).

| Category | Count | What it tests |
|---|---|---|
| Clear seniority signal | 8 | "5+ years experience" → mid |
| Ambiguous/conflicting seniority | 4 | Title says Junior, desc implies senior |
| Multi-location | 4 | Location parsing degrades gracefully |
| Remote-only roles | 4 | work_mode recall |
| Non-English titles (translated) | 6 | Robustness without multilingual complexity |
| Dense skills list (20+ skills) | 6 | Skills F1 regression target |
| Sparse/vague descriptions | 6 | Does model hallucinate nulls correctly |
| Executive roles | 4 | Seniority ceiling behavior |
| Internships | 4 | Edge of employment type enum |
| Salary in unusual format | 4 | Numeric parsing edge cases |

Format: JSONL, one record per line:
```json
{
  "id": "ex_001",
  "category": "clear_seniority",
  "input": {"raw_description": "...", "raw_title": "..."},
  "expected": {
    "title": "Data Engineer",
    "seniority": "mid",
    "work_mode": "hybrid",
    "location": "Berlin, Germany",
    "skills": ["Python", "Spark", "Airflow", "dbt"]
  },
  "notes": "5 years experience in description, hybrid explicitly stated"
}
```

---

## Build Phases

### Phase 1 — Foundation ✅
*Goal: runnable eval end-to-end against 10 examples*

- [x] `pyproject.toml` — uv project, deps: `openai`, `pydantic`, `logfire`, `rapidfuzz`, `numpy`
- [x] `llm_eval/schemas.py` — `JobInfo`, `GoldenExample`, `ScorerResult`, `FieldResult`, `ExampleReport`, `EvalReport`
- [x] `llm_eval/agent.py` — single-call extractor via `openai.beta.chat.completions.parse()`
- [x] `llm_eval/scoring/base.py` — `Scorer` and `AsyncScorer` protocols
- [x] `llm_eval/scoring/deterministic.py` — `ExactMatchScorer`, `FuzzyMatchScorer`
- [x] `data/golden_set.jsonl` — first 10 examples across 5 categories
- [x] `llm_eval/eval/dataset.py` — JSONL loader
- [x] `llm_eval/eval/runner.py` — async loop with semaphore-bounded concurrency
- [x] `evals/component/test_extraction.py` — pytest, session-scoped fixture, asserts pass rate >= 0.7
- [x] `tests/unit/test_scoring.py` — unit tests for deterministic scorers (no API calls)

Checkpoint: `uv run pytest evals/component/ -v` passes. ✅

### Phase 2 — Tracing + LLM Judge ✅
*Goal: observable pipeline, richer scoring*

- [x] `llm_eval/trace.py` — `configure_logfire()`, instruments OpenAI automatically
- [x] Tracing added to `agent.py` and `runner.py` via `logfire.span()`
- [x] `evals/conftest.py` — calls `configure_logfire()` + `logfire.force_flush()` for pytest
- [x] `run_eval.py` — standalone script entry point with Logfire tracing
- [x] `llm_eval/scoring/llm_judge.py` — `LLMJudgeScorer` with structured output and result caching
- [x] `llm_eval/scoring/embedding.py` — `EmbeddingScorer` (cosine sim) + `EmbeddingF1Scorer` (soft F1)
- [x] Title scoring: embedding similarity first, LLM judge only on failure
- [x] `evals/judge/test_judge.py` — judge consistency meta-eval (variance < 0.2 across 3 runs)
- [x] `tests/unit/test_scoring_async.py` — unit tests for async scorers (mocked API calls)
- [x] `notebooks/02_eval_report.py` — Marimo notebook: per-field scores, bar chart, distribution, failures
- [ ] Finish all 50 golden examples
- [ ] Rich terminal report from `run_eval.py`

Checkpoint: traces visible in Logfire dashboard at https://logfire-eu.pydantic.dev/qiuruihao/llm-eval ✅

### Phase 3 — Experiments + Model Comparison ✅
*Goal: the "so what" layer — make results diff-able across model runs*

- [x] `llm_eval/eval/experiment.py` — `load_experiment()`, `compare_experiments()` returns a delta table per field
- [x] `run_eval.py` — `--model` flag and `_save_artifact()` saving to `data/experiments/<model>_<timestamp>.json`
- [x] Run real comparison: `gpt-4o-mini` (mean 0.87, 92% pass) vs `gpt-4.1-mini` (mean 0.88, 96% pass)
- [x] `notebooks/03_model_comparison.py` — Marimo: side-by-side delta table + bar charts, regressions in red
- [x] `data/experiments/` — version-controlled JSON artifacts so notebooks render without API keys

Checkpoint: `uv run python run_eval.py --model gpt-4.1-mini` produces a versioned JSON artifact; `notebooks/03_model_comparison.py` renders the delta table offline. ✅

### Phase 4 — Polish + Publish ✅
*Goal: ready to share as a portfolio piece*

- [x] `README.md` — results table (mean score, pass rate per model), architecture diagram, quickstart
- [x] `docs/design.md` — scoring strategy rationale per field
- [x] `data/golden_set.jsonl` — 50 examples across 10 categories
- [x] `evals/component/test_router.py` — abstention/calibration eval: null vs hallucinate on sparse descriptions
- [x] `.env.example` — documents OPENAI_API_KEY, LOGFIRE_TOKEN, EVAL_MODEL

---

## What Makes This Stand Out

1. **Field-stratified golden set** — design and annotation rationale are visible in the `notes` field
2. **Scoring diversity** — right metric for each field type, rationale documented in `docs/design.md`
3. **The judge is evaluated** — `evals/judge/` checks consistency (most projects skip this)
4. **Experiments are version-controlled artifacts** — diff-able JSON, notebooks render without API keys
