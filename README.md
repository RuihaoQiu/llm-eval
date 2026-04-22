# llm-eval

Rigorous evaluation framework for LLM pipelines, demonstrated on a job-posting extraction use case.

> Rigorous LLM evaluation requires choosing the right measurement for each output type, questioning your judge, and treating eval results as versioned artifacts — not one-time metrics.

---

## What

A structured eval harness that runs a single-step LLM extractor against a hand-curated golden dataset and scores each output field with the appropriate metric.

The extractor takes a raw job posting and returns a structured `JobInfo`:

| Field | Type | Scorer |
|---|---|---|
| `title` | string | LLM judge |
| `seniority` | enum | Exact match |
| `work_mode` | enum | Exact match |
| `location` | string | Fuzzy match |
| `skills` | list[str] | LLM judge (list rubric) |

Every scorer returns a continuous score in `[0, 1]` — not just pass/fail. This enables partial credit, delta tables across experiments, and richer failure analysis.

---

## Why

Most LLM eval setups make the same mistakes:

- **One metric for everything** — using exact match on open-ended text, or LLM-as-judge on enum fields, produces misleading numbers.
- **The judge is never evaluated** — if your judge is inconsistent, your eval is noise. This project includes a consistency meta-eval on the judge itself.
- **Results are ephemeral** — eval outputs live in notebooks or CI logs. Here they are versioned JSON artifacts, diffable across model runs.

---

## How

### Architecture

```
data/golden_set.jsonl
        │
        ▼
 eval/dataset.py          ← load + validate golden examples
        │
        ▼
 eval/runner.py           ← async loop (semaphore, tracing)
        │        │
        ▼        ▼
  agent.py    scoring/    ← field-specific scorers
                ├── deterministic.py   ExactMatch, FuzzyMatch
                └── llm_judge.py       LLMJudgeScorer (cached)
        │
        ▼
   EvalReport             ← per-field scores + aggregate
```

### Scoring strategy

- **Enum fields** (`seniority`, `work_mode`): exact match — no partial credit for wrong enum values.
- **Text fields** (`location`): fuzzy match via RapidFuzz — handles abbreviations, punctuation, transliteration.
- **Set-valued fields** (`skills`): LLM judge with a list-aware rubric — handles abbreviations and synonyms ("k8s" = "Kubernetes"), scores overlap on a 0–2 scale.
- **Open-ended text** (`title`): LLM judge — handles translations, abbreviations, and seniority-modifier differences that string matching misses.

### LLM Judge

- Structured output: `JudgeVerdict(score: Literal[0, 1, 2], reasoning: str)`
- Score mapping: `0 → 0.0`, `1 → 0.5`, `2 → 1.0`
- Separate rubrics for title (semantic equivalence) and skills (list overlap with synonym handling)
- Judge model is configurable independently from extraction model (default: `gpt-4o`) — avoids self-evaluation bias
- Results cached by `hash(rubric + expected + actual)` — identical comparisons are never re-evaluated
- Consistency meta-eval in `evals/judge/`: runs the judge 3× on identical pairs and asserts variance < 0.2

### Golden dataset

50 synthetic examples across 10 categories (see `data/golden_set.jsonl`):

| Category | Count | What it tests |
|---|---|---|
| Clear seniority signal | 8 | "5+ years experience" → mid |
| Ambiguous/conflicting seniority | 4 | Title says Junior, desc implies Senior |
| Multi-location | 4 | Location parsing degrades gracefully |
| Remote-only roles | 4 | work_mode recall |
| Non-English titles (translated) | 6 | Robustness |
| Dense skills list (20+ skills) | 6 | Skills F1 regression target |
| Sparse/vague descriptions | 6 | Does the model hallucinate nulls correctly |
| Executive roles | 4 | Seniority ceiling behaviour |
| Internships | 4 | Edge of employment type enum |
| Salary in unusual format | 4 | Numeric parsing edge cases |

### Results

Latest run: `gpt-4o-mini` extraction, `gpt-4o` judge.

| Model | Mean score | Pass rate | Avg latency | Total tokens |
|---|---|---|---|---|
| `gpt-4o-mini` | 0.90 | 94% (47/50) | 1789ms | 16,224 |

Field-level breakdown:

| Field | Mean score | Pass rate |
|---|---|---|
| title | 1.00 | 100% |
| seniority | 0.92 | 92% |
| work_mode | 0.98 | 98% |
| location | 0.68 | 90% |
| skills | 0.91 | 92% |

Location is the weakest field — fuzzy match degrades on city+country vs city-only mismatches. Title scores perfectly with the LLM judge, which handles translations and reformulations that string matching would miss.

Versioned artifacts in `data/experiments/`. Visualise with `uv run marimo run notebooks/model_comparison.py`.

---

## Quickstart

```bash
# Install dependencies
uv sync

# Run unit tests (no API key required)
uv run pytest tests/unit/ -v

# Run the full eval CLI (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
uv run python run_eval.py --model gpt-4o-mini --judge-model gpt-4o

# Run component eval via pytest
uv run pytest evals/component/ -v

# Run judge consistency meta-eval
uv run pytest evals/judge/ -v

# Run abstention/calibration eval
uv run pytest evals/component/test_router.py -v

# Visualise results (no API key needed — reads versioned artifacts)
uv run marimo run notebooks/eval_report.py
uv run marimo run notebooks/model_comparison.py
```

---

## Project structure

```
llm-eval/
├── llm_eval/
│   ├── agent.py               # Single-step extractor (OpenAI structured output)
│   ├── schemas.py             # Pydantic models: JobInfo, GoldenExample, EvalReport
│   ├── trace.py               # Logfire tracing helpers
│   └── scoring/
│       ├── base.py            # Scorer / AsyncScorer protocols
│       ├── deterministic.py   # ExactMatchScorer, FuzzyMatchScorer
│       └── llm_judge.py       # LLMJudgeScorer with caching
│   └── eval/
│       ├── runner.py          # Async eval loop
│       ├── dataset.py         # Load/validate golden set from JSONL
│       └── experiment.py      # load_experiment(), compare_experiments()
├── docs/
│   ├── intro.md               # Why LLM evaluation is hard and how to solve it
│   ├── design.md              # Scoring strategy rationale per field
│   └── success_criteria.md    # Specific, measurable eval targets
├── evals/
│   ├── component/
│   │   ├── test_extraction.py # Field-level accuracy (requires API key)
│   │   └── test_router.py     # Abstention/calibration eval
│   └── judge/
│       └── test_judge.py      # Judge consistency meta-eval
├── data/
│   ├── golden_set.jsonl       # 50 hand-curated synthetic examples
│   └── experiments/           # Versioned JSON artifacts per model run
├── notebooks/
│   ├── eval_report.py         # Marimo: per-field scores, failure breakdown
│   └── model_comparison.py    # Marimo: side-by-side delta table
└── tests/
    └── unit/
        ├── test_scoring.py          # Deterministic scorer tests
        └── test_scoring_async.py    # Async scorer tests (mocked)
```
