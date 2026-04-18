# llm-eval

Rigorous evaluation framework for LLM pipelines, demonstrated on a job-posting extraction use case.

> Rigorous LLM evaluation requires choosing the right measurement for each output type, questioning your judge, and treating eval results as versioned artifacts — not one-time metrics.

---

## What

A structured eval harness that runs a single-step LLM extractor against a hand-curated golden dataset and scores each output field with the appropriate metric.

The extractor takes a raw job posting and returns a structured `JobInfo`:

| Field | Type | Scorer |
|---|---|---|
| `title` | string | Embedding similarity + LLM judge (on failure) |
| `seniority` | enum | Exact match |
| `work_mode` | enum | Exact match |
| `location` | string | Fuzzy match |
| `skills` | list[str] | Embedding F1 |

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
 llm_eval/eval/dataset.py  ← load + validate golden examples
        │
        ▼
 llm_eval/eval/runner.py  ← async loop (semaphore, tracing)
        │        │
        ▼        ▼
  agent.py    scoring/    ← field-specific scorers
                ├── deterministic.py   ExactMatch, FuzzyMatch
                ├── embedding.py       EmbeddingScorer, EmbeddingF1Scorer
                └── llm_judge.py       LLMJudgeScorer (cached)
        │
        ▼
   EvalReport             ← per-field scores + aggregate
```

### Scoring strategy

- **Enum fields** (`seniority`, `work_mode`): exact match — no partial credit for wrong enum values.
- **Text fields** (`location`): fuzzy match via RapidFuzz — handles abbreviations, punctuation, transliteration.
- **Set-valued fields** (`skills`): soft embedding F1 — order-invariant, semantic, computes precision and recall separately.
- **Open-ended text** (`title`): embedding cosine similarity first; LLM judge is called only when similarity fails. This saves tokens while catching semantic equivalences fuzzy match misses.

### LLM Judge

- Structured output: `JudgeVerdict(score: Literal[0, 1, 2], reasoning: str)`
- Score mapping: `0 → 0.0`, `1 → 0.5`, `2 → 1.0`
- Results cached by `hash(rubric + expected + actual)` — identical comparisons are never re-evaluated.
- Consistency meta-eval in `evals/judge/`: runs the judge 3× on identical pairs and asserts variance < 0.2.

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

| Model | Mean score | Pass rate |
|---|---|---|
| `gpt-4o-mini` | 0.87 | 92% (46/50) |
| `gpt-4.1-mini` | 0.88 | 96% (48/50) |

Pass threshold: overall score ≥ 0.7 (mean across all fields).

Field-level breakdown (gpt-4o-mini / gpt-4.1-mini):

| Field | gpt-4o-mini | gpt-4.1-mini | Δ |
|---|---|---|---|
| title | 0.93 | 0.94 | +0.01 |
| seniority | 0.86 | 0.92 | **+0.06** |
| work_mode | 0.98 | 1.00 | +0.02 |
| location | 0.68 | 0.66 | −0.02 |
| skills | 0.91 | 0.87 | −0.04 |

`gpt-4.1-mini` wins on seniority and work_mode classification. `gpt-4o-mini` is slightly better on skills (less conservative on sparse descriptions). Location is the weakest field for both — fuzzy match degrades on city+country vs city-only mismatches.

Versioned artifacts in `data/experiments/`. Visualise with `uv run marimo run notebooks/03_model_comparison.py`.

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- `OPENAI_API_KEY` for eval runs (unit tests don't need it)

## Quickstart

```bash
# Install dependencies
uv sync

# Run unit tests (no API key required)
uv run pytest tests/unit/ -v

# Run the full eval CLI (requires OPENAI_API_KEY)
export OPENAI_API_KEY=your-api-key-here
uv run python run_eval.py --model gpt-4o-mini

# Run component eval via pytest
uv run pytest evals/component/ -v

# Run judge consistency meta-eval
uv run pytest evals/judge/ -v

# Run abstention/calibration eval
uv run pytest evals/component/test_router.py -v

# Visualise results (no API key needed — reads versioned artifacts)
uv run marimo run notebooks/02_eval_report.py
uv run marimo run notebooks/03_model_comparison.py
```

---

## Project structure

```
llm-eval/
├── run_eval.py            # CLI entry point
├── llm_eval/
│   ├── agent.py               # Single-step extractor (OpenAI structured output)
│   ├── schemas.py             # Pydantic models: JobInfo, GoldenExample, EvalReport
│   ├── trace.py               # Logfire tracing helpers
│   ├── scoring/
│   │   ├── base.py            # Scorer / AsyncScorer protocols
│   │   ├── deterministic.py   # ExactMatchScorer, FuzzyMatchScorer
│   │   ├── embedding.py       # EmbeddingScorer, EmbeddingF1Scorer
│   │   └── llm_judge.py       # LLMJudgeScorer with caching
│   └── eval/
│       ├── runner.py          # Async eval loop
│       ├── dataset.py         # Load/validate golden set from JSONL
│       └── experiment.py      # load_experiment(), compare_experiments()
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
│   ├── 02_eval_report.py      # Marimo: per-field scores, failure breakdown
│   └── 03_model_comparison.py # Marimo: side-by-side delta table
└── tests/
    └── unit/
        ├── test_scoring.py          # Deterministic scorer tests
        └── test_scoring_async.py    # Async scorer tests (mocked)
```
