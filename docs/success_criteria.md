# Success Criteria

Specific, measurable targets for the job-extraction eval. These were set based on initial
baseline runs with `gpt-4o-mini` and informed by the field-level scoring design in
[design.md](design.md).

---

## Task fidelity

| Field | Scorer | Target | Rationale |
|---|---|---|---|
| `title` | LLM judge | mean >= 0.85 | Titles vary widely (translations, abbreviations); 0.85 allows minor reformulations |
| `seniority` | Exact match | mean >= 0.80 | Closed enum — errors are unambiguous; 0.80 accounts for genuinely ambiguous postings |
| `work_mode` | Exact match | mean >= 0.95 | Usually stated explicitly; failures here indicate extraction bugs |
| `location` | Fuzzy match | mean >= 0.60 | Weakest field by design — city-only vs city+country mismatch is a known limitation |
| `skills` | LLM judge | mean >= 0.80 | Long lists with synonym variance; 0.80 balances precision and recall |
| **Overall** | Weighted mean | mean >= 0.80, pass rate >= 85% | Pass threshold per example: 0.70 |

## Operational

| Metric | Target | Rationale |
|---|---|---|
| Latency (p95) | < 10s per example | Bounded by OpenAI API; semaphore limits to 5 concurrent |
| Total tokens per example | < 2000 | Budget constraint — title + skills judge calls dominate |
| Judge consistency | variance < 0.2 across 3 runs | Validated in `evals/judge/test_judge.py` |

## What "pass" means

An example **passes** when its overall score (mean of primary field scores) >= 0.70.
The eval **passes** when the pass rate across all 50 examples meets the targets above.

These thresholds are intentionally conservative for a demo project. Production targets
would be set higher after a larger golden set and cross-model validation.
