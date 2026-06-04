# Why LLM Evaluation Is Hard (And What to Do About It)

## The problem

You build an LLM pipeline. It looks great on a few examples. You ship it. Then users report
garbage outputs that your spot-checks never caught.

This happens because most teams evaluate LLMs the way they test traditional software — run a
handful of examples, eyeball the results, and move on. But LLMs are probabilistic. The same
prompt can produce different outputs each time. "Looks good" is not a metric.

## Why it matters

Bad evaluation leads to three failure modes:

**1. False confidence.** You measure one number (e.g. "90% accuracy") without asking what that
number actually captures. If your metric is exact string match on open-ended text, a correct
answer phrased differently counts as wrong. Your real accuracy is higher than reported — but
you don't know where the actual failures are.

**2. Silent regressions.** You swap models, tweak a prompt, or update a dependency. Output
quality drops on 5% of cases. Without automated evals that run on every change, you won't
notice until users complain.

**3. Misplaced trust in LLM judges.** Using an LLM to grade another LLM's output is powerful,
but the judge itself can be inconsistent. If your judge gives different scores for the same
input on different runs, your eval results are noise — you're making decisions on random numbers.

## How to solve it

This project demonstrates a structured approach to LLM evaluation:

### Pick the right metric for each output type

Not every field deserves the same scorer. Enum fields (e.g. "remote" / "hybrid" / "onsite")
have a single correct answer — exact match is the right metric, and using an LLM judge here
just adds variance. Free-text fields like job titles need semantic comparison because
"Senior Backend Developer" and "Sr. Backend Dev" are the same thing.

| Output type | Right metric | Wrong metric |
|---|---|---|
| Enum / categorical | Exact match | LLM judge (adds noise) |
| Short text with variants | Fuzzy match | Exact match (misses synonyms) |
| Open-ended text | LLM judge | Exact match (too strict) |
| Unordered lists | LLM judge with list rubric | Order-sensitive comparison |

### Evaluate the evaluator

If you use an LLM as a judge, you need to test the judge itself. Run the same comparison
multiple times and check that scores are consistent. If the variance is too high, your eval
is unreliable — you're measuring judge randomness, not model quality.

### Use a different model for judging

When the same model generates output and judges it, you risk self-evaluation bias — the model
may rate its own style of errors more favorably. Using a separate (often stronger) model as
the judge produces more objective scores.

### Track more than accuracy

Accuracy alone doesn't tell you if a model upgrade is worth the cost. Track latency and token
usage alongside scores so you can answer questions like: "Model B is 2% more accurate but 3x
more expensive — is the tradeoff worth it?"

### Version your results

Eval results should be JSON artifacts committed to version control, not numbers in a notebook
that disappear when you restart the kernel. This lets you diff results across model versions,
reproduce past runs, and build visualizations without re-running expensive API calls.

## What this project demonstrates

This repo implements all of the above on a concrete task: extracting structured job information
from raw job postings. The task is simple on purpose — the evaluation framework is the point.

- **50 hand-curated test cases** across 10 categories (ambiguous seniority, non-English titles,
  sparse descriptions, etc.)
- **Field-specific scoring** — exact match for enums, fuzzy match for location, LLM judge for
  titles and skills
- **Judge consistency meta-eval** — the judge is tested for reliability before trusting its scores
- **Separate extraction and judge models** — `gpt-4o-mini` extracts, `gpt-4o` judges
- **Latency and token tracking** — every run records operational metrics alongside accuracy
- **Versioned artifacts** — results are diffable JSON, notebooks render offline

See the [README](../README.md) for quickstart and results.
