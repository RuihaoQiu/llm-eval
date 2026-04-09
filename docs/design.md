# Scoring Design Rationale

One question every eval project should answer explicitly: *why this metric for this field?* Wrong choices produce misleading numbers — exact match on open-ended text misses synonyms; LLM judge on enum fields adds variance where there should be none.

---

## Field-by-field decisions

### `seniority` and `work_mode` — ExactMatchScorer

Both are closed enumerations with no semantic gradient. "mid" is not closer to "senior" than it is to "intern" in any way that the eval should reward. Partial credit would be meaningless: a model that returns "junior" when the answer is "senior" is wrong, full stop.

LLM judge is intentionally avoided here — a judge would introduce stochastic variance on a question that has a deterministic answer.

### `location` — FuzzyMatchScorer (RapidFuzz token_set_ratio)

Location strings have many legitimate surface-form variants: "Berlin" vs "Berlin, Germany", "New York" vs "NYC", "São Paulo" vs "Sao Paulo". Exact match would penalise all of these.

`token_set_ratio` is used rather than simple Levenshtein because it is order-invariant and handles subset matches gracefully — "Berlin" inside "Berlin, Germany" scores 100. This is better than embedding similarity for locations because embeddings conflate geographically distinct cities that happen to appear in similar contexts.

The weakest field in practice (mean ~0.67). The main failure mode is city-only vs city+country mismatches where the golden example has a full "City, Country" string but the model returns just the city. This is a deliberate known limitation — adding country inference would mask an actual extraction failure.

### `skills` — EmbeddingF1Scorer (soft F1)

Skills are an unordered set, so any order-sensitive metric is wrong by construction. The challenge is that surface forms differ: "PyTorch" vs "torch", "Kubernetes" vs "k8s", "scikit-learn" vs "sklearn".

Soft embedding F1 handles this:
1. For each predicted skill, find the highest-similarity golden skill (soft precision).
2. For each golden skill, find the highest-similarity predicted skill (soft recall).
3. Compute F1 from the two.

This separates over-extraction (low precision) from under-extraction (low recall), which matters for failure analysis. Pure cosine similarity without the F1 framing would not distinguish between a model that gets everything plus extra noise vs one that gets half the skills exactly right.

Exact match is not used because it would penalise semantically identical skills with different capitalisation or abbreviation — common in real job postings.

### `title` — EmbeddingScorer + LLMJudgeScorer (two-stage)

Job titles have the highest variance of any field: "Software Engineer", "SWE", "Développeur Logiciel", and "Ingénieur en Logiciel" can all be correct extractions of the same role. Pure string match fails. Pure LLM judge is expensive.

Two-stage approach:
1. Compute cosine similarity using `text-embedding-3-small`. If similarity ≥ 0.85 → pass without judge call.
2. On failure, call the LLM judge with a rubric that explicitly handles translation, abbreviation, and seniority-modifier stripping (e.g. "Senior Backend Developer" → "Backend Developer").

This saves ~60–70% of judge calls on a typical eval run while still catching semantic equivalences that embedding alone misses (e.g. non-English titles that embed into different regions of the space).

The judge returns `JudgeVerdict(score: Literal[0, 1, 2])` — a three-point scale rather than binary to allow partial credit for cases like "Backend Engineer" when the golden is "Backend Developer". Results are cached by `sha256(rubric + expected + actual)` so re-running the eval never re-spends tokens on identical comparisons.

---

## What was intentionally excluded

**Numeric scorer (salary)**: The golden set does not include salary because normalising currency, frequency (annual vs monthly), and precision across job postings is a separate problem that would dominate the eval. Including a noisy field would pull down overall scores without revealing anything useful about the extractor.

**ROUGE / BLEU**: These are sequence-overlap metrics designed for generation tasks. They penalise paraphrase and reward n-gram overlap, which is the opposite of what job-title extraction needs. They are not used anywhere in this project.

**Semantic similarity alone for skills**: Cosine similarity between the full predicted skills list and the full golden list would not distinguish precision from recall failures. The F1 framing is necessary for actionable diagnostics.
