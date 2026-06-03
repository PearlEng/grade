# GRADE Scoring Reference

**Package:** `grade 0.1.0.dev0` · Related: [`docs/methodology.md`](methodology.md)

This document defines the six GRADE scoring dimensions, their weights, and how each is
computed. The weights are locked for V1 and apply uniformly to the overall composite.
Individual tasks may shift weights within their rubrics, but the project-level weights
below determine leaderboard rankings.

---

## Dimension Weights (Locked)

| Dimension | Key | Weight |
|---|---|---|
| Grounding Accuracy | `grounding_accuracy` | 35% |
| Insight Quality | `insight_quality` | 20% |
| Evidence Linkage | `evidence_linkage` | 15% |
| Calibration & Limitation Handling | `calibration_limitation_handling` | 15% |
| Consistency | `consistency` | 10% |
| Structure & Usability | `structure_usability` | 5% |

All six dimension scores are in **[0, 1]**. The composite score for a task, track, or
overall result is the weighted sum:

```
composite = 0.35 × grounding_accuracy
          + 0.20 × insight_quality
          + 0.15 × evidence_linkage
          + 0.15 × calibration_limitation_handling
          + 0.10 × consistency
          + 0.05 × structure_usability
```

Track-level and pack-level composites are the arithmetic mean of task-level composites
within that track or pack. The overall composite is the mean over all tasks.

---

## Dimension Definitions

### 1. Grounding Accuracy (35%)

Measures whether the model's factual claims match the ground-truth values in the fixture
data. This is the highest-weighted dimension because factual errors — hallucinated numbers,
misread rates, incorrect entity counts — are the most harmful failure mode in an
analytics context.

**How it is scored (C1):**

Each task defines a list of `gold_facts`. Each gold fact has an expected numeric or
string value derived from the fixture data. For numeric facts, the scorer compares the
model's reported value against the gold value with an absolute or relative tolerance. For
string facts (e.g., entity identifiers), the scorer uses case-folded exact match.

The Grounding Accuracy score for a task is the mean per-fact score over all gold facts.
A fact scores 1.0 if the predicted value is within tolerance; 0.0 otherwise.

**Important:** Gold fact values are sourced from `ground_truth.json` files in the fixture
pack directories. These are computed from the generated data (seed 42) and may differ
from the generator's input target constants due to stochastic noise.

**Forbidden claims:** Tasks also specify `forbidden_claims` — statements the data cannot
support. Forbidden-claim violations are scored under Calibration & Limitation Handling,
not here. However, producing a forbidden claim alongside correct facts does not rescue
the Calibration score.

---

### 2. Insight Quality (20%)

Measures coverage and depth of the analytical insights the model is expected to surface.
Correct numeric retrieval without interpretation — listing numbers without explaining
their significance — scores poorly here.

**How it is scored (C2/C3):**

Each task defines `gold_insights`: higher-order findings the model should reach
(e.g., "the October cancellation rate is approximately 2.6× the September rate, which
represents the most significant operational anomaly in the reporting period"). The C3
claim validator checks whether each gold insight appears in the model's `key_findings`
using deterministic string-matching with an optional judge fallback for paraphrase
tolerance.

The dimension score is the mean appearance rate across all gold insights.

Insight Quality is elevated in Track 3 (recommendations) and Track 5 (research synthesis)
because the practical value of those tasks depends on the model going beyond data
recitation.

---

### 3. Evidence Linkage (15%)

Measures whether factual claims and recommendations are explicitly tied to specific
source files, columns, or entity IDs from the fixture data. A response that reaches
correct conclusions without naming its sources fails this dimension.

**How it is scored (C2):**

Evidence Linkage is evaluated by the judge model using task-specific guidance. The rubric
for each task describes what citation form is required (e.g., "the response must name
the source CSV file and the column used for each computed rate").

Track 3 tasks enforce a particularly strict evidence requirement: responses that omit
entity IDs (tutor IDs, group IDs, school IDs) and numeric rates receive zero on this
dimension regardless of whether the analytical conclusion is correct. This reflects the
practical need for traceability in operational decision-making.

---

### 4. Calibration & Limitation Handling (15%)

Measures whether the model acknowledges data limitations, avoids overconfident claims,
and refrains from assertions the data cannot support. This dimension is equally weighted
with Evidence Linkage because producing plausible-sounding but unsupportable claims is as
harmful as omitting citations.

**How it is scored (C3):**

This dimension is scored using three sub-checks, each in [0, 1]:

1. **Required limitations present** — tasks define `required_limitations` (caveats the
   model must acknowledge). The C3 validator checks whether each required limitation
   appears in the model's `limitations` field.

2. **Forbidden claims absent** — tasks define `forbidden_claims` (statements the data
   cannot support). The validator checks that no forbidden claim appears anywhere in the
   model output — in `key_findings`, `limitations`, or the raw response text.

3. **Gold insights present** — as in Insight Quality, but contributing to the
   calibration aggregate when claims are about uncertainty or scope boundaries.

The aggregate Calibration & Limitation Handling score is the mean of non-empty sub-check
scores. All three sub-checks use deterministic string-matching as the primary method,
with an optional judge fallback for paraphrase cases.

**Typical required limitations across tracks:**

- Track 1/2: Data covers a single quarter; year-over-year comparisons are not supported.
- Track 2: Correlation between two signals does not establish causation.
- Track 4: Subgroups below the n<10 reporting threshold are suppressed; no rates can be
  reported or estimated for them.
- Track 5: A single published study's effect size cannot be applied to this program
  without acknowledging differences in study population, design, and context.

---

### 5. Consistency (10%)

Measures whether the model produces stable outputs across 5 repeated runs on the same
task. Inconsistency — finding different "top priorities," reporting different numeric
values, or reversing rankings across runs — undermines practitioner trust.

**How it is scored (C4):**

The Consistency score is a weighted combination of three sub-metrics, each in [0, 1]:

| Sub-metric | Weight | Definition |
|---|---|---|
| Finding stability | 0.50 | Mean appearance rate of each unique top-3 finding across all 5 runs, using token-level Jaccard matching (threshold 0.50) |
| Ranking stability | 0.30 | Mean pairwise normalized Kendall tau-b across all 10 run pairs, computed over top-3 findings shared between each pair |
| Metric variance | 0.20 | Mean inverse coefficient of variation (1 / (1 + CV)) over all numeric `structured_metrics` keys present across runs |

A model that produces identical outputs on every run scores 1.0 on Consistency. A model
that produces substantially different findings or different numeric values on each run
scores proportionally lower.

Consistency is not a test of whether the model reaches the correct answer; it is a
test of output stability given a fixed input. A consistently wrong model can score well
here while scoring poorly on Grounding Accuracy.

---

### 6. Structure & Usability (5%)

Measures the practitioner-facing format quality of the response. This dimension has the
lowest weight because structure is the easiest failure to correct — it does not reflect
an underlying reasoning problem. However, a response that is factually accurate but
incomprehensible to a program analyst fails a practical usefulness test.

**How it is scored (C2):**

Evaluated by the judge model using task-specific rubric guidance. Typical criteria:

- Clear distinction between findings, caveats, and (for Track 3) recommendations.
- Numeric values reported with appropriate precision and units (percentages vs. decimal
  fractions; percentage points vs. ratios).
- Section structure that allows a practitioner to scan key findings and limitations
  without reading the full response.
- Absence of jargon that would be opaque to an educator or analyst audience.

---

## Score Aggregation

Scores are aggregated as follows in `result_schema.json`:

1. **Task-level scores**: each dimension score is the mean over the 5 repetition runs.
2. **Track-level scores**: arithmetic mean of task-level scores for tasks within the track.
3. **Pack-level scores**: arithmetic mean of task-level scores for tasks run against the pack.
4. **Overall scores**: arithmetic mean of all task-level scores.
5. **Composite**: the weighted sum (using the table above) applied to the overall dimension
   scores.

The scorecard generator (`benchmark/reports/scorecard.py`) applies these aggregations and
renders both a Markdown report and a `results.json` file conforming to
`benchmark/schemas/result_schema.json`.

---

## Per-Track Rubric Defaults

Individual track READMEs document the typical dimension weights used within that track's
rubric. While the project-level weights above determine the leaderboard, per-task rubrics
may redistribute weight to reflect the primary failure mode for each track. For reference:

| Track | Primary emphasis | Typical grounding weight | Typical calibration weight |
|---|---|---|---|
| 1 — Grounded Retrieval | Factual accuracy | 0.40–0.45 | 0.10 |
| 2 — Snapshot & Trends | Calibrated narration | 0.35 | 0.15 |
| 3 — Coaching Recommendations | Evidence citation | 0.35 | 0.10 |
| 4 — Equity Interpretation | Accuracy + calibration equally | 0.30–0.35 | 0.20–0.25 |
| 5 — Research Synthesis | Grounding + insight + evidence | 0.30 | 0.15 |

Rubric weight sums are validated at runtime; tasks whose weights do not sum to 1.0
(within 1e-6) are rejected before any scoring occurs.
