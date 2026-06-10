# GRADE Methodology

**GRADE — Grounded Reasoning & Analysis for Data in Education**

Version: `0.1.0.dev0` · Benchmark package: `PearlEng/grade`

---

## Mission

GRADE measures how accurately and usefully AI systems analyze realistic education program
data. The benchmark focuses on a practical question: given a set of structured program
records — attendance logs, session data, subgroup summaries, and published research
references — can a model produce answers that a district analyst or program manager could
trust and act on?

Trustworthy answers require five things simultaneously:

1. **Factual accuracy** — numbers match the source data.
2. **Evidence citation** — every claim is traced to a specific file and column.
3. **Analytical insight** — signals are interpreted, not just recited.
4. **Calibrated humility** — data limitations and caveats are acknowledged.
5. **Consistency** — the same question yields the same answer across repeated runs.

GRADE is purpose-built to test all five. It is not a general-purpose reasoning benchmark
and makes no claims about performance outside the education-analytics domain.

---

## Task Taxonomy — Five Tracks

GRADE V1 contains 26 tasks organized into five tracks. Each track isolates a distinct
mode of analytical reasoning.

### Track 1 — Grounded Retrieval & Computation

**Primary pack:** Operations (`pack_operations`)
**Tasks:** 6

Tasks ask the model to retrieve specific facts or perform straightforward arithmetic
(counts, rates, date-filtered aggregates) directly from structured CSV fixtures. There is
no multi-step inference; a correct answer is fully determinable from the provided data.
The primary failure mode is hallucinating or miscomputing a specific number.

Typical operations: direct counts of entities (students, sessions, groups); attendance and
cancellation rate calculations; date-range filtering; subgroup comparisons (e.g.,
IEP vs. non-IEP attendance rates).

### Track 2 — Program Snapshot & Trend Interpretation

**Primary pack:** Outcomes (`pack_outcomes`)
**Tasks:** 5

Tasks ask the model to summarize month-over-month signals, identify anomalies, and
produce calibrated trend narratives. The defining challenge is epistemic discipline:
the data contains correlations that can superficially motivate causal claims, but the
data alone cannot support them. A strong response reports realized values accurately,
characterizes patterns at the right confidence level, and explicitly declines to assert
causation.

Typical operations: month-over-month attendance and satisfaction trend summaries;
anomaly identification (e.g., the October cancellation spike); multi-signal snapshots;
progress-against-goal assessment.

### Track 3 — Operational Coaching & Recommendations

**Primary pack:** Operations (`pack_operations`)
**Tasks:** 5

Tasks require the model to synthesize operational evidence into actionable,
practitioner-facing recommendations — not merely to report what the data says, but to
rank options, assign confidence, and cite the specific data points supporting each
recommendation. Responses that do not name entity IDs, numeric rates, and source files
earn zero on the Evidence Linkage dimension.

Typical operations: tutor coaching prioritization; IEP-support planning; response to
a session-cancellation spike; confidence-banded school-level intervention recommendations;
multi-signal priority ranking.

### Track 4 — Equity & Subgroup Interpretation

**Primary pack:** Equity & Research (`pack_equity_research`)
**Tasks:** 5

Tasks evaluate whether a model reasons carefully about subgroup data, including correctly
handling suppressed low-N groups, quantifying attendance and outcome disparities, and
translating findings into plain language appropriate for practitioners. Every task includes
a required caveat about sample size — the core competency this track tests.

Typical operations: IEP attendance gap analysis; suppressed-subgroup identification and
explanation; racial/ethnic subgroup comparisons; proficiency gap quantification; fairness
and equity interpretation.

### Track 5 — Program Effectiveness & Research Reasoning

**Primary pack:** Equity & Research (`pack_equity_research`)
**Tasks:** 5

Tasks require synthesis across two evidence layers: local fixture data and a curated set
of 10 external research references (`research_refs.json`). A correct answer integrates
specific local numbers with specific research citations — neither generic literature
summaries nor local data presented in isolation are sufficient. Tasks deliberately include
references with contradictory findings to test whether models can distinguish supporting
from challenging evidence.

Typical operations: predictor ranking with research-to-data linkage; mixed-evidence
comparison; program-design alignment with the evidence base; data-gap identification;
IEP risk synthesis combining dosage-response findings with local data.

---

## Fixture Packs — Three Synthetic Datasets

All fixture data is fully synthetic and seeded. No real students, tutors, schools, or
program records are included. The generator produces byte-for-byte identical output given
the same seed and source code.

| Pack | Pack ID | Tasks | Tracks |
|---|---|---|---|
| Operations | `pack_operations` | 11 | 1, 3 |
| Outcomes | `pack_outcomes` | 5 | 2 |
| Equity & Research | `pack_equity_research` | 10 | 4, 5 |

The three packs are layered: Outcomes extends Operations with month-over-month summary
tables; Equity & Research extends Operations with subgroup summary tables and research
references.

Dataset cards with full file inventories, column contracts, intentional signal
descriptions, and known limitations are in `docs/dataset_cards/`:

- [`docs/dataset_cards/pack_operations.md`](dataset_cards/pack_operations.md)
- [`docs/dataset_cards/pack_outcomes.md`](dataset_cards/pack_outcomes.md)
- [`docs/dataset_cards/pack_equity_research.md`](dataset_cards/pack_equity_research.md)

---

## Scoring Design — Six Dimensions

Each task response is scored on six dimensions. The six dimensions are fully defined in
[`docs/scoring.md`](scoring.md). The locked project-level weights are:

| Dimension | Weight |
|---|---|
| Grounding Accuracy | 35% |
| Insight Quality | 20% |
| Evidence Linkage | 15% |
| Calibration & Limitation Handling | 15% |
| Consistency | 10% |
| Structure & Usability | 5% |

The composite score for a task is the weighted sum of its six dimension scores. Track-level
and pack-level composites are the mean over all tasks within that track or pack.

---

## Repeated-Run Methodology

Each (model × task) pair is evaluated **5 times** independently. Repetitions use the
same prompt and the same temperature-0 setting. The five runs serve two purposes:

1. **Consistency scoring** — the Consistency dimension (10% weight) measures how
   stable key findings, their ranking, and numeric metrics are across the five runs.
   See [`docs/scoring.md`](scoring.md) for the full consistency sub-metric definitions.

2. **Variance reduction** — dimension scores other than Consistency are averaged over
   the five runs to reduce single-run noise.

The `run_count` field in `result_schema.json` records the number of repetitions for each
scorecard. The expected value is 5.

---

## Ground-Truth Construction

Each task includes `gold_facts` (verifiable numeric or categorical claims), `gold_insights`
(higher-order findings), `forbidden_claims` (claims the data cannot support), and
`required_limitations` (caveats the model must acknowledge).

All `gold_facts` numeric values are derived from `ground_truth.json` files in the fixture
pack directories — not from the generator's input constants. Because the generator applies
stochastic noise, realized values differ from spec targets; the `ground_truth.json` files
record the authoritative computed values for seed 42.

The scoring pipeline:

- **C1 (fact scoring)** — compares structured numeric outputs against `gold_facts` with
  absolute or relative tolerance.
- **C2 (rubric scoring)** — a judge model evaluates each of the six dimensions using
  task-authored per-dimension guidance.  Judge selection is **cross-family**: Claude
  Opus 4.8 judges every candidate except Claude-family models, which are judged by
  GPT-5.5 at xhigh reasoning effort instead — no judge ever shares a model family
  (and hence a house style) with the model it grades.  Two caveats are inherent to
  this design and disclosed here: (1) both judges are themselves benchmark
  contestants, though neither ever judges its own family; (2) Claude rows are scored
  by a different judge than all other rows, so a systematic harshness difference
  between the two judges would shift the judged dimensions (40% of the composite)
  for those rows relative to the rest of the leaderboard.
- **C3 (claim validation)** — deterministic string-matching (with optional judge fallback)
  checks `gold_insights` presence, `forbidden_claims` absence, and `required_limitations`
  presence.
- **C4 (consistency scoring)** — measures finding stability, ranking stability, and
  numeric-metric variance across the 5 repeated runs.
- **C9 (scorecard)** — aggregates C1–C4 outputs into per-task, per-track, per-pack, and
  overall scores, then renders the Markdown scorecard and `results.json`.

Schema contracts for tasks, outputs, and results are in `benchmark/schemas/`.

---

## Limitations and Non-Goals

**What GRADE measures:** accuracy, evidence citation, calibration, insight quality, and
response consistency for education program analytics tasks using realistic synthetic data.

**What GRADE does not measure:**

- General reasoning ability or knowledge outside the education-analytics domain.
- Real-time data access, retrieval-augmented generation capabilities, or tool use.
- Downstream decision quality — whether a practitioner actually acts on the model's
  recommendations.
- Latency, cost, or token efficiency.
- Performance on program types, grade levels, or data structures not represented in the
  three fixture packs.

**Known limitations of the current fixture set:**

- All data covers a single fictional program across a single quarter (Q1 FY2025-26).
  Year-over-year comparisons and longitudinal analyses are out of scope.
- The three fixture packs share a common program context. Pack diversity (different
  programs, different states, different data models) is a planned V2 extension.
- Track 5 research references are linked to a single program. The `relevance_to_program`
  and `contradicts_claim` annotations represent plausible benchmark-purpose interpretations,
  not the original authors' claims about any specific program.
- The synthetic generator embeds specific effect-size targets. Realized values for seed 42
  are validated in `ground_truth.json`; a different seed produces different realized values
  that may shift task difficulty.

**Statistical precision.** GRADE V1 contains 26 tasks (11 operations, 5 outcomes,
10 equity & research), with per-track task counts as low as 5. Overall composites are
averaged over 26 tasks × 5 runs; per-track and per-pack scores rest on much smaller
samples and should be read as indicative rather than precise. Small leaderboard gaps
(roughly a few points) on per-track views are within plausible noise.

**Judge scope.** The LLM judge evaluates the judged dimensions against the task's gold
material (gold facts, gold insights, required limitations, forbidden claims, reference
outline) — it does **not** see the raw fixture CSVs. Verifying numbers against the data
is the deterministic C1 scorer's job; the judge assesses interpretation quality relative
to the gold material. This keeps judge prompts bounded and judging reproducible, at the
cost that `evidence_linkage` is assessed on plausibility against gold material rather
than re-derived from the data.

**Benchmark contamination.** Fixture data, gold facts, and task prompts are public in
this repository, so future model training runs may ingest them. Because every dataset is
produced by the synthetic generator (`benchmark/datagen/`), GRADE mitigates this by
regenerating fixtures (new seed, recomputed `ground_truth.json`) for each numbered
benchmark version; scores are only comparable within a benchmark version, and the
version is recorded in every result (`grade_version`).

The public/private boundary governing what can be published is documented in
[`docs/publication_policy.md`](publication_policy.md).
