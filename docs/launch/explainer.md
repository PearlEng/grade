# GRADE: An Open Benchmark for AI in Education Program Analytics

*What does it mean for an AI to give a trustworthy answer about a tutoring program?
GRADE measures exactly that.*

---

## The problem this benchmark addresses

Educators, program managers, and district analysts are increasingly asked to use AI tools
to make sense of program data — attendance records, satisfaction surveys, subgroup outcome
reports. The natural question follows: how do we know whether any given AI system is
actually good at this?

General-purpose reasoning benchmarks do not answer it. A model can score well on
mathematical reasoning tasks or reading comprehension while still hallucinating a
cancellation rate, misstating an attendance gap, or fabricating a causal claim the data
cannot support. When a program director uses an AI-generated summary to decide which
tutors need coaching support, factual errors and overconfident conclusions carry real
stakes.

GRADE — **G**rounded **R**easoning & **A**nalysis for **D**ata in **E**ducation — is an
open benchmark built to evaluate AI systems on the specific analytical tasks that matter
in education program analytics. It asks: given realistic program records, can a model
produce answers a district analyst could trust and act on?

---

## What GRADE measures

GRADE is not a test of general language understanding. It measures five concrete
analytical capabilities, each grounded in the kinds of questions practitioners actually
ask about tutoring and intervention programs:

**1. Grounded retrieval and computation.** Can the model accurately retrieve specific
facts — counts, rates, date-filtered aggregates — directly from structured data files
without hallucinating or miscomputing?

**2. Program snapshot and trend interpretation.** Can the model summarize month-over-month
signals, identify anomalies, and resist the temptation to assert causal claims that the
data cannot support?

**3. Operational coaching and recommendations.** Can the model synthesize data into
ranked, evidence-backed recommendations — and cite the specific data points (tutor IDs,
attendance rates, source files) that justify each recommendation?

**4. Equity and subgroup interpretation.** Can the model reason carefully about subgroup
data, including correctly handling suppressed low-N groups and translating attendance or
outcome disparities into practitioner-appropriate language?

**5. Program effectiveness and research reasoning.** Can the model integrate local program
data with external research references, connect observed patterns to published findings,
and identify where local data confirms or diverges from research expectations?

---

## How the benchmark is structured

### 26 tasks across five tracks

GRADE V1 contains 26 tasks organized into the five tracks above. Tasks range in
difficulty from straightforward fact retrieval (Track 1) to synthesis tasks that require
connecting a local IEP attendance gap to published dosage-response findings (Track 5).

### Three synthetic fixture packs

All data is fully synthetic and seeded. No real students, tutors, schools, or program
records are included. The synthetic data is designed to preserve the structural realism
of how tutoring programs are actually operated and reported — including intentional
messiness such as incomplete group assignments, uneven survey coverage, partial attendance
records, and an embedded October cancellation spike that mirrors the scheduling disruptions
common in fall semesters.

The three fixture packs are:

- **Operations pack** (`pack_operations`) — a single program quarter with 135 students, 9
  tutors, 14 tutoring groups, and 546 scheduled sessions across three schools. Primary pack
  for Tracks 1 and 3.
- **Outcomes pack** (`pack_outcomes`) — extends Operations with month-over-month summary
  tables covering attendance trends, satisfaction survey scores, and cancellation rates
  across Q1 (September–November 2025). Primary pack for Track 2.
- **Equity & Research pack** (`pack_equity_research`) — extends Operations with subgroup
  attendance and outcome summaries (including suppressed low-N racial/ethnic subgroups)
  and 10 structured research references. Primary pack for Tracks 4 and 5.

Each pack includes a `ground_truth.json` file with pre-validated signal values computed
directly from the generated data. These are the authoritative values used for scoring —
not spec targets or prose descriptions.

### Six scoring dimensions

Every task response is scored on six dimensions:

| Dimension | Weight | What it tests |
|---|---|---|
| Grounding Accuracy | 35% | Factual correctness against fixture data |
| Insight Quality | 20% | Coverage and depth of higher-order findings |
| Evidence Linkage | 15% | Whether claims cite specific sources, files, and entity IDs |
| Calibration & Limitation Handling | 15% | Whether the model acknowledges what the data cannot support |
| Consistency | 10% | Stability of findings across 5 repeated runs |
| Structure & Usability | 5% | Clarity and practitioner-facing format |

The Grounding Accuracy dimension carries the highest weight because factual errors are the
most harmful failure mode in an analytics context. The Calibration & Limitation Handling
dimension is weighted equally to Evidence Linkage because producing plausible-sounding
but unsupportable claims is as harmful as omitting citations.

### Repeated runs

Each (model × task) combination is run **5 times** independently at temperature 0. The
five runs serve two purposes: they power the Consistency dimension score, and they reduce
single-run noise for the other five dimensions by averaging.

---

## How to read a GRADE scorecard

A GRADE scorecard reports scores at four levels of aggregation: overall, per-track,
per-pack, and per-task.

**Overall composite score** is the weighted sum of the six dimension scores averaged
across all 26 tasks. A score of 1.0 means the model produced factually accurate,
well-cited, appropriately calibrated, consistently structured responses on every task.

**Track scores** reveal where a model is strong and where it struggles. A model might
score well on Track 1 (straightforward retrieval) but poorly on Track 4 (equity
interpretation), indicating difficulty with subgroup caveats and low-N handling. Track
profiles are more informative for model selection and improvement than the overall
composite.

**Dimension scores** (reported at the track and overall level) show the failure mode
profile. A model with high Grounding Accuracy but low Calibration is reliably factual
but overconfident. A model with low Evidence Linkage may reach correct conclusions
without citing its sources — a problem for practitioners who need traceability.

**Per-task detail** is available in the full `results.json` output. Each task record
includes the five individual run scores, the per-dimension means, and the gold-fact
comparison results for numeric assertions.

No GRADE scorecard should be read as a statement about general AI capability. It is a
measurement of specific analytical behavior on a specific class of education program
analytics tasks.

---

## What GRADE does not measure

Being clear about scope is part of responsible benchmark design.

GRADE does not measure:

- General reasoning ability outside the education analytics domain.
- Real-time data access, tool use, or retrieval-augmented generation capabilities.
- Latency, cost, or token efficiency.
- The quality of downstream decisions made by practitioners using AI outputs.
- Performance on program types, grade levels, or data structures not represented in the
  current fixture packs.

The current V1 fixture set covers a single fictional program across a single quarter.
Year-over-year comparisons, multi-year longitudinal analyses, and multi-program diversity
are planned V2 extensions.

---

## Who GRADE is for

GRADE is designed for three audiences:

**Researchers and developers** building or fine-tuning AI systems for education analytics
applications. GRADE provides a reproducible, open evaluation contract with explicit rubrics
and ground-truth validation.

**Education organizations** evaluating AI tools for program analysis. GRADE offers a
vocabulary and a structured methodology for asking concrete questions about AI behavior on
tasks that reflect real operational needs.

**Benchmark contributors** who want to extend the task set, add fixture packs representing
different program types, or adapt the evaluation methodology for related domains. The full
task schema, fixture schema, and scoring pipeline are open-source.

---

## How to participate

**Run the benchmark on a model of your choice**

The benchmark runner supports any model accessible via
[OpenRouter](https://openrouter.ai/). To run all 26 tasks with 5 repetitions:

```bash
git clone https://github.com/PearlEng/<NAME>
cd <NAME>
pip install -e ".[openrouter]"
grade-run --model <model-id> --reps 5 --out results/
```

<!-- NOTE: <NAME> is a placeholder pending the final repository name decision (G3). -->

**Contribute tasks or fixture packs**

New tasks must conform to the task schema in `benchmark/schemas/task_schema.json`.
New fixture packs must conform to the fixture schema contract in
`benchmark/schemas/fixture_schema.md`. See `CONTRIBUTING.md` for the full contribution
guide and the gold-fact validation requirements.

**Report issues or discuss methodology**

Use GitHub Issues for bug reports, task corrections, and methodology questions. See
`GOVERNANCE.md` for the decision-making process for changes to the scoring rubric or
track definitions.

---

## A note on the fixture data

All fixture data is fully synthetic and seeded. The generator produces deterministic
output: given the same seed and source code, the output is byte-for-byte identical. This
means any two researchers running the benchmark against the same model, on the same tasks,
using the same fixtures, will get comparable scores.

The synthetic data is designed to be structurally realistic — it reflects patterns
common in actual tutoring program records, including incomplete data, uneven survey
coverage, subgroup disparities, and seasonal operational anomalies. It does not represent
any real program, district, students, or tutors.

---

*Full methodology: [`docs/methodology.md`](../methodology.md)*
*Scoring reference: [`docs/scoring.md`](../scoring.md)*
*Dataset cards: [`docs/dataset_cards/`](../dataset_cards/)*
