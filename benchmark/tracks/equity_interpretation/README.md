# Track 4 — Equity & Subgroup Interpretation

## Purpose

Track 4 evaluates a model's ability to reason carefully and responsibly about subgroup
data, including comparing outcomes across student populations, identifying and correctly
handling low-sample subgroups, and distinguishing directional signals from robust
conclusions. Tasks in this track specifically test whether a model appropriately surfaces
limitations when data is sparse, suppressed, or insufficient for generalizable claims.

## Scope

Tasks cover the following analytical operations:

- **Subgroup comparisons** — contrasting attendance rates, proficiency rates, or other
  metrics across student subgroups defined by IEP status, race/ethnicity, ELL status,
  or other characteristics.
- **Low-N caveat handling** — correctly identifying suppressed subgroups (below the
  n<10 reporting threshold), explaining why suppression is applied, and refusing to
  draw conclusions from suppressed or borderline-small samples.
- **Directional vs. robust distinctions** — articulating when a finding is directionally
  suggestive versus when it can support an actionable program-level conclusion, given the
  sample sizes involved.
- **Fairness and equity implications** — translating subgroup disparities into plain
  language suitable for practitioner audiences (program directors, district analysts,
  board members) while maintaining appropriate analytical humility.
- **Gap quantification** — computing and interpreting attendance and proficiency gaps
  between subgroups (e.g., IEP vs. non-IEP) and contextualizing their magnitude.

## Fixture Pack

All Track 4 tasks use the **Equity & Research pack** located at
`fixtures/pack_equity_research/`. Relevant files include:

| File | Contents |
|---|---|
| `subgroup_attendance_summary.csv` | Monthly attendance rates by subgroup dimension and value, with suppression flags for n<10 groups |
| `subgroup_outcomes_summary.csv` | Fall 2025 and spring 2026 proficiency rates and scale scores by subgroup, with suppression flags |
| `ground_truth.json` | Pre-validated signal values including suppressed subgroup Ns, IEP gaps, and proficiency disparities |
| `students.csv` | One row per enrolled student with IEP, ELL, race/ethnicity, and other demographic flags |
| `program_context.json` | Program goals, reporting-period dates, and equity targets |

## Critical Design Principles

### Every task must test low-N caveat handling

Every Track 4 task includes at least one `required_limitation` explicitly about sample
size — the core competency this track tests. Low-N subgroups from the equity research
pack include:

- **AIAN** (American Indian/Alaska Native): n=4 students (suppressed)
- **NHPI** (Native Hawaiian/Pacific Islander): n=4 students (suppressed)
- **TwoOrMore** (Two or More Races): n=7 students (suppressed)
- **Unknown** race/ethnicity: n=1 student (suppressed)
- **IEP students**: n=14–15 students (reportable but small)

### Every task must forbid overgeneralization

Every Track 4 task includes at least one `forbidden_claim` prohibiting robust
conclusions from suppressed or small subgroups. The canonical forbidden claim form is:
"Robust conclusions about [subgroup] can be drawn from this dataset" when n is below
the threshold for reliable inference.

### Gold facts must match ground_truth.json

All numeric gold facts (IEP attendance gap, proficiency rates, suppressed subgroup Ns)
are validated against `fixtures/pack_equity_research/ground_truth.json`. The key
realized values are:

| Signal | Value |
|---|---|
| Program-wide attendance rate | 80.92% |
| IEP attendance rate | 68.39% |
| Non-IEP attendance rate | 82.43% |
| IEP attendance gap | 14.04 pp |
| IEP proficiency rate (spring 2026) | 39.29% |
| Non-IEP proficiency rate (spring 2026) | 61.77% |
| IEP proficiency gap (spring 2026) | 22.48 pp |

## Task Types

Track 4 uses two `task_type` values from the schema enum:

- `subgroup_comparison` — tasks that ask the model to compare rates or outcomes across
  two or more student subgroups.
- `limitation_assessment` — tasks that ask the model to evaluate what conclusions can
  and cannot be drawn from the available data, given suppression and sample-size
  constraints.

## Rubric Defaults

Track 4 tasks weight **Grounding Accuracy** (0.30–0.35) and
**Calibration & Limitation Handling** (0.20–0.25) more equally than other tracks,
reflecting that both precise facts and appropriate epistemic humility are equally
important in equity analysis. **Insight Quality** (0.20) is also weighted meaningfully
because translating subgroup findings into actionable practitioner language is a core
competency. **Evidence Linkage** (0.15) ensures claims are tied to specific fixture rows.

## Authoring Conventions

- `task_type` is `"subgroup_comparison"` for tasks primarily comparing group rates;
  `"limitation_assessment"` for tasks primarily asking what conclusions can be drawn.
- `allowed_inputs` includes only files the model needs; `ground_truth.json` is included
  when the task relies on pre-validated signal values (IEP gap, suppressed Ns).
- All terminology is practitioner-facing: "IEP designation", "benchmark proficiency",
  "attendance rate", "subgroup suppression" — never internal system nomenclature.
- Unique `fact_id` values within each task follow the `F1`, `F2`, … convention.
