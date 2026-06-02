# Track 5 — Program Effectiveness & Research Reasoning

## Purpose

Track 5 evaluates a model's ability to reason across two distinct evidence layers:
(1) local fixture data from the program under review and (2) a curated set of
external research references (`research_refs.json`).  Tasks require the model to
connect observed program patterns to published findings, identify where local data
confirms or diverges from research expectations, and produce calibrated assessments
that acknowledge both the strength of the evidence base and the limitations of
applying external findings to a single program context.

Unlike Track 1 (grounded retrieval) or Track 4 (equity interpretation), Track 5
tasks demand synthesis rather than computation.  A correct answer integrates specific
local numbers with specific research citations — neither generic research summaries
nor local data presented in isolation are sufficient.

## Scope

Tasks in this track cover the following analytical operations:

- **Predictor ranking** — ordering the evidence-based predictors of effectiveness
  most relevant to the program's current data profile, with explicit research-to-data
  linkage for each predictor.
- **Mixed-evidence comparison** — identifying where local patterns align with and
  where they conflict with published findings, including research references that
  contain `contradicts_claim` fields.
- **Design evaluation** — assessing how a specific program design choice (e.g.,
  stand-alone vs. bundled tutoring) aligns with the research evidence base.
- **Data gap identification** — specifying the additional data types that would most
  strengthen or challenge an effectiveness claim, linked to the specific research
  hypotheses they would allow local testing of.
- **Risk synthesis** — combining local attendance and outcome data with
  dosage-response and threshold evidence to produce a calibrated assessment of
  outcome risk for a specific student subgroup.

## Fixture Pack

All Track 5 tasks use the **Equity & Research pack** located at
`fixtures/pack_equity_research/`.  The critical files for this track are:

| File | Contents |
|---|---|
| `research_refs.json` | 10 research references (R1–R10), each with citation, summary, key_finding, relevance_to_program, supports_claim, contradicts_claim, and caution |
| `ground_truth.json` | Pre-validated program metrics (attendance rates, IEP gap, proficiency rates) used to anchor gold facts |
| `subgroup_attendance_summary.csv` | Monthly attendance rates by subgroup dimension and value |
| `subgroup_outcomes_summary.csv` | Fall 2025 and spring 2026 proficiency rates and score gains by subgroup |
| `students.csv` | Student-level demographic data including IEP, ELL, and FRL flags |
| `program_context.json` | Program goals, dosage targets, implementation challenges, and data notes |

## Research References Used

Each task integrates at least one reference from `research_refs.json`.  The
references most relevant to this track are:

| Ref ID | Key Finding | Used In |
|---|---|---|
| R1 | High-dosage tutoring (≥90 min/week) produces roughly twice the effect size of lower-dosage programs | T5-EFR-001, T5-EFR-003 |
| R2 | Weekly attendance monitoring is associated with 8–12 pp higher realized attendance rates | T5-EFR-001 |
| R3 | Students with IEPs require equity-targeted program design to achieve attendance and outcome parity | T5-EFR-002, T5-EFR-005 |
| R4 | Stand-alone high-dosage tutoring may produce smaller gains than tutoring embedded in a broader instructional bundle | T5-EFR-003 |
| R5 | Embedding tutoring within the school day reduces engagement barriers for high-need students | T5-EFR-002 |
| R6 | Weekly data review of attendance and progress is the single design lever most predictive of sustained effect sizes at scale | T5-EFR-001, T5-EFR-005 |
| R7 | Each additional 20 sessions attended corresponds to ~0.04 SD additional math gain (dosage-response) | T5-EFR-001, T5-EFR-002, T5-EFR-004, T5-EFR-005 |
| R8 | Small-group tutoring of 3–6 students is nearly as effective as one-on-one tutoring for elementary math | T5-EFR-003 |
| R9 | Tutoring interventions for low-SES students yield effect sizes roughly twice those of non-tutoring interventions | T5-EFR-004 |
| R10 | Students with IEPs require ≥80% session attendance to realize the full intensive-intervention effect | T5-EFR-002, T5-EFR-005 |

References with non-null `contradicts_claim` fields (R2, R4, R10) are specifically
used in tasks that ask the model to identify where local patterns diverge from
research expectations.

## Authoring Conventions

- `task_type` for all Track 5 tasks is `"research_reasoning"`.
- Every task's `rubric` or `reference_answer_outline` references at least one
  `ref_id` from `research_refs.json`.
- Every factual `gold_fact` about local data is grounded in `ground_truth.json`
  values where defined, or computed directly from fixture CSVs.  No values are
  invented or approximate: the `iep_attendance_rate`, `iep_attendance_gap_pp`,
  and `iep_proficiency_gap_pp_spring` values in `ground_truth.json` are the
  canonical sources for those metrics across all tasks in this track.
- `allowed_inputs` lists only the files the model needs to answer the specific
  question; files not listed are out of scope.
- All language uses practitioner-facing terminology (e.g., "students with IEPs",
  "attendance rate", "proficiency gap") rather than internal system identifiers.

## Rubric Defaults

Track 5 tasks distribute weight across all six dimensions, with `grounding_accuracy`
(0.30), `insight_quality` (0.25), and `evidence_linkage` (0.20) accounting for the
majority of the score.  This reflects that the primary failure modes are:

1. Asserting local data values that do not match the fixture
2. Citing research findings without connecting them to specific local patterns
3. Failing to distinguish supporting from contradicting evidence

`calibration_limitation_handling` (0.15) is elevated relative to Track 1 defaults
because Track 5 tasks specifically require the model to acknowledge the limits of
applying external effect sizes to a single-program context.
