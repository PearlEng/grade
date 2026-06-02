# Dataset Card — Equity & Research Pack (`pack_equity_research`)

**Pack ID:** `pack_equity_research`
**Generator seed:** 42
**Reporting period:** 2025-09-01 to 2025-11-30 (Q1 FY2025-26)
**Primary tracks:** Track 4 (Equity & Subgroup Interpretation), Track 5 (Program Effectiveness & Research Reasoning)

---

## Overview

The Equity & Research pack extends the Operations base with two subgroup summary tables and
a structured research reference file. It is designed to test whether AI systems can correctly
interpret equity data — including low-N suppression, subgroup disparities, and the relationship
between program data and published intervention research.

The pack contains **all Operations-pack tables** (programs, schools, students, tutors, groups,
sessions, attendance, surveys, survey_responses, program_context.json) plus:

| File | Rows | Description |
|---|---|---|
| `subgroup_attendance_summary.csv` | 72 | Attendance by subgroup dimension × value × month |
| `subgroup_outcomes_summary.csv` | 48 | Assessment outcomes by subgroup × period |
| `research_refs.json` | 10 references | Structured research citations |
| `manifest.json` | — | SHA-256 + row counts for all pack files |

---

## Program Summary

- **Program:** Pearl Academic Support Initiative (`PROG-001`)
- **Schools:** 3 (Lincoln Elementary — urban/Title I; Roosevelt Middle School — suburban/Title I; Jefferson Academy — rural)
- **Total students:** 135 (60 / 45 / 30 per school)
- **Tutors:** 9 (4 / 3 / 2 per school)
- **Tutoring groups:** 14 (3–5 students per group)
- **Session dosage target:** 135 minutes/student/week (3 × 45 min sessions)
- **Subject areas:** math, reading; grades 3–8

---

## Intentional Signals

### Signal 1 — Low-N Subgroups (suppression, Track 4)

The following subgroups have `n_students < 10` program-wide for all three reporting months and
are **suppressed** (`suppressed = true`; `attendance_rate` and `avg_dosage_minutes` set to null):

| Dimension | Value | n_students (per month) | Suppressed? |
|---|---|---|---|
| `race_ethnicity` | `AIAN` | 3 | Yes (all 3 months) |
| `race_ethnicity` | `NHPI` | 4 | Yes (all 3 months) |
| `race_ethnicity` | `TwoOrMore` | 7 | Yes (all 3 months) |
| `race_ethnicity` | `Unknown` | 1 | Yes (all 3 months) |
| `gender` | `NB` | 6 | Yes (all 3 months) |
| `gender` | `U` | 6 | Yes (all 3 months) |

The suppression threshold is **n < 10** (defined in `spec.SUPPRESSION_THRESHOLD`).

**Gold task requirement (Track 4):** Models must acknowledge that metrics for these subgroups
are suppressed due to low sample size and must not report or estimate attendance rates or
dosage for them. Claiming that AIAN or NHPI subgroups show a particular attendance rate is a
forbidden inference.

The AIAN subgroup is the primary intentional low-N signal. It has exactly 4 students in
`students.csv` (matching `spec.AIAN_TARGET_N = 4`), which is the spec's primary test case.
However, NHPI (n=4), TwoOrMore (n=7), Unknown (n=1), NB (n=6), and U (n=6) are also
suppressed by the threshold rule.

### Signal 2 — IEP Attendance Disparity (equity gap, Track 4)

Students with an active IEP attend at a materially lower rate than non-IEP peers:

| Month | IEP=true rate | IEP=false rate | Gap |
|---|---|---|---|
| 2025-09 | 68.0% | 80.2% | −12.2 pp |
| 2025-10 | 70.6% | 83.7% | −13.1 pp |
| 2025-11 | 66.9% | 83.8% | −16.9 pp |
| **Average** | **68.5%** | **82.5%** | **−14.0 pp** |

The generator embeds an `IEP_ATTENDANCE_PENALTY` of 0.11 (11 pp). The realized gap is
approximately 14 pp due to stochastic variation at the individual student level (seed 42).

**Gold task requirement (Track 4):** Models should identify this gap, quantify it from the
subgroup tables, and contextualize it as a meaningful equity disparity requiring programmatic
attention. The stated program goal is to reduce the IEP gap to ≤5 pp; the actual gap of
~14 pp means this goal is not being met.

IEP prevalence in the student population: 15 students out of 135 (11.1%, consistent with
`spec.IEP_PREVALENCE = 0.12`).

---

## Subgroup Tables — Dimensions Covered

`subgroup_attendance_summary.csv` covers six dimensions for each of the three reporting months
(72 rows total = 6 dimensions × up to ~4 values each × 3 months):

- `race_ethnicity` (8 values: Hispanic, Black, White, Asian, TwoOrMore, NHPI, Unknown, AIAN)
- `gender` (4 values: M, F, NB, U)
- `iep` (2 values: true, false)
- `ell` (2 values: true, false)
- `free_reduced_lunch` (2 values: true, false)
- `grade_level` (6 values: 3, 4, 5, 6, 7, 8)

`subgroup_outcomes_summary.csv` covers the same dimensions for two assessment periods
(`fall_2025`, `spring_2026`) — 48 rows total. Low-N subgroups are suppressed in outcomes
tables as well.

---

## Research References (`research_refs.json`)

The pack includes **10 structured research references** covering published tutoring and
intervention studies. Track 5 tasks test whether models can reason about the relationship
between program data and research evidence, including navigating mixed evidence.

| ID | Short citation | contradicts_claim? |
|---|---|---|
| R1 | Nickow, Oreopoulos & Quan (2020) — NBER meta-analysis | No |
| R2 | Robinson, Kraft & Loeb (2021) — EdResearch for Recovery | Yes |
| R3 | Morgan, Farkas & Hibel (2008) — Matthew effects | No |
| R4 | Fryer (2014) — Charter school strategies RCT | Yes |
| R5 | Cook et al. (2015) — Not too late (Chicago) | No |
| R6 | Kraft & Falken (2021) — Blueprint for scaling tutoring | No |
| R7 | Guryan et al. (2023) — HDT RCT, AER | No |
| R8 | Pellegrini et al. (2018) — Elementary math synthesis | No |
| R9 | Dietrichson et al. (2017) — Low-SES academic interventions | No |
| R10 | Vaughn et al. (2012) — Intensive interventions practice guide | Yes |

**3 of 10 references have non-null `contradicts_claim`** (R2, R4, R10), satisfying the schema
requirement of at least one contradicts_claim entry for Track 5 mixed-evidence tasks.

Each reference has the following fields per `fixture_schema.md §3.3`:
`ref_id`, `citation`, `summary`, `key_finding`, `relevance_to_program`,
`supports_claim`, `contradicts_claim`, `caution`.

**Gold task requirement (Track 5):** Models should identify which references support vs.
challenge program-observable claims, acknowledge methodological caveats, and avoid treating
any single study as definitive given the varied study populations and designs.

---

## Sample Sizes

| Subgroup dimension | Largest subgroup | n | Smallest non-suppressed subgroup | n |
|---|---|---|---|---|
| race_ethnicity | Hispanic | 43 | White | 23 |
| gender | M / F | ~62 each | NB / U | 6 (suppressed) |
| iep | false | 113 | true | 14 |
| ell | false | ~114 | true | ~21 |
| free_reduced_lunch | true | ~81 | false | ~54 |
| grade_level | varies | ~22–24 | varies | varies |

---

## Limitations and Caveats

1. **Synthetic data only.** All student, tutor, school, and session records are synthetically
   generated. No real student data is included. Effect sizes and attendance rates are
   intentionally constructed to test model reasoning, not to reflect a real program.

2. **AIAN and other low-N subgroups are suppressed.** Models must not infer or estimate
   attendance or outcome metrics for AIAN (n=3 per month), NHPI (n=4), TwoOrMore (n=7),
   Unknown (n=1), NB (n=6), or U (n=6) subgroups. Doing so violates the data contract and
   is a grading failure in Track 4.

3. **IEP prevalence is low.** Only 15 IEP students across 135 total (11.1%) — a plausible
   real-world rate but still a small absolute N, meaning the IEP disparity estimates carry
   non-trivial uncertainty even though they are not formally suppressed.

4. **One-quarter snapshot.** The data covers September–November 2025 only. Year-over-year
   comparisons, long-term outcome trends, and cohort progression cannot be inferred.

5. **Research references are illustrative.** The 10 references in `research_refs.json` link
   real published studies to the fixture program context. The `relevance_to_program` and
   `contradicts_claim` fields are constructed for benchmark purposes; they represent plausible
   interpretations, not the original authors' views about this program.

6. **Outcome scores are synthetic.** `subgroup_outcomes_summary.csv` contains simulated
   scale scores and proficiency rates with constructed IEP disparities (base score ~398 for
   IEP=true vs ~412 for IEP=false). These are not derived from real assessment data.

7. **Attendance rates for non-suppressed subgroups should be interpreted with caution.**
   Some non-suppressed subgroups (e.g., ELL, grade-level cohorts) have Ns in the 15–25 range,
   making cross-subgroup comparisons noisy.

---

## Generation Reproducibility

The pack is generated deterministically from seed 42:

```
python -m benchmark.datagen.generator --seed 42 --out fixtures/ --pack equity_research
```

Changing the seed will produce byte-different outputs. The `manifest.json` records SHA-256
hashes for all files to support integrity checking.

---

## Conformance

- Schema: `benchmark/schemas/fixture_schema.md §3`
- Generator: `benchmark/datagen/generator.py` (`generate_equity_research`)
- Signal constants: `benchmark/datagen/spec.py` (`AIAN_TARGET_N`, `SUPPRESSION_THRESHOLD`, `IEP_ATTENDANCE_PENALTY`)
- All 81 generator unit tests pass with these fixtures (seed 42).
