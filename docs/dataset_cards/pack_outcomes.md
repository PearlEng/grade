# Dataset Card — Outcomes Pack (`pack_outcomes`)

**Pack ID:** `pack_outcomes`
**Generator seed:** 42
**Grade version:** 0.1.0.dev0
**Reporting period:** 2025-09-01 through 2025-11-30 (Q1 FY2025-26)
**Primary benchmark tracks:** Track 2 (Program Snapshot & Trend)

---

## Overview

The Outcomes pack extends the Operations pack with two month-over-month (MoM) time-series
tables: `monthly_attendance_summary.csv` and `monthly_satisfaction_summary.csv`. These
tables are the primary artifact for Track 2 tasks, which require models to identify and
interpret trends, anomalies, and signal patterns across a three-month reporting window.

The pack covers the Pearl Academic Support Initiative (`PROG-001`), a high-dosage tutoring
program serving 135 students across 3 schools in grades 3–8.

**Source of truth for B-task assertions:** All realized signal values are recorded in
`ground_truth.json` (see §2.3 of `fixture_schema.md`). B-task and C-task authors **must**
use those computed values rather than the `spec.py` input constants — the generator applies
noise so realized aggregates differ from target constants.

---

## File Inventory

| File | Rows | Description |
|---|---|---|
| `programs.csv` | 1 | Single program record |
| `schools.csv` | 3 | Lincoln Elementary, Roosevelt Middle School, Jefferson Academy |
| `tutors.csv` | 9 | 4 + 3 + 2 tutors per school |
| `groups.csv` | 14 | 6 + 5 + 3 groups per school |
| `students.csv` | 135 | 60 + 45 + 30 students per school |
| `sessions.csv` | 546 | All scheduled session slots across the reporting period |
| `attendance.csv` | 4,156 | Per-student per-session attendance records (completed sessions only) |
| `surveys.csv` | 3 | Three survey instruments (student satisfaction, tutor self-eval, parent feedback) |
| `survey_responses.csv` | 528 | Individual survey responses (sparse; 18% skip rate per question) |
| `monthly_attendance_summary.csv` | 9 | 3 schools × 3 months = 9 rows |
| `monthly_satisfaction_summary.csv` | 9 | 3 survey types × 3 months = 9 rows |
| `ground_truth.json` | — | Realized signal values computed from the generated data |
| `program_context.json` | — | Program narrative, goals, and known data gaps |
| `manifest.json` | — | File inventory with SHA-256 hashes and row counts |

---

## Intentional Signal Patterns

The following signals are deterministic given seed 42 and are intended as `gold_facts` /
`gold_insights` for B-task and C-task authors. Changing any `spec.py` constant below
constitutes a schema-breaking change requiring fixture regeneration and B/C task updates.

**All realized numbers below are sourced from `ground_truth.json`.** They differ from the
spec.py input targets due to intentional per-school jitter and stochastic attendance draws.

### Signal 1: Month-over-month attendance trend

**Spec constants (inputs only):** `MOM_ATTENDANCE_START = 0.72`, `MOM_ATTENDANCE_DELTA = 0.05`,
`MOM_TREND_MONTHS = 3`

The generator embeds an upward attendance tendency via a base probability that rises each month,
but per-school noise makes the realized program-wide aggregate noisy and non-monotonic.

Realized values from `ground_truth.json > monthly_attendance_rate` (mean across schools):

| Month | Program-wide Attendance Rate | MoM Delta |
|---|---|---|
| 2025-09 (September) | 0.7871 (~78.7%) | — (baseline) |
| 2025-10 (October) | 0.8199 (~82.0%) | +0.0328 (+3.3 pp) |
| 2025-11 (November) | 0.8209 (~82.1%) | +0.0010 (+0.1 pp) |

Per-school values in `monthly_attendance_summary.csv`:

| Month | SCH-001 (Lincoln) | SCH-002 (Roosevelt) | SCH-003 (Jefferson) |
|---|---|---|---|
| 2025-09 | 0.8164 | 0.7477 | 0.7971 |
| 2025-10 | 0.8330 (+0.0167) | 0.8333 (+0.0857) | 0.7933 (−0.0037) |
| 2025-11 | 0.8271 (−0.0059) | 0.7961 (−0.0372) | 0.8394 (+0.0461) |

**Gold fact for B-tasks:** Attendance rose from approximately 79% in September to ~82% in
October and held roughly flat in November — a net improvement of approximately +3 percentage
points program-wide. The trend is non-monotonic at the school level (Jefferson dips in
October; Roosevelt dips in November). Roosevelt Middle School showed the sharpest single-month
jump (+8.6 pp, September to October). Do NOT assert a clean +10 pp monotonic rise.

### Signal 2: October session cancellation spike

**Spec constants:** `CANCELLATION_RATE_NORMAL = 0.08`, `CANCELLATION_RATE_DIP_MONTH = 0.18`,
`SATISFACTION_DIP_MONTH = 1` (0-indexed, i.e., October)

Realized values from `ground_truth.json > monthly_cancellation_rate`:

| Month | Realized Cancellation Rate |
|---|---|
| 2025-09 (September) | 10.6% |
| 2025-10 (October) | 27.9% |
| 2025-11 (November) | 9.5% |

October's cancellation rate is approximately 2.6× the September rate, consistent with the
`SESSION_STATUS_WEIGHTS_DIP` configuration.

**Note:** The `program_context.json` `implementation_challenges` field documents:
*"October session cancellation rate was elevated (18%) due to schedule conflicts."*

### Signal 3: Satisfaction dip and recovery (realized monthly series)

**Spec constants (inputs):** `SATISFACTION_BASELINE = 3.8`, `SATISFACTION_DIP_MONTH = 1`
(October), `SATISFACTION_DIP_MAGNITUDE = 0.4`, `SATISFACTION_RECOVERY = 0.4`

`monthly_satisfaction_summary.csv` now contains a genuine **three-month series** for each
survey type (September / October / November). A dedicated, isolated RNG (`seed + 9999`)
generates these scores so that the Operations and Equity & Research packs remain byte-identical
to their previous versions.

Realized values from `ground_truth.json > monthly_satisfaction`:

| Survey Type | 2025-09 | 2025-10 (dip) | 2025-11 (recovery) | Oct MoM delta |
|---|---|---|---|---|
| `student_satisfaction` | 3.8798 | 3.3827 | 3.8689 | −0.4971 |
| `parent_feedback` | 3.8765 | 3.4486 | 3.8731 | −0.4279 |
| `tutor_self_eval` | 3.8157 | 3.3202 | 3.7065 | −0.4955 |

`satisfaction_dip_realized.dip_confirmed = true` — the October score is below both September
and November for `student_satisfaction`.

**Gold insight for B-tasks:** Student satisfaction dipped by approximately −0.50 Likert
points in October (from ~3.88 to ~3.38) before recovering to ~3.87 in November. The dip
correlates with the October session cancellation spike (Signal 2). Use the exact values from
`ground_truth.json > monthly_satisfaction > student_satisfaction` for assertions.

---

## Schema Conformance

All files conform to the column contracts in `benchmark/schemas/fixture_schema.md`:

- `monthly_attendance_summary.csv`: 14 columns match §2.1 contract exactly; `summary_id`
  values (`MAS-0001` through `MAS-0009`) are unique; `mom_attendance_delta` and
  `mom_dosage_delta` are empty for the first month of each school and populated thereafter.
- `monthly_satisfaction_summary.csv`: 9 columns match §2.2 contract exactly; 9 rows (3 survey
  types × 3 months); `summary_id` values (`MSS-0001` through `MSS-0009`) are unique;
  `mom_score_delta` is empty for September (first month) and populated for October/November.
- `ground_truth.json`: conforms to §2.3 contract; all signal values computed from generated data.
- All Operations pack tables (programs, schools, tutors, groups, students, sessions,
  attendance, surveys, survey_responses) conform to §1.1–1.9 contracts.
- `program_context.json`: all required fields present per §1.10.
- `manifest.json`: all required fields present per §4; SHA-256 hashes and row counts
  included for all files.

---

## Known Limitations

1. **Attendance trend is non-monotonic at program-wide level.** The spec targets a clean
   +5 pp/month rise (72% → 82%), but realized program-wide rates are ~78.7% / ~82.0% / ~82.1%
   due to per-school noise. B-task gold facts must use the `ground_truth.json` values.

2. **Satisfaction response counts are synthetic.** The Sep/Oct monthly rows in
   `monthly_satisfaction_summary.csv` are generated from a dedicated RNG (not from actual
   survey responses — all three surveys were administered in November only). Response counts
   represent plausible monthly cohort sizes, not real monthly survey events.

3. **IEP attendance disparity** (spec signal 3) is embedded in `attendance.csv` but is
   not summarized in any Outcomes pack table. It is observable by joining `attendance.csv`
   to `students.csv` on `student_id` and grouping by `iep`. The Equity & Research pack
   (`pack_equity_research`) contains pre-aggregated subgroup summaries.
   Realized gap (from `ground_truth.json`): IEP rate ~68.4%, non-IEP rate ~82.4%,
   gap ~14.0 pp.

4. **AIAN low-N suppression** applies only to the Equity & Research pack. However, the
   `ground_truth.json > suppressed_low_n_subgroups` in this pack documents the four
   subgroups below the suppression threshold: AIAN (n=4), NHPI (n=4), TwoOrMore (n=7),
   Unknown (n=1).

5. **Unassigned students** (~6% per spec) have no `tutoring_group_id` and therefore
   no attendance records.

---

## Reproducibility

Re-running `python -m benchmark.datagen.generator --seed 42 --out fixtures/ --pack outcomes`
with the same generator source code produces byte-for-byte identical CSV and JSON files.
The `manifest.json` `generated_at_utc` timestamp will differ, but all other file content
hashes will match. The integer seed (42) is stored in `program_context.json`, `manifest.json`,
and `ground_truth.json`.
