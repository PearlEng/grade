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
| `monthly_satisfaction_summary.csv` | 3 | 3 survey types, all administered in 2025-11 |
| `program_context.json` | — | Program narrative, goals, and known data gaps |
| `manifest.json` | — | File inventory with SHA-256 hashes and row counts |

---

## Intentional Signal Patterns

The following signals are deterministic given seed 42 and are intended as `gold_facts` /
`gold_insights` for B-task and C-task authors. Changing any `spec.py` constant below
constitutes a schema-breaking change requiring fixture regeneration and B/C task updates.

### Signal 1: Month-over-month attendance trend

**Spec constants:** `MOM_ATTENDANCE_START = 0.72`, `MOM_ATTENDANCE_DELTA = 0.05`,
`MOM_TREND_MONTHS = 3`

The generator embeds an upward attendance trend via a base attendance probability that rises
each month. The actual values in `monthly_attendance_summary.csv` (aggregated program-wide
across all three schools) are:

| Month | Program-wide Attendance Rate | MoM Delta |
|---|---|---|
| 2025-09 (September) | 0.7879 (78.8%) | — (baseline) |
| 2025-10 (October) | 0.8236 (82.4%) | +0.0357 (+3.6 pp) |
| 2025-11 (November) | 0.8196 (82.0%) | −0.0040 (−0.4 pp) |

Per-school values in `monthly_attendance_summary.csv`:

| Month | SCH-001 (Lincoln) | SCH-002 (Roosevelt) | SCH-003 (Jefferson) |
|---|---|---|---|
| 2025-09 | 0.8164 | 0.7477 | 0.7971 |
| 2025-10 | 0.8330 (+0.0167) | 0.8333 (+0.0857) | 0.7933 (−0.0037) |
| 2025-11 | 0.8271 (−0.0059) | 0.7961 (−0.0372) | 0.8394 (+0.0461) |

**Gold fact for B-tasks:** Attendance rose from approximately 79% in September to 82% in
October and maintained roughly that level in November — a net improvement of approximately
+4 percentage points across the reporting period. Roosevelt Middle School showed the sharpest
single-month jump (+8.6 pp, September to October).

### Signal 2: October session cancellation spike (correlated with satisfaction context)

**Spec constants:** `CANCELLATION_RATE_NORMAL = 0.08`, `CANCELLATION_RATE_DIP_MONTH = 0.18`,
`SATISFACTION_DIP_MONTH = 1` (0-indexed, i.e., October)

October is the designated "dip month." Session status weights are shifted to reflect an
elevated cancellation rate. Verified values from `sessions.csv`:

| Month | Total Sessions | Completed | Cancellations + No-Shows | Effective Cancellation Rate |
|---|---|---|---|---|
| 2025-09 (September) | 188 | 168 | 20 | 10.6% |
| 2025-10 (October) | 190 | 137 | 53 | 27.9% |
| 2025-11 (November) | 168 | 152 | 16 | 9.5% |

October's cancellation rate is approximately 2.6× the September rate, consistent with the
`SESSION_STATUS_WEIGHTS_DIP` configuration (`completed` weight reduced from 0.88 to 0.78;
all cancellation weights doubled).

**Note:** The `program_context.json` `implementation_challenges` field explicitly documents:
*"October session cancellation rate was elevated (18%) due to schedule conflicts."*

### Signal 3: Satisfaction dip and recovery

**Spec constants:** `SATISFACTION_BASELINE = 3.8`, `SATISFACTION_DIP_MONTH = 1` (October),
`SATISFACTION_DIP_MAGNITUDE = 0.4`, `SATISFACTION_RECOVERY = 0.4`

The satisfaction dip-and-recovery signal is captured indirectly in this pack. All three
surveys (`SRV-001` student satisfaction, `SRV-002` tutor self-eval, `SRV-003` parent
feedback) were administered in a single end-of-period snapshot on November 15, 20, and 10
respectively. Consequently, `monthly_satisfaction_summary.csv` contains only November rows
with no prior month to compare against, and all `mom_score_delta` values are empty (null)
for the first and only month.

Verified values from `monthly_satisfaction_summary.csv` (2025-11 only):

| Survey Type | avg_score | responses_count | response_rate |
|---|---|---|---|
| `student_satisfaction` | 3.5121 | 248 | 1.8370 |
| `parent_feedback` | 3.1029 | 267 | 1.9778 |
| `tutor_self_eval` | 2.9231 | 13 | 1.4444 |

The contextual signal for the October dip is carried by the session cancellation data
(Signal 2 above) and by `program_context.json`:
*"Student satisfaction dipped in October, correlating with the cancellation spike."*

B-task and C-task authors should use the session-level cancellation signal as the observable
proxy for the October satisfaction dip. The `SATISFACTION_DIP_MAGNITUDE` and
`SATISFACTION_RECOVERY` constants describe the intended conceptual pattern; actual score
measurement exists only at the November endpoint.

---

## Schema Conformance

All files conform to the column contracts in `benchmark/schemas/fixture_schema.md`:

- `monthly_attendance_summary.csv`: 14 columns match §2.1 contract exactly; `summary_id`
  values (`MAS-0001` through `MAS-0009`) are unique; `mom_attendance_delta` and
  `mom_dosage_delta` are empty for the first month of each school and populated thereafter.
- `monthly_satisfaction_summary.csv`: 9 columns match §2.2 contract exactly; `summary_id`
  values (`MSS-0001` through `MSS-0003`) are unique.
- All Operations pack tables (programs, schools, tutors, groups, students, sessions,
  attendance, surveys, survey_responses) conform to §1.1–1.9 contracts.
- `program_context.json`: all required fields present per §1.10.
- `manifest.json`: all required fields present per §4; SHA-256 hashes and row counts
  included for all files.

---

## Known Limitations

1. **Satisfaction summary is single-month only.** All surveys were administered in November.
   The `monthly_satisfaction_summary.csv` has no multi-month trend and all
   `mom_score_delta` values are null. The dip-and-recovery narrative from spec.py is not
   directly observable in the satisfaction table; it is inferred from session cancellation
   rates and `program_context.json`.

2. **Attendance trend does not match spec target exactly.** The spec targets a rise from
   72% to 82% (+10 pp, +5 pp/month). Actual program-wide rates are ~79% / ~82% / ~82%.
   The per-school variance (especially Jefferson Academy's −0.4 pp October dip) causes
   the aggregate to not follow a monotone increase. B-task gold facts should use the
   actual rounded values above, not the spec target constants.

3. **Response-rate values exceed 1.0** in `monthly_satisfaction_summary.csv`. This arises
   because `responses_count` records per-respondent per-question rows (multiple per
   person), while `total_invited` counts distinct people. The `response_rate` column in
   this table is computed as `responses_count / total_invited` and is not a traditional
   response rate. B-tasks should not use `response_rate` from this table as a completion
   percentage.

4. **IEP attendance disparity** (spec signal 3) is embedded in attendance.csv but is
   not summarized in any Outcomes pack table. It is observable by joining `attendance.csv`
   to `students.csv` on `student_id` and grouping by `iep`. The Equity & Research pack
   (`pack_equity_research`) contains pre-aggregated subgroup summaries.

5. **AIAN low-N suppression** applies only to the Equity & Research pack.

6. **Unassigned students** (~6% per spec) have no `tutoring_group_id` and therefore
   no attendance records.

---

## Reproducibility

Re-running `python -m benchmark.datagen.generator --seed 42 --out fixtures/ --pack outcomes`
with the same generator source code produces byte-for-byte identical CSV and JSON files.
The `manifest.json` `generated_at_utc` timestamp will differ, but file content hashes will
match. The integer seed (42) is stored in `program_context.json` and `manifest.json`.
