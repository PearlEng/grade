# GRADE Fixture Schema Contracts

This document is the authoritative column contract for every CSV and JSON file produced by
the A3 synthetic data generator and consumed by the A4/A5/A6 fixture packs.

Downstream tickets that must conform to this contract:
- **A3** (`benchmark/datagen/`) — synthetic generator must produce these exact column names and types.
- **A4** (`fixtures/pack_operations/`) — Operations pack.
- **A5** (`fixtures/pack_outcomes/`) — Outcomes pack.
- **A6** (`fixtures/pack_equity_research/`) — Equity & Research pack.

Conventions used below:
- `PK` — primary key (unique, non-null).
- `FK → table.column` — foreign key reference.
- `nullable` — NULL/empty allowed.
- All date columns use ISO 8601 format (`YYYY-MM-DD`).
- All timestamp columns use ISO 8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`).
- Boolean columns use `true`/`false` strings in CSV; JSON booleans in JSON files.

---

## 1. Operations Pack (`fixtures/pack_operations/`)

The Operations pack represents a single tutoring/intervention program at a single point in time
(a "snapshot quarter"). It is the primary pack for Track 1 (Grounded Retrieval & Computation)
and Track 3 (Operational Coaching & Recommendations) tasks.

### 1.1 `programs.csv`

One row per program. V1 contains exactly one program per pack instance.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `program_id` | string | PK | Stable program identifier, e.g. `PROG-001`. |
| `program_name` | string | non-null | Human-readable program name. |
| `program_type` | string | non-null, enum | One of: `tutoring`, `intervention`, `after_school`, `in_school_support`. |
| `start_date` | date | non-null | Program launch date. |
| `end_date` | date | nullable | Program end date; null if ongoing. |
| `target_grade_levels` | string | non-null | Comma-separated grade levels, e.g. `3,4,5`. |
| `subject_areas` | string | non-null | Comma-separated subjects, e.g. `math,reading`. |
| `district_id` | string | non-null | Parent district identifier. |
| `school_count` | integer | non-null, ≥1 | Number of schools enrolled in the program. |
| `dosage_target_minutes_per_week` | integer | non-null, >0 | Intended instructional minutes per student per week. |
| `reporting_period_start` | date | non-null | Start of the snapshot reporting period. |
| `reporting_period_end` | date | non-null | End of the snapshot reporting period. |

### 1.2 `schools.csv`

One row per school enrolled in the program.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `school_id` | string | PK | Stable school identifier, e.g. `SCH-001`. |
| `school_name` | string | non-null | Human-readable school name. |
| `program_id` | string | FK → programs.program_id | Program this school belongs to. |
| `district_id` | string | non-null | Parent district identifier. |
| `enrollment` | integer | non-null, ≥0 | Total enrolled students at the school (not program-specific). |
| `locale` | string | nullable, enum | One of: `urban`, `suburban`, `rural`, `town`. |
| `title_i` | boolean | non-null | Whether the school qualifies for Title I funding. |
| `grade_span` | string | non-null | Grade span offered, e.g. `K-5`, `6-8`. |

### 1.3 `students.csv`

One row per student enrolled in the program during the reporting period.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `student_id` | string | PK | Anonymized student identifier, e.g. `STU-00001`. |
| `program_id` | string | FK → programs.program_id | Program enrollment. |
| `school_id` | string | FK → schools.school_id | Enrolled school. |
| `grade_level` | integer | non-null, 0–12 | Current grade level (0 = Kindergarten). |
| `gender` | string | nullable | Reported gender; `M`, `F`, `NB`, `U` (unreported). |
| `race_ethnicity` | string | nullable | Reported race/ethnicity code (see note below). |
| `iep` | boolean | non-null | Has an active Individualized Education Program. |
| `ell` | boolean | non-null | English Language Learner status. |
| `free_reduced_lunch` | boolean | non-null | Eligible for free/reduced-price lunch. |
| `enrollment_date` | date | non-null | Date the student was enrolled in the program. |
| `active` | boolean | non-null | Whether the student was active at reporting_period_end. |
| `tutoring_group_id` | string | nullable, FK → groups.group_id | Assigned tutoring group; null if not yet assigned. |

**Note on race_ethnicity codes:** `AIAN`, `Asian`, `Black`, `Hispanic`, `NHPI`, `White`,
`TwoOrMore`, `Unknown`. The generator must produce low-N subgroups (n < 10) for at least one
subgroup to support Track 4 suppression and calibration tasks.

### 1.4 `tutors.csv`

One row per tutor active during the reporting period.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `tutor_id` | string | PK | Anonymized tutor identifier, e.g. `TUT-001`. |
| `program_id` | string | FK → programs.program_id | Program assignment. |
| `school_id` | string | FK → schools.school_id | Primary assigned school. |
| `hire_date` | date | non-null | Date of hire. |
| `certification_level` | string | nullable, enum | One of: `none`, `paraprofessional`, `certified_teacher`, `specialist`. |
| `subject_specialization` | string | nullable | Subject specialism, e.g. `math`, `reading`, `both`. |
| `active` | boolean | non-null | Active at reporting_period_end. |
| `sessions_delivered_ytd` | integer | non-null, ≥0 | Sessions delivered year-to-date. |
| `avg_session_rating` | float | nullable, [1, 5] | Average student-reported session rating; null if no surveys. |

### 1.5 `groups.csv`

One row per tutoring group.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `group_id` | string | PK | Group identifier, e.g. `GRP-001`. |
| `program_id` | string | FK → programs.program_id | Program. |
| `school_id` | string | FK → schools.school_id | School. |
| `tutor_id` | string | FK → tutors.tutor_id | Assigned tutor. |
| `group_size` | integer | non-null, ≥1 | Current number of students in the group. |
| `subject` | string | non-null | Primary subject area: `math`, `reading`, `science`, `other`. |
| `grade_level` | integer | non-null | Grade level of students in this group. |
| `sessions_per_week` | integer | non-null, ≥1 | Scheduled sessions per week. |
| `session_duration_minutes` | integer | non-null, >0 | Scheduled duration per session. |
| `start_date` | date | non-null | Group formation date. |

### 1.6 `sessions.csv`

One row per scheduled session slot (attended or not).

| Column | Type | Constraints | Description |
|---|---|---|---|
| `session_id` | string | PK | Session identifier, e.g. `SES-00001`. |
| `group_id` | string | FK → groups.group_id | Group. |
| `tutor_id` | string | FK → tutors.tutor_id | Delivering tutor. |
| `scheduled_date` | date | non-null | Scheduled session date. |
| `scheduled_start_time` | string | non-null | `HH:MM` local time. |
| `duration_minutes` | integer | non-null, >0 | Scheduled duration. |
| `status` | string | non-null, enum | One of: `completed`, `cancelled_tutor`, `cancelled_student`, `cancelled_school`, `no_show`. |
| `actual_duration_minutes` | integer | nullable | Actual duration if completed; null otherwise. |
| `tutor_on_time` | boolean | nullable | Whether the tutor started within 5 minutes of schedule; null if cancelled. |
| `platform` | string | nullable | Delivery platform: `in_person`, `virtual`, `hybrid`. |

### 1.7 `attendance.csv`

One row per student per session (individual attendance records).

| Column | Type | Constraints | Description |
|---|---|---|---|
| `attendance_id` | string | PK | Attendance record identifier. |
| `session_id` | string | FK → sessions.session_id | Session. |
| `student_id` | string | FK → students.student_id | Student. |
| `attended` | boolean | non-null | Whether the student attended. |
| `minutes_attended` | integer | nullable | Minutes attended if partial attendance; null means full session or absent. |
| `absence_reason` | string | nullable, enum | One of: `illness`, `school_conflict`, `family`, `unknown`, `no_show`; null if attended. |
| `recorded_at` | timestamp | non-null | When the attendance record was submitted. |

### 1.8 `surveys.csv`

One row per survey instrument definition.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `survey_id` | string | PK | Survey instrument identifier, e.g. `SRV-001`. |
| `program_id` | string | FK → programs.program_id | Program. |
| `survey_name` | string | non-null | Human-readable survey name. |
| `survey_type` | string | non-null, enum | One of: `student_satisfaction`, `tutor_self_eval`, `parent_feedback`, `exit_survey`. |
| `administered_date` | date | non-null | Date the survey was administered. |
| `total_invited` | integer | non-null, ≥0 | Number of students/participants invited. |
| `total_responded` | integer | non-null, ≥0 | Number of responses received; must be ≤ total_invited. |
| `response_rate` | float | non-null, [0, 1] | Computed as total_responded / total_invited; 0 if no invitations. |

### 1.9 `survey_responses.csv`

One row per respondent per survey question. May be sparse (not all respondents answer all
questions).

| Column | Type | Constraints | Description |
|---|---|---|---|
| `response_id` | string | PK | Response record identifier. |
| `survey_id` | string | FK → surveys.survey_id | Survey instrument. |
| `student_id` | string | nullable, FK → students.student_id | Respondent (null for anonymous responses). |
| `question_code` | string | non-null | Short question identifier, e.g. `Q1`, `NPS_SCORE`. |
| `question_text` | string | non-null | Full question text. |
| `response_value` | string | non-null | Raw response value (numeric scale values stored as strings for uniformity). |
| `response_type` | string | non-null, enum | One of: `likert_5`, `likert_4`, `nps`, `binary`, `free_text`, `multiple_choice`. |
| `responded_at` | timestamp | non-null | Response submission timestamp. |

### 1.10 `program_context.json`

One JSON file per pack. Contains program-level context that cannot be naturally expressed as
tabular rows.

```json
{
  "program_id": "<string>",
  "narrative_description": "<string>",
  "stated_goals": ["<string>", "..."],
  "primary_contact": {
    "name": "<string>",
    "role": "<string>"
  },
  "data_collection_notes": "<string | null>",
  "known_data_gaps": ["<string>", "..."],
  "implementation_challenges": ["<string>", "..."],
  "reporting_period": {
    "start": "<YYYY-MM-DD>",
    "end": "<YYYY-MM-DD>",
    "label": "<string>"
  },
  "seed": "<integer>"
}
```

**Field notes:**
- `known_data_gaps` must document any intentional gaps (missing schools, low survey coverage)
  so that gold tasks can require the model to acknowledge them.
- `seed` is the integer random seed used to generate this pack instance; enables reproducibility.

### 1.11 `ground_truth.json` (Operations pack)

One JSON file per Operations pack. Contains **realized signal values** computed from the
generated data so that B-task authors can assert exact numbers.

```json
{
  "pack_id": "pack_operations",
  "seed": "<integer>",
  "note": "<string>",
  "monthly_attendance_rate": {
    "2025-09": "<float>",
    "2025-10": "<float>",
    "2025-11": "<float>"
  },
  "program_wide_attendance_rate": "<float>",
  "monthly_cancellation_rate": {
    "2025-09": "<float>",
    "2025-10": "<float>",
    "2025-11": "<float>"
  },
  "iep_attendance_rate": "<float>",
  "non_iep_attendance_rate": "<float>",
  "iep_attendance_gap_pp": "<float>",
  "suppressed_low_n_subgroups": [
    {
      "subgroup_dimension": "race_ethnicity",
      "subgroup_value": "<string>",
      "n": "<integer>"
    }
  ]
}
```

**Field notes:**
- `monthly_attendance_rate` — per-month realized attendance rate, computed directly from
  raw ``attendance.csv`` records (not from spec constants).
- `monthly_cancellation_rate` — fraction of scheduled sessions cancelled or no-show per month.
- `iep_attendance_gap_pp` — realized gap in percentage points: non-IEP rate minus IEP rate.
- `suppressed_low_n_subgroups` — race/ethnicity subgroups with n < suppression threshold.

**Important:** B-task authors MUST use ``ground_truth.json`` values for assertions, not
``spec.py`` constants or dataset-card prose.

---

## 2. Outcomes Pack (`fixtures/pack_outcomes/`)

The Outcomes pack extends the Operations data with month-over-month time-series tables. It is
the primary pack for Track 2 (Program Snapshot & Trend) tasks.

The Outcomes pack reuses all Operations pack tables (with the same column contracts) PLUS the
following additional tables:

### 2.1 `monthly_attendance_summary.csv`

One row per (program × school × month). Enables month-over-month attendance trend analysis.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `summary_id` | string | PK | Record identifier. |
| `program_id` | string | FK → programs.program_id | Program. |
| `school_id` | string | FK → schools.school_id | School. |
| `month_label` | string | non-null | ISO year-month, e.g. `2025-09`. |
| `month_start` | date | non-null | First day of the month. |
| `month_end` | date | non-null | Last day of the month. |
| `sessions_scheduled` | integer | non-null, ≥0 | Total sessions scheduled in this period. |
| `sessions_completed` | integer | non-null, ≥0 | Sessions completed (status = `completed`). |
| `sessions_cancelled` | integer | non-null, ≥0 | Sessions cancelled for any reason. |
| `attendance_rate` | float | non-null, [0, 1] | Fraction of student-session slots attended. |
| `avg_dosage_minutes` | float | non-null, ≥0 | Average minutes of instruction per enrolled student. |
| `active_students` | integer | non-null, ≥0 | Students with ≥1 attended session this month. |
| `mom_attendance_delta` | float | nullable | Month-over-month change in attendance_rate; null for the first month. |
| `mom_dosage_delta` | float | nullable | Month-over-month change in avg_dosage_minutes; null for the first month. |

**Generator note (A3):** The generator must embed at least one statistically notable
month-over-month trend (positive or negative) that can serve as a `gold_fact` in Track 2
tasks. The trend value and direction must be deterministic given the pack seed.

### 2.2 `monthly_satisfaction_summary.csv`

One row per (program × school × survey_type × month).

| Column | Type | Constraints | Description |
|---|---|---|---|
| `summary_id` | string | PK | Record identifier. |
| `program_id` | string | FK → programs.program_id | Program. |
| `school_id` | string | FK → schools.school_id | School. |
| `survey_type` | string | non-null | Matches surveys.survey_type. |
| `month_label` | string | non-null | ISO year-month. |
| `responses_count` | integer | non-null, ≥0 | Number of responses collected this month. |
| `avg_score` | float | nullable | Mean Likert/NPS/binary score; null if no responses. |
| `response_rate` | float | nullable, [0, 1] | Response rate for this month; null if not applicable. |
| `mom_score_delta` | float | nullable | Month-over-month change in avg_score; null for the first month. |

### 2.3 `ground_truth.json`

One JSON file per Outcomes pack. Contains **realized signal values** computed from the
generated data — not the spec.py input constants — so B-task authors can assert exact numbers.

```json
{
  "pack_id": "pack_outcomes",
  "seed": "<integer>",
  "note": "<string>",
  "monthly_attendance_rate": {
    "2025-09": "<float>",
    "2025-10": "<float>",
    "2025-11": "<float>"
  },
  "program_wide_attendance_rate": "<float>",
  "monthly_satisfaction": {
    "<survey_type>": {
      "2025-09": "<float>",
      "2025-10": "<float>",
      "2025-11": "<float>"
    }
  },
  "satisfaction_dip_realized": {
    "dip_month": "2025-10",
    "scores_by_month": {
      "2025-09": "<float>",
      "2025-10": "<float>",
      "2025-11": "<float>"
    },
    "dip_confirmed": "<boolean>"
  },
  "monthly_cancellation_rate": {
    "2025-09": "<float>",
    "2025-10": "<float>",
    "2025-11": "<float>"
  },
  "iep_attendance_rate": "<float>",
  "non_iep_attendance_rate": "<float>",
  "iep_attendance_gap_pp": "<float>",
  "suppressed_low_n_subgroups": [
    {
      "subgroup_dimension": "race_ethnicity",
      "subgroup_value": "<string>",
      "n": "<integer>"
    }
  ]
}
```

**Field notes:**
- `monthly_attendance_rate` — program-wide average attendance rate per month (mean across schools).
- `program_wide_attendance_rate` — overall rate across all months and schools.
- `monthly_satisfaction` — per-survey-type per-month realized avg_score from ``monthly_satisfaction_summary.csv``.
- `satisfaction_dip_realized.dip_confirmed` — boolean; true if the October score is lower than both September and November scores for ``student_satisfaction``.
- `monthly_cancellation_rate` — fraction of scheduled sessions cancelled or no-show each month.
- `iep_attendance_gap_pp` — realized gap in percentage points: non-IEP rate minus IEP rate.
- `suppressed_low_n_subgroups` — race/ethnicity subgroups with n < suppression threshold.

**Important:** B-task authors MUST use ``ground_truth.json`` values for assertions, not ``spec.py``
constants.  The generator applies noise so realized values differ from input targets.

---

## 3. Equity & Research Pack (`fixtures/pack_equity_research/`)

The Equity & Research pack extends the Operations base with subgroup tables and research
references. It is the primary pack for Track 4 (Equity & Subgroup Interpretation) and
Track 5 (Program Effectiveness & Research Reasoning) tasks.

The Equity & Research pack reuses all Operations pack tables PLUS:

### 3.1 `subgroup_attendance_summary.csv`

One row per (program × subgroup dimension × subgroup value × month).

| Column | Type | Constraints | Description |
|---|---|---|---|
| `summary_id` | string | PK | Record identifier. |
| `program_id` | string | FK → programs.program_id | Program. |
| `school_id` | string | nullable, FK → schools.school_id | School; null for program-wide aggregates. |
| `month_label` | string | non-null | ISO year-month. |
| `subgroup_dimension` | string | non-null, enum | One of: `race_ethnicity`, `gender`, `iep`, `ell`, `free_reduced_lunch`, `grade_level`. |
| `subgroup_value` | string | non-null | Value within the dimension, e.g. `Hispanic`, `M`, `true`. |
| `n_students` | integer | non-null, ≥0 | Students in this subgroup this month. |
| `n_sessions_attended` | integer | non-null, ≥0 | Total sessions attended by this subgroup. |
| `attendance_rate` | float | nullable, [0, 1] | Attendance rate; null if n_students < suppression threshold. |
| `avg_dosage_minutes` | float | nullable | Average dosage; null if n_students < suppression threshold. |
| `suppressed` | boolean | non-null | Whether values are suppressed due to low N (n < 10). |
| `suppression_reason` | string | nullable | Reason for suppression if suppressed = true, e.g. `n < 10`. |

**Generator note (A3):** At least one subgroup must have n_students < 10 for ≥1 month, with
`suppressed = true` and `attendance_rate = null`. Gold tasks for Track 4 must require the
model to acknowledge low-N suppression.

### 3.2 `subgroup_outcomes_summary.csv`

One row per (program × subgroup × assessment period). Captures outcome disparities.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `summary_id` | string | PK | Record identifier. |
| `program_id` | string | FK → programs.program_id | Program. |
| `school_id` | string | nullable | School; null for program-wide. |
| `assessment_period` | string | non-null | Label for the assessment window, e.g. `fall_2025`, `spring_2026`. |
| `subgroup_dimension` | string | non-null | See subgroup_attendance_summary. |
| `subgroup_value` | string | non-null | Value within dimension. |
| `n_students` | integer | non-null, ≥0 | Students assessed. |
| `avg_scale_score` | float | nullable | Mean standardized scale score; null if suppressed. |
| `avg_score_gain` | float | nullable | Mean gain from baseline; null if no baseline or suppressed. |
| `benchmark_proficiency_rate` | float | nullable, [0, 1] | Fraction meeting proficiency benchmark; null if suppressed. |
| `suppressed` | boolean | non-null | Low-N suppression flag. |
| `suppression_reason` | string | nullable | Suppression reason. |

### 3.3 `research_refs.json`

One JSON file per pack. Contains structured references to published research that Track 5
tasks may ask the model to reason about in relation to the fixture data.

```json
{
  "program_id": "<string>",
  "references": [
    {
      "ref_id": "<string>",
      "citation": "<string>",
      "summary": "<string>",
      "key_finding": "<string>",
      "relevance_to_program": "<string>",
      "supports_claim": "<string | null>",
      "contradicts_claim": "<string | null>",
      "caution": "<string | null>"
    }
  ]
}
```

**Field notes:**
- `ref_id` — short identifier, e.g. `R1`, `R2`.
- `citation` — author-year style citation for the synthetic reference.
- `summary` — 2–4 sentence abstract of the referenced work.
- `key_finding` — the single most relevant finding for this program context.
- `relevance_to_program` — why this reference is pertinent to the fixture program.
- `supports_claim` — if non-null, the reference supports this program-observable claim.
- `contradicts_claim` — if non-null, the reference is in tension with an observable claim.
- `caution` — methodological or generalizability caveat the model should acknowledge.

**Generator note (A3):** At least two research references must be included per pack; at least
one must have a non-null `contradicts_claim` to enable Track 5 research-reasoning tasks that
require the model to navigate mixed evidence.

### 3.4 `ground_truth.json` (Equity & Research pack)

One JSON file per Equity & Research pack. Contains **realized signal values** computed from
the generated data so that B-task authors can assert exact numbers.

```json
{
  "pack_id": "pack_equity_research",
  "seed": "<integer>",
  "note": "<string>",
  "monthly_attendance_rate": {
    "2025-09": "<float>",
    "2025-10": "<float>",
    "2025-11": "<float>"
  },
  "program_wide_attendance_rate": "<float>",
  "iep_attendance_rate": "<float>",
  "non_iep_attendance_rate": "<float>",
  "iep_attendance_gap_pp": "<float>",
  "suppressed_low_n_subgroups": [
    {
      "subgroup_dimension": "race_ethnicity",
      "subgroup_value": "<string>",
      "n": "<integer>"
    }
  ],
  "iep_proficiency_gap_pp_spring": "<float>",
  "subgroup_outcome_disparities": [
    {
      "assessment_period": "<string>",
      "subgroup_dimension": "<string>",
      "subgroup_value": "<string>",
      "n_students": "<integer>",
      "benchmark_proficiency_rate": "<float | null>"
    }
  ]
}
```

**Field notes:**
- `iep_attendance_gap_pp` — realized attendance gap: non-IEP rate minus IEP rate (pp).
- `iep_proficiency_gap_pp_spring` — realized proficiency rate gap (spring_2026):
  non-IEP benchmark_proficiency_rate minus IEP rate, in percentage points.
- `subgroup_outcome_disparities` — IEP subgroup rows from ``subgroup_outcomes_summary.csv``
  for ``spring_2026`` (non-suppressed only), for asserting outcome parity tasks.

**Important:** B-task authors MUST use ``ground_truth.json`` values for assertions, not
``spec.py`` constants.

---

## 4. Cross-pack Notes

### Suppression threshold
The default low-N suppression threshold is **n < 10**. The generator must respect this across
all subgroup tables and set `suppressed = true` and null out metric columns accordingly.

### Seed and reproducibility
Every pack is generated with a deterministic integer seed stored in `program_context.json`.
Given the same seed and generator version, outputs must be byte-for-byte identical.

### Pack file inventory
Each pack directory must contain a `manifest.json` listing every file in the pack with its
SHA-256 hash and row count (for CSV files). This enables integrity checking by the runner.

```json
{
  "pack_id": "<string>",
  "grade_version": "<string>",
  "seed": "<integer>",
  "generated_at_utc": "<ISO 8601>",
  "files": [
    {
      "filename": "<string>",
      "sha256": "<hex string>",
      "row_count": "<integer | null>"
    }
  ]
}
```
