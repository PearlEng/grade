# Dataset Card — Operations Pack (`pack_operations`)

**Pack ID:** `pack_operations`
**Generator seed:** 42
**Generator version:** `grade 0.1.0.dev0`
**Reporting period:** 2025-09-01 to 2025-11-30 (Q1 FY2025-26)
**Primary benchmark tracks:** Track 1 (Grounded Retrieval & Computation), Track 3 (Operational Coaching & Recommendations)

---

## Realism intent

The Operations pack simulates the **first operational quarter of a high-dosage tutoring program** as seen by a district analyst or program manager. The fictional program — Pearl Academic Support Initiative (`PROG-001`) — serves 135 students across three schools in a single district, with nine tutors delivering small-group sessions (3–5 students) three times per week at 45 minutes per session (135 min/student/week target dosage).

The framing is deliberately grounded in the realities of how tutoring programs are actually run and reported:

- **Incomplete data at snapshot time.** Six percent of students are not yet assigned to a tutoring group, mimicking the real-world lag between enrollment and group placement.
- **Uneven survey coverage.** Twenty-two percent of tutors have no session-rating data because surveys had not yet been collected for them. Response rates for the three survey instruments range between 55–80%.
- **Messy attendance.** Eight percent of attendance records record partial minutes rather than a simple present/absent flag, as occurs when a student leaves early or arrives late.
- **School diversity.** Three schools with different locales (urban, suburban, rural), two Title I and one non-Title I, and different enrollment sizes (420, 310, 185) provide a realistic distribution for school-level comparisons.

The October dip in session completions (elevated cancellation rate ~18% vs. the normal ~8–12%) reflects scheduling pressures common in the fall semester and is intentionally embedded as a detectable signal for benchmark tasks.

---

## Tables and row counts

All files live in `fixtures/pack_operations/`. Row counts are for data rows (header excluded).

| File | Type | Rows | Description |
|---|---|---|---|
| `programs.csv` | CSV | 1 | Single program record for Pearl Academic Support Initiative. |
| `schools.csv` | CSV | 3 | One row per school (Lincoln Elementary, Roosevelt Middle, Jefferson Academy). |
| `tutors.csv` | CSV | 9 | Nine tutors across the three schools; 4/3/2 per school. |
| `groups.csv` | CSV | 14 | Fourteen tutoring groups; 6/5/3 per school. |
| `students.csv` | CSV | 135 | One row per enrolled student (60/45/30 per school). |
| `sessions.csv` | CSV | 546 | One row per scheduled session slot over the 13-week quarter. |
| `attendance.csv` | CSV | 4,156 | One row per student per completed session. |
| `surveys.csv` | CSV | 3 | One row per survey instrument (student satisfaction, tutor self-eval, parent feedback). |
| `survey_responses.csv` | CSV | 528 | Per-respondent per-question response rows; sparse by design. |
| `program_context.json` | JSON | — | Narrative context, stated goals, data gaps, implementation challenges. |
| `manifest.json` | JSON | — | SHA-256 hashes and row counts for all files in the pack. |

---

## Column contracts

All column names and types conform to `benchmark/schemas/fixture_schema.md`. Key conventions:

- Boolean columns use `true`/`false` strings in CSV.
- Nullable fields are empty strings in CSV (never `"null"`).
- All dates are ISO 8601 `YYYY-MM-DD`; timestamps are ISO 8601 UTC `YYYY-MM-DDTHH:MM:SSZ`.
- Foreign-key relationships: `students.tutoring_group_id → groups.group_id`, `groups.tutor_id → tutors.tutor_id`, `sessions.group_id → groups.group_id`, `attendance.session_id → sessions.session_id`, `attendance.student_id → students.student_id`, `survey_responses.survey_id → surveys.survey_id`.

---

## Intentional messiness and known limitations

### Synthetic data
All data is fully synthetic and seeded. No real students, tutors, schools, or program records are included. The data is designed to look realistic but does not represent any actual program.

### Seeded determinism
Re-running `.venv/bin/python -m benchmark.datagen.generator --seed 42 --out fixtures/` with the same generator source code produces byte-for-byte identical files. Changing the seed produces a different but equally valid fixture set.

### Missing survey responses (sparsity)
`survey_responses.csv` is intentionally sparse: 18% of question-respondent slots are dropped to simulate real survey non-completion. Not every respondent answers every question. Benchmark tasks that rely on survey averages must handle missing items correctly.

### Unassigned students
About 6% of students in `students.csv` have an empty `tutoring_group_id`. These students appear in the enrollment roster but have no attendance records (since attendance is only generated for group-assigned students with completed sessions). Analysts should not interpret missing attendance records as absences for this cohort.

### No tutor ratings for some tutors
22% of tutors have an empty `avg_session_rating`, representing tutors for whom no survey data was collected by snapshot time. This is different from a zero rating.

### Sessions data covers completed sessions only for attendance
`attendance.csv` contains one row per student per *completed* session only. Cancelled or no-show sessions do not generate attendance rows. The `sessions.csv` table contains all scheduled session slots regardless of status.

### October cancellation signal
Sessions in October (month index 1) have an elevated cancellation rate (~18% vs. the normal ~8–12%). This is intentionally embedded as a detectable operational signal. The `sessions.csv` `status` column carries `cancelled_tutor`, `cancelled_student`, `cancelled_school`, and `no_show` values for non-completed sessions.

### Race/ethnicity subgroup: AIAN low-N
The AIAN (American Indian/Alaska Native) race-ethnicity subgroup contains exactly 4 students program-wide — intentionally below the n < 10 suppression threshold. This subgroup appears in `students.csv` with normal rows but any aggregated metrics computed from this subgroup should be suppressed. (Formal suppression flags are only present in the Equity & Research pack's `subgroup_attendance_summary.csv`; Operations pack users must apply the threshold themselves.)

### IEP attendance disparity
Students with an active IEP (`iep = true`, approximately 12% of students) attend at roughly 11 percentage points below non-IEP peers. This disparity is embedded via the generator's `IEP_ATTENDANCE_PENALTY` constant and is a detectable signal in `attendance.csv`.

### Survey response timing
All three surveys were administered in November 2025 (the final five weeks of the reporting period). Survey responses therefore reflect end-of-quarter sentiment and do not capture within-quarter trends. The parent feedback survey was administered earliest (2025-11-10) and had lower-than-target response rates due to translation delays for non-English-speaking households.

### No outcome or subgroup summary tables
The Operations pack contains raw operational tables only. Month-over-month trend summaries (`monthly_attendance_summary.csv`, `monthly_satisfaction_summary.csv`) are in the Outcomes pack (`pack_outcomes`). Subgroup aggregates with suppression flags are in the Equity & Research pack (`pack_equity_research`).
