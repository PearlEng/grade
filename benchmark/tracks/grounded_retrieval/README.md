# Track 1 — Grounded Retrieval & Computation

## Purpose

Track 1 evaluates a model's ability to accurately retrieve specific facts and perform
straightforward arithmetic computations from structured tabular data (CSV files) and
program context documents.  Tasks in this track require no multi-step inference or
causal reasoning; a correct answer is fully determinable from the provided fixture data.

## Scope

Tasks cover the following analytical operations:

- **Direct counts** — enumeration of entities (students, groups, schools, sessions) by
  filtering and aggregating rows from CSV fixtures.
- **Rate computation** — calculation of attendance and cancellation rates using
  attendance.csv joined to sessions.csv with appropriate filters (e.g., completed
  sessions only, specific date ranges, subgroup membership).
- **Date-range filtering** — isolation of records within a calendar month or the full
  reporting quarter (September–November 2025).
- **Relative-time interpretation** — resolving natural-language references such as
  "last month" to a concrete date range given an assumed reporting date.
- **Scope comparisons** — contrasting rates or counts across months, schools, or
  student subgroups (e.g., students with and without an IEP designation).

## Fixture Pack

All Track 1 tasks use the **Operations pack** located at `fixtures/pack_operations/`.
Relevant files include:

| File | Contents |
|---|---|
| `students.csv` | One row per enrolled student with school, grade, IEP/ELL/FRL flags, and group assignment |
| `sessions.csv` | One row per scheduled session with date, group, status, and delivery platform |
| `attendance.csv` | One row per student per completed session with an attended flag and optional minutes |
| `groups.csv` | One row per tutoring group with school, tutor, subject, and group size |
| `program_context.json` | Reporting-period dates, program goals, and known data gaps |

## Authoring Conventions

- `task_type` for direct lookups is `"retrieval"`; tasks requiring arithmetic
  (e.g., computing a rate) use `"computation"`.
- `allowed_inputs` lists only the files the model needs; files not listed are out of scope.
- Every `gold_fact` numeric value was computed directly from the fixture CSVs
  (see `fixtures/pack_operations/ground_truth.json` for pre-validated signal values).
- All terminology uses practitioner-facing language (e.g., "tutoring group", "attendance
  rate", "IEP designation") rather than internal system nomenclature.

## Rubric Defaults

Track 1 tasks weight **Grounding Accuracy** most heavily (typically 0.40–0.45) because
the primary failure mode is hallucinating or miscomputing a specific number.
Evidence Linkage is also elevated (0.15–0.20) to ensure the model cites the specific
columns and files it used.
