# Track 2 — Program Snapshot & Trend Interpretation

## Overview

Track 2 evaluates a model's ability to produce accurate, well-calibrated summaries of
month-over-month program signals.  Tasks ask for attendance trends, satisfaction snapshots,
and multi-signal integration across the Q1 reporting period (September–November 2025).

The defining challenge is **epistemic discipline**: the data contains correlations and
co-occurrences that can superficially motivate causal narratives.  A strong response
reports realized values accurately, characterizes patterns at the right level of confidence,
and explicitly refuses causal claims that the data cannot support.

## Task List

| Task ID     | Title                                                              | Type              |
|-------------|---------------------------------------------------------------------|-------------------|
| T2-OUT-001  | Q1 Attendance Trend: Month-over-Month Change Summary               | trend_analysis    |
| T2-OUT-002  | October Satisfaction Dip: Executive Snapshot with Caveats          | snapshot_summary  |
| T2-OUT-003  | October Cancellation Spike: What the Data Shows and Does Not Show  | trend_analysis    |
| T2-OUT-004  | Program-Wide Q1 Attendance Snapshot Against Stated Goals           | snapshot_summary  |
| T2-OUT-005  | Multi-Signal October Snapshot: Correlating Cancellations, Attendance, and Satisfaction | snapshot_summary |

## Fixture Pack

All tasks draw from `fixtures/pack_outcomes/`.  Key files:

- `ground_truth.json` — realized monthly values computed from generated data (canonical source).
- `monthly_attendance_summary.csv` — school-level session counts, completion rates, and dosage by month.
- `monthly_satisfaction_summary.csv` — survey averages by type and month.
- `program_context.json` — program narrative, stated goals, known data gaps.

## Key Realized Values (ground_truth.json)

| Signal                        | Sep 2025 | Oct 2025 | Nov 2025 |
|-------------------------------|----------|----------|----------|
| Student attendance rate       | 78.71%   | 81.99%   | 82.09%   |
| Parent feedback (avg, 1–5)    | 3.88     | 3.45     | 3.87     |
| Student satisfaction (avg)    | 3.88     | 3.38     | 3.87     |
| Tutor self-eval (avg)         | 3.82     | 3.32     | 3.71     |
| Session cancellation rate     | 10.64%   | 27.89%   | 9.52%    |

Program-wide quarterly attendance: **80.92%** (narrowly above the ≥80% goal).

## Design Notes — Forbidden Claims and Causal Language

The core test of Track 2 is whether a model appropriately withholds causal attribution
when only correlational evidence exists.  Each task includes `forbidden_claims` that
prohibit language such as:

- "The October cancellations **caused** the satisfaction dip"
- "Attendance improvements were **driven by** programmatic changes"
- "The program **achieved** its attendance goal" (when only the aggregate, not every month, cleared 80%)
- "The dip proves a lasting negative effect" (when November showed recovery)

Models that produce these claims score zero on the `calibration_limitation_handling` dimension
regardless of whether their numerical facts are correct.

## Authoring Conventions

- `fact_id` values are unique within each task (`F1`…`F5`).
- `numeric_value` + `tolerance` are set from `ground_truth.json`; never from spec constants.
- Rubric weights sum to 1.0 for every task; `calibration_limitation_handling` is weighted at 0.15
  (above the project default of 0.10) to reflect the causal-reasoning emphasis of this track.
- Educator/analyst vocabulary throughout; no product-specific terminology.
