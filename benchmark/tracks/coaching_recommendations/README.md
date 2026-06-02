# Track 3 — Operational Coaching & Recommendations

## Overview

Track 3 tasks require a model to synthesize quantitative evidence from program
operational data into actionable, educator-facing recommendations. Unlike Track 1
(retrieval and computation) or Track 2 (trend analysis), Track 3 demands judgment:
the model must rank options, assign confidence levels, and produce recommendations
that are both grounded in specific fixture data and appropriately caveated.

**Tiers:** 5–6 (judgment-heavy)
**Task type:** `recommendation`
**Pack:** `pack_operations` (primary fixture source for all five tasks)

---

## Task inventory

| Task ID    | Title                                                                              | Key signals used                              |
|------------|------------------------------------------------------------------------------------|-----------------------------------------------|
| T3-OPS-001 | Identify highest-priority tutor for coaching (attendance)                          | Per-tutor attendance rate, group-level rates  |
| T3-OPS-002 | Prioritize tutors for IEP student support                                          | Per-tutor IEP attendance rate, subgroup gap   |
| T3-OPS-003 | Recommend actions to address the October cancellation spike                        | Monthly cancellation rates, tutor exposure    |
| T3-OPS-004 | Confidence-banded recommendation for school-level attendance intervention          | School-level attendance, sample sizes         |
| T3-OPS-005 | Multi-signal coaching priority list (attendance + cancellation + session volume)   | All three operational signals, tutors.csv YTD |

---

## Design rationale

### Evidence-citation requirement

Every task's `rubric.evidence_linkage` dimension includes an **explicit citation
requirement** stated in the `guidance` field, marked with **REQUIRED**. Responses
that do not name specific entity IDs (tutor, group, or school IDs), numeric rates,
and source fixture files earn zero credit on the evidence-linkage dimension. This
enforces a core property of useful operational recommendations: a practitioner must
be able to trace every recommendation back to a specific data point.

### Confidence banding (T3-OPS-004)

Task T3-OPS-004 explicitly requires the model to assign high/medium/low confidence
bands to its recommendations and justify each band with both a numeric rate and a
sample size. This tests whether the model distinguishes statistical reliability
from point-estimate magnitude — a skill critical for advising on small-school or
small-subgroup data.

### Multi-signal synthesis (T3-OPS-005)

Task T3-OPS-005 requires ranking using three partially conflicting signals. A tutor
may rank highly on attendance but poorly on cancellation rate, or vice versa.
The task tests whether the model can articulate why signals conflict and avoid
false precision in composite scoring.

### IEP subgroup focus (T3-OPS-002)

Task T3-OPS-002 operationalizes the program's stated goal of reducing the IEP
attendance gap to ≤5 percentage points. The realized gap is 14.04 pp
(ground_truth.json), substantially exceeding the goal. The task requires the model
to identify which specific tutors bear the highest coaching priority for IEP
support based on fixture-derived per-tutor IEP attendance rates.

---

## Key ground-truth values (from `fixtures/pack_operations/ground_truth.json`)

These values are authoritative for automated test assertions:

| Metric                      | Value   |
|-----------------------------|---------|
| Program-wide attendance rate | 80.92% |
| September cancellation rate  | 10.64% |
| October cancellation rate    | 27.89% |
| November cancellation rate   | 9.52%  |
| IEP attendance rate          | 68.39% |
| Non-IEP attendance rate      | 82.43% |
| IEP attendance gap           | 14.04 pp |

Additional facts computed from fixture CSVs (recomputed in verification tests):

| Metric                             | Value   |
|------------------------------------|---------|
| TUT-005 attendance rate             | 76.39% |
| TUT-004 quarterly cancellation rate | 23.08% |
| SCH-002 attendance rate             | 78.89% |
| SCH-001 attendance rate             | 82.49% |
| SCH-003 attendance rate             | 81.03% |
| GRP-007 attendance rate             | 73.96% |
| TUT-005 IEP attendance rate         | 62.50% |
| TUT-008 GRP-012 IEP attendance rate | 57.58% |

---

## Running the verification tests

```bash
.venv/bin/pytest tests/unit/test_track_coaching_recommendations.py -v
```

All gold facts are independently recomputed from the Operations pack fixtures and
asserted to match. Facts that also appear in `ground_truth.json` (attendance rates,
cancellation rates, IEP gap) assert the ground_truth value, not a divergent
definition.
