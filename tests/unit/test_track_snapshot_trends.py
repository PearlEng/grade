"""Verification tests for Track 2 — Snapshot & Trend Interpretation tasks.

Checks:
1. Every task validates against task_schema.json via validate_task().
2. Every numeric gold_fact matches the value recomputed from the Outcomes fixtures
   / ground_truth.json within the task's own tolerance.
3. Every task has non-empty required_limitations and at least one forbidden_claim.
4. fact_id values are unique within each task.
5. Exactly 5 tasks are present.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.schemas import validate_task

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_TASKS_FILE = _REPO_ROOT / "benchmark" / "tracks" / "snapshot_trends" / "tasks.jsonl"
_FIXTURES = _REPO_ROOT / "fixtures" / "pack_outcomes"
_GT_FILE = _FIXTURES / "ground_truth.json"
_MONTHLY_ATT = _FIXTURES / "monthly_attendance_summary.csv"
_MONTHLY_SAT = _FIXTURES / "monthly_satisfaction_summary.csv"


# ---------------------------------------------------------------------------
# Fixtures — load once
# ---------------------------------------------------------------------------


def _load_tasks() -> list[dict]:
    tasks = []
    with _TASKS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def _load_ground_truth() -> dict[str, Any]:
    with _GT_FILE.open(encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Recomputed reference values from fixtures
# (these mirror the logic used in ground_truth.json generation)
# ---------------------------------------------------------------------------


def _reference_values() -> dict[str, float]:
    """Return a dict of fact_id → expected numeric value recomputed from fixtures.

    Values are derived from ground_truth.json (primary) and
    monthly_attendance_summary.csv / monthly_satisfaction_summary.csv (secondary).
    The ground_truth.json is the canonical source per the pack README.
    """
    gt = _load_ground_truth()
    att_rows = _load_csv(_MONTHLY_ATT)
    sat_rows = _load_csv(_MONTHLY_SAT)

    # ---- Attendance rates from ground_truth.json ----
    att_sep = gt["monthly_attendance_rate"]["2025-09"]  # 0.7871
    att_oct = gt["monthly_attendance_rate"]["2025-10"]  # 0.8199
    att_nov = gt["monthly_attendance_rate"]["2025-11"]  # 0.8209
    att_program = gt["program_wide_attendance_rate"]  # 0.8092

    mom_sep_oct = att_oct - att_sep  # +0.0328
    mom_oct_nov = att_nov - att_oct  # +0.001

    # ---- Cancellation rates from ground_truth.json ----
    canc_sep = gt["monthly_cancellation_rate"]["2025-09"]  # 0.1064
    canc_oct = gt["monthly_cancellation_rate"]["2025-10"]  # 0.2789
    canc_nov = gt["monthly_cancellation_rate"]["2025-11"]  # 0.0952

    # October cancelled session count from monthly_attendance_summary.csv
    oct_canc_total = sum(
        int(r["sessions_cancelled"]) for r in att_rows if r["month_label"] == "2025-10"
    )  # should be 53

    # ---- Satisfaction scores from ground_truth.json (student_satisfaction is canonical) ----
    sat_student_oct = gt["monthly_satisfaction"]["student_satisfaction"]["2025-10"]  # 3.3827
    sat_student_nov = gt["monthly_satisfaction"]["student_satisfaction"]["2025-11"]  # 3.8689

    sat_parent_sep = gt["monthly_satisfaction"]["parent_feedback"]["2025-09"]  # 3.8765
    sat_parent_oct = gt["monthly_satisfaction"]["parent_feedback"]["2025-10"]  # 3.4486
    sat_parent_nov = gt["monthly_satisfaction"]["parent_feedback"]["2025-11"]  # 3.8731

    # Cross-check: parent_feedback Oct avg from monthly_satisfaction_summary.csv
    csv_parent_oct = next(
        float(r["avg_score"])
        for r in sat_rows
        if r["survey_type"] == "parent_feedback" and r["month_label"] == "2025-10"
    )
    assert abs(csv_parent_oct - sat_parent_oct) < 0.001, (
        f"ground_truth parent Oct {sat_parent_oct} != CSV {csv_parent_oct}"
    )

    return {
        # T2-OUT-001 facts
        "T2-OUT-001:F1": att_sep,
        "T2-OUT-001:F2": att_oct,
        "T2-OUT-001:F3": att_nov,
        "T2-OUT-001:F4": mom_sep_oct,
        "T2-OUT-001:F5": mom_oct_nov,
        # T2-OUT-002 facts
        "T2-OUT-002:F1": sat_parent_sep,
        "T2-OUT-002:F2": sat_parent_oct,
        "T2-OUT-002:F3": sat_parent_nov,
        "T2-OUT-002:F4": sat_student_oct,
        "T2-OUT-002:F5": sat_student_nov,
        # T2-OUT-003 facts
        "T2-OUT-003:F1": canc_oct,
        "T2-OUT-003:F2": canc_sep,
        "T2-OUT-003:F3": canc_nov,
        "T2-OUT-003:F4": float(oct_canc_total),
        # T2-OUT-004 facts
        "T2-OUT-004:F1": att_program,
        "T2-OUT-004:F2": att_sep,
        "T2-OUT-004:F3": att_oct,
        "T2-OUT-004:F4": att_nov,
        # T2-OUT-005 facts
        "T2-OUT-005:F1": canc_oct,
        "T2-OUT-005:F2": att_oct,
        "T2-OUT-005:F3": sat_student_oct,
        "T2-OUT-005:F4": sat_parent_oct,
    }


# ---------------------------------------------------------------------------
# Parameterised helpers
# ---------------------------------------------------------------------------

_TASKS = _load_tasks()
_REFERENCE = _reference_values()


# ---------------------------------------------------------------------------
# Test 1: task count
# ---------------------------------------------------------------------------


def test_task_count() -> None:
    """There should be exactly 5 tasks in the snapshot_trends JSONL."""
    assert len(_TASKS) == 5, f"Expected 5 tasks, found {len(_TASKS)}"


# ---------------------------------------------------------------------------
# Test 2: schema validation for every task
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _TASKS, ids=[t["task_id"] for t in _TASKS])
def test_task_validates_against_schema(task: dict) -> None:
    """Every task must pass validate_task() without raising."""
    validate_task(task)


# ---------------------------------------------------------------------------
# Test 3: required_limitations and forbidden_claims non-empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _TASKS, ids=[t["task_id"] for t in _TASKS])
def test_required_limitations_non_empty(task: dict) -> None:
    """Every task must have at least one required_limitation."""
    assert len(task["required_limitations"]) >= 1, (
        f"{task['task_id']}: required_limitations is empty"
    )


@pytest.mark.parametrize("task", _TASKS, ids=[t["task_id"] for t in _TASKS])
def test_forbidden_claims_non_empty(task: dict) -> None:
    """Every task must have at least one forbidden_claim."""
    assert len(task["forbidden_claims"]) >= 1, f"{task['task_id']}: forbidden_claims is empty"


# ---------------------------------------------------------------------------
# Test 4: fact_id uniqueness within each task
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _TASKS, ids=[t["task_id"] for t in _TASKS])
def test_fact_ids_are_unique(task: dict) -> None:
    """fact_id values must be unique within a task."""
    ids = [f["fact_id"] for f in task["gold_facts"]]
    assert len(ids) == len(set(ids)), f"{task['task_id']}: duplicate fact_ids: {ids}"


# ---------------------------------------------------------------------------
# Test 5: numeric gold_facts match recomputed fixture values
# ---------------------------------------------------------------------------


def _numeric_facts(task: dict) -> list[tuple[str, float, float, float]]:
    """Return (key, task_value, expected, tolerance) for facts with numeric_value."""
    result = []
    for fact in task["gold_facts"]:
        if fact.get("numeric_value") is None:
            continue
        key = f"{task['task_id']}:{fact['fact_id']}"
        if key not in _REFERENCE:
            continue
        tol = fact.get("tolerance") or 0.0
        result.append((key, float(fact["numeric_value"]), _REFERENCE[key], tol))
    return result


_ALL_NUMERIC_FACTS: list[tuple[str, float, float, float]] = [
    entry for task in _TASKS for entry in _numeric_facts(task)
]


@pytest.mark.parametrize(
    "key,task_value,expected,tolerance",
    _ALL_NUMERIC_FACTS,
    ids=[e[0] for e in _ALL_NUMERIC_FACTS],
)
def test_numeric_gold_fact_matches_fixture(
    key: str, task_value: float, expected: float, tolerance: float
) -> None:
    """Each numeric gold_fact must match the value recomputed from fixture data."""
    assert abs(task_value - expected) <= tolerance, (
        f"{key}: task value {task_value} differs from fixture-recomputed "
        f"{expected} by {abs(task_value - expected):.6f} (tolerance {tolerance})"
    )


# ---------------------------------------------------------------------------
# Test 6: rubric weights sum to 1.0 for every task
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _TASKS, ids=[t["task_id"] for t in _TASKS])
def test_rubric_weights_sum_to_one(task: dict) -> None:
    """All six rubric dimension weights must sum to 1.0 (within floating-point tolerance)."""
    total = sum(dim["weight"] for dim in task["rubric"].values())
    assert abs(total - 1.0) < 1e-9, (
        f"{task['task_id']}: rubric weights sum to {total}, expected 1.0"
    )


# ---------------------------------------------------------------------------
# Test 7: track field is 2 for every task
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _TASKS, ids=[t["task_id"] for t in _TASKS])
def test_track_is_2(task: dict) -> None:
    """All snapshot_trends tasks must have track == 2."""
    assert task["track"] == 2, f"{task['task_id']}: track={task['track']}, expected 2"


# ---------------------------------------------------------------------------
# Test 8: task_type is a valid Track 2 type
# ---------------------------------------------------------------------------

_VALID_T2_TASK_TYPES = {"trend_analysis", "snapshot_summary"}


@pytest.mark.parametrize("task", _TASKS, ids=[t["task_id"] for t in _TASKS])
def test_task_type_valid_for_track2(task: dict) -> None:
    """task_type must be trend_analysis or snapshot_summary for Track 2."""
    assert task["task_type"] in _VALID_T2_TASK_TYPES, (
        f"{task['task_id']}: task_type='{task['task_type']}' not in {_VALID_T2_TASK_TYPES}"
    )


# ---------------------------------------------------------------------------
# Test 9: causal forbidden_claims — at least one per task mentions causal language
# ---------------------------------------------------------------------------

_CAUSAL_KEYWORDS = {"caused", "cause", "causing", "driven by", "due to", "led to", "result of"}


@pytest.mark.parametrize("task", _TASKS, ids=[t["task_id"] for t in _TASKS])
def test_has_causal_forbidden_claim(task: dict) -> None:
    """Every task must forbid at least one claim that uses explicit causal language."""
    causal_found = any(
        any(kw in claim.lower() for kw in _CAUSAL_KEYWORDS) for claim in task["forbidden_claims"]
    )
    assert causal_found, (
        f"{task['task_id']}: no forbidden_claim contains causal language "
        f"(keywords checked: {_CAUSAL_KEYWORDS})"
    )
