"""Verification tests for Track 1 (Grounded Retrieval & Computation) tasks.

These tests enforce two guarantees:

1. **Schema validity** — every task in tasks.jsonl must pass ``validate_task``.
2. **Gold-fact correctness** — every ``numeric_value`` gold fact is independently
   recomputed from the Operations pack fixture CSVs and asserted to match.

Running ``pytest tests/unit/test_track_grounded_retrieval.py`` must pass before any
Track 1 task is merged.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import pytest

from benchmark.schemas import validate_task

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[2]
_TASKS_PATH = _REPO_ROOT / "benchmark" / "tracks" / "grounded_retrieval" / "tasks.jsonl"
_PACK_DIR = _REPO_ROOT / "fixtures" / "pack_operations"

# ---------------------------------------------------------------------------
# Helpers — load fixture data once
# ---------------------------------------------------------------------------


def _load_csv(name: str) -> list[dict[str, str]]:
    """Load a pack CSV file and return a list of row dicts."""
    with (_PACK_DIR / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_tasks() -> list[dict]:
    """Load all tasks from the JSONL file."""
    tasks = []
    with _TASKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


# Fixture data loaded once at module level so individual tests stay fast.
_STUDENTS = _load_csv("students.csv")
_SESSIONS = _load_csv("sessions.csv")
_ATTENDANCE = _load_csv("attendance.csv")
_GROUPS = _load_csv("groups.csv")

# ---------------------------------------------------------------------------
# Recomputation helpers
# ---------------------------------------------------------------------------


def _attendance_rate(session_ids: set[str], attendance: list[dict]) -> tuple[int, int]:
    """Return (attended_count, total_count) for the given session IDs."""
    rows = [a for a in attendance if a["session_id"] in session_ids]
    attended = sum(1 for a in rows if a["attended"] == "true")
    return attended, len(rows)


def _completed_sessions_in_month(sessions: list[dict], year_month: str) -> set[str]:
    """Return session_ids for completed sessions whose date starts with year_month."""
    return {
        s["session_id"]
        for s in sessions
        if s["status"] == "completed" and s["scheduled_date"].startswith(year_month)
    }


def _all_completed_sessions(sessions: list[dict]) -> set[str]:
    return {s["session_id"] for s in sessions if s["status"] == "completed"}


def _sessions_in_month(sessions: list[dict], year_month: str) -> list[dict]:
    return [s for s in sessions if s["scheduled_date"].startswith(year_month)]


# ---------------------------------------------------------------------------
# Task loading and schema validation
# ---------------------------------------------------------------------------


class TestTasksLoading:
    """Structural tests for the tasks.jsonl file itself."""

    def test_tasks_file_exists(self) -> None:
        """The tasks.jsonl file must exist at the expected path."""
        assert _TASKS_PATH.exists(), f"tasks.jsonl not found at {_TASKS_PATH}"

    def test_approximately_six_tasks(self) -> None:
        """The file must contain approximately six tasks (at least 5, at most 8)."""
        tasks = _load_tasks()
        assert 5 <= len(tasks) <= 8, f"Expected ~6 tasks, found {len(tasks)}"

    def test_all_tasks_are_track_1(self) -> None:
        """Every task must declare track=1."""
        for task in _load_tasks():
            assert task["track"] == 1, (
                f"Task {task['task_id']} has track={task['track']}, expected 1"
            )

    def test_all_task_ids_unique(self) -> None:
        """task_id values must be globally unique within the file."""
        tasks = _load_tasks()
        ids = [t["task_id"] for t in tasks]
        assert len(ids) == len(set(ids)), f"Duplicate task IDs found: {ids}"

    def test_fact_ids_unique_within_each_task(self) -> None:
        """fact_id values must be unique within each task."""
        for task in _load_tasks():
            fact_ids = [f["fact_id"] for f in task.get("gold_facts", [])]
            assert len(fact_ids) == len(set(fact_ids)), (
                f"Task {task['task_id']} has duplicate fact_ids: {fact_ids}"
            )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _load_tasks(), ids=[t["task_id"] for t in _load_tasks()])
def test_task_schema_valid(task: dict) -> None:
    """Every task in tasks.jsonl must pass validate_task without error."""
    validate_task(task)  # raises jsonschema.ValidationError on failure


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests
# ---------------------------------------------------------------------------


class TestT1OPS001GoldFacts:
    """Recompute gold facts for T1-OPS-001 (student and group counts by school)."""

    def test_f1_total_students(self) -> None:
        """F1: 135 total enrolled students."""
        assert len(_STUDENTS) == 135

    def test_f2_sch001_students(self) -> None:
        """F2: 60 students enrolled at SCH-001."""
        count = sum(1 for s in _STUDENTS if s["school_id"] == "SCH-001")
        assert count == 60

    def test_f3_sch002_students(self) -> None:
        """F3: 45 students enrolled at SCH-002."""
        count = sum(1 for s in _STUDENTS if s["school_id"] == "SCH-002")
        assert count == 45

    def test_f4_sch003_students(self) -> None:
        """F4: 30 students enrolled at SCH-003."""
        count = sum(1 for s in _STUDENTS if s["school_id"] == "SCH-003")
        assert count == 30

    def test_f5_total_groups(self) -> None:
        """F5: 14 tutoring groups across all schools."""
        assert len(_GROUPS) == 14

    def test_f6_groups_by_school(self) -> None:
        """F6: SCH-001=6 groups, SCH-002=5 groups, SCH-003=3 groups."""
        by_school: dict[str, int] = defaultdict(int)
        for g in _GROUPS:
            by_school[g["school_id"]] += 1
        assert by_school["SCH-001"] == 6
        assert by_school["SCH-002"] == 5
        assert by_school["SCH-003"] == 3


class TestT1OPS002GoldFacts:
    """Recompute gold facts for T1-OPS-002 (program-wide quarterly attendance rate)."""

    def test_f1_completed_session_count(self) -> None:
        """F1: 457 completed sessions across the quarter."""
        completed = sum(1 for s in _SESSIONS if s["status"] == "completed")
        assert completed == 457

    def test_f2_attendance_record_count(self) -> None:
        """F2: 4156 attendance records linked to completed sessions."""
        completed_ids = _all_completed_sessions(_SESSIONS)
        att = [a for a in _ATTENDANCE if a["session_id"] in completed_ids]
        assert len(att) == 4156

    def test_f3_attended_count(self) -> None:
        """F3: 3363 attended records among completed sessions."""
        completed_ids = _all_completed_sessions(_SESSIONS)
        att = [a for a in _ATTENDANCE if a["session_id"] in completed_ids]
        attended = sum(1 for a in att if a["attended"] == "true")
        assert attended == 3363

    def test_f4_program_wide_attendance_rate(self) -> None:
        """F4: Program-wide attendance rate is 0.8092 ± 0.0005."""
        completed_ids = _all_completed_sessions(_SESSIONS)
        attended, total = _attendance_rate(completed_ids, _ATTENDANCE)
        rate = attended / total
        assert abs(rate - 0.8092) <= 0.0005, f"Rate was {rate:.4f}"


class TestT1OPS003GoldFacts:
    """Recompute gold facts for T1-OPS-003 (monthly attendance rate comparison)."""

    def test_f1_september_attendance_rate(self) -> None:
        """F1: September attendance rate is 0.7879 ± 0.0005."""
        sep_ids = _completed_sessions_in_month(_SESSIONS, "2025-09")
        attended, total = _attendance_rate(sep_ids, _ATTENDANCE)
        rate = attended / total
        assert abs(rate - 0.7879) <= 0.0005, f"Sep rate was {rate:.4f}"

    def test_f2_october_attendance_rate(self) -> None:
        """F2: October attendance rate is 0.8236 ± 0.0005."""
        oct_ids = _completed_sessions_in_month(_SESSIONS, "2025-10")
        attended, total = _attendance_rate(oct_ids, _ATTENDANCE)
        rate = attended / total
        assert abs(rate - 0.8236) <= 0.0005, f"Oct rate was {rate:.4f}"

    def test_f3_november_attendance_rate(self) -> None:
        """F3: November attendance rate is 0.8196 ± 0.0005."""
        nov_ids = _completed_sessions_in_month(_SESSIONS, "2025-11")
        attended, total = _attendance_rate(nov_ids, _ATTENDANCE)
        rate = attended / total
        assert abs(rate - 0.8196) <= 0.0005, f"Nov rate was {rate:.4f}"

    def test_f4_sep_to_oct_gap(self) -> None:
        """F4: Sep is lowest (78.8%), Oct is highest (82.4%), gap ≈ 3.57 pp."""
        sep_ids = _completed_sessions_in_month(_SESSIONS, "2025-09")
        oct_ids = _completed_sessions_in_month(_SESSIONS, "2025-10")
        sep_attended, sep_total = _attendance_rate(sep_ids, _ATTENDANCE)
        oct_attended, oct_total = _attendance_rate(oct_ids, _ATTENDANCE)
        sep_rate = sep_attended / sep_total
        oct_rate = oct_attended / oct_total
        gap_pp = (oct_rate - sep_rate) * 100
        assert abs(gap_pp - 3.57) <= 0.1, f"Gap was {gap_pp:.2f} pp"


class TestT1OPS004GoldFacts:
    """Recompute gold facts for T1-OPS-004 (October cancellation spike)."""

    def test_f1_september_cancellation_rate(self) -> None:
        """F1: September cancellation rate ≈ 0.0798 ± 0.001."""
        sep_sess = _sessions_in_month(_SESSIONS, "2025-09")
        cancelled = sum(1 for s in sep_sess if "cancelled" in s["status"])
        rate = cancelled / len(sep_sess)
        assert abs(rate - 0.0798) <= 0.001, f"Sep cancel rate was {rate:.4f}"

    def test_f2_october_cancellation_rate(self) -> None:
        """F2: October cancellation rate ≈ 0.2105 ± 0.001."""
        oct_sess = _sessions_in_month(_SESSIONS, "2025-10")
        cancelled = sum(1 for s in oct_sess if "cancelled" in s["status"])
        rate = cancelled / len(oct_sess)
        assert abs(rate - 0.2105) <= 0.001, f"Oct cancel rate was {rate:.4f}"

    def test_f3_november_cancellation_rate(self) -> None:
        """F3: November cancellation rate ≈ 0.0655 ± 0.001."""
        nov_sess = _sessions_in_month(_SESSIONS, "2025-11")
        cancelled = sum(1 for s in nov_sess if "cancelled" in s["status"])
        rate = cancelled / len(nov_sess)
        assert abs(rate - 0.0655) <= 0.001, f"Nov cancel rate was {rate:.4f}"

    def test_f4_october_vs_sep_nov_average(self) -> None:
        """F4: October rate minus average of Sep/Nov is approximately 13.8 pp ± 0.3."""
        sep_sess = _sessions_in_month(_SESSIONS, "2025-09")
        oct_sess = _sessions_in_month(_SESSIONS, "2025-10")
        nov_sess = _sessions_in_month(_SESSIONS, "2025-11")

        def rate(sess: list[dict]) -> float:
            cancelled = sum(1 for s in sess if "cancelled" in s["status"])
            return cancelled / len(sess)

        sep_rate = rate(sep_sess)
        oct_rate = rate(oct_sess)
        nov_rate = rate(nov_sess)
        diff = (oct_rate - (sep_rate + nov_rate) / 2) * 100
        assert abs(diff - 13.8) <= 0.3, f"Difference was {diff:.2f} pp"


class TestT1OPS005GoldFacts:
    """Recompute gold facts for T1-OPS-005 (IEP vs non-IEP attendance gap)."""

    def test_f1_iep_and_non_iep_counts(self) -> None:
        """F1: 15 students with IEP=true, 120 with IEP=false."""
        iep = sum(1 for s in _STUDENTS if s["iep"] == "true")
        non_iep = sum(1 for s in _STUDENTS if s["iep"] == "false")
        assert iep == 15
        assert non_iep == 120

    def test_f2_iep_attendance_rate(self) -> None:
        """F2: IEP student attendance rate ≈ 0.6839 ± 0.001."""
        iep_ids = {s["student_id"] for s in _STUDENTS if s["iep"] == "true"}
        completed_ids = _all_completed_sessions(_SESSIONS)
        iep_att = [
            a
            for a in _ATTENDANCE
            if a["session_id"] in completed_ids and a["student_id"] in iep_ids
        ]
        attended = sum(1 for a in iep_att if a["attended"] == "true")
        rate = attended / len(iep_att)
        assert abs(rate - 0.6839) <= 0.001, f"IEP attendance rate was {rate:.4f}"

    def test_f3_non_iep_attendance_rate(self) -> None:
        """F3: Non-IEP student attendance rate ≈ 0.8243 ± 0.001."""
        non_iep_ids = {s["student_id"] for s in _STUDENTS if s["iep"] == "false"}
        completed_ids = _all_completed_sessions(_SESSIONS)
        non_iep_att = [
            a
            for a in _ATTENDANCE
            if a["session_id"] in completed_ids and a["student_id"] in non_iep_ids
        ]
        attended = sum(1 for a in non_iep_att if a["attended"] == "true")
        rate = attended / len(non_iep_att)
        assert abs(rate - 0.8243) <= 0.001, f"Non-IEP attendance rate was {rate:.4f}"

    def test_f4_iep_gap(self) -> None:
        """F4: The non-IEP minus IEP gap is approximately 14.04 pp ± 0.1."""
        iep_ids = {s["student_id"] for s in _STUDENTS if s["iep"] == "true"}
        non_iep_ids = {s["student_id"] for s in _STUDENTS if s["iep"] == "false"}
        completed_ids = _all_completed_sessions(_SESSIONS)

        def subgroup_rate(student_ids: set[str]) -> float:
            att = [
                a
                for a in _ATTENDANCE
                if a["session_id"] in completed_ids and a["student_id"] in student_ids
            ]
            return sum(1 for a in att if a["attended"] == "true") / len(att)

        gap_pp = (subgroup_rate(non_iep_ids) - subgroup_rate(iep_ids)) * 100
        assert abs(gap_pp - 14.04) <= 0.1, f"IEP gap was {gap_pp:.2f} pp"


class TestT1OPS006GoldFacts:
    """Recompute gold facts for T1-OPS-006 (November as last month)."""

    def test_f1_november_scheduled_sessions(self) -> None:
        """F1: 168 sessions were scheduled in November 2025."""
        nov = _sessions_in_month(_SESSIONS, "2025-11")
        assert len(nov) == 168

    def test_f2_november_completed_sessions(self) -> None:
        """F2: 152 of the November sessions were completed."""
        nov_completed = [
            s
            for s in _SESSIONS
            if s["scheduled_date"].startswith("2025-11") and s["status"] == "completed"
        ]
        assert len(nov_completed) == 152

    def test_f3_november_attendance_rate(self) -> None:
        """F3: November attendance rate for completed sessions ≈ 0.8196 ± 0.001."""
        nov_ids = _completed_sessions_in_month(_SESSIONS, "2025-11")
        attended, total = _attendance_rate(nov_ids, _ATTENDANCE)
        rate = attended / total
        assert abs(rate - 0.8196) <= 0.001, f"November attendance rate was {rate:.4f}"

    def test_f4_november_vs_quarter_delta(self) -> None:
        """F4: November rate exceeds the quarter rate by approximately 1.04 pp ± 0.2."""
        # Quarter rate
        completed_all = _all_completed_sessions(_SESSIONS)
        q_attended, q_total = _attendance_rate(completed_all, _ATTENDANCE)
        quarter_rate = q_attended / q_total

        # November rate
        nov_ids = _completed_sessions_in_month(_SESSIONS, "2025-11")
        nov_attended, nov_total = _attendance_rate(nov_ids, _ATTENDANCE)
        nov_rate = nov_attended / nov_total

        delta_pp = (nov_rate - quarter_rate) * 100
        assert abs(delta_pp - 1.04) <= 0.2, f"Delta was {delta_pp:.2f} pp"
