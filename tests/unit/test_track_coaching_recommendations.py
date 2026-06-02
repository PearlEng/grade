"""Verification tests for Track 3 (Operational Coaching & Recommendations) tasks.

These tests enforce three guarantees:

1. **Schema validity** — every task in tasks.jsonl must pass ``validate_task``.
2. **Gold-fact correctness** — every numeric gold fact is independently recomputed
   from the Operations pack fixture CSVs and asserted to match.  Where a metric is
   also defined in ``ground_truth.json`` (program-wide attendance, IEP gap,
   cancellation rates), the ground_truth value is used as the authoritative
   assertion target.
3. **Rubric evidence-citation requirement** — every task's ``rubric`` must include
   an ``evidence_linkage`` guidance string that contains an explicit citation
   requirement (the word "REQUIRED").

Running ``pytest tests/unit/test_track_coaching_recommendations.py`` must pass
before any Track 3 task is merged.
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

_REPO_ROOT = Path(__file__).parents[2]
_TASKS_PATH = _REPO_ROOT / "benchmark" / "tracks" / "coaching_recommendations" / "tasks.jsonl"
_PACK_DIR = _REPO_ROOT / "fixtures" / "pack_operations"
_GROUND_TRUTH_PATH = _PACK_DIR / "ground_truth.json"

# ---------------------------------------------------------------------------
# Helpers — load fixture data once at module level
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


def _load_ground_truth() -> dict[str, Any]:
    """Load the authoritative ground_truth.json for the Operations pack."""
    return json.load(_GROUND_TRUTH_PATH.open(encoding="utf-8"))  # type: ignore[no-any-return]


# Fixture data loaded once so individual tests stay fast.
_SESSIONS = _load_csv("sessions.csv")
_ATTENDANCE = _load_csv("attendance.csv")
_GROUPS = _load_csv("groups.csv")
_TUTORS = _load_csv("tutors.csv")
_STUDENTS = _load_csv("students.csv")
_GROUND_TRUTH = _load_ground_truth()

# ---------------------------------------------------------------------------
# Shared computation helpers
# ---------------------------------------------------------------------------


def _completed_session_ids() -> set[str]:
    """Return session_ids for all completed sessions."""
    return {s["session_id"] for s in _SESSIONS if s["status"] == "completed"}


def _session_to_group() -> dict[str, str]:
    return {s["session_id"]: s["group_id"] for s in _SESSIONS}


def _group_to_tutor() -> dict[str, str]:
    return {g["group_id"]: g["tutor_id"] for g in _GROUPS}


def _group_to_school() -> dict[str, str]:
    return {g["group_id"]: g["school_id"] for g in _GROUPS}


def _iep_student_ids() -> set[str]:
    return {s["student_id"] for s in _STUDENTS if s["iep"] == "true"}


def _attendance_rate(att_records: list[dict]) -> float:
    """Return attended/total for a list of attendance records."""
    if not att_records:
        raise ValueError("No attendance records provided.")
    attended = sum(1 for a in att_records if a["attended"] == "true")
    return attended / len(att_records)


def _tutor_attendance_records(
    tutor_id: str,
    completed_ids: set[str],
    sess_to_grp: dict[str, str],
    grp_to_tutor: dict[str, str],
) -> list[dict]:
    """Return all attendance records for completed sessions led by tutor_id."""
    return [
        a
        for a in _ATTENDANCE
        if a["session_id"] in completed_ids
        and grp_to_tutor.get(sess_to_grp.get(a["session_id"], ""), "") == tutor_id
    ]


def _cancel_or_noshow(sessions: list[dict]) -> int:
    """Count sessions that were cancelled (any party) or a no-show.

    Matches the canonical ``monthly_cancellation_rate`` definition used in
    both packs' ``ground_truth.json``.
    """
    return sum(1 for s in sessions if "cancelled" in s["status"] or s["status"] == "no_show")


def _sessions_in_month(year_month: str) -> list[dict]:
    return [s for s in _SESSIONS if s["scheduled_date"].startswith(year_month)]


# ---------------------------------------------------------------------------
# Task loading and structural validation
# ---------------------------------------------------------------------------


class TestTasksLoading:
    """Structural tests for the tasks.jsonl file itself."""

    def test_tasks_file_exists(self) -> None:
        """The tasks.jsonl file must exist at the expected path."""
        assert _TASKS_PATH.exists(), f"tasks.jsonl not found at {_TASKS_PATH}"

    def test_approximately_five_tasks(self) -> None:
        """The file must contain approximately five tasks (at least 5, at most 8)."""
        tasks = _load_tasks()
        assert 5 <= len(tasks) <= 8, f"Expected ~5 tasks, found {len(tasks)}"

    def test_all_tasks_are_track_3(self) -> None:
        """Every task must declare track=3."""
        for task in _load_tasks():
            assert task["track"] == 3, (
                f"Task {task['task_id']} has track={task['track']}, expected 3"
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

    def test_all_tasks_are_recommendation_type(self) -> None:
        """Every Track 3 task must use task_type='recommendation'."""
        for task in _load_tasks():
            assert task["task_type"] == "recommendation", (
                f"Task {task['task_id']} has task_type='{task['task_type']}', "
                "expected 'recommendation'"
            )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _load_tasks(), ids=[t["task_id"] for t in _load_tasks()])
def test_task_schema_valid(task: dict) -> None:
    """Every task in tasks.jsonl must pass validate_task without error."""
    validate_task(task)


# ---------------------------------------------------------------------------
# Rubric evidence-citation requirement
# ---------------------------------------------------------------------------


class TestRubricEvidenceCitationRequirement:
    """Every task's rubric.evidence_linkage must include an explicit citation requirement.

    Specifically, the ``guidance`` field must contain the word 'REQUIRED'.
    """

    @pytest.mark.parametrize("task", _load_tasks(), ids=[t["task_id"] for t in _load_tasks()])
    def test_evidence_linkage_has_required_keyword(self, task: dict) -> None:
        """rubric.evidence_linkage.guidance must contain 'REQUIRED'."""
        rubric = task.get("rubric", {})
        el = rubric.get("evidence_linkage", {})
        guidance = el.get("guidance", "")
        assert "REQUIRED" in guidance, (
            f"Task {task['task_id']}: rubric.evidence_linkage.guidance does not contain "
            f"'REQUIRED'. Actual guidance: {guidance!r}"
        )

    @pytest.mark.parametrize("task", _load_tasks(), ids=[t["task_id"] for t in _load_tasks()])
    def test_evidence_linkage_weight_elevated(self, task: dict) -> None:
        """evidence_linkage weight should be at least 0.15 for coaching tasks."""
        rubric = task.get("rubric", {})
        el = rubric.get("evidence_linkage", {})
        weight = el.get("weight", 0.0)
        assert weight >= 0.15, f"Task {task['task_id']}: evidence_linkage weight {weight} < 0.15"

    @pytest.mark.parametrize("task", _load_tasks(), ids=[t["task_id"] for t in _load_tasks()])
    def test_rubric_weights_sum_to_one(self, task: dict) -> None:
        """All six rubric dimension weights must sum to 1.0 ± 0.001."""
        rubric = task.get("rubric", {})
        dimensions = [
            "grounding_accuracy",
            "insight_quality",
            "evidence_linkage",
            "calibration_limitation_handling",
            "consistency",
            "structure_usability",
        ]
        total = sum(rubric.get(d, {}).get("weight", 0.0) for d in dimensions)
        assert abs(total - 1.0) <= 0.001, (
            f"Task {task['task_id']}: rubric weights sum to {total:.4f}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T3-OPS-001
# ---------------------------------------------------------------------------


class TestT3OPS001GoldFacts:
    """Recompute gold facts for T3-OPS-001 (priority tutor by attendance rate)."""

    def test_f1_tut005_attendance_rate(self) -> None:
        """F1: TUT-005 attendance rate is approximately 76.4% (440/576)."""
        completed_ids = _completed_session_ids()
        sess_to_grp = _session_to_group()
        grp_to_tutor = _group_to_tutor()

        records = _tutor_attendance_records("TUT-005", completed_ids, sess_to_grp, grp_to_tutor)
        assert len(records) == 576, f"Expected 576 records, got {len(records)}"
        attended = sum(1 for a in records if a["attended"] == "true")
        assert attended == 440, f"Expected 440 attended, got {attended}"
        rate = attended / len(records)
        assert abs(rate - 0.7639) <= 0.001, f"TUT-005 rate was {rate:.4f}"

    def test_f2_tut005_vs_program_wide_gap(self) -> None:
        """F2: TUT-005 is approximately 4.5 pp below the program-wide rate."""
        completed_ids = _completed_session_ids()
        sess_to_grp = _session_to_group()
        grp_to_tutor = _group_to_tutor()

        # TUT-005 rate
        tut005_records = _tutor_attendance_records(
            "TUT-005", completed_ids, sess_to_grp, grp_to_tutor
        )
        tut005_rate = _attendance_rate(tut005_records)

        # Program-wide rate — assert against ground_truth.json authoritative value
        program_rate = _GROUND_TRUTH["program_wide_attendance_rate"]
        assert abs(program_rate - 0.8092) <= 0.0005

        gap_pp = (program_rate - tut005_rate) * 100
        assert abs(gap_pp - 4.53) <= 0.1, f"Gap was {gap_pp:.2f} pp"

    def test_f3_grp007_attendance_rate(self) -> None:
        """F3: GRP-007 (TUT-005) is the lowest-attending group at 73.96%."""
        completed_ids = _completed_session_ids()
        grp007_records = [
            a
            for a in _ATTENDANCE
            if a["session_id"] in completed_ids
            and _session_to_group().get(a["session_id"]) == "GRP-007"
        ]
        assert len(grp007_records) == 288, f"Expected 288, got {len(grp007_records)}"
        attended = sum(1 for a in grp007_records if a["attended"] == "true")
        assert attended == 213, f"Expected 213, got {attended}"
        rate = attended / len(grp007_records)
        assert abs(rate - 0.7396) <= 0.001, f"GRP-007 rate was {rate:.4f}"

    def test_f4_program_wide_attendance_rate_matches_ground_truth(self) -> None:
        """F4: Program-wide rate from fixtures matches ground_truth.json (0.8092)."""
        # Ground truth is authoritative
        gt_rate = _GROUND_TRUTH["program_wide_attendance_rate"]
        assert abs(gt_rate - 0.8092) <= 0.0005

        # Recompute from fixtures and confirm consistency
        completed_ids = _completed_session_ids()
        att = [a for a in _ATTENDANCE if a["session_id"] in completed_ids]
        attended = sum(1 for a in att if a["attended"] == "true")
        computed_rate = attended / len(att)
        assert abs(computed_rate - gt_rate) <= 0.0005, (
            f"Computed rate {computed_rate:.4f} diverges from ground_truth {gt_rate}"
        )

    def test_f3_tut005_is_lowest_attending_tutor(self) -> None:
        """TUT-005 must have the strictly lowest attendance rate among all tutors."""
        completed_ids = _completed_session_ids()
        sess_to_grp = _session_to_group()
        grp_to_tutor = _group_to_tutor()

        all_tutor_ids = list({t["tutor_id"] for t in _TUTORS})
        rates = {}
        for tid in all_tutor_ids:
            records = _tutor_attendance_records(tid, completed_ids, sess_to_grp, grp_to_tutor)
            if records:
                rates[tid] = _attendance_rate(records)

        lowest_tutor = min(rates, key=lambda t: rates[t])
        assert lowest_tutor == "TUT-005", (
            f"Expected TUT-005 to be lowest, but {lowest_tutor} had rate {rates[lowest_tutor]:.4f}"
        )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T3-OPS-002
# ---------------------------------------------------------------------------


class TestT3OPS002GoldFacts:
    """Recompute gold facts for T3-OPS-002 (IEP student attendance by tutor)."""

    def test_f1_program_wide_iep_rate_matches_ground_truth(self) -> None:
        """F1: Program-wide IEP attendance rate matches ground_truth.json (0.6839)."""
        # Assert ground_truth value
        gt_iep_rate = _GROUND_TRUTH["iep_attendance_rate"]
        assert abs(gt_iep_rate - 0.6839) <= 0.001

        # Recompute to confirm consistency
        completed_ids = _completed_session_ids()
        iep_ids = _iep_student_ids()
        iep_att = [
            a
            for a in _ATTENDANCE
            if a["session_id"] in completed_ids and a["student_id"] in iep_ids
        ]
        computed_rate = _attendance_rate(iep_att)
        assert abs(computed_rate - gt_iep_rate) <= 0.001, (
            f"Computed IEP rate {computed_rate:.4f} diverges from ground_truth {gt_iep_rate}"
        )

    def test_f2_tut005_iep_attendance_rate(self) -> None:
        """F2: TUT-005 IEP attendance rate is approximately 62.5% (100/160)."""
        completed_ids = _completed_session_ids()
        sess_to_grp = _session_to_group()
        iep_ids = _iep_student_ids()

        tut005_iep_att = [
            a
            for a in _ATTENDANCE
            if a["session_id"] in completed_ids
            and sess_to_grp.get(a["session_id"]) in ("GRP-007", "GRP-010")
            and a["student_id"] in iep_ids
        ]
        assert len(tut005_iep_att) == 160, (
            f"Expected 160 IEP records for TUT-005, got {len(tut005_iep_att)}"
        )
        attended = sum(1 for a in tut005_iep_att if a["attended"] == "true")
        assert attended == 100, f"Expected 100 attended, got {attended}"
        rate = attended / len(tut005_iep_att)
        assert abs(rate - 0.6250) <= 0.001, f"TUT-005 IEP rate was {rate:.4f}"

    def test_f3_tut008_grp012_iep_attendance_rate(self) -> None:
        """F3: TUT-008 GRP-012 IEP attendance rate is approximately 57.6% (19/33)."""
        completed_ids = _completed_session_ids()
        sess_to_grp = _session_to_group()
        iep_ids = _iep_student_ids()

        grp012_iep_att = [
            a
            for a in _ATTENDANCE
            if a["session_id"] in completed_ids
            and sess_to_grp.get(a["session_id"]) == "GRP-012"
            and a["student_id"] in iep_ids
        ]
        assert len(grp012_iep_att) == 33, (
            f"Expected 33 IEP records in GRP-012, got {len(grp012_iep_att)}"
        )
        attended = sum(1 for a in grp012_iep_att if a["attended"] == "true")
        assert attended == 19, f"Expected 19 attended, got {attended}"
        rate = attended / len(grp012_iep_att)
        assert abs(rate - 0.5758) <= 0.001, f"GRP-012 IEP rate was {rate:.4f}"

    def test_f4_iep_gap_matches_ground_truth(self) -> None:
        """F4: IEP attendance gap is approximately 14.04 pp — assert ground_truth value."""
        gt_gap = _GROUND_TRUTH["iep_attendance_gap_pp"]
        assert abs(gt_gap - 14.04) <= 0.1

        # Recompute to confirm consistency
        completed_ids = _completed_session_ids()
        iep_ids = _iep_student_ids()
        non_iep_ids = {s["student_id"] for s in _STUDENTS if s["iep"] == "false"}

        iep_att = [
            a
            for a in _ATTENDANCE
            if a["session_id"] in completed_ids and a["student_id"] in iep_ids
        ]
        non_iep_att = [
            a
            for a in _ATTENDANCE
            if a["session_id"] in completed_ids and a["student_id"] in non_iep_ids
        ]
        computed_gap = (_attendance_rate(non_iep_att) - _attendance_rate(iep_att)) * 100
        assert abs(computed_gap - gt_gap) <= 0.1, (
            f"Computed gap {computed_gap:.2f} pp diverges from ground_truth {gt_gap}"
        )

    def test_f3_tut008_grp012_is_lowest_iep_group(self) -> None:
        """GRP-012 must have the lowest IEP attendance rate among groups with IEP students."""
        completed_ids = _completed_session_ids()
        sess_to_grp = _session_to_group()
        iep_ids = _iep_student_ids()

        group_ids = {g["group_id"] for g in _GROUPS}
        iep_rates: dict[str, float] = {}
        for grp_id in group_ids:
            records = [
                a
                for a in _ATTENDANCE
                if a["session_id"] in completed_ids
                and sess_to_grp.get(a["session_id"]) == grp_id
                and a["student_id"] in iep_ids
            ]
            if len(records) >= 5:  # minimum sample threshold
                iep_rates[grp_id] = _attendance_rate(records)

        lowest_grp = min(iep_rates, key=lambda g: iep_rates[g])
        assert lowest_grp == "GRP-012", (
            f"Expected GRP-012 as lowest IEP rate, but {lowest_grp} had {iep_rates[lowest_grp]:.4f}"
        )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T3-OPS-003
# ---------------------------------------------------------------------------


class TestT3OPS003GoldFacts:
    """Recompute gold facts for T3-OPS-003 (October cancellation spike + tutor exposure)."""

    def test_f1_october_cancellation_rate_matches_ground_truth(self) -> None:
        """F1: October cancellation rate is 27.89% — assert ground_truth value."""
        gt_oct_rate = _GROUND_TRUTH["monthly_cancellation_rate"]["2025-10"]
        assert abs(gt_oct_rate - 0.2789) <= 0.001

        # Recompute to confirm consistency
        oct_sess = _sessions_in_month("2025-10")
        computed_rate = _cancel_or_noshow(oct_sess) / len(oct_sess)
        assert abs(computed_rate - gt_oct_rate) <= 0.001, (
            f"Computed Oct rate {computed_rate:.4f} diverges from ground_truth {gt_oct_rate}"
        )

    def test_f2_tut004_quarterly_cancellation_rate(self) -> None:
        """F2: TUT-004 quarterly cancellation rate is approximately 23.1% (9/39)."""
        grp_to_tutor = _group_to_tutor()
        tut004_sess = [s for s in _SESSIONS if grp_to_tutor.get(s["group_id"]) == "TUT-004"]
        assert len(tut004_sess) == 39, f"Expected 39 TUT-004 sessions, got {len(tut004_sess)}"
        cancelled = _cancel_or_noshow(tut004_sess)
        assert cancelled == 9, f"Expected 9 cancelled/no-show, got {cancelled}"
        rate = cancelled / len(tut004_sess)
        assert abs(rate - 0.2308) <= 0.001, f"TUT-004 cancel rate was {rate:.4f}"

    def test_f2_tut004_october_cancellation_rate(self) -> None:
        """F2 (sub): TUT-004 October cancellation rate is 5/13 = 38.5%."""
        grp_to_tutor = _group_to_tutor()
        oct_tut004 = [
            s
            for s in _SESSIONS
            if s["scheduled_date"].startswith("2025-10")
            and grp_to_tutor.get(s["group_id"]) == "TUT-004"
        ]
        assert len(oct_tut004) == 13, f"Expected 13 TUT-004 October sessions, got {len(oct_tut004)}"
        cancelled = _cancel_or_noshow(oct_tut004)
        assert cancelled == 5, f"Expected 5 TUT-004 October cancellations, got {cancelled}"
        rate = cancelled / len(oct_tut004)
        assert abs(rate - 0.3846) <= 0.01, f"TUT-004 Oct rate was {rate:.4f}"

    def test_f3_sep_nov_cancellation_rates_match_ground_truth(self) -> None:
        """F3: Sep and Nov cancellation rates match ground_truth.json."""
        gt_sep = _GROUND_TRUTH["monthly_cancellation_rate"]["2025-09"]
        gt_nov = _GROUND_TRUTH["monthly_cancellation_rate"]["2025-11"]
        assert abs(gt_sep - 0.1064) <= 0.001
        assert abs(gt_nov - 0.0952) <= 0.001

        sep_sess = _sessions_in_month("2025-09")
        nov_sess = _sessions_in_month("2025-11")
        sep_computed = _cancel_or_noshow(sep_sess) / len(sep_sess)
        nov_computed = _cancel_or_noshow(nov_sess) / len(nov_sess)
        assert abs(sep_computed - gt_sep) <= 0.001
        assert abs(nov_computed - gt_nov) <= 0.001

    def test_f3_october_vs_sep_nov_gap(self) -> None:
        """F3: October rate is approximately 17.8 pp above average of Sep+Nov."""
        sep_sess = _sessions_in_month("2025-09")
        oct_sess = _sessions_in_month("2025-10")
        nov_sess = _sessions_in_month("2025-11")
        sep_rate = _cancel_or_noshow(sep_sess) / len(sep_sess)
        oct_rate = _cancel_or_noshow(oct_sess) / len(oct_sess)
        nov_rate = _cancel_or_noshow(nov_sess) / len(nov_sess)
        gap_pp = (oct_rate - (sep_rate + nov_rate) / 2) * 100
        assert abs(gap_pp - 17.8) <= 0.3, f"Gap was {gap_pp:.2f} pp"

    def test_f4_tut008_cancellation_rate(self) -> None:
        """F4: TUT-008 quarterly cancellation rate is approximately 20.5% (16/78)."""
        grp_to_tutor = _group_to_tutor()
        tut008_sess = [s for s in _SESSIONS if grp_to_tutor.get(s["group_id"]) == "TUT-008"]
        assert len(tut008_sess) == 78, f"Expected 78 TUT-008 sessions, got {len(tut008_sess)}"
        cancelled = _cancel_or_noshow(tut008_sess)
        assert cancelled == 16, f"Expected 16 cancelled/no-show, got {cancelled}"
        rate = cancelled / len(tut008_sess)
        assert abs(rate - 0.2051) <= 0.001, f"TUT-008 cancel rate was {rate:.4f}"

    def test_f2_tut004_has_highest_cancellation_rate(self) -> None:
        """TUT-004 must have the strictly highest quarterly cancellation rate."""
        grp_to_tutor = _group_to_tutor()
        all_tutor_ids = list({t["tutor_id"] for t in _TUTORS})
        rates = {}
        for tid in all_tutor_ids:
            sess = [s for s in _SESSIONS if grp_to_tutor.get(s["group_id"]) == tid]
            if sess:
                rates[tid] = _cancel_or_noshow(sess) / len(sess)

        highest_tutor = max(rates, key=lambda t: rates[t])
        assert highest_tutor == "TUT-004", (
            f"Expected TUT-004 as highest cancel rate, "
            f"but {highest_tutor} had {rates[highest_tutor]:.4f}"
        )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T3-OPS-004
# ---------------------------------------------------------------------------


class TestT3OPS004GoldFacts:
    """Recompute gold facts for T3-OPS-004 (school-level attendance with confidence bands)."""

    @staticmethod
    def _school_attendance_records(school_id: str) -> list[dict]:
        """Return attendance records for completed sessions at school_id."""
        completed_ids = _completed_session_ids()
        sess_to_grp = _session_to_group()
        grp_to_school = _group_to_school()
        return [
            a
            for a in _ATTENDANCE
            if a["session_id"] in completed_ids
            and grp_to_school.get(sess_to_grp.get(a["session_id"], ""), "") == school_id
        ]

    def test_f1_sch001_attendance_rate(self) -> None:
        """F1: SCH-001 attendance rate is approximately 82.5% (1456/1765)."""
        records = self._school_attendance_records("SCH-001")
        assert len(records) == 1765, f"Expected 1765, got {len(records)}"
        attended = sum(1 for a in records if a["attended"] == "true")
        assert attended == 1456, f"Expected 1456, got {attended}"
        rate = attended / len(records)
        assert abs(rate - 0.8249) <= 0.001, f"SCH-001 rate was {rate:.4f}"

    def test_f2_sch002_attendance_rate(self) -> None:
        """F2: SCH-002 attendance rate is approximately 78.9% (1121/1421), below 80% goal."""
        records = self._school_attendance_records("SCH-002")
        assert len(records) == 1421, f"Expected 1421, got {len(records)}"
        attended = sum(1 for a in records if a["attended"] == "true")
        assert attended == 1121, f"Expected 1121, got {attended}"
        rate = attended / len(records)
        assert abs(rate - 0.7889) <= 0.001, f"SCH-002 rate was {rate:.4f}"
        # SCH-002 must be below the 80% program goal
        assert rate < 0.80, f"SCH-002 rate {rate:.4f} should be below 0.80"

    def test_f3_sch003_attendance_rate(self) -> None:
        """F3: SCH-003 attendance rate is approximately 81.0% (786/970)."""
        records = self._school_attendance_records("SCH-003")
        assert len(records) == 970, f"Expected 970, got {len(records)}"
        attended = sum(1 for a in records if a["attended"] == "true")
        assert attended == 786, f"Expected 786, got {attended}"
        rate = attended / len(records)
        assert abs(rate - 0.8103) <= 0.001, f"SCH-003 rate was {rate:.4f}"

    def test_f4_sch001_vs_sch002_gap(self) -> None:
        """F4: SCH-001 to SCH-002 gap is approximately 3.6 pp."""
        sch001_records = self._school_attendance_records("SCH-001")
        sch002_records = self._school_attendance_records("SCH-002")
        sch001_rate = _attendance_rate(sch001_records)
        sch002_rate = _attendance_rate(sch002_records)
        gap_pp = (sch001_rate - sch002_rate) * 100
        assert abs(gap_pp - 3.6) <= 0.2, f"Gap was {gap_pp:.2f} pp"

    def test_f2_sch002_is_only_school_below_80_percent(self) -> None:
        """SCH-002 must be the only school below the 80% program goal."""
        schools_below_80 = []
        for sch in ("SCH-001", "SCH-002", "SCH-003"):
            records = self._school_attendance_records(sch)
            if _attendance_rate(records) < 0.80:
                schools_below_80.append(sch)
        assert schools_below_80 == ["SCH-002"], (
            f"Expected only SCH-002 below 80%, found: {schools_below_80}"
        )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T3-OPS-005
# ---------------------------------------------------------------------------


class TestT3OPS005GoldFacts:
    """Recompute gold facts for T3-OPS-005 (multi-signal coaching priority list)."""

    def test_f1_tut005_attendance_rate_and_session_count(self) -> None:
        """F1: TUT-005 rate ≈ 76.4% and sessions_ytd = 78."""
        completed_ids = _completed_session_ids()
        sess_to_grp = _session_to_group()
        grp_to_tutor = _group_to_tutor()

        records = _tutor_attendance_records("TUT-005", completed_ids, sess_to_grp, grp_to_tutor)
        rate = _attendance_rate(records)
        assert abs(rate - 0.7639) <= 0.001, f"TUT-005 rate was {rate:.4f}"

        # sessions_ytd from tutors.csv
        tut005_info = next(t for t in _TUTORS if t["tutor_id"] == "TUT-005")
        assert int(tut005_info["sessions_delivered_ytd"]) == 78, (
            f"TUT-005 sessions_ytd expected 78, got {tut005_info['sessions_delivered_ytd']}"
        )

    def test_f2_tut004_cancellation_rate_and_attendance_rate(self) -> None:
        """F2: TUT-004 cancel rate ≈ 23.1%; attendance rate ≈ 81.9%."""
        grp_to_tutor = _group_to_tutor()

        # Cancellation rate
        tut004_sess = [s for s in _SESSIONS if grp_to_tutor.get(s["group_id"]) == "TUT-004"]
        cancel_rate = _cancel_or_noshow(tut004_sess) / len(tut004_sess)
        assert abs(cancel_rate - 0.2308) <= 0.001, f"TUT-004 cancel rate was {cancel_rate:.4f}"

        # Attendance rate (completed sessions)
        completed_ids = _completed_session_ids()
        sess_to_grp = _session_to_group()
        att_records = _tutor_attendance_records("TUT-004", completed_ids, sess_to_grp, grp_to_tutor)
        att_rate = _attendance_rate(att_records)
        assert abs(att_rate - 0.8185) <= 0.001, f"TUT-004 att rate was {att_rate:.4f}"

    def test_f3_tut002_and_tut008_lowest_sessions_ytd(self) -> None:
        """F3: TUT-002 and TUT-008 each have 35 sessions YTD, the joint lowest."""
        sessions_ytd = {t["tutor_id"]: int(t["sessions_delivered_ytd"]) for t in _TUTORS}
        tut002_ytd = sessions_ytd["TUT-002"]
        tut008_ytd = sessions_ytd["TUT-008"]
        assert tut002_ytd == 35, f"TUT-002 sessions_ytd expected 35, got {tut002_ytd}"
        assert tut008_ytd == 35, f"TUT-008 sessions_ytd expected 35, got {tut008_ytd}"

        # Both must be the joint minimum
        min_ytd = min(sessions_ytd.values())
        assert min_ytd == 35, f"Minimum sessions_ytd expected 35, got {min_ytd}"

    def test_f4_tut003_highest_attendance_rate_and_missing_rating(self) -> None:
        """F4: TUT-003 has highest attendance rate at 85.5%; no session rating recorded."""
        completed_ids = _completed_session_ids()
        sess_to_grp = _session_to_group()
        grp_to_tutor = _group_to_tutor()

        records = _tutor_attendance_records("TUT-003", completed_ids, sess_to_grp, grp_to_tutor)
        rate = _attendance_rate(records)
        assert abs(rate - 0.8545) <= 0.001, f"TUT-003 rate was {rate:.4f}"

        # TUT-003 must have the highest rate
        all_tutor_ids = list({t["tutor_id"] for t in _TUTORS})
        all_rates = {}
        for tid in all_tutor_ids:
            recs = _tutor_attendance_records(tid, completed_ids, sess_to_grp, grp_to_tutor)
            if recs:
                all_rates[tid] = _attendance_rate(recs)
        highest_tutor = max(all_rates, key=lambda t: all_rates[t])
        assert highest_tutor == "TUT-003", (
            f"Expected TUT-003 as highest, but {highest_tutor} had {all_rates[highest_tutor]:.4f}"
        )

        # TUT-003 avg_session_rating must be empty
        tut003_info = next(t for t in _TUTORS if t["tutor_id"] == "TUT-003")
        assert tut003_info["avg_session_rating"] == "", (
            f"TUT-003 expected empty rating, got '{tut003_info['avg_session_rating']}'"
        )

    def test_f4_tut009_also_missing_rating(self) -> None:
        """TUT-009 also has no session rating recorded."""
        tut009_info = next(t for t in _TUTORS if t["tutor_id"] == "TUT-009")
        assert tut009_info["avg_session_rating"] == "", (
            f"TUT-009 expected empty rating, got '{tut009_info['avg_session_rating']}'"
        )
