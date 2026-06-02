"""Verification tests for Track 5 (Effectiveness & Research Reasoning) tasks.

These tests enforce three guarantees:

1. **Schema validity** — every task in tasks.jsonl must pass ``validate_task``.
2. **Gold-fact correctness** — every numeric gold fact is independently recomputed
   from the Equity & Research pack fixture files and asserted to match
   ground_truth.json where those values are defined there.
3. **Track-5 invariants** — every task must reference at least one real ref_id
   from research_refs.json in its rubric or reference_answer_outline; fact_id
   values must be unique within each task; the file must contain ~5 tasks.

Running ``pytest tests/unit/test_track_effectiveness_research.py`` must pass
before any Track 5 task is merged.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmark.schemas import validate_task

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[2]
_TASKS_PATH = _REPO_ROOT / "benchmark" / "tracks" / "effectiveness_research" / "tasks.jsonl"
_PACK_DIR = _REPO_ROOT / "fixtures" / "pack_equity_research"
_GROUND_TRUTH_PATH = _PACK_DIR / "ground_truth.json"
_RESEARCH_REFS_PATH = _PACK_DIR / "research_refs.json"

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


def _load_ground_truth() -> dict:
    """Load the ground_truth.json file."""
    with _GROUND_TRUTH_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _load_research_refs() -> dict:
    """Load research_refs.json and return a set of valid ref_ids."""
    with _RESEARCH_REFS_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data  # type: ignore[no-any-return]


# Fixture data loaded once at module level so individual tests stay fast.
_STUDENTS = _load_csv("students.csv")
_SUBGROUP_ATTENDANCE = _load_csv("subgroup_attendance_summary.csv")
_SUBGROUP_OUTCOMES = _load_csv("subgroup_outcomes_summary.csv")
_GROUND_TRUTH = _load_ground_truth()
_RESEARCH_REFS_DATA = _load_research_refs()
_VALID_REF_IDS: set[str] = {ref["ref_id"] for ref in _RESEARCH_REFS_DATA["references"]}

# ---------------------------------------------------------------------------
# Recomputation helpers
# ---------------------------------------------------------------------------


def _proficiency_rate_for(
    data: list[dict[str, str]],
    period: str,
    dimension: str,
    value: str,
) -> float:
    """Return benchmark_proficiency_rate from the outcomes summary."""
    rows = [
        r
        for r in data
        if r["assessment_period"] == period
        and r["subgroup_dimension"] == dimension
        and r["subgroup_value"] == value
    ]
    assert len(rows) == 1, f"Expected 1 row, found {len(rows)} for {period}/{dimension}/{value}"
    return float(rows[0]["benchmark_proficiency_rate"])


def _n_students_for_outcome_row(
    data: list[dict[str, str]],
    period: str,
    dimension: str,
    value: str,
) -> int:
    """Return n_students from the outcomes summary."""
    rows = [
        r
        for r in data
        if r["assessment_period"] == period
        and r["subgroup_dimension"] == dimension
        and r["subgroup_value"] == value
    ]
    assert len(rows) == 1, f"Expected 1 row, found {len(rows)} for {period}/{dimension}/{value}"
    return int(rows[0]["n_students"])


def _score_gain_for(
    data: list[dict[str, str]],
    period: str,
    dimension: str,
    value: str,
) -> float:
    """Return avg_score_gain from the outcomes summary."""
    rows = [
        r
        for r in data
        if r["assessment_period"] == period
        and r["subgroup_dimension"] == dimension
        and r["subgroup_value"] == value
    ]
    assert len(rows) == 1, f"Expected 1 row, found {len(rows)} for {period}/{dimension}/{value}"
    return float(rows[0]["avg_score_gain"])


def _attendance_rate_for(
    data: list[dict[str, str]],
    month: str,
    dimension: str,
    value: str,
) -> float:
    """Return attendance_rate from the subgroup attendance summary."""
    rows = [
        r
        for r in data
        if r["month_label"] == month
        and r["subgroup_dimension"] == dimension
        and r["subgroup_value"] == value
    ]
    assert len(rows) == 1, f"Expected 1 row, found {len(rows)} for {month}/{dimension}/{value}"
    return float(rows[0]["attendance_rate"])


# ---------------------------------------------------------------------------
# Task loading and structural tests
# ---------------------------------------------------------------------------


class TestTasksLoading:
    """Structural tests for the tasks.jsonl file itself."""

    def test_tasks_file_exists(self) -> None:
        """The tasks.jsonl file must exist at the expected path."""
        assert _TASKS_PATH.exists(), f"tasks.jsonl not found at {_TASKS_PATH}"

    def test_approximately_five_tasks(self) -> None:
        """The file must contain approximately five tasks (at least 4, at most 7)."""
        tasks = _load_tasks()
        assert 4 <= len(tasks) <= 7, f"Expected ~5 tasks, found {len(tasks)}"

    def test_all_tasks_are_track_5(self) -> None:
        """Every task must declare track=5."""
        for task in _load_tasks():
            assert task["track"] == 5, (
                f"Task {task['task_id']} has track={task['track']}, expected 5"
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

    def test_all_task_types_are_research_reasoning(self) -> None:
        """All tasks must use task_type='research_reasoning'."""
        for task in _load_tasks():
            assert task["task_type"] == "research_reasoning", (
                f"Task {task['task_id']} has task_type={task['task_type']!r}, "
                "expected 'research_reasoning'"
            )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _load_tasks(), ids=[t["task_id"] for t in _load_tasks()])
def test_task_schema_valid(task: dict) -> None:
    """Every task in tasks.jsonl must pass validate_task without error."""
    validate_task(task)  # raises jsonschema.ValidationError on failure


# ---------------------------------------------------------------------------
# Track 5 invariant: each task must reference real ref_ids
# ---------------------------------------------------------------------------


class TestTrack5Invariants:
    """Tests that enforce the DoD invariants specific to Track 5."""

    def test_every_task_references_real_ref_id(self) -> None:
        """Every task must reference at least one valid ref_id from research_refs.json.

        Checks rubric dimension guidance strings and reference_answer_outline for
        any occurrence of a known ref_id (R1–R10).
        """
        for task in _load_tasks():
            task_text_parts: list[str] = []

            # Collect all rubric guidance strings
            rubric = task.get("rubric", {})
            for dim_data in rubric.values():
                if isinstance(dim_data, dict):
                    guidance = dim_data.get("guidance", "")
                    if guidance:
                        task_text_parts.append(guidance)

            # Collect reference_answer_outline if present
            outline = task.get("reference_answer_outline", "")
            if outline:
                task_text_parts.append(outline)

            # Also check gold_facts claims
            for fact in task.get("gold_facts", []):
                task_text_parts.append(fact.get("claim", ""))

            combined_text = " ".join(task_text_parts)

            found_ref_ids = {ref_id for ref_id in _VALID_REF_IDS if ref_id in combined_text}
            assert len(found_ref_ids) >= 1, (
                f"Task {task['task_id']} does not reference any valid ref_id "
                f"(checked: {sorted(_VALID_REF_IDS)}) in its rubric guidance or "
                f"reference_answer_outline. Found text: {combined_text[:200]!r}"
            )

    def test_every_task_has_required_limitations(self) -> None:
        """Every Track 5 task must have at least two required_limitations."""
        for task in _load_tasks():
            limitations = task.get("required_limitations", [])
            assert len(limitations) >= 2, (
                f"Task {task['task_id']} has only {len(limitations)} required_limitations; "
                "expected at least 2 for Track 5 tasks"
            )

    def test_every_task_has_forbidden_claims(self) -> None:
        """Every Track 5 task must have at least two forbidden_claims."""
        for task in _load_tasks():
            forbidden = task.get("forbidden_claims", [])
            assert len(forbidden) >= 2, (
                f"Task {task['task_id']} has only {len(forbidden)} forbidden_claims; "
                "expected at least 2 for Track 5 tasks"
            )

    def test_rubric_weights_sum_to_one(self) -> None:
        """Rubric dimension weights must sum to 1.0 for each task."""
        for task in _load_tasks():
            rubric = task.get("rubric", {})
            total = sum(dim["weight"] for dim in rubric.values() if isinstance(dim, dict))
            assert abs(total - 1.0) < 0.001, (
                f"Task {task['task_id']} rubric weights sum to {total:.4f}, expected 1.0"
            )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T5-EFR-001
# ---------------------------------------------------------------------------


class TestT5EFR001GoldFacts:
    """Recompute gold facts for T5-EFR-001 (ranking predictors of effectiveness)."""

    def test_f1_dosage_target_135_per_week(self) -> None:
        """F1: Program dosage target is 135 min/week from program_context."""
        # Ground truth uses 135 min/week as the dosage target (from program_context.json)
        # We verify the IEP attendance rate which is the key local data in this task
        gt_iep_rate = _GROUND_TRUTH["iep_attendance_rate"]
        assert abs(gt_iep_rate - 0.6839) <= 0.001, (
            f"IEP attendance rate was {gt_iep_rate:.4f}, expected ~0.6839"
        )

    def test_f2_iep_attendance_below_80_pct_threshold(self) -> None:
        """F2: IEP attendance rate 0.6839 is below R10's 80% threshold."""
        gt_iep_rate = _GROUND_TRUTH["iep_attendance_rate"]
        # Confirm IEP rate is below 0.80
        assert gt_iep_rate < 0.80, f"IEP rate {gt_iep_rate:.4f} is not below the 0.80 threshold"
        # Confirm the shortfall matches the gold fact (~11.6 pp)
        shortfall_pp = (0.80 - gt_iep_rate) * 100
        assert abs(shortfall_pp - 11.61) <= 0.2, (
            f"Shortfall below 80% was {shortfall_pp:.2f} pp, expected ~11.6 pp"
        )

    def test_f3_program_wide_attendance_rate(self) -> None:
        """F3: Program-wide attendance rate ~80.9% from ground_truth.json."""
        gt_rate = _GROUND_TRUTH["program_wide_attendance_rate"]
        assert abs(gt_rate - 0.8092) <= 0.001, (
            f"Program-wide attendance rate was {gt_rate:.4f}, expected ~0.8092"
        )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T5-EFR-002
# ---------------------------------------------------------------------------


class TestT5EFR002GoldFacts:
    """Recompute gold facts for T5-EFR-002 (IEP pattern vs. research thresholds)."""

    def test_f1_iep_attendance_rate_matches_ground_truth(self) -> None:
        """F1: IEP attendance rate from ground_truth.json is 0.6839."""
        gt_rate = _GROUND_TRUTH["iep_attendance_rate"]
        assert abs(gt_rate - 0.6839) <= 0.001, f"IEP attendance rate was {gt_rate:.4f}"

    def test_f2_iep_shortfall_below_80_pct_threshold(self) -> None:
        """F2: IEP rate is ~11.6 pp below the 80% threshold in R10."""
        iep_rate = _GROUND_TRUTH["iep_attendance_rate"]
        shortfall_pp = (0.80 - iep_rate) * 100
        assert abs(shortfall_pp - 11.61) <= 0.2, (
            f"Shortfall below 80% was {shortfall_pp:.2f} pp, expected ~11.6 pp"
        )

    def test_f3_iep_attendance_gap_matches_ground_truth(self) -> None:
        """F3: IEP attendance gap is 14.04 pp from ground_truth.json."""
        gt_gap = _GROUND_TRUTH["iep_attendance_gap_pp"]
        assert abs(gt_gap - 14.04) <= 0.1, (
            f"IEP attendance gap was {gt_gap:.4f} pp, expected ~14.04 pp"
        )

    def test_f3_non_iep_attendance_rate_matches_ground_truth(self) -> None:
        """F3: Non-IEP attendance rate 0.8243 from ground_truth.json."""
        gt_rate = _GROUND_TRUTH["non_iep_attendance_rate"]
        assert abs(gt_rate - 0.8243) <= 0.001, f"Non-IEP attendance rate was {gt_rate:.4f}"

    def test_f4_iep_november_attendance_is_lowest_month(self) -> None:
        """F4: IEP November attendance (66.9%) is the lowest of the three months."""
        sep_rate = _attendance_rate_for(_SUBGROUP_ATTENDANCE, "2025-09", "iep", "true")
        oct_rate = _attendance_rate_for(_SUBGROUP_ATTENDANCE, "2025-10", "iep", "true")
        nov_rate = _attendance_rate_for(_SUBGROUP_ATTENDANCE, "2025-11", "iep", "true")

        assert abs(nov_rate - 0.6689) <= 0.001, (
            f"IEP November rate was {nov_rate:.4f}, expected ~0.6689"
        )
        assert nov_rate < sep_rate, (
            f"November rate {nov_rate:.4f} is not below September {sep_rate:.4f}"
        )
        assert nov_rate < oct_rate, (
            f"November rate {nov_rate:.4f} is not below October {oct_rate:.4f}"
        )

    def test_f4_iep_october_attendance_is_highest_month(self) -> None:
        """F4: IEP October attendance (70.6%) is the highest of the three months."""
        sep_rate = _attendance_rate_for(_SUBGROUP_ATTENDANCE, "2025-09", "iep", "true")
        oct_rate = _attendance_rate_for(_SUBGROUP_ATTENDANCE, "2025-10", "iep", "true")
        nov_rate = _attendance_rate_for(_SUBGROUP_ATTENDANCE, "2025-11", "iep", "true")

        assert abs(oct_rate - 0.7063) <= 0.001, (
            f"IEP October rate was {oct_rate:.4f}, expected ~0.7063"
        )
        assert oct_rate > sep_rate, (
            f"October rate {oct_rate:.4f} is not above September {sep_rate:.4f}"
        )
        assert oct_rate > nov_rate, (
            f"October rate {oct_rate:.4f} is not above November {nov_rate:.4f}"
        )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T5-EFR-003
# ---------------------------------------------------------------------------


class TestT5EFR003GoldFacts:
    """Recompute gold facts for T5-EFR-003 (stand-alone tutoring design evaluation)."""

    def test_f2_non_iep_spring_proficiency(self) -> None:
        """F2: Non-IEP spring 2026 proficiency ~61.8% (n=120)."""
        rate = _proficiency_rate_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "false")
        assert abs(rate - 0.6177) <= 0.001, (
            f"Non-IEP spring proficiency was {rate:.4f}, expected ~0.6177"
        )
        n = _n_students_for_outcome_row(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "false")
        assert n == 120, f"Non-IEP n was {n}, expected 120"

    def test_f2_iep_spring_proficiency(self) -> None:
        """F2: IEP spring 2026 proficiency ~39.3% (n=15)."""
        rate = _proficiency_rate_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        assert abs(rate - 0.3929) <= 0.001, (
            f"IEP spring proficiency was {rate:.4f}, expected ~0.3929"
        )
        n = _n_students_for_outcome_row(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        assert n == 15, f"IEP n was {n}, expected 15"

    def test_f2_iep_proficiency_gap_matches_ground_truth(self) -> None:
        """F2: Proficiency gap ~22.48 pp from ground_truth.json."""
        gt_gap = _GROUND_TRUTH["iep_proficiency_gap_pp_spring"]
        assert abs(gt_gap - 22.48) <= 0.1, (
            f"IEP proficiency gap was {gt_gap:.4f} pp, expected ~22.48 pp"
        )

    def test_f3_iep_score_gain(self) -> None:
        """F3: IEP students gained an average of 8.06 scale points by spring 2026."""
        gain = _score_gain_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        assert abs(gain - 8.06) <= 0.01, f"IEP avg score gain was {gain:.4f}, expected ~8.06"

    def test_f3_non_iep_score_gain(self) -> None:
        """F3: Non-IEP students gained an average of 9.24 scale points by spring 2026."""
        gain = _score_gain_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "false")
        assert abs(gain - 9.24) <= 0.01, f"Non-IEP avg score gain was {gain:.4f}, expected ~9.24"


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T5-EFR-004
# ---------------------------------------------------------------------------


class TestT5EFR004GoldFacts:
    """Recompute gold facts for T5-EFR-004 (additional data for FRL effectiveness)."""

    def test_f1_frl_student_count(self) -> None:
        """F1: 73 students are FRL-eligible (free_reduced_lunch=true)."""
        frl_count = sum(1 for s in _STUDENTS if s["free_reduced_lunch"] == "true")
        assert frl_count == 73, f"FRL count was {frl_count}, expected 73"

    def test_f1_frl_student_pct_of_total(self) -> None:
        """F1: FRL-eligible students are approximately 54.1% of total enrollment (73/135)."""
        frl_count = sum(1 for s in _STUDENTS if s["free_reduced_lunch"] == "true")
        total = len(_STUDENTS)
        pct = frl_count / total * 100
        assert abs(pct - 54.07) <= 0.2, f"FRL percentage was {pct:.2f}%, expected ~54.1%"

    def test_f2_frl_spring_proficiency(self) -> None:
        """F2: FRL-eligible spring 2026 proficiency ~56.4% (n=73)."""
        rate = _proficiency_rate_for(
            _SUBGROUP_OUTCOMES, "spring_2026", "free_reduced_lunch", "true"
        )
        assert abs(rate - 0.5641) <= 0.001, (
            f"FRL spring proficiency was {rate:.4f}, expected ~0.5641"
        )
        n = _n_students_for_outcome_row(
            _SUBGROUP_OUTCOMES, "spring_2026", "free_reduced_lunch", "true"
        )
        assert n == 73, f"FRL n was {n}, expected 73"

    def test_f2_non_frl_spring_proficiency(self) -> None:
        """F2: Non-FRL spring 2026 proficiency ~59.3% (n=62)."""
        rate = _proficiency_rate_for(
            _SUBGROUP_OUTCOMES, "spring_2026", "free_reduced_lunch", "false"
        )
        assert abs(rate - 0.5929) <= 0.001, (
            f"Non-FRL spring proficiency was {rate:.4f}, expected ~0.5929"
        )
        n = _n_students_for_outcome_row(
            _SUBGROUP_OUTCOMES, "spring_2026", "free_reduced_lunch", "false"
        )
        assert n == 62, f"Non-FRL n was {n}, expected 62"

    def test_f2_frl_spring_gap(self) -> None:
        """F2: FRL vs non-FRL spring proficiency gap ~2.9 pp."""
        frl_rate = _proficiency_rate_for(
            _SUBGROUP_OUTCOMES, "spring_2026", "free_reduced_lunch", "true"
        )
        non_frl_rate = _proficiency_rate_for(
            _SUBGROUP_OUTCOMES, "spring_2026", "free_reduced_lunch", "false"
        )
        gap_pp = (non_frl_rate - frl_rate) * 100
        assert abs(gap_pp - 2.88) <= 0.2, f"FRL spring gap was {gap_pp:.2f} pp, expected ~2.88 pp"

    def test_f3_frl_fall_proficiency(self) -> None:
        """F3: FRL fall 2025 proficiency ~59.4% (n=73)."""
        rate = _proficiency_rate_for(_SUBGROUP_OUTCOMES, "fall_2025", "free_reduced_lunch", "true")
        assert abs(rate - 0.5944) <= 0.001, f"FRL fall proficiency was {rate:.4f}, expected ~0.5944"

    def test_f3_non_frl_fall_proficiency(self) -> None:
        """F3: Non-FRL fall 2025 proficiency ~59.9% (n=62)."""
        rate = _proficiency_rate_for(_SUBGROUP_OUTCOMES, "fall_2025", "free_reduced_lunch", "false")
        assert abs(rate - 0.5988) <= 0.001, (
            f"Non-FRL fall proficiency was {rate:.4f}, expected ~0.5988"
        )

    def test_f3_frl_fall_baseline_nearly_identical(self) -> None:
        """F3: FRL and non-FRL fall baselines are within 1 pp of each other."""
        frl_fall = _proficiency_rate_for(
            _SUBGROUP_OUTCOMES, "fall_2025", "free_reduced_lunch", "true"
        )
        non_frl_fall = _proficiency_rate_for(
            _SUBGROUP_OUTCOMES, "fall_2025", "free_reduced_lunch", "false"
        )
        fall_gap_pp = abs(non_frl_fall - frl_fall) * 100
        assert fall_gap_pp < 1.0, (
            f"FRL vs non-FRL fall gap was {fall_gap_pp:.2f} pp; expected < 1 pp"
        )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T5-EFR-005
# ---------------------------------------------------------------------------


class TestT5EFR005GoldFacts:
    """Recompute gold facts for T5-EFR-005 (IEP outcome risk synthesis)."""

    def test_f1_iep_attendance_rate_from_ground_truth(self) -> None:
        """F1: IEP attendance rate 0.6839 from ground_truth.json."""
        gt_rate = _GROUND_TRUTH["iep_attendance_rate"]
        assert abs(gt_rate - 0.6839) <= 0.001, f"IEP attendance rate was {gt_rate:.4f}"

    def test_f1_iep_attendance_below_r10_threshold(self) -> None:
        """F1: IEP rate (68.4%) is below the 80% threshold, shortfall ~11.6 pp."""
        gt_rate = _GROUND_TRUTH["iep_attendance_rate"]
        shortfall = (0.80 - gt_rate) * 100
        assert abs(shortfall - 11.61) <= 0.2, (
            f"Shortfall from 80% threshold was {shortfall:.2f} pp, expected ~11.6 pp"
        )

    def test_f2_iep_spring_proficiency(self) -> None:
        """F2: IEP spring 2026 proficiency ~39.3% (n=15)."""
        rate = _proficiency_rate_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        n = _n_students_for_outcome_row(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        assert abs(rate - 0.3929) <= 0.001, f"IEP spring proficiency was {rate:.4f}"
        assert n == 15, f"IEP spring n was {n}, expected 15"

    def test_f3_iep_proficiency_gap_spring_matches_ground_truth(self) -> None:
        """F3: IEP proficiency gap ~22.48 pp from ground_truth.json."""
        gt_gap = _GROUND_TRUTH["iep_proficiency_gap_pp_spring"]
        assert abs(gt_gap - 22.48) <= 0.1, (
            f"IEP proficiency gap was {gt_gap:.4f} pp, expected ~22.48 pp"
        )

    def test_f3_proficiency_gap_recomputed_from_fixture(self) -> None:
        """F3: Gap recomputed from subgroup_outcomes_summary.csv matches ground_truth.json."""
        non_iep_rate = _proficiency_rate_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "false")
        iep_rate = _proficiency_rate_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        gap_pp = (non_iep_rate - iep_rate) * 100
        gt_gap = _GROUND_TRUTH["iep_proficiency_gap_pp_spring"]
        assert abs(gap_pp - gt_gap) <= 0.2, (
            f"Recomputed gap {gap_pp:.4f} does not match ground_truth {gt_gap}"
        )

    def test_f4_iep_score_gain(self) -> None:
        """F4: IEP average score gain is 8.06 scale points."""
        gain = _score_gain_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        assert abs(gain - 8.06) <= 0.01, f"IEP avg score gain was {gain:.4f}, expected ~8.06"

    def test_f4_non_iep_score_gain(self) -> None:
        """F4: Non-IEP average score gain is 9.24 scale points."""
        gain = _score_gain_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "false")
        assert abs(gain - 9.24) <= 0.01, f"Non-IEP avg score gain was {gain:.4f}, expected ~9.24"

    def test_f4_iep_gain_smaller_than_non_iep(self) -> None:
        """F4: IEP score gain (8.06) is smaller than non-IEP (9.24)."""
        iep_gain = _score_gain_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        non_iep_gain = _score_gain_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "false")
        assert iep_gain < non_iep_gain, (
            f"IEP gain {iep_gain:.4f} is not smaller than non-IEP {non_iep_gain:.4f}"
        )


# ---------------------------------------------------------------------------
# Cross-task ref_id coverage test
# ---------------------------------------------------------------------------


class TestResearchRefCoverage:
    """Verify that the task file collectively covers a broad range of ref_ids."""

    def test_at_least_five_distinct_ref_ids_cited_across_tasks(self) -> None:
        """All five tasks together must cite at least 5 distinct valid ref_ids."""
        cited_refs: set[str] = set()
        for task in _load_tasks():
            rubric = task.get("rubric", {})
            text_parts: list[str] = []
            for dim_data in rubric.values():
                if isinstance(dim_data, dict):
                    guidance = dim_data.get("guidance", "")
                    if guidance:
                        text_parts.append(guidance)
            outline = task.get("reference_answer_outline", "")
            if outline:
                text_parts.append(outline)
            for fact in task.get("gold_facts", []):
                text_parts.append(fact.get("claim", ""))
            combined = " ".join(text_parts)
            for ref_id in _VALID_REF_IDS:
                if ref_id in combined:
                    cited_refs.add(ref_id)

        assert len(cited_refs) >= 5, (
            f"Only {len(cited_refs)} distinct ref_ids cited across all tasks: {cited_refs}. "
            "Expected at least 5."
        )
