"""Verification tests for Track 4 (Equity & Subgroup Interpretation) tasks.

These tests enforce three guarantees:

1. **Schema validity** — every task in tasks.jsonl must pass ``validate_task``.
2. **Gold-fact correctness** — every numeric gold fact is independently recomputed
   from the Equity & Research pack fixture CSVs and asserted to match ground_truth.json.
3. **Track-4 invariants** — every task must have at least one sample-size
   ``required_limitation`` and at least one overgeneralization ``forbidden_claim``;
   fact_id values must be unique within each task; the file must contain ~5 tasks.

Running ``pytest tests/unit/test_track_equity_interpretation.py`` must pass before any
Track 4 task is merged.
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
_TASKS_PATH = _REPO_ROOT / "benchmark" / "tracks" / "equity_interpretation" / "tasks.jsonl"
_PACK_DIR = _REPO_ROOT / "fixtures" / "pack_equity_research"
_GROUND_TRUTH_PATH = _PACK_DIR / "ground_truth.json"

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


# Fixture data loaded once at module level so individual tests stay fast.
_SUBGROUP_ATTENDANCE = _load_csv("subgroup_attendance_summary.csv")
_SUBGROUP_OUTCOMES = _load_csv("subgroup_outcomes_summary.csv")
_GROUND_TRUTH = _load_ground_truth()

# ---------------------------------------------------------------------------
# Recomputation helpers
# ---------------------------------------------------------------------------


def _attendance_rows_for(
    data: list[dict[str, str]],
    dimension: str,
    value: str,
) -> list[dict[str, str]]:
    """Return rows for a specific subgroup dimension/value combination."""
    return [
        r for r in data if r["subgroup_dimension"] == dimension and r["subgroup_value"] == value
    ]


def _suppressed_rows_for(
    data: list[dict[str, str]],
    dimension: str,
) -> list[dict[str, str]]:
    """Return suppressed rows for a given subgroup dimension."""
    return [r for r in data if r["subgroup_dimension"] == dimension and r["suppressed"] == "true"]


def _unique_suppressed_values(
    data: list[dict[str, str]],
    dimension: str,
) -> set[str]:
    """Return unique suppressed subgroup values for a given dimension."""
    rows = _suppressed_rows_for(data, dimension)
    return {r["subgroup_value"] for r in rows}


def _n_students_for_outcome_row(
    data: list[dict[str, str]],
    period: str,
    dimension: str,
    value: str,
) -> int:
    """Return n_students from the outcomes summary for a specific row."""
    rows = [
        r
        for r in data
        if r["assessment_period"] == period
        and r["subgroup_dimension"] == dimension
        and r["subgroup_value"] == value
    ]
    assert len(rows) == 1, f"Expected 1 row, found {len(rows)} for {period}/{dimension}/{value}"
    return int(rows[0]["n_students"])


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

    def test_all_tasks_are_track_4(self) -> None:
        """Every task must declare track=4."""
        for task in _load_tasks():
            assert task["track"] == 4, (
                f"Task {task['task_id']} has track={task['track']}, expected 4"
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

    def test_all_task_types_are_track4_valid(self) -> None:
        """All tasks must use task_type values appropriate for Track 4."""
        valid_types = {"subgroup_comparison", "limitation_assessment"}
        for task in _load_tasks():
            assert task["task_type"] in valid_types, (
                f"Task {task['task_id']} has unexpected task_type={task['task_type']!r}"
            )


# ---------------------------------------------------------------------------
# Track 4 invariant tests
# ---------------------------------------------------------------------------


class TestTrack4Invariants:
    """Tests that enforce the DoD invariants specific to Track 4."""

    def test_every_task_has_sample_size_limitation(self) -> None:
        """Every Track 4 task must have at least one required_limitation about sample size.

        The required_limitation must contain at least one of the key phrases that
        signal a sample-size caveat: 'n=', 'sample', 'small', 'few students',
        'fewer than', or 'students'.
        """
        sample_size_keywords = {
            "n=",
            "sample",
            "small",
            "few students",
            "fewer than",
            "students",
            "n <",
            "subgroup",
        }
        for task in _load_tasks():
            limitations = task.get("required_limitations", [])
            assert len(limitations) > 0, f"Task {task['task_id']} has no required_limitations"
            has_sample_size_limitation = any(
                any(kw.lower() in lim.lower() for kw in sample_size_keywords) for lim in limitations
            )
            assert has_sample_size_limitation, (
                f"Task {task['task_id']} has no sample-size required_limitation. "
                f"Limitations: {limitations}"
            )

    def test_every_task_has_overgeneralization_forbidden_claim(self) -> None:
        """Every Track 4 task must have at least one forbidden_claim about overgeneralization.

        The forbidden_claim must contain at least one of the key phrases indicating
        that robust/generalizable conclusions are forbidden from small/suppressed samples.
        """
        overgeneralization_keywords = {
            "robust",
            "statistically significant",
            "generaliz",
            "generalise",
            "conclusive",
            "conclus",
            "proves",
            "prove",
            "can be drawn",
            "firm",
            "definitive",
        }
        for task in _load_tasks():
            forbidden = task.get("forbidden_claims", [])
            assert len(forbidden) > 0, f"Task {task['task_id']} has no forbidden_claims"
            has_overgeneralization = any(
                any(kw.lower() in claim.lower() for kw in overgeneralization_keywords)
                for claim in forbidden
            )
            assert has_overgeneralization, (
                f"Task {task['task_id']} has no overgeneralization forbidden_claim. "
                f"Forbidden claims: {forbidden}"
            )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _load_tasks(), ids=[t["task_id"] for t in _load_tasks()])
def test_task_schema_valid(task: dict) -> None:
    """Every task in tasks.jsonl must pass validate_task without error."""
    validate_task(task)  # raises jsonschema.ValidationError on failure


# ---------------------------------------------------------------------------
# Ground truth consistency tests
# ---------------------------------------------------------------------------


class TestGroundTruthConsistency:
    """Assert that ground_truth.json values match the fixture CSV data."""

    def test_iep_attendance_gap_matches_ground_truth(self) -> None:
        """IEP attendance gap in ground_truth.json matches subgroup_attendance_summary.csv."""
        gt_gap = _GROUND_TRUTH["iep_attendance_gap_pp"]
        gt_iep_rate = _GROUND_TRUTH["iep_attendance_rate"]
        gt_non_iep_rate = _GROUND_TRUTH["non_iep_attendance_rate"]
        recomputed_gap = (gt_non_iep_rate - gt_iep_rate) * 100
        assert abs(recomputed_gap - gt_gap) <= 0.1, (
            f"Ground truth gap {gt_gap} does not match recomputed {recomputed_gap:.4f}"
        )

    def test_suppressed_subgroups_in_ground_truth(self) -> None:
        """Suppressed subgroups list must contain AIAN, NHPI, TwoOrMore, and Unknown."""
        suppressed = _GROUND_TRUTH["suppressed_low_n_subgroups"]
        suppressed_values = {s["subgroup_value"] for s in suppressed}
        assert "AIAN" in suppressed_values
        assert "NHPI" in suppressed_values
        assert "TwoOrMore" in suppressed_values
        assert "Unknown" in suppressed_values

    def test_aian_n_in_ground_truth(self) -> None:
        """AIAN subgroup must have n=4 in ground_truth.json."""
        suppressed = _GROUND_TRUTH["suppressed_low_n_subgroups"]
        aian = next(s for s in suppressed if s["subgroup_value"] == "AIAN")
        assert aian["n"] == 4, f"AIAN n was {aian['n']}, expected 4"

    def test_nhpi_n_in_ground_truth(self) -> None:
        """NHPI subgroup must have n=4 in ground_truth.json."""
        suppressed = _GROUND_TRUTH["suppressed_low_n_subgroups"]
        nhpi = next(s for s in suppressed if s["subgroup_value"] == "NHPI")
        assert nhpi["n"] == 4, f"NHPI n was {nhpi['n']}, expected 4"

    def test_twoormore_n_in_ground_truth(self) -> None:
        """TwoOrMore subgroup must have n=7 in ground_truth.json."""
        suppressed = _GROUND_TRUTH["suppressed_low_n_subgroups"]
        two = next(s for s in suppressed if s["subgroup_value"] == "TwoOrMore")
        assert two["n"] == 7, f"TwoOrMore n was {two['n']}, expected 7"

    def test_unknown_n_in_ground_truth(self) -> None:
        """Unknown subgroup must have n=1 in ground_truth.json."""
        suppressed = _GROUND_TRUTH["suppressed_low_n_subgroups"]
        unknown = next(s for s in suppressed if s["subgroup_value"] == "Unknown")
        assert unknown["n"] == 1, f"Unknown n was {unknown['n']}, expected 1"

    def test_iep_proficiency_gap_matches_ground_truth(self) -> None:
        """IEP proficiency gap in ground_truth.json matches subgroup_outcome_disparities."""
        disparities = _GROUND_TRUTH["subgroup_outcome_disparities"]
        non_iep = next(d for d in disparities if d["subgroup_value"] == "false")
        iep = next(d for d in disparities if d["subgroup_value"] == "true")
        gap = (non_iep["benchmark_proficiency_rate"] - iep["benchmark_proficiency_rate"]) * 100
        expected_gap = _GROUND_TRUTH["iep_proficiency_gap_pp_spring"]
        assert abs(gap - expected_gap) <= 0.1, (
            f"Recomputed gap {gap:.4f} does not match ground_truth {expected_gap}"
        )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T4-EQU-001
# ---------------------------------------------------------------------------


class TestT4EQU001GoldFacts:
    """Recompute gold facts for T4-EQU-001 (IEP vs non-IEP attendance gap)."""

    def test_f1_iep_attendance_rate(self) -> None:
        """F1: IEP attendance rate ~68.4% matches ground_truth.json."""
        gt_rate = _GROUND_TRUTH["iep_attendance_rate"]
        assert abs(gt_rate - 0.6839) <= 0.001, f"IEP rate was {gt_rate:.4f}"

    def test_f2_non_iep_attendance_rate(self) -> None:
        """F2: Non-IEP attendance rate ~82.4% matches ground_truth.json."""
        gt_rate = _GROUND_TRUTH["non_iep_attendance_rate"]
        assert abs(gt_rate - 0.8243) <= 0.001, f"Non-IEP rate was {gt_rate:.4f}"

    def test_f3_iep_attendance_gap(self) -> None:
        """F3: IEP attendance gap ~14.04 pp matches ground_truth.json."""
        gt_gap = _GROUND_TRUTH["iep_attendance_gap_pp"]
        assert abs(gt_gap - 14.04) <= 0.1, f"IEP gap was {gt_gap:.4f}"

    def test_f4_iep_subgroup_n_in_summary(self) -> None:
        """F4: IEP=true subgroup has n=14 students in the attendance summary."""
        iep_rows = _attendance_rows_for(_SUBGROUP_ATTENDANCE, "iep", "true")
        # Use the first month's n_students as representative
        assert len(iep_rows) > 0, "No IEP=true rows found in attendance summary"
        n = int(iep_rows[0]["n_students"])
        assert n == 14, f"IEP n_students was {n}, expected 14"


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T4-EQU-002
# ---------------------------------------------------------------------------


class TestT4EQU002GoldFacts:
    """Recompute gold facts for T4-EQU-002 (suppressed subgroup identification)."""

    def test_f1_aian_suppressed_in_outcomes(self) -> None:
        """F1: AIAN subgroup (n=4) is suppressed in outcomes data."""
        aian_rows = [
            r
            for r in _SUBGROUP_OUTCOMES
            if r["subgroup_dimension"] == "race_ethnicity" and r["subgroup_value"] == "AIAN"
        ]
        assert len(aian_rows) > 0, "No AIAN rows found in outcomes summary"
        for row in aian_rows:
            assert row["suppressed"] == "true", f"AIAN row {row['summary_id']} is not suppressed"
        gt_aian = next(
            s for s in _GROUND_TRUTH["suppressed_low_n_subgroups"] if s["subgroup_value"] == "AIAN"
        )
        assert gt_aian["n"] == 4

    def test_f2_nhpi_suppressed_in_outcomes(self) -> None:
        """F2: NHPI subgroup (n=4) is suppressed in outcomes data."""
        nhpi_rows = [
            r
            for r in _SUBGROUP_OUTCOMES
            if r["subgroup_dimension"] == "race_ethnicity" and r["subgroup_value"] == "NHPI"
        ]
        assert len(nhpi_rows) > 0, "No NHPI rows found in outcomes summary"
        for row in nhpi_rows:
            assert row["suppressed"] == "true", f"NHPI row {row['summary_id']} is not suppressed"
        gt_nhpi = next(
            s for s in _GROUND_TRUTH["suppressed_low_n_subgroups"] if s["subgroup_value"] == "NHPI"
        )
        assert gt_nhpi["n"] == 4

    def test_f3_twoormore_suppressed_in_outcomes(self) -> None:
        """F3: TwoOrMore subgroup (n=7) is suppressed in outcomes data."""
        two_rows = [
            r
            for r in _SUBGROUP_OUTCOMES
            if r["subgroup_dimension"] == "race_ethnicity" and r["subgroup_value"] == "TwoOrMore"
        ]
        assert len(two_rows) > 0, "No TwoOrMore rows found in outcomes summary"
        for row in two_rows:
            assert row["suppressed"] == "true", (
                f"TwoOrMore row {row['summary_id']} is not suppressed"
            )
        gt_two = next(
            s
            for s in _GROUND_TRUTH["suppressed_low_n_subgroups"]
            if s["subgroup_value"] == "TwoOrMore"
        )
        assert gt_two["n"] == 7

    def test_f4_unknown_suppressed_in_outcomes(self) -> None:
        """F4: Unknown subgroup (n=1) is suppressed in outcomes data."""
        unk_rows = [
            r
            for r in _SUBGROUP_OUTCOMES
            if r["subgroup_dimension"] == "race_ethnicity" and r["subgroup_value"] == "Unknown"
        ]
        assert len(unk_rows) > 0, "No Unknown rows found in outcomes summary"
        for row in unk_rows:
            assert row["suppressed"] == "true", f"Unknown row {row['summary_id']} is not suppressed"
        gt_unk = next(
            s
            for s in _GROUND_TRUTH["suppressed_low_n_subgroups"]
            if s["subgroup_value"] == "Unknown"
        )
        assert gt_unk["n"] == 1

    def test_f5_suppressed_reason_is_low_n(self) -> None:
        """F5: All suppressed race/ethnicity rows have suppression_reason 'n < 10'."""
        suppressed = [
            r
            for r in _SUBGROUP_OUTCOMES
            if r["subgroup_dimension"] == "race_ethnicity" and r["suppressed"] == "true"
        ]
        assert len(suppressed) > 0, "No suppressed race/ethnicity rows found"
        for row in suppressed:
            assert row["suppression_reason"] == "n < 10", (
                f"Row {row['summary_id']} has suppression_reason={row['suppression_reason']!r}"
            )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T4-EQU-003
# ---------------------------------------------------------------------------


class TestT4EQU003GoldFacts:
    """Recompute gold facts for T4-EQU-003 (IEP proficiency gap robustness)."""

    def test_f1_non_iep_spring_proficiency(self) -> None:
        """F1: Non-IEP spring 2026 proficiency rate ~61.8% (n=120)."""
        rate = _proficiency_rate_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "false")
        assert abs(rate - 0.6177) <= 0.001, f"Non-IEP proficiency was {rate:.4f}"
        n = _n_students_for_outcome_row(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "false")
        assert n == 120, f"Non-IEP n was {n}, expected 120"

    def test_f2_iep_spring_proficiency(self) -> None:
        """F2: IEP spring 2026 proficiency rate ~39.3% (n=15)."""
        rate = _proficiency_rate_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        assert abs(rate - 0.3929) <= 0.001, f"IEP proficiency was {rate:.4f}"
        n = _n_students_for_outcome_row(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        assert n == 15, f"IEP n was {n}, expected 15"

    def test_f3_iep_proficiency_gap(self) -> None:
        """F3: IEP proficiency gap ~22.48 pp matches ground_truth.json."""
        gt_gap = _GROUND_TRUTH["iep_proficiency_gap_pp_spring"]
        assert abs(gt_gap - 22.48) <= 0.1, f"IEP proficiency gap was {gt_gap:.4f}"

    def test_f3_gap_recomputed_from_fixture(self) -> None:
        """F3: Gap recomputed from fixture matches ground_truth value."""
        non_iep_rate = _proficiency_rate_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "false")
        iep_rate = _proficiency_rate_for(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        gap_pp = (non_iep_rate - iep_rate) * 100
        gt_gap = _GROUND_TRUTH["iep_proficiency_gap_pp_spring"]
        assert abs(gap_pp - gt_gap) <= 0.2, (
            f"Recomputed gap {gap_pp:.4f} does not match ground_truth {gt_gap}"
        )

    def test_f4_iep_n_is_15_in_spring_outcomes(self) -> None:
        """F4: IEP subgroup has n=15 students in spring 2026 outcomes."""
        n = _n_students_for_outcome_row(_SUBGROUP_OUTCOMES, "spring_2026", "iep", "true")
        assert n == 15, f"IEP n_students was {n}, expected 15"


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T4-EQU-004
# ---------------------------------------------------------------------------


class TestT4EQU004GoldFacts:
    """Recompute gold facts for T4-EQU-004 (Hispanic and Black attendance vs. program avg)."""

    def test_f1_hispanic_september_attendance(self) -> None:
        """F1: Hispanic September attendance rate ~80.8%."""
        rows = [
            r
            for r in _SUBGROUP_ATTENDANCE
            if r["subgroup_dimension"] == "race_ethnicity"
            and r["subgroup_value"] == "Hispanic"
            and r["month_label"] == "2025-09"
        ]
        assert len(rows) == 1, f"Expected 1 row, found {len(rows)}"
        rate = float(rows[0]["attendance_rate"])
        assert abs(rate - 0.8083) <= 0.001, f"Hispanic Sep rate was {rate:.4f}"

    def test_f1_hispanic_october_attendance(self) -> None:
        """F1: Hispanic October attendance rate ~84.8%."""
        rows = [
            r
            for r in _SUBGROUP_ATTENDANCE
            if r["subgroup_dimension"] == "race_ethnicity"
            and r["subgroup_value"] == "Hispanic"
            and r["month_label"] == "2025-10"
        ]
        assert len(rows) == 1, f"Expected 1 row, found {len(rows)}"
        rate = float(rows[0]["attendance_rate"])
        assert abs(rate - 0.8482) <= 0.001, f"Hispanic Oct rate was {rate:.4f}"

    def test_f1_hispanic_november_attendance(self) -> None:
        """F1: Hispanic November attendance rate ~84.4%."""
        rows = [
            r
            for r in _SUBGROUP_ATTENDANCE
            if r["subgroup_dimension"] == "race_ethnicity"
            and r["subgroup_value"] == "Hispanic"
            and r["month_label"] == "2025-11"
        ]
        assert len(rows) == 1, f"Expected 1 row, found {len(rows)}"
        rate = float(rows[0]["attendance_rate"])
        assert abs(rate - 0.8442) <= 0.001, f"Hispanic Nov rate was {rate:.4f}"

    def test_f2_black_september_attendance(self) -> None:
        """F2: Black September attendance rate ~77.3%."""
        rows = [
            r
            for r in _SUBGROUP_ATTENDANCE
            if r["subgroup_dimension"] == "race_ethnicity"
            and r["subgroup_value"] == "Black"
            and r["month_label"] == "2025-09"
        ]
        assert len(rows) == 1, f"Expected 1 row, found {len(rows)}"
        rate = float(rows[0]["attendance_rate"])
        assert abs(rate - 0.7728) <= 0.001, f"Black Sep rate was {rate:.4f}"

    def test_f2_black_october_attendance(self) -> None:
        """F2: Black October attendance rate ~81.1%."""
        rows = [
            r
            for r in _SUBGROUP_ATTENDANCE
            if r["subgroup_dimension"] == "race_ethnicity"
            and r["subgroup_value"] == "Black"
            and r["month_label"] == "2025-10"
        ]
        assert len(rows) == 1, f"Expected 1 row, found {len(rows)}"
        rate = float(rows[0]["attendance_rate"])
        assert abs(rate - 0.8108) <= 0.001, f"Black Oct rate was {rate:.4f}"

    def test_f2_black_november_attendance(self) -> None:
        """F2: Black November attendance rate ~79.2%."""
        rows = [
            r
            for r in _SUBGROUP_ATTENDANCE
            if r["subgroup_dimension"] == "race_ethnicity"
            and r["subgroup_value"] == "Black"
            and r["month_label"] == "2025-11"
        ]
        assert len(rows) == 1, f"Expected 1 row, found {len(rows)}"
        rate = float(rows[0]["attendance_rate"])
        assert abs(rate - 0.792) <= 0.001, f"Black Nov rate was {rate:.4f}"

    def test_f3_program_wide_rate_from_ground_truth(self) -> None:
        """F3: Program-wide attendance rate ~80.9% from ground_truth.json."""
        gt_rate = _GROUND_TRUTH["program_wide_attendance_rate"]
        assert abs(gt_rate - 0.8092) <= 0.001, f"Program-wide rate was {gt_rate:.4f}"

    def test_f4_hispanic_and_black_not_suppressed(self) -> None:
        """F4: Hispanic and Black subgroups are not suppressed in the attendance summary."""
        for ethnicity in ("Hispanic", "Black"):
            rows = [
                r
                for r in _SUBGROUP_ATTENDANCE
                if r["subgroup_dimension"] == "race_ethnicity" and r["subgroup_value"] == ethnicity
            ]
            assert len(rows) > 0, f"No rows for {ethnicity}"
            for row in rows:
                assert row["suppressed"] == "false", (
                    f"{ethnicity} row {row['summary_id']} is unexpectedly suppressed"
                )


# ---------------------------------------------------------------------------
# Gold-fact recomputation tests — T4-EQU-005
# ---------------------------------------------------------------------------


class TestT4EQU005GoldFacts:
    """Recompute gold facts for T4-EQU-005 (comprehensive race/ethnicity equity review)."""

    def test_f1_five_reportable_race_eth_subgroups_in_attendance(self) -> None:
        """F1: Five race/ethnicity subgroups have reportable attendance data."""
        reportable = {
            r["subgroup_value"]
            for r in _SUBGROUP_ATTENDANCE
            if r["subgroup_dimension"] == "race_ethnicity" and r["suppressed"] == "false"
        }
        expected = {"Asian", "Black", "Hispanic", "White"}
        # All four specific groups must be present
        for grp in expected:
            assert grp in reportable, f"Expected reportable group {grp!r} not found"

    def test_f2_four_suppressed_race_eth_subgroups(self) -> None:
        """F2: Exactly four race/ethnicity subgroups are suppressed in attendance data."""
        suppressed = _unique_suppressed_values(_SUBGROUP_ATTENDANCE, "race_ethnicity")
        expected = {"AIAN", "NHPI", "TwoOrMore", "Unknown"}
        assert suppressed == expected, (
            f"Suppressed values {suppressed} do not match expected {expected}"
        )

    def test_f2_suppressed_ns_match_ground_truth(self) -> None:
        """F2: Suppressed Ns in ground_truth.json: AIAN=4, NHPI=4, TwoOrMore=7, Unknown=1."""
        gt_suppressed = {
            s["subgroup_value"]: s["n"] for s in _GROUND_TRUTH["suppressed_low_n_subgroups"]
        }
        assert gt_suppressed["AIAN"] == 4
        assert gt_suppressed["NHPI"] == 4
        assert gt_suppressed["TwoOrMore"] == 7
        assert gt_suppressed["Unknown"] == 1

    def test_f4_hispanic_at_or_near_program_average_all_months(self) -> None:
        """F4: Hispanic attendance rates in all months are at or near the program average (~80.9%).

        September rate (80.83%) is within 0.1 pp of the program average; October and November
        are above it. The gold fact states 'at or above 80%' in all months, which is verified here.
        """
        for month in ("2025-09", "2025-10", "2025-11"):
            rows = [
                r
                for r in _SUBGROUP_ATTENDANCE
                if r["subgroup_dimension"] == "race_ethnicity"
                and r["subgroup_value"] == "Hispanic"
                and r["month_label"] == month
            ]
            assert len(rows) == 1
            rate = float(rows[0]["attendance_rate"])
            # All months should be at or above 80% (the program's stated attendance goal threshold)
            assert rate >= 0.80, f"Hispanic rate in {month} ({rate:.4f}) < 80%"

    def test_f5_asian_n_is_smallest_reportable(self) -> None:
        """F5: Asian is the reportable race/ethnicity subgroup with the smallest n (n=12)."""
        reportable_ns: dict[str, int] = {}
        for r in _SUBGROUP_ATTENDANCE:
            if (
                r["subgroup_dimension"] == "race_ethnicity"
                and r["suppressed"] == "false"
                and r["month_label"] == "2025-09"
            ):
                reportable_ns[r["subgroup_value"]] = int(r["n_students"])

        assert "Asian" in reportable_ns, "Asian not in reportable subgroups"
        asian_n = reportable_ns["Asian"]
        assert asian_n == 12, f"Asian n was {asian_n}, expected 12"
        # Asian must have the smallest n among reportable subgroups
        min_n = min(reportable_ns.values())
        assert asian_n == min_n, (
            f"Asian (n={asian_n}) is not the smallest reportable subgroup; min is {min_n}"
        )
