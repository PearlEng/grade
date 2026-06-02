"""Unit tests for the GRADE synthetic data generator (A3).

Three test categories:

1.  **Determinism** — same seed → byte-identical output across two independent runs.
2.  **Schema conformance** — every CSV and JSON file produced by the generator
    contains exactly the columns (and JSON keys) specified in
    ``benchmark/schemas/fixture_schema.md``.
3.  **Signal presence** — spot-checks that the three intentional signals baked in
    by ``spec.py`` are actually present in the generated data:

    * MoM attendance trend (Outcomes pack): attendance rises across the 3-month window.
    * Low-N subgroup (Equity & Research pack): AIAN subgroup is suppressed.
    * IEP disparity (Equity & Research pack): IEP students attend at a lower rate.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.datagen import spec
from benchmark.datagen.generator import (
    generate_equity_research,
    generate_operations,
    generate_outcomes,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex string.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts.

    Args:
        path: CSV file to read.

    Returns:
        List of row dicts.
    """
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file.

    Args:
        path: JSON file to read.

    Returns:
        Parsed dictionary.
    """
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ops_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the Operations pack once per test session.

    Returns:
        Path to ``pack_operations/`` directory.
    """
    out = tmp_path_factory.mktemp("grade_ops")
    return generate_operations(out, SEED)


@pytest.fixture(scope="module")
def outcomes_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the Outcomes pack once per test session.

    Returns:
        Path to ``pack_outcomes/`` directory.
    """
    out = tmp_path_factory.mktemp("grade_out")
    return generate_outcomes(out, SEED)


@pytest.fixture(scope="module")
def equity_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the Equity & Research pack once per test session.

    Returns:
        Path to ``pack_equity_research/`` directory.
    """
    out = tmp_path_factory.mktemp("grade_eq")
    return generate_equity_research(out, SEED)


# ---------------------------------------------------------------------------
# 1. Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed → byte-identical files on two independent runs."""

    def test_operations_byte_identical(self, tmp_path: Path) -> None:
        """Operations pack re-run with same seed produces identical files."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        pack_a = generate_operations(dir_a, SEED)
        pack_b = generate_operations(dir_b, SEED)

        files_a = sorted(f.name for f in pack_a.iterdir() if f.name != "manifest.json")
        files_b = sorted(f.name for f in pack_b.iterdir() if f.name != "manifest.json")
        assert files_a == files_b, "File inventory differs between runs"

        for fname in files_a:
            sha_a = _sha256(pack_a / fname)
            sha_b = _sha256(pack_b / fname)
            assert sha_a == sha_b, f"{fname}: hash differs between runs"

    def test_outcomes_byte_identical(self, tmp_path: Path) -> None:
        """Outcomes pack re-run with same seed produces identical files."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        pack_a = generate_outcomes(dir_a, SEED)
        pack_b = generate_outcomes(dir_b, SEED)

        files_a = sorted(f.name for f in pack_a.iterdir() if f.name != "manifest.json")
        for fname in files_a:
            assert _sha256(pack_a / fname) == _sha256(pack_b / fname), (
                f"{fname}: hash differs between runs"
            )

    def test_equity_byte_identical(self, tmp_path: Path) -> None:
        """Equity & Research pack re-run with same seed produces identical files."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        pack_a = generate_equity_research(dir_a, SEED)
        pack_b = generate_equity_research(dir_b, SEED)

        files_a = sorted(f.name for f in pack_a.iterdir() if f.name != "manifest.json")
        for fname in files_a:
            assert _sha256(pack_a / fname) == _sha256(pack_b / fname), (
                f"{fname}: hash differs between runs"
            )

    def test_different_seeds_differ(self, tmp_path: Path) -> None:
        """Different seeds must produce different students.csv."""
        pack_42 = generate_operations(tmp_path / "s42", 42)
        pack_99 = generate_operations(tmp_path / "s99", 99)
        assert _sha256(pack_42 / "students.csv") != _sha256(pack_99 / "students.csv")

    def test_cli_byte_identical(self, tmp_path: Path) -> None:
        """CLI invocation with same seed produces identical operations pack."""
        dir_a = tmp_path / "cli_a"
        dir_b = tmp_path / "cli_b"
        main(["--seed", "42", "--out", str(dir_a), "--pack", "operations"])
        main(["--seed", "42", "--out", str(dir_b), "--pack", "operations"])

        pack_a = dir_a / "pack_operations"
        pack_b = dir_b / "pack_operations"
        for fname in sorted(f.name for f in pack_a.iterdir() if f.name != "manifest.json"):
            assert _sha256(pack_a / fname) == _sha256(pack_b / fname), (
                f"CLI: {fname} differs between runs"
            )


# ---------------------------------------------------------------------------
# 2. Schema conformance tests
# ---------------------------------------------------------------------------


class TestSchemaConformanceOperations:
    """Operations pack CSV/JSON column contracts from fixture_schema.md §1."""

    def test_programs_columns(self, ops_dir: Path) -> None:
        """programs.csv must have all required columns."""
        required = {
            "program_id",
            "program_name",
            "program_type",
            "start_date",
            "end_date",
            "target_grade_levels",
            "subject_areas",
            "district_id",
            "school_count",
            "dosage_target_minutes_per_week",
            "reporting_period_start",
            "reporting_period_end",
        }
        rows = _read_csv(ops_dir / "programs.csv")
        assert rows, "programs.csv is empty"
        assert required <= set(rows[0].keys())

    def test_schools_columns(self, ops_dir: Path) -> None:
        """schools.csv must have all required columns."""
        required = {
            "school_id",
            "school_name",
            "program_id",
            "district_id",
            "enrollment",
            "locale",
            "title_i",
            "grade_span",
        }
        rows = _read_csv(ops_dir / "schools.csv")
        assert required <= set(rows[0].keys())

    def test_students_columns(self, ops_dir: Path) -> None:
        """students.csv must have all required columns."""
        required = {
            "student_id",
            "program_id",
            "school_id",
            "grade_level",
            "gender",
            "race_ethnicity",
            "iep",
            "ell",
            "free_reduced_lunch",
            "enrollment_date",
            "active",
            "tutoring_group_id",
        }
        rows = _read_csv(ops_dir / "students.csv")
        assert required <= set(rows[0].keys())

    def test_tutors_columns(self, ops_dir: Path) -> None:
        """tutors.csv must have all required columns."""
        required = {
            "tutor_id",
            "program_id",
            "school_id",
            "hire_date",
            "certification_level",
            "subject_specialization",
            "active",
            "sessions_delivered_ytd",
            "avg_session_rating",
        }
        rows = _read_csv(ops_dir / "tutors.csv")
        assert required <= set(rows[0].keys())

    def test_groups_columns(self, ops_dir: Path) -> None:
        """groups.csv must have all required columns."""
        required = {
            "group_id",
            "program_id",
            "school_id",
            "tutor_id",
            "group_size",
            "subject",
            "grade_level",
            "sessions_per_week",
            "session_duration_minutes",
            "start_date",
        }
        rows = _read_csv(ops_dir / "groups.csv")
        assert required <= set(rows[0].keys())

    def test_sessions_columns(self, ops_dir: Path) -> None:
        """sessions.csv must have all required columns."""
        required = {
            "session_id",
            "group_id",
            "tutor_id",
            "scheduled_date",
            "scheduled_start_time",
            "duration_minutes",
            "status",
            "actual_duration_minutes",
            "tutor_on_time",
            "platform",
        }
        rows = _read_csv(ops_dir / "sessions.csv")
        assert required <= set(rows[0].keys())

    def test_attendance_columns(self, ops_dir: Path) -> None:
        """attendance.csv must have all required columns."""
        required = {
            "attendance_id",
            "session_id",
            "student_id",
            "attended",
            "minutes_attended",
            "absence_reason",
            "recorded_at",
        }
        rows = _read_csv(ops_dir / "attendance.csv")
        assert required <= set(rows[0].keys())

    def test_surveys_columns(self, ops_dir: Path) -> None:
        """surveys.csv must have all required columns."""
        required = {
            "survey_id",
            "program_id",
            "survey_name",
            "survey_type",
            "administered_date",
            "total_invited",
            "total_responded",
            "response_rate",
        }
        rows = _read_csv(ops_dir / "surveys.csv")
        assert required <= set(rows[0].keys())

    def test_survey_responses_columns(self, ops_dir: Path) -> None:
        """survey_responses.csv must have all required columns."""
        required = {
            "response_id",
            "survey_id",
            "student_id",
            "question_code",
            "question_text",
            "response_value",
            "response_type",
            "responded_at",
        }
        rows = _read_csv(ops_dir / "survey_responses.csv")
        assert required <= set(rows[0].keys())

    def test_program_context_keys(self, ops_dir: Path) -> None:
        """program_context.json must have all required top-level keys."""
        required = {
            "program_id",
            "narrative_description",
            "stated_goals",
            "primary_contact",
            "data_collection_notes",
            "known_data_gaps",
            "implementation_challenges",
            "reporting_period",
            "seed",
        }
        data = _read_json(ops_dir / "program_context.json")
        assert required <= set(data.keys())
        assert isinstance(data["stated_goals"], list)
        assert data["seed"] == SEED

    def test_manifest_present_and_valid(self, ops_dir: Path) -> None:
        """manifest.json must be present and contain all pack files."""
        manifest = _read_json(ops_dir / "manifest.json")
        assert manifest["pack_id"] == "pack_operations"
        assert manifest["seed"] == SEED
        assert isinstance(manifest["files"], list)
        filenames = {f["filename"] for f in manifest["files"]}
        assert "programs.csv" in filenames
        assert "manifest.json" not in filenames  # self-excluded

    def test_pk_uniqueness_students(self, ops_dir: Path) -> None:
        """student_id values must be unique."""
        rows = _read_csv(ops_dir / "students.csv")
        ids = [r["student_id"] for r in rows]
        assert len(ids) == len(set(ids)), "Duplicate student_id values found"

    def test_pk_uniqueness_sessions(self, ops_dir: Path) -> None:
        """session_id values must be unique."""
        rows = _read_csv(ops_dir / "sessions.csv")
        ids = [r["session_id"] for r in rows]
        assert len(ids) == len(set(ids)), "Duplicate session_id values found"

    def test_boolean_format_iep(self, ops_dir: Path) -> None:
        """IEP column must contain only 'true' or 'false' strings."""
        rows = _read_csv(ops_dir / "students.csv")
        values = {r["iep"] for r in rows}
        assert values <= {"true", "false"}

    def test_session_status_enum(self, ops_dir: Path) -> None:
        """Session status values must be within the allowed enum."""
        allowed = {
            "completed",
            "cancelled_tutor",
            "cancelled_student",
            "cancelled_school",
            "no_show",
        }
        rows = _read_csv(ops_dir / "sessions.csv")
        statuses = {r["status"] for r in rows}
        assert statuses <= allowed

    def test_student_count(self, ops_dir: Path) -> None:
        """Total enrolled students must match spec.STUDENTS_PER_SCHOOL sum."""
        rows = _read_csv(ops_dir / "students.csv")
        assert len(rows) == sum(spec.STUDENTS_PER_SCHOOL)


class TestSchemaConformanceOutcomes:
    """Outcomes pack additional table contracts from fixture_schema.md §2."""

    def test_monthly_attendance_columns(self, outcomes_dir: Path) -> None:
        """monthly_attendance_summary.csv must have all required columns."""
        required = {
            "summary_id",
            "program_id",
            "school_id",
            "month_label",
            "month_start",
            "month_end",
            "sessions_scheduled",
            "sessions_completed",
            "sessions_cancelled",
            "attendance_rate",
            "avg_dosage_minutes",
            "active_students",
            "mom_attendance_delta",
            "mom_dosage_delta",
        }
        rows = _read_csv(outcomes_dir / "monthly_attendance_summary.csv")
        assert required <= set(rows[0].keys())

    def test_monthly_satisfaction_columns(self, outcomes_dir: Path) -> None:
        """monthly_satisfaction_summary.csv must have all required columns."""
        required = {
            "summary_id",
            "program_id",
            "school_id",
            "survey_type",
            "month_label",
            "responses_count",
            "avg_score",
            "response_rate",
            "mom_score_delta",
        }
        rows = _read_csv(outcomes_dir / "monthly_satisfaction_summary.csv")
        assert required <= set(rows[0].keys())

    def test_monthly_satisfaction_three_months(self, outcomes_dir: Path) -> None:
        """monthly_satisfaction_summary.csv must contain rows for all three months."""
        rows = _read_csv(outcomes_dir / "monthly_satisfaction_summary.csv")
        months = {r["month_label"] for r in rows}
        assert months == set(spec.MONTHS), (
            f"Expected satisfaction rows for months {spec.MONTHS}, got {sorted(months)}"
        )

    def test_monthly_satisfaction_pk_unique(self, outcomes_dir: Path) -> None:
        """monthly_satisfaction_summary summary_id must be unique."""
        rows = _read_csv(outcomes_dir / "monthly_satisfaction_summary.csv")
        ids = [r["summary_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_ground_truth_present(self, outcomes_dir: Path) -> None:
        """ground_truth.json must exist in the outcomes pack."""
        assert (outcomes_dir / "ground_truth.json").exists(), "Missing ground_truth.json"

    def test_ground_truth_keys(self, outcomes_dir: Path) -> None:
        """ground_truth.json must contain all required top-level keys."""
        required = {
            "pack_id",
            "seed",
            "monthly_attendance_rate",
            "program_wide_attendance_rate",
            "monthly_satisfaction",
            "satisfaction_dip_realized",
            "monthly_cancellation_rate",
            "iep_attendance_rate",
            "non_iep_attendance_rate",
            "iep_attendance_gap_pp",
            "suppressed_low_n_subgroups",
        }
        data = _read_json(outcomes_dir / "ground_truth.json")
        assert required <= set(data.keys())
        assert data["pack_id"] == "pack_outcomes"
        assert data["seed"] == SEED

    def test_ground_truth_cancellation_rate_keys(self, outcomes_dir: Path) -> None:
        """ground_truth.json monthly_cancellation_rate must have entries for all three months."""
        data = _read_json(outcomes_dir / "ground_truth.json")
        assert set(data["monthly_cancellation_rate"].keys()) == set(spec.MONTHS)

    def test_monthly_attendance_pk_unique(self, outcomes_dir: Path) -> None:
        """monthly_attendance_summary summary_id must be unique."""
        rows = _read_csv(outcomes_dir / "monthly_attendance_summary.csv")
        ids = [r["summary_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_operations_tables_present_in_outcomes(self, outcomes_dir: Path) -> None:
        """Outcomes pack must include all Operations table files."""
        for fname in [
            "programs.csv",
            "schools.csv",
            "students.csv",
            "tutors.csv",
            "groups.csv",
            "sessions.csv",
            "attendance.csv",
            "surveys.csv",
            "survey_responses.csv",
            "program_context.json",
        ]:
            assert (outcomes_dir / fname).exists(), f"Missing {fname} in outcomes pack"


class TestGroundTruthOperations:
    """Operations pack ground_truth.json contract."""

    def test_ground_truth_present(self, ops_dir: Path) -> None:
        """ground_truth.json must exist in the operations pack."""
        assert (ops_dir / "ground_truth.json").exists(), "Missing ground_truth.json in operations"

    def test_ground_truth_keys(self, ops_dir: Path) -> None:
        """ground_truth.json must contain all required top-level keys."""
        required = {
            "pack_id",
            "seed",
            "monthly_attendance_rate",
            "program_wide_attendance_rate",
            "monthly_cancellation_rate",
            "iep_attendance_rate",
            "non_iep_attendance_rate",
            "iep_attendance_gap_pp",
            "suppressed_low_n_subgroups",
        }
        data = _read_json(ops_dir / "ground_truth.json")
        assert required <= set(data.keys())
        assert data["pack_id"] == "pack_operations"
        assert data["seed"] == SEED

    def test_ground_truth_monthly_attendance_all_months(self, ops_dir: Path) -> None:
        """monthly_attendance_rate must have entries for all three months."""
        data = _read_json(ops_dir / "ground_truth.json")
        assert set(data["monthly_attendance_rate"].keys()) == set(spec.MONTHS)

    def test_ground_truth_monthly_cancellation_all_months(self, ops_dir: Path) -> None:
        """monthly_cancellation_rate must have entries for all three months."""
        data = _read_json(ops_dir / "ground_truth.json")
        assert set(data["monthly_cancellation_rate"].keys()) == set(spec.MONTHS)

    def test_ground_truth_october_cancellation_elevated(self, ops_dir: Path) -> None:
        """October cancellation rate must be higher than September and November."""
        data = _read_json(ops_dir / "ground_truth.json")
        canc = data["monthly_cancellation_rate"]
        months = sorted(spec.MONTHS)
        assert canc[months[1]] > canc[months[0]], (
            f"October cancellation ({canc[months[1]]}) not > September ({canc[months[0]]})"
        )
        assert canc[months[1]] > canc[months[2]], (
            f"October cancellation ({canc[months[1]]}) not > November ({canc[months[2]]})"
        )

    def test_ground_truth_iep_gap_positive(self, ops_dir: Path) -> None:
        """IEP attendance gap must be positive (non-IEP attends more)."""
        data = _read_json(ops_dir / "ground_truth.json")
        assert data["iep_attendance_gap_pp"] is not None
        assert data["iep_attendance_gap_pp"] > 0, (
            f"IEP gap should be positive, got {data['iep_attendance_gap_pp']}"
        )

    def test_ground_truth_suppressed_subgroups_present(self, ops_dir: Path) -> None:
        """suppressed_low_n_subgroups must include AIAN with n < suppression threshold."""
        data = _read_json(ops_dir / "ground_truth.json")
        aian = [s for s in data["suppressed_low_n_subgroups"] if s["subgroup_value"] == "AIAN"]
        assert aian, "AIAN not found in suppressed_low_n_subgroups"
        assert aian[0]["n"] == spec.AIAN_TARGET_N

    def test_ground_truth_in_manifest(self, ops_dir: Path) -> None:
        """manifest.json must include an entry for ground_truth.json."""
        manifest = _read_json(ops_dir / "manifest.json")
        filenames = {f["filename"] for f in manifest["files"]}
        assert "ground_truth.json" in filenames, "ground_truth.json not listed in manifest"


class TestGroundTruthEquity:
    """Equity & Research pack ground_truth.json contract."""

    def test_ground_truth_present(self, equity_dir: Path) -> None:
        """ground_truth.json must exist in the equity pack."""
        assert (equity_dir / "ground_truth.json").exists(), (
            "Missing ground_truth.json in equity pack"
        )

    def test_ground_truth_keys(self, equity_dir: Path) -> None:
        """ground_truth.json must contain all required top-level keys."""
        required = {
            "pack_id",
            "seed",
            "monthly_attendance_rate",
            "program_wide_attendance_rate",
            "iep_attendance_rate",
            "non_iep_attendance_rate",
            "iep_attendance_gap_pp",
            "suppressed_low_n_subgroups",
            "iep_proficiency_gap_pp_spring",
            "subgroup_outcome_disparities",
        }
        data = _read_json(equity_dir / "ground_truth.json")
        assert required <= set(data.keys())
        assert data["pack_id"] == "pack_equity_research"
        assert data["seed"] == SEED

    def test_ground_truth_iep_gap_positive(self, equity_dir: Path) -> None:
        """IEP attendance gap must be positive."""
        data = _read_json(equity_dir / "ground_truth.json")
        assert data["iep_attendance_gap_pp"] is not None
        assert data["iep_attendance_gap_pp"] > 0

    def test_ground_truth_proficiency_gap_positive(self, equity_dir: Path) -> None:
        """IEP proficiency gap (spring) must be positive (non-IEP outperforms IEP)."""
        data = _read_json(equity_dir / "ground_truth.json")
        assert data["iep_proficiency_gap_pp_spring"] is not None
        assert data["iep_proficiency_gap_pp_spring"] > 0, (
            f"Expected positive proficiency gap, got {data['iep_proficiency_gap_pp_spring']}"
        )

    def test_ground_truth_suppressed_subgroups_aian(self, equity_dir: Path) -> None:
        """suppressed_low_n_subgroups must include AIAN."""
        data = _read_json(equity_dir / "ground_truth.json")
        aian = [s for s in data["suppressed_low_n_subgroups"] if s["subgroup_value"] == "AIAN"]
        assert aian, "AIAN not found in suppressed_low_n_subgroups"
        assert aian[0]["n"] == spec.AIAN_TARGET_N

    def test_ground_truth_subgroup_outcome_disparities_iep(self, equity_dir: Path) -> None:
        """subgroup_outcome_disparities must contain IEP true/false spring rows."""
        data = _read_json(equity_dir / "ground_truth.json")
        disparities = data["subgroup_outcome_disparities"]
        values = {d["subgroup_value"] for d in disparities}
        assert "true" in values, "IEP=true not in subgroup_outcome_disparities"
        assert "false" in values, "IEP=false not in subgroup_outcome_disparities"

    def test_ground_truth_in_manifest(self, equity_dir: Path) -> None:
        """manifest.json must include an entry for ground_truth.json."""
        manifest = _read_json(equity_dir / "manifest.json")
        filenames = {f["filename"] for f in manifest["files"]}
        assert "ground_truth.json" in filenames, "ground_truth.json not listed in manifest"


class TestSchemaConformanceEquity:
    """Equity & Research pack contracts from fixture_schema.md §3."""

    def test_subgroup_attendance_columns(self, equity_dir: Path) -> None:
        """subgroup_attendance_summary.csv must have all required columns."""
        required = {
            "summary_id",
            "program_id",
            "school_id",
            "month_label",
            "subgroup_dimension",
            "subgroup_value",
            "n_students",
            "n_sessions_attended",
            "attendance_rate",
            "avg_dosage_minutes",
            "suppressed",
            "suppression_reason",
        }
        rows = _read_csv(equity_dir / "subgroup_attendance_summary.csv")
        assert required <= set(rows[0].keys())

    def test_subgroup_outcomes_columns(self, equity_dir: Path) -> None:
        """subgroup_outcomes_summary.csv must have all required columns."""
        required = {
            "summary_id",
            "program_id",
            "school_id",
            "assessment_period",
            "subgroup_dimension",
            "subgroup_value",
            "n_students",
            "avg_scale_score",
            "avg_score_gain",
            "benchmark_proficiency_rate",
            "suppressed",
            "suppression_reason",
        }
        rows = _read_csv(equity_dir / "subgroup_outcomes_summary.csv")
        assert required <= set(rows[0].keys())

    def test_research_refs_structure(self, equity_dir: Path) -> None:
        """research_refs.json must have program_id and at least 2 references."""
        data = _read_json(equity_dir / "research_refs.json")
        assert "program_id" in data
        assert "references" in data
        assert len(data["references"]) >= 2

    def test_research_refs_has_contradicts_claim(self, equity_dir: Path) -> None:
        """At least one reference must have a non-null contradicts_claim."""
        data = _read_json(equity_dir / "research_refs.json")
        contradicts = [r for r in data["references"] if r.get("contradicts_claim") is not None]
        assert contradicts, "No reference with a contradicts_claim found"

    def test_operations_tables_present_in_equity(self, equity_dir: Path) -> None:
        """Equity pack must include all Operations table files."""
        for fname in [
            "programs.csv",
            "schools.csv",
            "students.csv",
            "tutors.csv",
            "groups.csv",
            "sessions.csv",
            "attendance.csv",
            "surveys.csv",
            "survey_responses.csv",
            "program_context.json",
        ]:
            assert (equity_dir / fname).exists(), f"Missing {fname} in equity pack"


# ---------------------------------------------------------------------------
# 3. Signal presence tests
# ---------------------------------------------------------------------------


class TestSignals:
    """Intentional signals from spec.py are present in the generated data."""

    # Signal 1: MoM attendance trend
    def test_mom_attendance_trend_positive(self, outcomes_dir: Path) -> None:
        """Attendance rate in September must be lower than in November (MoM rise)."""
        rows = _read_csv(outcomes_dir / "monthly_attendance_summary.csv")
        # Aggregate across schools by month
        by_month: dict[str, list[float]] = {}
        for r in rows:
            ml = r["month_label"]
            val = r["attendance_rate"]
            if val != "":
                by_month.setdefault(ml, []).append(float(val))

        months = sorted(by_month.keys())
        assert len(months) >= 2, "Need at least 2 months to check MoM trend"
        avg_first = sum(by_month[months[0]]) / len(by_month[months[0]])
        avg_last = sum(by_month[months[-1]]) / len(by_month[months[-1]])
        assert avg_last > avg_first, (
            f"Expected attendance to rise from {months[0]} ({avg_first:.3f}) "
            f"to {months[-1]} ({avg_last:.3f})"
        )

    def test_mom_attendance_delta_present(self, outcomes_dir: Path) -> None:
        """Non-first months must have a non-empty mom_attendance_delta."""
        rows = _read_csv(outcomes_dir / "monthly_attendance_summary.csv")
        months = sorted({r["month_label"] for r in rows})
        non_first = [r for r in rows if r["month_label"] != months[0]]
        assert non_first, "No non-first month rows found"
        with_delta = [r for r in non_first if r["mom_attendance_delta"] != ""]
        assert with_delta, "No mom_attendance_delta values found in non-first months"

    # Signal 4: Monthly satisfaction dip and recovery
    def test_satisfaction_dip_present(self, outcomes_dir: Path) -> None:
        """October satisfaction must be below both September and November (dip signal)."""
        rows = _read_csv(outcomes_dir / "monthly_satisfaction_summary.csv")
        # Focus on student_satisfaction as the primary signal
        student_rows = [r for r in rows if r["survey_type"] == "student_satisfaction"]
        assert len(student_rows) == 3, (
            f"Expected 3 student_satisfaction rows (one per month), got {len(student_rows)}"
        )
        by_month = {r["month_label"]: float(r["avg_score"]) for r in student_rows}
        months = sorted(spec.MONTHS)
        sep_score = by_month[months[0]]  # 2025-09
        oct_score = by_month[months[1]]  # 2025-10 (dip month)
        nov_score = by_month[months[2]]  # 2025-11
        assert oct_score < sep_score, (
            f"Expected Oct ({oct_score}) < Sep ({sep_score}) for satisfaction dip"
        )
        assert oct_score < nov_score, (
            f"Expected Oct ({oct_score}) < Nov ({nov_score}) for satisfaction recovery"
        )

    def test_satisfaction_dip_magnitude(self, outcomes_dir: Path) -> None:
        """October satisfaction dip must be at least half of SATISFACTION_DIP_MAGNITUDE."""
        rows = _read_csv(outcomes_dir / "monthly_satisfaction_summary.csv")
        student_rows = [r for r in rows if r["survey_type"] == "student_satisfaction"]
        by_month = {r["month_label"]: float(r["avg_score"]) for r in student_rows}
        months = sorted(spec.MONTHS)
        sep_score = by_month[months[0]]
        oct_score = by_month[months[1]]
        dip = sep_score - oct_score
        min_dip = spec.SATISFACTION_DIP_MAGNITUDE * 0.5
        assert dip >= min_dip, (
            f"Satisfaction dip ({dip:.4f}) is below minimum expected ({min_dip:.4f})"
        )

    def test_ground_truth_dip_confirmed(self, outcomes_dir: Path) -> None:
        """ground_truth.json must confirm the satisfaction dip."""
        data = _read_json(outcomes_dir / "ground_truth.json")
        assert data["satisfaction_dip_realized"]["dip_confirmed"] is True, (
            "ground_truth.json: satisfaction_dip_realized.dip_confirmed is not True"
        )

    # Signal 2: Low-N AIAN subgroup suppressed
    def test_aian_subgroup_suppressed(self, equity_dir: Path) -> None:
        """AIAN race_ethnicity subgroup must be suppressed in at least one month."""
        rows = _read_csv(equity_dir / "subgroup_attendance_summary.csv")
        aian_rows = [
            r
            for r in rows
            if r["subgroup_dimension"] == "race_ethnicity" and r["subgroup_value"] == "AIAN"
        ]
        assert aian_rows, "No AIAN subgroup rows found"
        suppressed = [r for r in aian_rows if r["suppressed"] == "true"]
        assert suppressed, "AIAN subgroup is not suppressed"
        # All suppressed rows must have null attendance_rate
        for r in suppressed:
            assert r["attendance_rate"] == "", (
                f"AIAN suppressed row has non-null attendance_rate: {r['attendance_rate']}"
            )

    def test_aian_n_below_threshold(self, equity_dir: Path) -> None:
        """AIAN n_students must be below suppression threshold in all months."""
        rows = _read_csv(equity_dir / "subgroup_attendance_summary.csv")
        aian_rows = [
            r
            for r in rows
            if r["subgroup_dimension"] == "race_ethnicity" and r["subgroup_value"] == "AIAN"
        ]
        for r in aian_rows:
            assert int(r["n_students"]) < spec.SUPPRESSION_THRESHOLD, (
                f"AIAN n_students={r['n_students']} >= suppression threshold"
            )

    def test_aian_student_count_in_students_csv(self, equity_dir: Path) -> None:
        """students.csv must have exactly AIAN_TARGET_N AIAN students."""
        rows = _read_csv(equity_dir / "students.csv")
        aian_count = sum(1 for r in rows if r["race_ethnicity"] == "AIAN")
        assert aian_count == spec.AIAN_TARGET_N, (
            f"Expected {spec.AIAN_TARGET_N} AIAN students, got {aian_count}"
        )

    # Signal 3: IEP attendance disparity
    def test_iep_attendance_disparity(self, equity_dir: Path) -> None:
        """IEP=true subgroup must have lower attendance rate than IEP=false."""
        rows = _read_csv(equity_dir / "subgroup_attendance_summary.csv")
        iep_rows = [r for r in rows if r["subgroup_dimension"] == "iep"]
        true_rates = [
            float(r["attendance_rate"])
            for r in iep_rows
            if r["subgroup_value"] == "true" and r["attendance_rate"] != ""
        ]
        false_rates = [
            float(r["attendance_rate"])
            for r in iep_rows
            if r["subgroup_value"] == "false" and r["attendance_rate"] != ""
        ]
        assert true_rates, "No IEP=true attendance rows with data"
        assert false_rates, "No IEP=false attendance rows with data"
        avg_iep = sum(true_rates) / len(true_rates)
        avg_non_iep = sum(false_rates) / len(false_rates)
        assert avg_iep < avg_non_iep, (
            f"Expected IEP attendance ({avg_iep:.3f}) < non-IEP ({avg_non_iep:.3f})"
        )

    def test_iep_disparity_magnitude(self, equity_dir: Path) -> None:
        """IEP attendance gap must be at least half the specified penalty."""
        rows = _read_csv(equity_dir / "subgroup_attendance_summary.csv")
        iep_rows = [r for r in rows if r["subgroup_dimension"] == "iep"]
        true_rates = [
            float(r["attendance_rate"])
            for r in iep_rows
            if r["subgroup_value"] == "true" and r["attendance_rate"] != ""
        ]
        false_rates = [
            float(r["attendance_rate"])
            for r in iep_rows
            if r["subgroup_value"] == "false" and r["attendance_rate"] != ""
        ]
        avg_iep = sum(true_rates) / len(true_rates)
        avg_non_iep = sum(false_rates) / len(false_rates)
        gap = avg_non_iep - avg_iep
        # Gap should be at least half the intended penalty (allows for noise)
        min_gap = spec.IEP_ATTENDANCE_PENALTY * 0.5
        assert gap >= min_gap, f"IEP gap ({gap:.3f}) is below minimum expected ({min_gap:.3f})"

    # Additional: suppression consistency
    def test_suppressed_rows_have_null_metrics(self, equity_dir: Path) -> None:
        """All suppressed=true rows in subgroup_attendance_summary must have empty metrics."""
        rows = _read_csv(equity_dir / "subgroup_attendance_summary.csv")
        for r in rows:
            if r["suppressed"] == "true":
                assert r["attendance_rate"] == "", (
                    f"Suppressed row has non-null attendance_rate: {r}"
                )
                assert r["avg_dosage_minutes"] == "", (
                    f"Suppressed row has non-null avg_dosage_minutes: {r}"
                )

    def test_suppressed_rows_outcomes(self, equity_dir: Path) -> None:
        """All suppressed=true rows in subgroup_outcomes_summary must have empty metrics."""
        rows = _read_csv(equity_dir / "subgroup_outcomes_summary.csv")
        for r in rows:
            if r["suppressed"] == "true":
                assert r["avg_scale_score"] == "", (
                    f"Suppressed row has non-null avg_scale_score: {r}"
                )

    # CLI invocation test
    def test_cli_all_packs(self, tmp_path: Path) -> None:
        """CLI with no --pack flag must generate all three pack directories."""
        main(["--seed", "42", "--out", str(tmp_path)])
        assert (tmp_path / "pack_operations").is_dir()
        assert (tmp_path / "pack_outcomes").is_dir()
        assert (tmp_path / "pack_equity_research").is_dir()

    def test_cli_single_pack(self, tmp_path: Path) -> None:
        """CLI with --pack operations must generate only pack_operations."""
        main(["--seed", "42", "--out", str(tmp_path), "--pack", "operations"])
        assert (tmp_path / "pack_operations").is_dir()
        assert not (tmp_path / "pack_outcomes").exists()
        assert not (tmp_path / "pack_equity_research").exists()
