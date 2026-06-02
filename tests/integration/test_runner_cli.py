"""Integration tests for the GRADE runner CLI (C5).

Tests run end-to-end against the stub adapter and a fixture task pack.
No network access or API keys are required.

Coverage:
- CLI entry point (argparse): --pack, --adapter, --runs, --out flags.
- Stub adapter × 1 task × 1 run → valid output + valid result JSON.
- Stub adapter × full operations pack × 1 run → valid result JSON.
- --dry-run flag exits without writing files.
- Unknown adapter name → non-zero exit.
- Unknown pack name → non-zero exit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.schemas import validate_output, validate_result
from runner.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ops_pack_path() -> Path:
    """Return the absolute path to the operations task pack."""
    return Path(__file__).parent.parent.parent / "benchmark" / "tasks" / "operations_pack.jsonl"


# ---------------------------------------------------------------------------
# Smoke test: 1 task × 1 run (fast, always CI-safe)
# ---------------------------------------------------------------------------


class TestStubAdapterSmokeRun:
    """Smoke tests using stub adapter × single task × 1 run."""

    def test_single_task_single_run_produces_valid_result(self, tmp_path: Path) -> None:
        """Running stub adapter on one task produces a result_schema-valid JSON."""
        # Create a minimal single-task pack.
        task = {
            "task_id": "T1-OPS-001",
            "track": 1,
            "title": "Count enrolled students",
            "user_prompt": ("Using students.csv, how many students are enrolled in the program?"),
            "task_type": "retrieval",
            "allowed_inputs": ["students.csv"],
            "gold_facts": [
                {
                    "fact_id": "F1",
                    "claim": "The program enrolls 135 students in total.",
                    "source_files": ["students.csv"],
                    "numeric_value": 135,
                    "tolerance": 0,
                }
            ],
            "gold_insights": ["The program has a large cohort."],
            "required_limitations": ["Count reflects snapshot date."],
            "forbidden_claims": ["All students improved."],
            "rubric": {
                "grounding_accuracy": {"weight": 0.35},
                "insight_quality": {"weight": 0.20},
                "evidence_linkage": {"weight": 0.15},
                "calibration_limitation_handling": {"weight": 0.15},
                "consistency": {"weight": 0.10},
                "structure_usability": {"weight": 0.05},
            },
        }
        pack_file = tmp_path / "single_task.jsonl"
        pack_file.write_text(json.dumps(task) + "\n", encoding="utf-8")
        out_dir = tmp_path / "out"

        exit_code = main(
            [
                "--pack",
                str(pack_file),
                "--adapter",
                "stub",
                "--runs",
                "1",
                "--out",
                str(out_dir),
            ]
        )

        assert exit_code == 0
        assert (out_dir / "result.json").exists()
        assert (out_dir / "raw_outputs.jsonl").exists()

        result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        validate_result(result)

        assert result["task_count"] == 1
        assert len(result["per_task_scores"]) == 1

    def test_raw_outputs_validate_against_output_schema(self, tmp_path: Path) -> None:
        """Each line in raw_outputs.jsonl must be a valid output_schema dict."""
        task = {
            "task_id": "T1-OPS-001",
            "track": 1,
            "title": "Count enrolled students",
            "user_prompt": ("Using students.csv, how many students are enrolled in the program?"),
            "task_type": "retrieval",
            "allowed_inputs": ["students.csv"],
            "gold_facts": [],
            "gold_insights": [],
            "required_limitations": [],
            "forbidden_claims": [],
            "rubric": {
                "grounding_accuracy": {"weight": 0.35},
                "insight_quality": {"weight": 0.20},
                "evidence_linkage": {"weight": 0.15},
                "calibration_limitation_handling": {"weight": 0.15},
                "consistency": {"weight": 0.10},
                "structure_usability": {"weight": 0.05},
            },
        }
        pack_file = tmp_path / "pack.jsonl"
        pack_file.write_text(json.dumps(task) + "\n", encoding="utf-8")
        out_dir = tmp_path / "out"

        exit_code = main(
            [
                "--pack",
                str(pack_file),
                "--adapter",
                "stub",
                "--runs",
                "2",
                "--out",
                str(out_dir),
            ]
        )

        assert exit_code == 0
        raw_lines = (out_dir / "raw_outputs.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(raw_lines) == 2  # 1 task × 2 runs

        for line in raw_lines:
            output = json.loads(line)
            validate_output(output)  # should not raise

    def test_result_has_per_track_and_per_pack_scores(self, tmp_path: Path) -> None:
        """result.json must contain per_track_scores and per_pack_scores."""
        task = {
            "task_id": "T1-OPS-001",
            "track": 1,
            "title": "Count enrolled students",
            "user_prompt": "Count students from students.csv",
            "task_type": "retrieval",
            "allowed_inputs": ["students.csv"],
            "gold_facts": [],
            "gold_insights": [],
            "required_limitations": [],
            "forbidden_claims": [],
            "rubric": {
                "grounding_accuracy": {"weight": 0.35},
                "insight_quality": {"weight": 0.20},
                "evidence_linkage": {"weight": 0.15},
                "calibration_limitation_handling": {"weight": 0.15},
                "consistency": {"weight": 0.10},
                "structure_usability": {"weight": 0.05},
            },
        }
        pack_file = tmp_path / "pack.jsonl"
        pack_file.write_text(json.dumps(task) + "\n", encoding="utf-8")
        out_dir = tmp_path / "out"

        main(
            [
                "--pack",
                str(pack_file),
                "--adapter",
                "stub",
                "--runs",
                "1",
                "--out",
                str(out_dir),
                "--model-id",
                "stub/echo-v1",
            ]
        )

        result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        assert "per_track_scores" in result
        assert "per_pack_scores" in result
        # Track 1 should be present (task.track=1).
        assert "1" in result["per_track_scores"]
        # No pack_id in this pack file → per_pack_scores may be empty.


# ---------------------------------------------------------------------------
# Full operations pack smoke test
# ---------------------------------------------------------------------------


class TestFullPackRun:
    """Smoke test against the real operations pack JSONL (stub adapter)."""

    def test_operations_pack_produces_valid_result(self, tmp_path: Path) -> None:
        """The full operations pack with stub adapter and 1 run must produce valid result."""
        ops_path = _ops_pack_path()
        if not ops_path.exists():
            pytest.skip("operations_pack.jsonl not found")

        out_dir = tmp_path / "out"
        exit_code = main(
            [
                "--pack",
                str(ops_path),
                "--adapter",
                "stub",
                "--runs",
                "1",
                "--out",
                str(out_dir),
                "--model-id",
                "stub/echo-v1",
            ]
        )

        assert exit_code == 0
        result_path = out_dir / "result.json"
        assert result_path.exists()

        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result(result)

        # All tasks in the operations pack should be scored.
        from runner.dispatcher import load_pack

        expected_task_count = len(load_pack(ops_path))
        assert result["task_count"] == expected_task_count


# ---------------------------------------------------------------------------
# CLI argument tests
# ---------------------------------------------------------------------------


class TestCLIArguments:
    """Tests for CLI argument handling and error cases."""

    def test_dry_run_exits_zero_without_writing_files(self, tmp_path: Path) -> None:
        """--dry-run must exit 0 without creating output files."""
        out_dir = tmp_path / "out"
        exit_code = main(
            [
                "--pack",
                "operations",
                "--adapter",
                "stub",
                "--runs",
                "5",
                "--out",
                str(out_dir),
                "--dry-run",
            ]
        )
        assert exit_code == 0
        assert not (out_dir / "result.json").exists()

    def test_unknown_adapter_exits_nonzero(self, tmp_path: Path) -> None:
        """Passing an unknown --adapter name must exit with code 1."""
        out_dir = tmp_path / "out"
        exit_code = main(
            [
                "--pack",
                "operations",
                "--adapter",
                "nonexistent_adapter_xyz",
                "--runs",
                "1",
                "--out",
                str(out_dir),
            ]
        )
        assert exit_code == 1

    def test_unknown_pack_exits_nonzero(self, tmp_path: Path) -> None:
        """Passing a non-existent pack file path must exit with code 1."""
        out_dir = tmp_path / "out"
        exit_code = main(
            [
                "--pack",
                "/no/such/pack.jsonl",
                "--adapter",
                "stub",
                "--runs",
                "1",
                "--out",
                str(out_dir),
            ]
        )
        assert exit_code == 1

    def test_named_pack_operations_resolves(self, tmp_path: Path) -> None:
        """Short name 'operations' should resolve to the real operations pack."""
        out_dir = tmp_path / "out"
        exit_code = main(
            [
                "--pack",
                "operations",
                "--adapter",
                "stub",
                "--runs",
                "1",
                "--out",
                str(out_dir),
                "--model-id",
                "stub/echo-v1",
            ]
        )
        assert exit_code == 0
        result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        # Operations pack tasks should have pack_id=pack_operations in per_pack_scores.
        assert "pack_operations" in result["per_pack_scores"]

    def test_custom_grade_version(self, tmp_path: Path) -> None:
        """--grade-version override must appear in the result JSON."""
        out_dir = tmp_path / "out"
        main(
            [
                "--pack",
                "operations",
                "--adapter",
                "stub",
                "--runs",
                "1",
                "--out",
                str(out_dir),
                "--grade-version",
                "1.2.3",
                "--model-id",
                "stub/echo-v1",
            ]
        )
        result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        assert result["grade_version"] == "1.2.3"
