"""Unit tests for runner.dispatcher — load_pack, run_task, scoring helpers.

All tests use the StubAdapter (no network, no API keys required).

Coverage:
- :func:`~runner.dispatcher.load_pack`: valid JSONL, empty file, bad JSON.
- :func:`~runner.dispatcher.run_task`: single run, multi-run, output validation,
  C1 score wiring, consistency scoring, result fields.
- :class:`~runner.dispatcher._NullJudge`: always returns 0.5.
- :func:`~runner.dispatcher._measure_consistency`: identical runs, diverse runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from runner.adapters.stub_adapter import StubAdapter
from runner.dispatcher import (
    TaskRunResult,
    _measure_consistency,
    load_pack,
    run_task,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VALID_TASK: dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# load_pack tests
# ---------------------------------------------------------------------------


class TestLoadPack:
    """Tests for :func:`load_pack`."""

    def test_load_single_task(self, tmp_path: Path) -> None:
        """A JSONL file with one task line should return a list with one dict."""
        f = tmp_path / "pack.jsonl"
        f.write_text(json.dumps(_VALID_TASK) + "\n", encoding="utf-8")
        tasks = load_pack(f)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "T1-OPS-001"

    def test_load_multiple_tasks(self, tmp_path: Path) -> None:
        """Multiple lines should return one dict per non-empty line."""
        task2 = dict(_VALID_TASK, task_id="T1-OPS-002")
        f = tmp_path / "pack.jsonl"
        lines = json.dumps(_VALID_TASK) + "\n" + json.dumps(task2) + "\n"
        f.write_text(lines, encoding="utf-8")
        tasks = load_pack(f)
        assert len(tasks) == 2

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        """Blank lines interspersed with task lines should be ignored."""
        f = tmp_path / "pack.jsonl"
        f.write_text("\n" + json.dumps(_VALID_TASK) + "\n\n", encoding="utf-8")
        tasks = load_pack(f)
        assert len(tasks) == 1

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        """An entirely empty file should return an empty list."""
        f = tmp_path / "pack.jsonl"
        f.write_text("", encoding="utf-8")
        tasks = load_pack(f)
        assert tasks == []

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        """A malformed JSON line should raise json.JSONDecodeError."""
        f = tmp_path / "pack.jsonl"
        f.write_text("{not valid json}\n", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_pack(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_pack(tmp_path / "no_such_file.jsonl")

    def test_loads_real_operations_pack(self) -> None:
        """The real operations pack JSONL should load without errors."""
        pack_path = (
            Path(__file__).parent.parent.parent / "benchmark" / "tasks" / "operations_pack.jsonl"
        )
        tasks = load_pack(pack_path)
        assert len(tasks) >= 5
        for task in tasks:
            assert "task_id" in task
            assert "rubric" in task


# ---------------------------------------------------------------------------
# run_task tests
# ---------------------------------------------------------------------------


class TestRunTask:
    """Tests for :func:`run_task` using the StubAdapter."""

    def test_single_run_returns_task_run_result(self) -> None:
        """run_task with runs=1 must return a TaskRunResult with 1 output."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=1, pack_id="pack_operations")
        assert isinstance(result, TaskRunResult)
        assert result.run_count == 1
        assert len(result.outputs) == 1

    def test_five_runs_returns_five_outputs(self) -> None:
        """run_task with runs=5 must return 5 outputs."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=5)
        assert result.run_count == 5
        assert len(result.outputs) == 5

    def test_task_id_matches(self) -> None:
        """TaskRunResult.task_id must match the input task."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=1)
        assert result.task_id == "T1-OPS-001"

    def test_track_matches(self) -> None:
        """TaskRunResult.track must match the input task's track."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=1)
        assert result.track == 1

    def test_pack_id_stored(self) -> None:
        """pack_id passed to run_task should be stored in the result."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=1, pack_id="pack_operations")
        assert result.pack_id == "pack_operations"

    def test_output_has_correct_run_indices(self) -> None:
        """Each output's run_index should be sequential starting from 0."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=3)
        indices = [o["run_index"] for o in result.outputs]
        assert indices == [0, 1, 2]

    def test_scores_dict_has_all_six_dimensions(self) -> None:
        """The scores dict must contain all six GRADE dimension keys."""
        from benchmark.rubrics.rubric_scoring import RUBRIC_DIMENSIONS

        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=1)
        assert set(result.scores.keys()) == set(RUBRIC_DIMENSIONS)

    def test_all_scores_in_unit_interval(self) -> None:
        """Every dimension score must be in [0, 1]."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=2)
        for dim, score in result.scores.items():
            assert 0.0 <= score <= 1.0, f"{dim}={score} out of [0,1]"

    def test_composite_in_unit_interval(self) -> None:
        """The composite score must be in [0, 1]."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=1)
        assert result.composite is not None
        assert 0.0 <= result.composite <= 1.0

    def test_outputs_validate_against_output_schema(self) -> None:
        """All returned outputs must pass output_schema validation (no raises)."""
        from benchmark.schemas import validate_output

        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=3)
        for output in result.outputs:
            validate_output(output)  # should not raise

    def test_runs_zero_raises_value_error(self) -> None:
        """Passing runs=0 must raise ValueError."""
        adapter = StubAdapter()
        with pytest.raises(ValueError, match="runs must be >= 1"):
            run_task(_VALID_TASK, adapter, runs=0)

    def test_null_judge_used_when_no_judge_provided(self) -> None:
        """Without a judge_client, C2 scoring should use the null judge (0.5)."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=1)
        # insight_quality from null judge = 0.5; grounding_accuracy from C1 differs.
        assert result.scores.get("insight_quality") == pytest.approx(0.5)

    def test_consistency_is_one_for_single_run(self) -> None:
        """Consistency must be 1.0 when there is only one run."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=1)
        assert result.scores.get("consistency") == pytest.approx(1.0)

    def test_consistency_nonzero_for_multiple_runs(self) -> None:
        """Consistency with identical stub outputs should be > 0."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=3)
        # Stub always emits the same first key_finding → consistency close to 1.0.
        assert result.scores.get("consistency", 0.0) > 0.0


# ---------------------------------------------------------------------------
# _measure_consistency tests
# ---------------------------------------------------------------------------


class TestMeasureConsistency:
    """Tests for :func:`_measure_consistency`."""

    def _make_output(self, first_finding: str) -> dict[str, Any]:
        """Create a minimal output dict with the given first key_finding.

        Args:
            first_finding: First key_finding string.

        Returns:
            A minimal output dict.
        """
        return {"key_findings": [first_finding]}

    def test_single_output_returns_one(self) -> None:
        """A single output is trivially consistent (returns 1.0)."""
        outputs = [self._make_output("hello world")]
        assert _measure_consistency(outputs) == pytest.approx(1.0)

    def test_identical_outputs_return_one(self) -> None:
        """Identical outputs should return exactly 1.0."""
        outputs = [self._make_output("the cat sat on the mat")] * 3
        assert _measure_consistency(outputs) == pytest.approx(1.0)

    def test_disjoint_outputs_return_zero(self) -> None:
        """Completely disjoint token sets should yield 0.0."""
        outputs = [self._make_output("aaa bbb"), self._make_output("ccc ddd")]
        assert _measure_consistency(outputs) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Partial overlap should return a score between 0 and 1."""
        outputs = [
            self._make_output("the cat sat"),
            self._make_output("the dog ran"),
        ]
        score = _measure_consistency(outputs)
        assert 0.0 < score < 1.0

    def test_empty_key_findings(self) -> None:
        """Outputs with empty key_findings (empty token sets) should return 1.0."""
        outputs: list[dict[str, Any]] = [{"key_findings": []}, {"key_findings": []}]
        # Both empty → union=0 → treated as 1.0 similarity.
        score = _measure_consistency(outputs)
        assert score == pytest.approx(1.0)
