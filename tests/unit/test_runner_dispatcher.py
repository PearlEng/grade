"""Unit tests for runner.dispatcher — load_pack, run_task, scoring helpers.

All tests use the StubAdapter (no network, no API keys required).

Coverage:
- :func:`~runner.dispatcher.load_pack`: valid JSONL, empty file, bad JSON.
- :func:`~runner.dispatcher.run_task`: single run, multi-run, output validation,
  C1 score wiring, C3 calibration wiring, C4 consistency wiring, result fields.
- :class:`~runner.dispatcher._NullJudge`: always returns 0.5.
- C3 integration: calibration_limitation_handling driven by validate_claims.
- C4 integration: consistency driven by score_consistency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from runner.adapters.stub_adapter import StubAdapter
from runner.dispatcher import (
    TaskRunResult,
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

# Task with no claim lists — C3 should return 1.0 (nothing to violate).
_TASK_NO_CLAIMS: dict[str, Any] = {
    "task_id": "T1-OPS-NOCLAIM",
    "track": 1,
    "title": "Minimal task",
    "user_prompt": "What is the total?",
    "task_type": "retrieval",
    "allowed_inputs": [],
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(
    task_id: str = "T1-OPS-001",
    key_findings: list[str] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Return a minimal schema-valid output dict."""
    return {
        "task_id": task_id,
        "model_id": "stub/echo-v1",
        "run_index": 0,
        "structured_metrics": {},
        "key_findings": key_findings or ["analysis complete"],
        "limitations": limitations or ["no real inference performed"],
        "evidence_citations": [],
        "runtime_metadata": {
            "adapter_version": "0.1.0",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "latency_ms": 1.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "model_temperature": None,
            "provider": "stub",
            "pack_id": None,
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
        """C4 consistency must be 1.0 when there is only one run."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=1)
        assert result.scores.get("consistency") == pytest.approx(1.0)

    def test_single_run_flags_trivial_consistency_and_excludes_it(self) -> None:
        """runs=1 must stamp consistency_trivial and drop the dimension from the composite (H-3).

        The composite must equal the renormalized weighted sum of the other
        five dimensions — a trivial consistency=1.0 must contribute nothing.
        """
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=1)
        assert "consistency_trivial" in result.scorer_flags

        rubric = _VALID_TASK["rubric"]
        included = [d for d in result.scores if d != "consistency"]
        weight_sum = sum(rubric[d]["weight"] for d in included)
        expected = sum(rubric[d]["weight"] * result.scores[d] for d in included) / weight_sum
        assert result.composite == pytest.approx(expected)

    def test_multi_run_composite_includes_consistency(self) -> None:
        """runs>=2 must keep consistency in the composite and not flag it."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=2)
        assert "consistency_trivial" not in result.scorer_flags

        rubric = _VALID_TASK["rubric"]
        expected = sum(rubric[d]["weight"] * result.scores[d] for d in result.scores)
        assert result.composite == pytest.approx(expected)

    def test_consistency_nonzero_for_multiple_runs(self) -> None:
        """Consistency with identical stub outputs should be > 0."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=3)
        # Stub always emits the same first key_finding → consistency close to 1.0.
        assert result.scores.get("consistency", 0.0) > 0.0

    def test_truncated_output_flagged(self) -> None:
        """A run whose finish_reason is 'length' must stamp the truncated_output flag (H-4)."""

        class _TruncatingAdapter:
            name = "stub"

            def __init__(self) -> None:
                self._inner = StubAdapter()

            def run(self, task: dict[str, Any], run_index: int = 0) -> dict[str, Any]:
                output = self._inner.run(task, run_index=run_index)
                output["runtime_metadata"]["finish_reason"] = "length"
                return output

        result = run_task(_VALID_TASK, _TruncatingAdapter(), runs=2)
        assert "truncated_output" in result.scorer_flags

    def test_non_truncated_output_not_flagged(self) -> None:
        """Normal stub runs must not carry the truncated_output flag."""
        result = run_task(_VALID_TASK, StubAdapter(), runs=1)
        assert "truncated_output" not in result.scorer_flags

    def test_c1_grounding_searches_limitations(self) -> None:
        """A gold-fact value stated only in limitations must be credited (M-1)."""
        from runner.dispatcher import _score_c1_grounding

        task = {
            "task_id": "T-LIM",
            "gold_facts": [
                {
                    "fact_id": "F1",
                    "claim": "The program enrolls 977 students.",
                    "source_files": ["students.csv"],
                    "numeric_value": 977,
                    "tolerance": 0,
                }
            ],
        }
        output = {
            "structured_metrics": {},
            "key_findings": ["Enrollment is healthy this quarter."],
            "limitations": ["Note that only 977 students are reflected in this snapshot."],
        }
        assert _score_c1_grounding(task, output) == pytest.approx(1.0)

    def test_c2_judge_not_called_for_non_c2_dimensions(self) -> None:
        """The judge must only be called for C2-owned dimensions (M-2)."""
        adapter = StubAdapter()
        mock_judge = MagicMock()
        mock_judge.judge.return_value = 0.5
        run_task(_VALID_TASK, adapter, runs=1, judge_client=mock_judge)

        called_dims = {call.kwargs["dimension"] for call in mock_judge.judge.call_args_list}
        assert "grounding_accuracy" not in called_dims
        assert "consistency" not in called_dims
        # C3 may route calibration through the judge as a Stage-2 paraphrase
        # fallback, but C2 itself must cover exactly the three owned dims.
        assert {"insight_quality", "evidence_linkage", "structure_usability"} <= called_dims


# ---------------------------------------------------------------------------
# C3 integration tests — calibration_limitation_handling driven by validate_claims
# ---------------------------------------------------------------------------


class TestC3CalibrationIntegration:
    """Assert that calibration_limitation_handling comes from C3 validate_claims."""

    def test_no_claims_yields_perfect_calibration(self) -> None:
        """A task with empty claim lists should score 1.0 calibration (C3 default)."""
        adapter = StubAdapter()
        result = run_task(_TASK_NO_CLAIMS, adapter, runs=1)
        assert result.scores["calibration_limitation_handling"] == pytest.approx(1.0)

    def test_forbidden_claim_present_lowers_calibration(self) -> None:
        """An output containing a forbidden claim should lower calibration below 1.0."""
        # The stub output's key_findings contain the task title; we craft a task
        # whose forbidden_claim matches the stub output text exactly so Stage 1
        # string-match detects it.
        task = dict(_VALID_TASK)
        task = {
            **_VALID_TASK,
            "task_id": "T1-OPS-FORBIDDEN",
            # StubAdapter key_findings = "[stub run 0] Count enrolled students: analysis complete."
            # We set a forbidden claim that is a substring of that text.
            "forbidden_claims": ["analysis complete"],
            "gold_insights": [],
            "required_limitations": [],
        }
        adapter = StubAdapter()
        result = run_task(task, adapter, runs=1)
        # C3 detects the forbidden claim → calibration < 1.0
        assert result.scores["calibration_limitation_handling"] < 1.0

    def test_required_limitation_missing_lowers_calibration(self) -> None:
        """An output missing a required limitation should lower calibration below 1.0."""
        task = {
            **_VALID_TASK,
            "task_id": "T1-OPS-LIMIT",
            "gold_insights": [],
            "forbidden_claims": [],
            # Stub limitations = "This is a stub response; no real model inference was performed."
            # We require something that is NOT in that text.
            "required_limitations": ["data may be outdated please verify independently"],
        }
        adapter = StubAdapter()
        result = run_task(task, adapter, runs=1)
        # C3 finds required limitation absent → score < 1.0
        assert result.scores["calibration_limitation_handling"] < 1.0

    def test_calibration_averaged_across_runs(self) -> None:
        """calibration_limitation_handling should be the average over all N runs."""
        # With a deterministic StubAdapter and no claims to violate, each run
        # produces 1.0 → average is 1.0.
        adapter = StubAdapter()
        result = run_task(_TASK_NO_CLAIMS, adapter, runs=3)
        assert result.scores["calibration_limitation_handling"] == pytest.approx(1.0)

    def test_c3_not_c2_drives_calibration(self) -> None:
        """Changing the judge_client score must not affect calibration when C3 is deterministic."""
        # A judge that always returns 0.0 should NOT lower calibration for a
        # task with no claim lists, because C3 deterministic path fires first
        # and returns 1.0 (no claims → perfect score).
        mock_judge = MagicMock()
        mock_judge.judge.return_value = 0.0

        adapter = StubAdapter()
        result = run_task(_TASK_NO_CLAIMS, adapter, runs=1, judge_client=mock_judge)
        # C3 returns 1.0 (no claims); judge_client is passed through but
        # deterministic path resolves all checks without calling judge when
        # there are no claim lists at all.
        assert result.scores["calibration_limitation_handling"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# C4 integration tests — consistency driven by score_consistency
# ---------------------------------------------------------------------------


class TestC4ConsistencyIntegration:
    """Assert that consistency comes from C4 score_consistency."""

    def test_identical_runs_yield_high_consistency(self) -> None:
        """Identical outputs across runs should yield consistency close to 1.0."""
        adapter = StubAdapter()
        # StubAdapter is deterministic for the same task → all runs identical.
        result = run_task(_VALID_TASK, adapter, runs=5)
        # Finding stability and ranking stability will both be 1.0 for identical
        # outputs; metric variance also 1.0 → overall consistency ≈ 1.0.
        assert result.scores["consistency"] > 0.9

    def test_consistency_in_unit_interval(self) -> None:
        """Consistency score must always be in [0, 1]."""
        adapter = StubAdapter()
        result = run_task(_VALID_TASK, adapter, runs=3)
        score = result.scores["consistency"]
        assert 0.0 <= score <= 1.0

    def test_measure_consistency_removed(self) -> None:
        """_measure_consistency must NOT be exported from dispatcher."""
        import runner.dispatcher as disp

        assert not hasattr(disp, "_measure_consistency"), (
            "_measure_consistency was removed in favour of C4 score_consistency "
            "but still appears in the module namespace"
        )
