"""Unit tests for the C2 rubric-based structured scorer.

All tests use a mock judge client — no live API calls are made in CI.

Coverage:
- :func:`~benchmark.rubrics.rubric_scoring.validate_rubric_weights`: valid and
  invalid weight-sum scenarios (including authored task rubrics from all five
  benchmark tracks).
- :func:`~benchmark.rubrics.rubric_scoring.score_rubric`: all-pass case,
  all-fail case, partial-pass case, composite computation, and correct weight
  application.
- Edge cases: dimension with guidance vs. without, uniform weights, non-uniform
  weights.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from benchmark.rubrics.rubric_scoring import (
    RUBRIC_DIMENSIONS,
    WEIGHT_SUM_TOLERANCE,
    score_rubric,
    validate_rubric_weights,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VALID_RUBRIC: dict[str, Any] = {
    "grounding_accuracy": {"weight": 0.35},
    "insight_quality": {"weight": 0.20},
    "evidence_linkage": {"weight": 0.15},
    "calibration_limitation_handling": {"weight": 0.15},
    "consistency": {"weight": 0.10},
    "structure_usability": {"weight": 0.05},
}

_VALID_TASK: dict[str, Any] = {
    "task_id": "T1-OPS-001",
    "track": 1,
    "title": "Count active students in the program",
    "user_prompt": "Using the provided fixture data, how many students are active?",
    "task_type": "retrieval",
    "allowed_inputs": ["students.csv"],
    "gold_facts": [
        {
            "fact_id": "F1",
            "claim": "There are 42 active students.",
            "source_files": ["students.csv"],
            "numeric_value": 42,
            "tolerance": 0,
        }
    ],
    "gold_insights": ["The program has a small cohort."],
    "required_limitations": ["Count is a snapshot."],
    "forbidden_claims": ["All students improved."],
    "rubric": _VALID_RUBRIC,
}

_VALID_OUTPUT: dict[str, Any] = {
    "task_id": "T1-OPS-001",
    "model_id": "test/model",
    "run_index": 0,
    "structured_metrics": {},
    "key_findings": ["There are 42 active students."],
    "limitations": ["Count reflects the snapshot period."],
    "evidence_citations": [],
    "runtime_metadata": {
        "adapter_version": "0.1.0",
        "timestamp_utc": "2026-06-01T12:00:00Z",
        "latency_ms": 500.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "model_temperature": 0.0,
        "provider": "test",
        "pack_id": "pack_operations",
    },
}


def _make_mock_judge(return_value: float = 1.0) -> MagicMock:
    """Return a mock that satisfies JudgeClientProtocol, always returning *return_value*.

    Args:
        return_value: The score to return for every judge call.

    Returns:
        A :class:`~unittest.mock.MagicMock` with a ``judge`` method.
    """
    mock = MagicMock()
    mock.judge.return_value = return_value
    return mock


def _make_mock_judge_sequence(scores: list[float]) -> MagicMock:
    """Return a mock whose ``judge`` method returns successive values from *scores*.

    Args:
        scores: List of floats returned in order, one per ``judge`` call.

    Returns:
        A :class:`~unittest.mock.MagicMock` with a ``judge`` side_effect.
    """
    mock = MagicMock()
    mock.judge.side_effect = scores
    return mock


# ---------------------------------------------------------------------------
# validate_rubric_weights tests
# ---------------------------------------------------------------------------


class TestValidateRubricWeights:
    """Tests for :func:`validate_rubric_weights`."""

    def test_valid_weights_pass(self) -> None:
        """Default project weights (0.35+0.20+0.15+0.15+0.10+0.05=1.0) must pass."""
        validate_rubric_weights(_VALID_RUBRIC)  # should not raise

    def test_uniform_weights_pass(self) -> None:
        """Six equal weights (each 1/6 ≈ 0.1667) should pass within tolerance."""
        w = 1.0 / 6.0
        rubric = {dim: {"weight": w} for dim in RUBRIC_DIMENSIONS}
        validate_rubric_weights(rubric)

    def test_weights_sum_slightly_off_fails(self) -> None:
        """Weights summing to 0.99 should raise ValueError."""
        rubric = copy.deepcopy(_VALID_RUBRIC)
        rubric["structure_usability"]["weight"] = 0.04  # 0.99 total
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_rubric_weights(rubric)

    def test_weights_sum_over_one_fails(self) -> None:
        """Weights summing to 1.06 should raise ValueError."""
        rubric = copy.deepcopy(_VALID_RUBRIC)
        rubric["grounding_accuracy"]["weight"] = 0.41  # 1.06 total
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_rubric_weights(rubric)

    def test_weights_sum_zero_fails(self) -> None:
        """All-zero weights should raise ValueError."""
        rubric = {dim: {"weight": 0.0} for dim in RUBRIC_DIMENSIONS}
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_rubric_weights(rubric)

    def test_missing_dimension_fails(self) -> None:
        """Rubric missing 'consistency' dimension should raise ValueError."""
        rubric = copy.deepcopy(_VALID_RUBRIC)
        del rubric["consistency"]
        with pytest.raises(ValueError, match="consistency"):
            validate_rubric_weights(rubric)

    def test_weights_within_tolerance_pass(self) -> None:
        """Weights summing to exactly 1.0 + tolerance/2 should pass."""
        rubric = copy.deepcopy(_VALID_RUBRIC)
        # Add a tiny epsilon well within tolerance.
        rubric["structure_usability"]["weight"] = 0.05 + WEIGHT_SUM_TOLERANCE / 2
        validate_rubric_weights(rubric)  # should not raise

    def test_weights_just_outside_tolerance_fails(self) -> None:
        """Weights summing to 1.0 + 2*tolerance should raise ValueError."""
        rubric = copy.deepcopy(_VALID_RUBRIC)
        rubric["structure_usability"]["weight"] = 0.05 + WEIGHT_SUM_TOLERANCE * 2
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_rubric_weights(rubric)


# ---------------------------------------------------------------------------
# score_rubric tests
# ---------------------------------------------------------------------------


class TestScoreRubric:
    """Tests for :func:`score_rubric`."""

    def test_all_pass_case(self) -> None:
        """All dimensions scoring 1.0 should yield composite == 1.0."""
        mock = _make_mock_judge(1.0)
        result = score_rubric(_VALID_TASK, _VALID_OUTPUT, mock)

        assert result["composite"] == pytest.approx(1.0)
        for dim in RUBRIC_DIMENSIONS:
            assert result["dimension_scores"][dim] == pytest.approx(1.0)

    def test_all_fail_case(self) -> None:
        """All dimensions scoring 0.0 should yield composite == 0.0."""
        mock = _make_mock_judge(0.0)
        result = score_rubric(_VALID_TASK, _VALID_OUTPUT, mock)

        assert result["composite"] == pytest.approx(0.0)
        for dim in RUBRIC_DIMENSIONS:
            assert result["dimension_scores"][dim] == pytest.approx(0.0)

    def test_partial_pass_case(self) -> None:
        """Partial scores should produce the correct weighted composite.

        Weights: 0.35, 0.20, 0.15, 0.15, 0.10, 0.05
        Scores:  1.0,  1.0,  0.0,  0.0,  1.0,  1.0
        Expected: 0.35 + 0.20 + 0.0 + 0.0 + 0.10 + 0.05 = 0.70
        """
        # grounding_accuracy=1, insight_quality=1, evidence_linkage=0,
        # calibration_limitation_handling=0, consistency=1, structure_usability=1
        scores = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        mock = _make_mock_judge_sequence(scores)
        result = score_rubric(_VALID_TASK, _VALID_OUTPUT, mock)

        assert result["composite"] == pytest.approx(0.70)

    def test_composite_matches_manual_calculation(self) -> None:
        """Composite should equal sum(weight_i * score_i) for all i."""
        per_dim_scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        mock = _make_mock_judge_sequence(per_dim_scores)
        result = score_rubric(_VALID_TASK, _VALID_OUTPUT, mock)

        weights = [_VALID_RUBRIC[dim]["weight"] for dim in RUBRIC_DIMENSIONS]
        expected = sum(w * s for w, s in zip(weights, per_dim_scores, strict=True))
        assert result["composite"] == pytest.approx(expected)

    def test_result_contains_task_id(self) -> None:
        """Result dict must include the task_id from the task definition."""
        mock = _make_mock_judge(0.5)
        result = score_rubric(_VALID_TASK, _VALID_OUTPUT, mock)
        assert result["task_id"] == "T1-OPS-001"

    def test_result_contains_rubric_weights(self) -> None:
        """Result must include rubric_weights matching the task rubric."""
        mock = _make_mock_judge(0.5)
        result = score_rubric(_VALID_TASK, _VALID_OUTPUT, mock)

        for dim in RUBRIC_DIMENSIONS:
            assert result["rubric_weights"][dim] == pytest.approx(_VALID_RUBRIC[dim]["weight"])

    def test_result_contains_all_dimensions(self) -> None:
        """dimension_scores must contain all six GRADE dimensions."""
        mock = _make_mock_judge(0.75)
        result = score_rubric(_VALID_TASK, _VALID_OUTPUT, mock)

        assert set(result["dimension_scores"].keys()) == set(RUBRIC_DIMENSIONS)

    def test_judge_called_once_per_dimension(self) -> None:
        """Exactly one judge call must be made per rubric dimension (six total)."""
        mock = _make_mock_judge(0.5)
        score_rubric(_VALID_TASK, _VALID_OUTPUT, mock)
        assert mock.judge.call_count == len(RUBRIC_DIMENSIONS)

    def test_dimensions_subset_judges_only_those(self) -> None:
        """A dimensions subset must restrict judge calls and returned scores.

        The dispatcher passes only the C2-owned dimensions so judge calls for
        C1/C3/C4-owned dimensions are never made (methodology finding M-2).
        """
        subset = ("insight_quality", "evidence_linkage", "structure_usability")
        mock = _make_mock_judge(0.5)
        result = score_rubric(_VALID_TASK, _VALID_OUTPUT, mock, dimensions=subset)

        assert mock.judge.call_count == len(subset)
        called_dims = {call.kwargs["dimension"] for call in mock.judge.call_args_list}
        assert called_dims == set(subset)
        assert set(result["dimension_scores"].keys()) == set(subset)
        assert set(result["rubric_weights"].keys()) == set(subset)

    def test_dimensions_subset_unknown_dimension_raises(self) -> None:
        """An unknown dimension in the subset must raise before judge calls."""
        mock = _make_mock_judge(0.5)
        with pytest.raises(ValueError, match="Unknown rubric dimension"):
            score_rubric(_VALID_TASK, _VALID_OUTPUT, mock, dimensions=("not_a_dim",))
        assert mock.judge.call_count == 0

    def test_judge_called_with_correct_dimension_names(self) -> None:
        """Each judge call must receive the correct dimension name as the first arg."""
        mock = _make_mock_judge(0.5)
        score_rubric(_VALID_TASK, _VALID_OUTPUT, mock)

        called_dims = [call.kwargs["dimension"] for call in mock.judge.call_args_list]
        assert called_dims == list(RUBRIC_DIMENSIONS)

    def test_judge_receives_guidance_when_present(self) -> None:
        """Judge must receive the guidance string for dimensions that have one."""
        task = copy.deepcopy(_VALID_TASK)
        task["rubric"]["grounding_accuracy"]["guidance"] = "Check TUT-005 numeric values."
        mock = _make_mock_judge(0.8)
        score_rubric(task, _VALID_OUTPUT, mock)

        first_call = mock.judge.call_args_list[0]
        assert first_call.kwargs["guidance"] == "Check TUT-005 numeric values."

    def test_judge_receives_empty_guidance_when_absent(self) -> None:
        """Judge must receive an empty string for dimensions without guidance."""
        mock = _make_mock_judge(0.8)
        score_rubric(_VALID_TASK, _VALID_OUTPUT, mock)

        # All dimensions in _VALID_RUBRIC have no guidance key.
        for call in mock.judge.call_args_list:
            assert call.kwargs["guidance"] == ""

    def test_non_normalized_rubric_raises_before_judge_calls(self) -> None:
        """A rubric whose weights don't sum to 1.0 must raise ValueError before judge calls."""
        task = copy.deepcopy(_VALID_TASK)
        task["rubric"]["grounding_accuracy"]["weight"] = 0.99  # way over
        mock = _make_mock_judge(1.0)

        with pytest.raises(ValueError, match="sum to 1.0"):
            score_rubric(task, _VALID_OUTPUT, mock)

        mock.judge.assert_not_called()

    def test_non_uniform_weights_applied_correctly(self) -> None:
        """Non-uniform weights must be applied correctly in the composite.

        Scenario: grounding_accuracy gets weight 0.80, all others share 0.20.
        A perfect grounding_accuracy score (1.0) and zero elsewhere should
        yield composite == 0.80.
        """
        residual = 0.20 / (len(RUBRIC_DIMENSIONS) - 1)
        task = copy.deepcopy(_VALID_TASK)
        for dim in RUBRIC_DIMENSIONS:
            task["rubric"][dim]["weight"] = residual
        task["rubric"]["grounding_accuracy"]["weight"] = 0.80

        scores = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock = _make_mock_judge_sequence(scores)
        result = score_rubric(task, _VALID_OUTPUT, mock)

        assert result["composite"] == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Authored task rubric validation (integration-style, no API calls)
# ---------------------------------------------------------------------------


class TestAuthoredTaskRubrics:
    """Verify that all authored task rubrics in benchmark/tracks/** sum to 1.0."""

    @staticmethod
    def _load_all_tasks() -> list[tuple[str, dict[str, Any]]]:
        """Load all task dicts from every tasks.jsonl in benchmark/tracks/.

        Returns:
            List of (task_id, task_dict) pairs across all tracks.
        """
        tracks_dir = Path(__file__).parent.parent.parent / "benchmark" / "tracks"
        tasks: list[tuple[str, dict[str, Any]]] = []
        for jsonl_file in sorted(tracks_dir.rglob("tasks.jsonl")):
            with jsonl_file.open(encoding="utf-8") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if raw_line:
                        task = json.loads(raw_line)
                        tasks.append((task["task_id"], task))
        return tasks

    def test_all_authored_rubrics_sum_to_one(self) -> None:
        """Every authored task rubric must have weights summing to ~1.0.

        This test catches authoring mistakes that JSON Schema cannot express
        (cross-field numeric sum constraints).  If any task fails, the test
        output will identify the task_id.
        """
        tasks = self._load_all_tasks()
        assert len(tasks) > 0, "No tasks found — check benchmark/tracks structure."

        failures: list[str] = []
        for task_id, task in tasks:
            try:
                validate_rubric_weights(task["rubric"])
            except ValueError as exc:
                failures.append(f"{task_id}: {exc}")

        assert not failures, "Rubric weight validation failed for:\n" + "\n".join(failures)

    def test_authored_task_count(self) -> None:
        """Sanity check: should find at least 20 authored tasks across all tracks."""
        tasks = self._load_all_tasks()
        assert len(tasks) >= 20, f"Expected ≥20 tasks, found {len(tasks)}"
