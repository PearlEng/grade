"""Unit tests for runner.aggregator — mean_scores and aggregate.

Coverage:
- :func:`~runner.aggregator.mean_scores`: empty list, single item, averaging.
- :func:`~runner.aggregator.aggregate`: per-task/track/pack/overall fields,
  schema validation, multi-track grouping, empty pack_id handling.
"""

from __future__ import annotations

import pytest

from benchmark.rubrics.rubric_scoring import RUBRIC_DIMENSIONS
from benchmark.schemas import validate_result
from runner.aggregator import aggregate, mean_scores
from runner.dispatcher import TaskRunResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZERO_SCORES: dict[str, float] = dict.fromkeys(RUBRIC_DIMENSIONS, 0.0)
_HALF_SCORES: dict[str, float] = dict.fromkeys(RUBRIC_DIMENSIONS, 0.5)
_ONE_SCORES: dict[str, float] = dict.fromkeys(RUBRIC_DIMENSIONS, 1.0)


def _make_task_result(
    task_id: str = "T1-OPS-001",
    track: int = 1,
    pack_id: str | None = "pack_operations",
    scores: dict[str, float] | None = None,
    run_count: int = 1,
) -> TaskRunResult:
    """Create a minimal TaskRunResult for testing.

    Args:
        task_id: Task identifier.
        track: Track number.
        pack_id: Pack identifier.
        scores: Dimension scores dict.  Defaults to all-0.5.
        run_count: Number of runs.

    Returns:
        A :class:`TaskRunResult` with the given fields.
    """
    s = scores if scores is not None else dict(_HALF_SCORES)
    return TaskRunResult(
        task_id=task_id,
        track=track,
        pack_id=pack_id,
        run_count=run_count,
        outputs=[],
        scores=s,
        composite=0.5,
        scorer_flags=[],
    )


# ---------------------------------------------------------------------------
# mean_scores tests
# ---------------------------------------------------------------------------


class TestMeanScores:
    """Tests for :func:`mean_scores`."""

    def test_empty_list_returns_zero_scores(self) -> None:
        """Empty input should return zero for every dimension."""
        result = mean_scores([])
        assert all(v == 0.0 for v in result.values())
        assert set(result.keys()) == set(RUBRIC_DIMENSIONS)

    def test_single_item_returns_same_scores(self) -> None:
        """A single-item list should return the same scores unchanged."""
        result = mean_scores([_HALF_SCORES])
        for dim in RUBRIC_DIMENSIONS:
            assert result[dim] == pytest.approx(0.5)

    def test_averages_correctly(self) -> None:
        """Mean of [0.0, 1.0] per dimension should be 0.5."""
        result = mean_scores([_ZERO_SCORES, _ONE_SCORES])
        for dim in RUBRIC_DIMENSIONS:
            assert result[dim] == pytest.approx(0.5)

    def test_three_items(self) -> None:
        """Average of 0.0, 0.5, 1.0 should be 0.5."""
        result = mean_scores([_ZERO_SCORES, _HALF_SCORES, _ONE_SCORES])
        for dim in RUBRIC_DIMENSIONS:
            assert result[dim] == pytest.approx(0.5)

    def test_returns_all_six_dimensions(self) -> None:
        """Result must always contain all six GRADE dimension keys."""
        result = mean_scores([_HALF_SCORES])
        assert set(result.keys()) == set(RUBRIC_DIMENSIONS)


# ---------------------------------------------------------------------------
# aggregate tests
# ---------------------------------------------------------------------------


class TestAggregate:
    """Tests for :func:`aggregate`."""

    def test_empty_task_results_raises(self) -> None:
        """aggregate([]) must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            aggregate([], model_id="stub/echo-v1")

    def test_single_task_result_validates(self) -> None:
        """A single-task result must pass result_schema validation."""
        tr = _make_task_result()
        scorecard = aggregate([tr], model_id="stub/echo-v1")
        validate_result(scorecard)  # should not raise

    def test_model_id_embedded(self) -> None:
        """The model_id passed to aggregate must appear in the result."""
        tr = _make_task_result()
        scorecard = aggregate([tr], model_id="test/model-007")
        assert scorecard["model_id"] == "test/model-007"

    def test_per_task_scores_count(self) -> None:
        """per_task_scores should contain one entry per TaskRunResult."""
        trs = [_make_task_result(task_id=f"T1-OPS-{i:03d}") for i in range(3)]
        scorecard = aggregate(trs, model_id="stub/echo-v1")
        assert len(scorecard["per_task_scores"]) == 3

    def test_task_count_field(self) -> None:
        """task_count must equal the number of TaskRunResult objects."""
        trs = [_make_task_result(task_id=f"T1-OPS-{i:03d}") for i in range(5)]
        scorecard = aggregate(trs, model_id="stub/echo-v1")
        assert scorecard["task_count"] == 5

    def test_per_track_grouping(self) -> None:
        """Tasks across two tracks must appear in two separate per_track_scores entries."""
        tr1 = _make_task_result(task_id="T1-OPS-001", track=1)
        tr2 = _make_task_result(task_id="T2-OPS-001", track=2)
        scorecard = aggregate([tr1, tr2], model_id="stub/echo-v1")
        assert "1" in scorecard["per_track_scores"]
        assert "2" in scorecard["per_track_scores"]

    def test_per_pack_grouping(self) -> None:
        """Tasks from two packs must appear in two per_pack_scores entries."""
        tr1 = _make_task_result(task_id="T1-OPS-001", pack_id="pack_operations")
        tr2 = _make_task_result(task_id="T2-OUT-001", pack_id="pack_outcomes")
        scorecard = aggregate([tr1, tr2], model_id="stub/echo-v1")
        assert "pack_operations" in scorecard["per_pack_scores"]
        assert "pack_outcomes" in scorecard["per_pack_scores"]

    def test_none_pack_id_not_in_per_pack(self) -> None:
        """Tasks with pack_id=None must not appear in per_pack_scores."""
        tr = _make_task_result(pack_id=None)
        scorecard = aggregate([tr], model_id="stub/echo-v1")
        assert scorecard["per_pack_scores"] == {}

    def test_overall_scores_are_averages(self) -> None:
        """overall_scores should equal the mean of all task scores."""
        tr1 = _make_task_result(scores=dict.fromkeys(RUBRIC_DIMENSIONS, 0.0))
        tr2 = _make_task_result(scores=dict.fromkeys(RUBRIC_DIMENSIONS, 1.0))
        scorecard = aggregate([tr1, tr2], model_id="stub/echo-v1")
        for dim in RUBRIC_DIMENSIONS:
            assert scorecard["overall_scores"][dim] == pytest.approx(0.5)

    def test_overall_composite_in_unit_interval(self) -> None:
        """overall_composite must be in [0, 1]."""
        trs = [_make_task_result()]
        scorecard = aggregate(trs, model_id="stub/echo-v1")
        assert 0.0 <= scorecard["overall_composite"] <= 1.0

    def test_result_id_generated_when_not_provided(self) -> None:
        """A UUID result_id must be generated when none is provided."""
        trs = [_make_task_result()]
        scorecard = aggregate(trs, model_id="stub/echo-v1")
        assert len(scorecard["result_id"]) >= 8

    def test_custom_result_id(self) -> None:
        """A provided result_id must appear verbatim in the scorecard."""
        trs = [_make_task_result()]
        scorecard = aggregate(trs, model_id="stub/echo-v1", result_id="my-custom-id")
        assert scorecard["result_id"] == "my-custom-id"

    def test_grade_version_embedded(self) -> None:
        """The grade_version passed must appear in the scorecard."""
        trs = [_make_task_result()]
        scorecard = aggregate(trs, model_id="stub/echo-v1", grade_version="99.0.0")
        assert scorecard["grade_version"] == "99.0.0"

    def test_multi_task_validates(self) -> None:
        """A 5-task, 3-track, 2-pack scorecard must pass result_schema validation."""
        trs = [
            _make_task_result("T1-OPS-001", track=1, pack_id="pack_operations"),
            _make_task_result("T1-OPS-002", track=1, pack_id="pack_operations"),
            _make_task_result("T2-OUT-001", track=2, pack_id="pack_outcomes"),
            _make_task_result("T3-OPS-001", track=3, pack_id="pack_operations"),
            _make_task_result("T4-EQU-001", track=4, pack_id="pack_equity_research"),
        ]
        scorecard = aggregate(trs, model_id="stub/echo-v1")
        validate_result(scorecard)  # should not raise
