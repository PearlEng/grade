"""Result aggregator — assembles per-task results into the result_schema shape.

:func:`aggregate` takes a list of :class:`~runner.dispatcher.TaskRunResult`
objects and produces a single scorecard dict that validates against
``benchmark/schemas/result_schema.json``.

The scorecard includes:

- ``per_task_scores`` — one entry per task.
- ``per_track_scores`` — dimension averages grouped by track (1–5).
- ``per_pack_scores`` — dimension averages grouped by pack ID.
- ``overall_scores`` — global dimension averages across all tasks.
- ``overall_composite`` — global weighted composite.

Public API
----------
- :func:`aggregate` — build a full result dict from task run results.
- :func:`mean_scores` — average a list of dimension score dicts.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from benchmark.rubrics.rubric_scoring import RUBRIC_DIMENSIONS
from runner.dispatcher import DimensionScores, TaskRunResult

#: Project-level default dimension weights used for composite computation
#: when a task-level rubric is unavailable.
DEFAULT_WEIGHTS: dict[str, float] = {
    "grounding_accuracy": 0.35,
    "insight_quality": 0.20,
    "evidence_linkage": 0.15,
    "calibration_limitation_handling": 0.15,
    "consistency": 0.10,
    "structure_usability": 0.05,
}

#: Map from track number to human-readable track name.
TRACK_NAMES: dict[int, str] = {
    1: "Grounded Retrieval & Computation",
    2: "Program Snapshot & Trend",
    3: "Operational Coaching & Recommendations",
    4: "Equity & Subgroup Interpretation",
    5: "Program Effectiveness & Research Reasoning",
}


def mean_scores(score_list: list[DimensionScores]) -> DimensionScores:
    """Average a list of per-dimension score dicts.

    Args:
        score_list: Non-empty list of dicts each mapping dimension name to
            a float in [0, 1].

    Returns:
        A single dict with the arithmetic mean for each dimension.  Returns
        all-zero scores if *score_list* is empty.

    Example::

        avg = mean_scores([{"grounding_accuracy": 0.8}, {"grounding_accuracy": 0.6}])
        assert avg["grounding_accuracy"] == 0.7
    """
    if not score_list:
        return dict.fromkeys(RUBRIC_DIMENSIONS, 0.0)

    result: DimensionScores = {}
    for dim in RUBRIC_DIMENSIONS:
        values = [s.get(dim, 0.0) for s in score_list]
        result[dim] = sum(values) / len(values)
    return result


def _composite(scores: DimensionScores, weights: dict[str, float] = DEFAULT_WEIGHTS) -> float:
    """Compute the weighted composite from dimension scores.

    Args:
        scores: Dict mapping dimension name to score in [0, 1].
        weights: Per-dimension weights.  Defaults to :data:`DEFAULT_WEIGHTS`.

    Returns:
        Weighted sum in [0, 1].
    """
    return sum(weights.get(dim, 0.0) * scores.get(dim, 0.0) for dim in RUBRIC_DIMENSIONS)


def aggregate(
    task_results: list[TaskRunResult],
    model_id: str,
    grade_version: str = "0.1.0",
    result_id: str | None = None,
) -> dict[str, Any]:
    """Assemble per-task results into a ``result_schema.json``-shaped scorecard.

    Args:
        task_results: List of :class:`~runner.dispatcher.TaskRunResult`
            objects, one per task.  Must not be empty.
        model_id: Fully qualified model identifier (e.g.
            ``"openrouter/anthropic/claude-opus-4-7"`` or
            ``"stub/echo-v1"``).
        grade_version: GRADE benchmark package version string.  Defaults to
            ``"0.1.0"``.
        result_id: Unique identifier for this result record.  If ``None``,
            a random UUID is generated.

    Returns:
        A dict conforming to ``result_schema.json`` with all required and
        optional aggregate fields populated.

    Raises:
        ValueError: If *task_results* is empty.

    Example::

        scorecard = aggregate(task_results, model_id="stub/echo-v1")
        validate_result(scorecard)
    """
    if not task_results:
        raise ValueError("task_results must not be empty")

    rid = result_id if result_id is not None else str(uuid.uuid4())
    scored_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- Per-task entries ---
    per_task_scores: list[dict[str, Any]] = []
    for tr in task_results:
        per_task_scores.append(
            {
                "task_id": tr.task_id,
                "track": tr.track,
                "pack_id": tr.pack_id,
                "run_count": tr.run_count,
                "scores": dict(tr.scores),
                "composite": tr.composite,
                "scorer_flags": tr.scorer_flags if tr.scorer_flags else None,
            }
        )

    # --- Per-track aggregation ---
    track_buckets: dict[int, list[DimensionScores]] = {}
    for tr in task_results:
        track_buckets.setdefault(tr.track, []).append(tr.scores)

    per_track_scores: dict[str, Any] = {}
    for track_num, score_list in track_buckets.items():
        avg = mean_scores(score_list)
        per_track_scores[str(track_num)] = {
            "track": track_num,
            "track_name": TRACK_NAMES.get(track_num),
            "task_count": len(score_list),
            "scores": avg,
            "composite": _composite(avg),
        }

    # --- Per-pack aggregation ---
    pack_buckets: dict[str, list[DimensionScores]] = {}
    for tr in task_results:
        if tr.pack_id is not None:
            pack_buckets.setdefault(tr.pack_id, []).append(tr.scores)

    valid_pack_ids = {"pack_operations", "pack_outcomes", "pack_equity_research"}
    per_pack_scores: dict[str, Any] = {}
    for pid, score_list in pack_buckets.items():
        if pid not in valid_pack_ids:
            continue
        avg = mean_scores(score_list)
        per_pack_scores[pid] = {
            "pack_id": pid,
            "task_count": len(score_list),
            "scores": avg,
            "composite": _composite(avg),
        }

    # --- Overall aggregation ---
    all_scores = [tr.scores for tr in task_results]
    overall = mean_scores(all_scores)

    return {
        "result_id": rid,
        "model_id": model_id,
        "grade_version": grade_version,
        "scored_at_utc": scored_at,
        "run_count": task_results[0].run_count if task_results else None,
        "task_count": len(task_results),
        "overall_scores": overall,
        "overall_composite": _composite(overall),
        "per_track_scores": per_track_scores,
        "per_pack_scores": per_pack_scores,
        "per_task_scores": per_task_scores,
    }
