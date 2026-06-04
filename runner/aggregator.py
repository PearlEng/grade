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
- ``cost_metrics`` — aggregate API cost, token counts, and latency (when available).

Public API
----------
- :func:`aggregate` — build a full result dict from task run results.
- :func:`mean_scores` — average a list of dimension score dicts.
- :func:`aggregate_cost_metrics` — collect cost/token/latency from task outputs.
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


def aggregate_cost_metrics(
    task_results: list[TaskRunResult],
    judge_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect and aggregate API cost, token counts, and latency from task outputs.

    Iterates over every per-run output stored in each :class:`TaskRunResult` and
    reads ``cost_usd``, ``prompt_tokens``, ``completion_tokens``, and
    ``latency_ms`` from each output's ``runtime_metadata``.

    Cost split — test-model vs judge overhead
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The result now carries **three** cost fields:

    - ``model_cost_usd`` — test-model inference cost only.  This is the
      **headline leaderboard number** — it reflects the cost of the model
      under test, not GRADE's evaluation infrastructure.
    - ``judge_cost_usd`` — accumulated judge (Opus) overhead across all
      :meth:`~benchmark.rubrics.judge_client.JudgeClient.judge` calls.
      ``None`` when ``--judge`` was not used or the provider never reported
      cost.
    - ``total_eval_cost_usd`` — ``model_cost_usd + judge_cost_usd`` when
      both are available; otherwise the best-effort partial sum or ``None``.

    The legacy ``total_cost_usd`` key is kept as an alias for
    ``model_cost_usd`` to avoid breaking existing callers that read it.

    Cost handling
    ~~~~~~~~~~~~~
    Cost data is genuinely absent for some adapters (e.g. the stub) and for
    providers that do not return a ``cost`` field.  To distinguish "zero cost"
    from "cost not reported", this function tracks whether *any* output in the
    batch had a non-None ``cost_usd``:

    - If **no** output reported cost → ``model_cost_usd`` is ``None`` and
      ``cost_available`` is ``False``.
    - If **some but not all** outputs reported cost → ``model_cost_usd`` is the
      sum of the available values, ``cost_available`` is ``True``, and
      ``cost_partial`` is ``True`` (a warning flag).
    - If **all** outputs reported cost → ``cost_available`` is ``True``,
      ``cost_partial`` is ``False``.

    Token handling
    ~~~~~~~~~~~~~~
    ``None`` token counts (stub adapter, providers without usage reporting) are
    treated as 0 for the purpose of summation and excluded from the
    ``tokens_available`` flag.

    Latency handling
    ~~~~~~~~~~~~~~~~
    ``latency_ms`` is always present in the schema (required field, ``minimum: 0``),
    so ``mean_latency_ms`` is always computed.  Stub latency values are included.

    Args:
        task_results: List of :class:`~runner.dispatcher.TaskRunResult` objects.
        judge_metrics: Optional dict of accumulated judge-client totals, keyed
            by ``"cumulative_cost_usd"``, ``"cumulative_prompt_tokens"``,
            ``"cumulative_completion_tokens"``, and ``"judge_call_count"``.
            Pass ``None`` (the default) when no live judge was used.

    Returns:
        A dict with the following keys:

        - ``model_cost_usd`` (``float | None``): Test-model inference cost —
          **the headline leaderboard figure**.  Alias ``total_cost_usd`` points
          to the same value.
        - ``judge_cost_usd`` (``float | None``): Accumulated judge-model
          overhead.  ``None`` if no judge was used or cost not reported.
        - ``total_eval_cost_usd`` (``float | None``): Sum of ``model_cost_usd``
          and ``judge_cost_usd`` when both are available.
        - ``cost_available`` (``bool``): ``True`` if at least one model output
          had a non-None ``cost_usd``.
        - ``cost_partial`` (``bool``): ``True`` if cost data was missing for at
          least one output in a batch where other outputs *did* have cost data.
        - ``total_prompt_tokens`` (``int``): Summed prompt token count (0 when
          tokens are not reported).
        - ``total_completion_tokens`` (``int``): Summed completion token count.
        - ``total_tokens`` (``int``): ``total_prompt_tokens + total_completion_tokens``.
        - ``tokens_available`` (``bool``): ``True`` if at least one output reported
          non-None token counts.
        - ``mean_latency_ms`` (``float | None``): Mean wall-clock latency in ms
          across all outputs.  ``None`` only if there are no outputs at all.
        - ``p50_latency_ms`` (``float | None``): Median latency.  ``None`` if no
          outputs.
        - ``max_latency_ms`` (``float | None``): Maximum latency.  ``None`` if no
          outputs.
        - ``per_task_cost`` (``dict[str, Any]``): Mapping of ``task_id`` →
          ``{"cost_usd": float | None, "prompt_tokens": int, "completion_tokens": int,
          "total_tokens": int, "mean_latency_ms": float | None}``.

    Example::

        metrics = aggregate_cost_metrics(task_results)
        if metrics["cost_available"]:
            print(f"Model cost (headline): ${metrics['model_cost_usd']:.5f}")
        if metrics["judge_cost_usd"] is not None:
            print(f"Judge overhead: ${metrics['judge_cost_usd']:.5f}")
    """
    all_latencies: list[float] = []
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    cost_values: list[float] = []
    cost_missing_count: int = 0
    tokens_available: bool = False

    per_task_cost: dict[str, Any] = {}

    for tr in task_results:
        task_cost_values: list[float] = []
        task_prompt_tokens: int = 0
        task_completion_tokens: int = 0
        task_latencies: list[float] = []
        task_cost_missing: int = 0

        for output in tr.outputs:
            meta: dict[str, Any] = output.get("runtime_metadata", {})

            # Latency — always present per schema.
            lat = meta.get("latency_ms")
            if lat is not None:
                all_latencies.append(float(lat))
                task_latencies.append(float(lat))

            # Tokens — may be None.
            pt = meta.get("prompt_tokens")
            ct = meta.get("completion_tokens")
            if pt is not None:
                total_prompt_tokens += int(pt)
                task_prompt_tokens += int(pt)
                tokens_available = True
            if ct is not None:
                total_completion_tokens += int(ct)
                task_completion_tokens += int(ct)
                tokens_available = True

            # Cost — may be None.
            cost = meta.get("cost_usd")
            if cost is not None:
                cost_values.append(float(cost))
                task_cost_values.append(float(cost))
            else:
                cost_missing_count += 1
                task_cost_missing += 1

        task_total_cost: float | None = sum(task_cost_values) if task_cost_values else None
        task_mean_latency: float | None = (
            sum(task_latencies) / len(task_latencies) if task_latencies else None
        )
        per_task_cost[tr.task_id] = {
            "cost_usd": task_total_cost,
            "prompt_tokens": task_prompt_tokens,
            "completion_tokens": task_completion_tokens,
            "total_tokens": task_prompt_tokens + task_completion_tokens,
            "mean_latency_ms": task_mean_latency,
        }

    # Global totals — test-model cost is the headline.
    cost_available: bool = len(cost_values) > 0
    model_cost_usd: float | None = sum(cost_values) if cost_available else None
    cost_partial: bool = cost_available and cost_missing_count > 0

    mean_latency_ms: float | None = (
        sum(all_latencies) / len(all_latencies) if all_latencies else None
    )

    # Median (p50) — sort and pick middle element.
    p50_latency_ms: float | None = None
    max_latency_ms: float | None = None
    if all_latencies:
        sorted_lats = sorted(all_latencies)
        n = len(sorted_lats)
        mid = n // 2
        if n % 2:
            p50_latency_ms = sorted_lats[mid]
        else:
            p50_latency_ms = (sorted_lats[mid - 1] + sorted_lats[mid]) / 2
        max_latency_ms = sorted_lats[-1]

    # Judge-overhead cost from the optional judge_metrics dict.
    judge_cost_usd: float | None = None
    if judge_metrics is not None:
        judge_cost_usd = judge_metrics.get("cumulative_cost_usd")

    # total_eval_cost_usd = model + judge when both present; best-effort otherwise.
    total_eval_cost_usd: float | None = None
    if model_cost_usd is not None and judge_cost_usd is not None:
        total_eval_cost_usd = model_cost_usd + judge_cost_usd
    elif model_cost_usd is not None:
        total_eval_cost_usd = model_cost_usd
    elif judge_cost_usd is not None:
        total_eval_cost_usd = judge_cost_usd

    return {
        # Headline test-model cost (leaderboard figure).
        "model_cost_usd": model_cost_usd,
        # Legacy alias — existing callers reading total_cost_usd still work.
        "total_cost_usd": model_cost_usd,
        # Judge evaluation overhead.
        "judge_cost_usd": judge_cost_usd,
        # Combined eval cost (model + judge).
        "total_eval_cost_usd": total_eval_cost_usd,
        "cost_available": cost_available,
        "cost_partial": cost_partial,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "tokens_available": tokens_available,
        "mean_latency_ms": mean_latency_ms,
        "p50_latency_ms": p50_latency_ms,
        "max_latency_ms": max_latency_ms,
        "per_task_cost": per_task_cost,
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
    judge_metrics: dict[str, Any] | None = None,
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
        judge_metrics: Optional dict of accumulated judge-client totals (from
            :class:`~benchmark.rubrics.judge_client.JudgeClient` instance
            state after the run), keyed by ``"cumulative_cost_usd"``,
            ``"cumulative_prompt_tokens"``, ``"cumulative_completion_tokens"``,
            and ``"judge_call_count"``.  Pass ``None`` (default) when no live
            judge was used — ``judge_cost_usd`` will be ``None`` in that case.

    Returns:
        A dict conforming to ``result_schema.json`` with all required and
        optional aggregate fields populated.  The ``cost_metrics`` sub-dict
        contains separate ``model_cost_usd`` (headline), ``judge_cost_usd``,
        and ``total_eval_cost_usd`` fields.

    Raises:
        ValueError: If *task_results* is empty.

    Example::

        scorecard = aggregate(task_results, model_id="stub/echo-v1")
        validate_result(scorecard)

        # With a live judge:
        scorecard = aggregate(
            task_results,
            model_id="stub/echo-v1",
            judge_metrics={
                "cumulative_cost_usd": judge_client.cumulative_cost_usd,
                "cumulative_prompt_tokens": judge_client.cumulative_prompt_tokens,
                "cumulative_completion_tokens": judge_client.cumulative_completion_tokens,
                "judge_call_count": judge_client.judge_call_count,
            },
        )
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

    cost_metrics = aggregate_cost_metrics(task_results, judge_metrics=judge_metrics)

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
        "cost_metrics": cost_metrics,
    }
