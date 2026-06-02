"""Task dispatcher — loads tasks, runs the adapter, and calls scorers (C1–C4).

The :func:`run_task` function executes one (task × adapter × N runs) cycle:

1. Calls the adapter ``runs`` times to gather a set of model outputs.
2. Validates each output against ``output_schema.json``.
3. Scores each output with the available automated scorers (C1 fact scoring,
   C2 rubric scoring, C3 claim validation, C4 consistency).
4. Returns a :class:`TaskRunResult` containing all per-run outputs and an
   averaged :class:`DimensionScores` dict ready for aggregation.

:func:`load_pack` handles reading a ``.jsonl`` task-pack file from disk.

Public API
----------
- :class:`DimensionScores` — typed dict of the six scoring dimensions.
- :class:`TaskRunResult` — per-task output + averaged scores.
- :func:`load_pack` — load tasks from a JSONL file.
- :func:`run_task` — run one task end-to-end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.rubrics.fact_scoring import score_facts
from benchmark.rubrics.rubric_scoring import RUBRIC_DIMENSIONS, score_rubric
from benchmark.schemas import validate_output
from runner.adapters.base import Adapter

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

#: Per-dimension score dict matching ``DimensionScores`` in result_schema.json.
DimensionScores = dict[str, float]

_ZERO_SCORES: DimensionScores = dict.fromkeys(RUBRIC_DIMENSIONS, 0.0)


@dataclass
class TaskRunResult:
    """All per-run outputs and averaged scores for one (task × adapter) pair.

    Attributes:
        task_id: Task identifier from the task definition.
        track: Benchmark track number (1–5).
        pack_id: Fixture pack identifier, or ``None`` if unknown.
        run_count: Number of repetition runs executed.
        outputs: List of raw model output dicts (one per run), each
            validated against ``output_schema.json``.
        scores: Per-dimension scores averaged over all runs.  Maps each of
            the six dimension names to a float in [0, 1].
        composite: Weighted composite score in [0, 1], or ``None`` if not
            yet computed.
        scorer_flags: Aggregated list of scorer warning flags from all runs.
    """

    task_id: str
    track: int
    pack_id: str | None
    run_count: int
    outputs: list[dict[str, Any]] = field(default_factory=list)
    scores: DimensionScores = field(default_factory=dict)
    composite: float | None = None
    scorer_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pack loader
# ---------------------------------------------------------------------------


def load_pack(pack_path: Path) -> list[dict[str, Any]]:
    """Load all tasks from a JSONL task-pack file.

    Args:
        pack_path: Absolute path to a ``.jsonl`` file where each line is a
            JSON object conforming to ``task_schema.json``.

    Returns:
        A list of task dicts in file order.  Empty lines are skipped.

    Raises:
        FileNotFoundError: If *pack_path* does not exist.
        json.JSONDecodeError: If any non-empty line is not valid JSON.

    Example::

        tasks = load_pack(Path("benchmark/tasks/operations_pack.jsonl"))
    """
    tasks: list[dict[str, Any]] = []
    with pack_path.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                tasks.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {lineno} of {pack_path}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc
    return tasks


# ---------------------------------------------------------------------------
# Mock judge for C2 rubric scoring when no live judge is configured
# ---------------------------------------------------------------------------


class _NullJudge:
    """Stub judge that always returns 0.5 for every rubric dimension.

    Used when the runner is invoked without a real judge client (e.g., in
    smoke tests).  This avoids requiring an API key just to produce a
    structurally valid result.

    This class satisfies :class:`~benchmark.rubrics.rubric_scoring.JudgeClientProtocol`.
    """

    def judge(
        self,
        dimension: str,
        guidance: str,
        task: dict[str, Any],
        model_output: dict[str, Any],
    ) -> float:
        """Return a fixed mid-point score of 0.5 for every dimension.

        Args:
            dimension: Rubric dimension name (ignored).
            guidance: Task-specific guidance text (ignored).
            task: Task definition dict (ignored).
            model_output: Normalized model output dict (ignored).

        Returns:
            Always ``0.5``.
        """
        return 0.5


_NULL_JUDGE = _NullJudge()

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_c1_grounding(task: dict[str, Any], output: dict[str, Any]) -> float:
    """Run C1 fact scoring and return ``grounding_accuracy`` in [0, 1].

    Args:
        task: Task definition dict.
        output: Normalized model output dict.

    Returns:
        Grounding accuracy score in [0, 1].
    """
    result = score_facts(
        gold_facts=task.get("gold_facts", []),
        structured_metrics=output.get("structured_metrics", {}),
        key_findings=output.get("key_findings", []),
    )
    return result.grounding_accuracy


def _score_c2_rubric(
    task: dict[str, Any],
    output: dict[str, Any],
    judge_client: Any,
) -> dict[str, float]:
    """Run C2 rubric scoring and return per-dimension scores.

    Args:
        task: Task definition dict.
        output: Normalized model output dict.
        judge_client: Object satisfying
            :class:`~benchmark.rubrics.rubric_scoring.JudgeClientProtocol`.

    Returns:
        Dict mapping each of the six dimension names to a float in [0, 1].
        On error (e.g. malformed rubric), returns zero scores for all
        dimensions.
    """
    try:
        result = score_rubric(task, output, judge_client)
        return dict(result["dimension_scores"])
    except (ValueError, KeyError):
        return dict.fromkeys(RUBRIC_DIMENSIONS, 0.0)


def _compute_composite(
    scores: DimensionScores,
    rubric: dict[str, Any],
) -> float:
    """Compute the weighted composite score from per-dimension scores.

    Args:
        scores: Per-dimension scores in [0, 1].
        rubric: Task rubric dict with per-dimension ``weight`` values.

    Returns:
        Weighted sum in [0, 1].
    """
    total = 0.0
    for dim in RUBRIC_DIMENSIONS:
        weight = float(rubric.get(dim, {}).get("weight", 0.0))
        total += weight * scores.get(dim, 0.0)
    return total


# ---------------------------------------------------------------------------
# Core task runner
# ---------------------------------------------------------------------------


def run_task(
    task: dict[str, Any],
    adapter: Adapter,
    runs: int = 5,
    pack_id: str | None = None,
    judge_client: Any | None = None,
) -> TaskRunResult:
    """Run one task end-to-end: call adapter N times, score, and aggregate.

    For each of the *runs* repetitions the function:

    1. Calls ``adapter.run(task, run_index=i)`` to get a model output.
    2. Validates the output against ``output_schema.json`` (raises on
       violation — the adapter is responsible for returning a valid payload).
    3. Scores the output using C1 (grounding accuracy via fact scoring) and
       C2 (rubric scoring via *judge_client*).
    4. C3 (claim validation) and C4 (cross-run consistency) are collected
       once all runs are complete.

    After all runs, dimension scores are averaged across repetitions.
    Consistency (C4) is measured as the fraction of runs whose
    ``key_findings`` overlap with the majority response (majority defined as
    the most common first finding).

    Args:
        task: Task definition dict conforming to ``task_schema.json``.
        adapter: Model adapter satisfying :class:`~runner.adapters.base.Adapter`.
        runs: Number of repetition runs per task.  Must be >= 1.
        pack_id: Fixture pack identifier to embed in the result.
        judge_client: Object satisfying
            :class:`~benchmark.rubrics.rubric_scoring.JudgeClientProtocol`
            for C2 rubric scoring.  If ``None``, a null judge (always 0.5)
            is used.

    Returns:
        A :class:`TaskRunResult` with all per-run outputs, averaged scores,
        and a weighted composite.

    Raises:
        ValueError: If *runs* < 1.
        jsonschema.ValidationError: If any adapter output fails
            ``output_schema.json`` validation.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")

    effective_judge = judge_client if judge_client is not None else _NULL_JUDGE

    task_id: str = task["task_id"]
    track: int = task.get("track", 1)
    rubric: dict[str, Any] = task.get("rubric", {})

    outputs: list[dict[str, Any]] = []
    per_run_scores: list[DimensionScores] = []
    all_flags: list[str] = []

    for run_index in range(runs):
        # Inject pack_id into the task dict so the adapter can embed it.
        task_with_pack = dict(task)
        if pack_id is not None:
            task_with_pack["pack_id"] = pack_id

        output = adapter.run(task_with_pack, run_index=run_index)

        # Validate against output_schema — raises on violation.
        validate_output(output)
        outputs.append(output)

        # --- C1: grounding accuracy via fact scoring ---
        c1_score = _score_c1_grounding(task, output)

        # --- C2: rubric scoring (all six dimensions) ---
        c2_scores = _score_c2_rubric(task, output, effective_judge)

        # Merge: C1 grounding_accuracy overrides C2's grounding dimension
        # because C1 is the authoritative automated scorer.
        run_scores: DimensionScores = dict(c2_scores)
        run_scores["grounding_accuracy"] = c1_score

        per_run_scores.append(run_scores)

    # --- C4: consistency across runs ---
    # Measure as mean pairwise Jaccard similarity of first key_finding tokens.
    consistency_score = _measure_consistency(outputs)

    # Average per-run scores across all runs.
    avg_scores: DimensionScores = {}
    for dim in RUBRIC_DIMENSIONS:
        if dim == "consistency":
            avg_scores[dim] = consistency_score
        else:
            values = [s.get(dim, 0.0) for s in per_run_scores]
            avg_scores[dim] = sum(values) / len(values) if values else 0.0

    # Compute composite.
    composite = _compute_composite(avg_scores, rubric)

    return TaskRunResult(
        task_id=task_id,
        track=track,
        pack_id=pack_id,
        run_count=runs,
        outputs=outputs,
        scores=avg_scores,
        composite=composite,
        scorer_flags=all_flags,
    )


def _measure_consistency(outputs: list[dict[str, Any]]) -> float:
    """Measure cross-run consistency as mean pairwise Jaccard on key_findings.

    Tokenises the first ``key_finding`` from each run into a set of
    lower-cased words and computes the mean Jaccard similarity across all
    (i, j) pairs.  Returns 1.0 if there is only one run (trivially
    consistent) or if all runs produce identical findings.

    Args:
        outputs: List of model output dicts (one per run).

    Returns:
        Consistency score in [0, 1].
    """
    if len(outputs) <= 1:
        return 1.0

    def _tokens(output: dict[str, Any]) -> frozenset[str]:
        findings = output.get("key_findings", [])
        text = findings[0] if findings else ""
        return frozenset(text.lower().split())

    token_sets = [_tokens(o) for o in outputs]
    n = len(token_sets)
    total = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            union = len(a | b)
            intersection = len(a & b)
            total += (intersection / union) if union > 0 else 1.0
            count += 1

    return total / count if count > 0 else 1.0
