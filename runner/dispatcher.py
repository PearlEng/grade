"""Task dispatcher — loads tasks, runs the adapter, and calls scorers (C1–C4).

The :func:`run_task` function executes one (task × adapter × N runs) cycle:

1. Calls the adapter ``runs`` times to gather a set of model outputs.
2. Validates each output against ``output_schema.json``.
3. Scores each output with the automated scorers:

   - **C1** (``score_facts``) → ``grounding_accuracy``
   - **C2** (``score_rubric``) → ``insight_quality``, ``evidence_linkage``,
     ``structure_usability``
   - **C3** (``validate_claims``) → ``calibration_limitation_handling``
     (averaged over the N runs)
   - **C4** (``score_consistency``) → ``consistency`` (computed once over
     all N run outputs)

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

from benchmark.rubrics.claim_validation import validate_claims
from benchmark.rubrics.consistency_scoring import score_consistency
from benchmark.rubrics.fact_scoring import score_facts
from benchmark.rubrics.rubric_scoring import RUBRIC_DIMENSIONS, score_rubric
from benchmark.schemas import validate_output
from runner.adapters.base import Adapter

# ---------------------------------------------------------------------------
# Fixture resolution
# ---------------------------------------------------------------------------

#: Canonical pack_id → fixtures subdirectory name.
_PACK_FIXTURE_DIRS: dict[str, str] = {
    "pack_operations": "pack_operations",
    "pack_outcomes": "pack_outcomes",
    "pack_equity_research": "pack_equity_research",
}

#: Repository root (two levels above this file: runner/ → repo root).
_REPO_ROOT: Path = Path(__file__).parent.parent


def _resolve_fixtures(
    allowed_inputs: list[str],
    pack_id: str | None,
) -> dict[str, str]:
    """Read fixture files for the given pack and return their full contents.

    Each filename in *allowed_inputs* is resolved from
    ``fixtures/<pack_dir>/<filename>`` where *pack_dir* is looked up from
    *pack_id* via :data:`_PACK_FIXTURE_DIRS`.  Files that do not exist in the
    pack directory are silently omitted from the returned dict.

    Args:
        allowed_inputs: List of bare filenames from the task's
            ``allowed_inputs`` field (e.g. ``["students.csv", "groups.csv"]``).
        pack_id: Canonical pack identifier (e.g. ``"pack_operations"``).
            When ``None`` or not in :data:`_PACK_FIXTURE_DIRS`, returns an
            empty dict.

    Returns:
        A ``{filename: contents}`` dict mapping each resolved filename to its
        full UTF-8 text contents.
    """
    if not pack_id or pack_id not in _PACK_FIXTURE_DIRS:
        return {}

    fixture_dir = _REPO_ROOT / "fixtures" / _PACK_FIXTURE_DIRS[pack_id]
    result: dict[str, str] = {}
    for filename in allowed_inputs:
        path = fixture_dir / filename
        if path.is_file():
            result[filename] = path.read_text(encoding="utf-8")
    return result


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

    The output's ``limitations`` list is passed through as a documented
    fallback search target — models often state caveated numbers ("only 127
    of the 135 enrolled students attended") in their limitations section.

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
        limitations=output.get("limitations", []),
    )
    return result.grounding_accuracy


#: The rubric dimensions whose authoritative score comes from the C2 judge.
#: The other three (grounding_accuracy, calibration_limitation_handling,
#: consistency) are owned by C1/C3/C4, so judging them would be wasted API
#: spend — at 26 tasks × 5 runs that's 390 discarded judge calls per model.
_C2_OWNED_DIMENSIONS: tuple[str, ...] = (
    "insight_quality",
    "evidence_linkage",
    "structure_usability",
)


def _score_c2_rubric(
    task: dict[str, Any],
    output: dict[str, Any],
    judge_client: Any,
) -> dict[str, float]:
    """Run C2 rubric scoring and return per-dimension scores.

    Only the C2-owned dimensions are judged: ``insight_quality``,
    ``evidence_linkage``, and ``structure_usability``.  The remaining three
    dimensions are owned by C1/C3/C4 and are never sent to the judge.

    Args:
        task: Task definition dict.
        output: Normalized model output dict.
        judge_client: Object satisfying
            :class:`~benchmark.rubrics.rubric_scoring.JudgeClientProtocol`.

    Returns:
        Dict mapping each C2-owned dimension name to a float in [0, 1].
        On error (e.g. malformed rubric), returns zero scores for the
        C2-owned dimensions.
    """
    try:
        result = score_rubric(task, output, judge_client, dimensions=_C2_OWNED_DIMENSIONS)
        return dict(result["dimension_scores"])
    except (ValueError, KeyError):
        return dict.fromkeys(_C2_OWNED_DIMENSIONS, 0.0)


def _score_c3_calibration(
    task: dict[str, Any],
    output: dict[str, Any],
    judge_client: Any,
) -> float:
    """Run C3 claim validation and return ``calibration_limitation_handling``.

    Args:
        task: Task definition dict.
        output: Normalized model output dict.
        judge_client: Object satisfying
            :class:`~benchmark.rubrics.claim_validation.JudgeClientProtocol`.
            If ``None``, only deterministic Stage 1 matching is used.

    Returns:
        ``calibration_limitation_handling`` score in [0, 1].
    """
    result = validate_claims(task, output, judge_client=judge_client)
    return result.calibration_limitation_handling


def _compute_composite(
    scores: DimensionScores,
    rubric: dict[str, Any],
    exclude: tuple[str, ...] = (),
) -> float:
    """Compute the weighted composite score from per-dimension scores.

    Args:
        scores: Per-dimension scores in [0, 1].
        rubric: Task rubric dict with per-dimension ``weight`` values.
        exclude: Dimensions to drop from the composite.  The remaining
            weights are renormalized to sum to 1, so the composite stays on
            the same [0, 1] scale.  Used for single-run executions where
            ``consistency`` is trivially 1.0 and would otherwise be free
            credit.

    Returns:
        Weighted sum in [0, 1].
    """
    total = 0.0
    included_weight = 0.0
    for dim in RUBRIC_DIMENSIONS:
        if dim in exclude:
            continue
        weight = float(rubric.get(dim, {}).get("weight", 0.0))
        included_weight += weight
        total += weight * scores.get(dim, 0.0)
    if exclude and included_weight > 0:
        return total / included_weight
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
    3. Scores the output using:

       - **C1** ``score_facts`` → ``grounding_accuracy``
       - **C2** ``score_rubric`` → ``insight_quality``, ``evidence_linkage``,
         ``structure_usability``
       - **C3** ``validate_claims`` → ``calibration_limitation_handling``
         (per run, averaged at the end)

    After all runs, the function calls:

    - **C4** ``score_consistency`` once over all N run outputs → ``consistency``

    Dimension scores are then averaged across repetitions (except
    ``consistency`` which is a single cross-run measurement).

    Scorer precedence when dimensions overlap:

    - ``grounding_accuracy``: C1 overrides C2.
    - ``calibration_limitation_handling``: C3 overrides C2.
    - ``consistency``: C4 overrides C2.
    - ``insight_quality``, ``evidence_linkage``, ``structure_usability``: C2.

    Args:
        task: Task definition dict conforming to ``task_schema.json``.
        adapter: Model adapter satisfying :class:`~runner.adapters.base.Adapter`.
        runs: Number of repetition runs per task.  Must be >= 1.
        pack_id: Fixture pack identifier to embed in the result.
        judge_client: Object satisfying
            :class:`~benchmark.rubrics.rubric_scoring.JudgeClientProtocol`
            for C2 rubric scoring and C3 claim validation.  If ``None``, a
            null judge (always 0.5) is used for C2, and C3 runs in
            deterministic-only mode.

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

    # Make null-judge runs visible in the result: when no live judge is
    # configured, the C2-owned dimensions (insight_quality, evidence_linkage,
    # structure_usability — 40% of the composite) are a flat 0.5 placeholder.
    # The flag lands in per_task_scores[].scorer_flags so downstream consumers
    # (leaderboard, website) can detect and refuse placeholder scorecards.
    if judge_client is None:
        all_flags.append("null_judge")

    # Resolve fixture contents once (same for all runs of this task).
    fixtures: dict[str, str] = _resolve_fixtures(
        task.get("allowed_inputs", []),
        pack_id,
    )

    for run_index in range(runs):
        # Inject pack_id and resolved fixture contents into the task dict so
        # the adapter can embed the actual data in its prompt.
        task_with_pack = dict(task)
        if pack_id is not None:
            task_with_pack["pack_id"] = pack_id
        # Always set fixtures key (empty dict when pack is unknown/files absent).
        task_with_pack["fixtures"] = fixtures

        output = adapter.run(task_with_pack, run_index=run_index)

        # Validate against output_schema — raises on violation.
        validate_output(output)
        outputs.append(output)

        # Surface truncation: a finish_reason of "length" means the response
        # hit max_tokens mid-analysis.  Limitations sections come last in
        # prose responses, so truncation silently deflates calibration scores
        # — make it visible in the scorecard instead.
        if output.get("runtime_metadata", {}).get("finish_reason") == "length":
            if "truncated_output" not in all_flags:
                all_flags.append("truncated_output")

        # --- C1: grounding accuracy via fact scoring ---
        c1_score = _score_c1_grounding(task, output)

        # --- C2: rubric scoring (judged dimensions only) ---
        c2_scores = _score_c2_rubric(task, output, effective_judge)

        # --- C3: claim validation → calibration_limitation_handling ---
        c3_score = _score_c3_calibration(task, output, judge_client)

        # Merge scores with authoritative-scorer precedence:
        #   C1 overrides grounding_accuracy
        #   C3 overrides calibration_limitation_handling
        #   consistency is set after all runs (C4); placeholder 0.0 here
        run_scores: DimensionScores = dict(c2_scores)
        run_scores["grounding_accuracy"] = c1_score
        run_scores["calibration_limitation_handling"] = c3_score

        per_run_scores.append(run_scores)

    # --- C4: consistency across all N runs (computed once) ---
    c4_result = score_consistency(task, outputs)
    consistency_score: float = c4_result["consistency_score"]

    # Average per-run scores across all runs, then override consistency from C4.
    avg_scores: DimensionScores = {}
    for dim in RUBRIC_DIMENSIONS:
        if dim == "consistency":
            avg_scores[dim] = consistency_score
        else:
            values = [s.get(dim, 0.0) for s in per_run_scores]
            avg_scores[dim] = sum(values) / len(values) if values else 0.0

    # Compute composite.  With a single run, all three C4 sub-metrics
    # trivially default to 1.0 — that's not measured consistency, it's free
    # credit (typically 10% of the composite).  Exclude the dimension and
    # renormalize the remaining weights so single-run composites stay
    # comparable, and flag the run so downstream consumers can tell.
    if runs < 2:
        all_flags.append("consistency_trivial")
        composite = _compute_composite(avg_scores, rubric, exclude=("consistency",))
    else:
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
