"""Rubric-based structured scorer for GRADE benchmark tasks (C2).

This module implements the rubric harness that evaluates model outputs against
the six GRADE scoring dimensions defined in each task's ``rubric`` field.  For
each dimension, the harness delegates to a *judge client* that returns a
structured sub-score in [0, 1]; the harness then computes a weighted aggregate
consistent with ``result_schema.json``.

Design decisions
----------------
- **Weight-sum validation** is enforced at runtime here because JSON Schema
  cannot express cross-field numeric constraints.  Any rubric whose dimension
  weights do not sum to approximately 1.0 (within ``WEIGHT_SUM_TOLERANCE``) is
  rejected with :exc:`ValueError` before any judge calls are made.
- **Judge calls are capped** at one call per scored dimension, providing a
  deterministic upper bound on API usage.  Callers may restrict scoring to a
  subset of dimensions via the ``dimensions`` parameter; the runner judges
  only the three C2-owned dimensions, since the other three are computed
  deterministically by C1/C3/C4.
- **Temperature 0 and a fixed random seed** are passed through to the judge
  client to ensure reproducible results; see :mod:`benchmark.rubrics.judge_client`
  for the live-judge integration wrapper.

Live-judge invocation
---------------------
In production, pass a real :class:`~benchmark.rubrics.judge_client.JudgeClient`
instance::

    from benchmark.rubrics.judge_client import JudgeClient
    from benchmark.rubrics.rubric_scoring import score_rubric

    client = JudgeClient(api_key="...")
    result = score_rubric(task, model_output, client)

In tests, use a mock that follows the same interface — a callable (or object
with a ``judge`` method) that accepts a prompt string and returns a float in
[0, 1].  See ``tests/unit/test_rubric_scoring.py`` for examples.
"""

from __future__ import annotations

from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Ordered list of the six canonical GRADE rubric dimension names.
RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "grounding_accuracy",
    "insight_quality",
    "evidence_linkage",
    "calibration_limitation_handling",
    "consistency",
    "structure_usability",
)

#: Absolute tolerance used when checking that rubric weights sum to 1.0.
WEIGHT_SUM_TOLERANCE: float = 1e-6


# ---------------------------------------------------------------------------
# Judge client protocol
# ---------------------------------------------------------------------------


class JudgeClientProtocol(Protocol):
    """Structural protocol for judge clients consumed by :func:`score_rubric`.

    Any object that implements :meth:`judge` satisfies this protocol, making
    the scorer independent of the concrete judge implementation (Anthropic SDK,
    OpenRouter, or a test mock).
    """

    def judge(
        self,
        dimension: str,
        guidance: str,
        task: dict[str, Any],
        model_output: dict[str, Any],
    ) -> float:
        """Evaluate *model_output* for *task* on a single rubric dimension.

        Args:
            dimension: The rubric dimension name (e.g. ``"grounding_accuracy"``).
            guidance: The task-specific scorer guidance text from the rubric, or
                an empty string if no guidance was authored for this dimension.
            task: The full task definition dict (validated against
                ``task_schema.json``).
            model_output: The normalized model output dict (validated against
                ``output_schema.json``).

        Returns:
            A float in ``[0.0, 1.0]`` representing the model's performance on
            this dimension for this task.
        """
        ...


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------


def validate_rubric_weights(rubric: dict[str, Any]) -> None:
    """Validate that a task rubric's dimension weights sum to approximately 1.0.

    JSON Schema cannot express cross-field numeric sum constraints, so this
    check is enforced at runtime.  The check is applied before any judge calls
    are made.

    Args:
        rubric: The ``rubric`` sub-dict from a task definition.  Each key is a
            dimension name; each value is a dict with at least a ``"weight"``
            key (float in [0, 1]).

    Raises:
        ValueError: If the weights do not sum to ``1.0 ± WEIGHT_SUM_TOLERANCE``,
            or if any dimension listed in :data:`RUBRIC_DIMENSIONS` is absent
            from *rubric*.

    Example::

        validate_rubric_weights(task["rubric"])  # raises if invalid
    """
    for dim in RUBRIC_DIMENSIONS:
        if dim not in rubric:
            raise ValueError(
                f"Rubric is missing required dimension '{dim}'. "
                f"Expected all of: {RUBRIC_DIMENSIONS}"
            )

    total: float = sum(rubric[dim]["weight"] for dim in RUBRIC_DIMENSIONS)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"Rubric dimension weights must sum to 1.0 (±{WEIGHT_SUM_TOLERANCE}), "
            f"but got {total:.10f}.  Adjust the per-dimension weights so they sum "
            "to exactly 1.0."
        )


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------


def score_rubric(
    task: dict[str, Any],
    model_output: dict[str, Any],
    judge_client: JudgeClientProtocol,
    dimensions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Score *model_output* against *task*'s rubric using *judge_client*.

    For each of the six GRADE rubric dimensions the harness:

    1. Reads the dimension's ``weight`` and optional ``guidance`` from the task
       rubric.
    2. Calls ``judge_client.judge(dimension, guidance, task, model_output)``
       to obtain a per-dimension sub-score in [0, 1].
    3. Accumulates a weighted aggregate score.

    The returned dict contains per-dimension sub-scores (keyed by dimension
    name), the weighted ``composite`` aggregate, and the ``rubric_weights`` used
    so that callers can audit the computation.

    Args:
        task: The full task definition dict.  Must contain a ``"rubric"`` key
            whose value is a dict with one entry per GRADE dimension, each
            having at least a ``"weight"`` float.  The weights must sum to
            approximately 1.0; a :exc:`ValueError` is raised otherwise.
        model_output: The normalized model output dict.  Passed through to the
            judge client unchanged.
        judge_client: An object implementing :class:`JudgeClientProtocol`.  The
            scorer makes exactly one ``judge`` call per scored dimension; cap
            and determinism settings (temperature 0, fixed seed) should be
            configured on the client itself — see
            :mod:`benchmark.rubrics.judge_client`.
        dimensions: Optional subset of :data:`RUBRIC_DIMENSIONS` to judge.
            When ``None`` (default) all six dimensions are judged.  The
            dispatcher passes only the C2-owned dimensions
            (``insight_quality``, ``evidence_linkage``,
            ``structure_usability``) since the other three are owned by
            C1/C3/C4 — judging them would be wasted API spend.  Weight
            validation always runs over the full rubric regardless.

    Returns:
        A dict with the following keys:

        - ``"task_id"`` (str): Copied from *task*.
        - ``"dimension_scores"`` (dict[str, float]): Per-dimension sub-scores
          in [0, 1] keyed by dimension name (scored dimensions only).
        - ``"rubric_weights"`` (dict[str, float]): The per-dimension weights
          extracted from the task rubric (scored dimensions only).
        - ``"composite"`` (float): The weighted aggregate score in [0, 1],
          computed as ``sum(weight_i * score_i)`` over the scored dimensions
          (partial when a ``dimensions`` subset is requested).

    Raises:
        ValueError: If the task rubric weights do not sum to
            ``1.0 ± WEIGHT_SUM_TOLERANCE``, or if a required dimension is
            absent from the rubric.
        KeyError: If *task* does not have a ``"rubric"`` key or *task* lacks a
            ``"task_id"`` key.

    Example::

        from benchmark.rubrics.rubric_scoring import score_rubric

        result = score_rubric(task, model_output, judge_client=mock_judge)
        print(result["composite"])       # e.g. 0.82
        print(result["dimension_scores"]["grounding_accuracy"])  # e.g. 0.9
    """
    rubric: dict[str, Any] = task["rubric"]
    task_id: str = task["task_id"]

    # Validate weight sum before making any judge calls.  Always validates
    # the full rubric, even when only a subset of dimensions is judged.
    validate_rubric_weights(rubric)

    scored_dimensions: tuple[str, ...] = dimensions if dimensions is not None else RUBRIC_DIMENSIONS
    unknown = [d for d in scored_dimensions if d not in RUBRIC_DIMENSIONS]
    if unknown:
        raise ValueError(f"Unknown rubric dimension(s) requested: {unknown}")

    dimension_scores: dict[str, float] = {}
    rubric_weights: dict[str, float] = {}
    composite: float = 0.0

    for dim in scored_dimensions:
        dim_spec: dict[str, Any] = rubric[dim]
        weight: float = float(dim_spec["weight"])
        guidance: str = dim_spec.get("guidance", "")

        sub_score: float = judge_client.judge(
            dimension=dim,
            guidance=guidance,
            task=task,
            model_output=model_output,
        )

        dimension_scores[dim] = sub_score
        rubric_weights[dim] = weight
        composite += weight * sub_score

    return {
        "task_id": task_id,
        "dimension_scores": dimension_scores,
        "rubric_weights": rubric_weights,
        "composite": composite,
    }
