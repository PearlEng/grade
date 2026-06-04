"""C9-Leaderboard — multi-model ranked leaderboard for the GRADE benchmark.

This module computes and renders an N-way ranked leaderboard from a collection
of per-model :func:`~runner.aggregator.aggregate` result dicts.

The leaderboard is sorted by **composite score** (descending) by default; the
caller may request any alternative sort key via ``sort_by``.

Cost-efficiency formula (consistent with the single-model scorecard)
--------------------------------------------------------------------
When ``model_cost_usd`` is available:

    cost_efficiency = overall_composite × 100 / model_cost_usd

This is ``composite percentage-points per US dollar of test-model cost``.  A
higher value is better.  Judge overhead (``judge_cost_usd``) is explicitly
excluded so that comparisons are fair across runs with and without a live judge.

When cost is absent, ``cost_efficiency`` is ``None``.

Public API
----------
- :func:`build_leaderboard` — compute the leaderboard dict from results.
- :func:`render_leaderboard_markdown` — render the dict as a ranked table.
"""

from __future__ import annotations

from typing import Any

from benchmark.reports.scorecard import DIMENSION_WEIGHTS

# ---------------------------------------------------------------------------
# Dimension abbreviations used in the table header
# ---------------------------------------------------------------------------

#: Short column labels for the six dimensions.
_DIM_LABELS: dict[str, str] = {
    "grounding_accuracy": "GA",
    "insight_quality": "IQ",
    "evidence_linkage": "EL",
    "calibration_limitation_handling": "CLH",
    "consistency": "Con",
    "structure_usability": "SU",
}

#: Canonical dimension order for display.
_DIM_ORDER: tuple[str, ...] = (
    "grounding_accuracy",
    "insight_quality",
    "evidence_linkage",
    "calibration_limitation_handling",
    "consistency",
    "structure_usability",
)

#: Valid sort_by values that the caller may request.
_VALID_SORT_BY: frozenset[str] = frozenset(
    {"composite", "cost_efficiency"} | set(_DIM_LABELS.keys()) | set(_DIM_LABELS.values())
)


def _compute_composite(scores: dict[str, float]) -> float:
    """Compute the weighted composite from dimension scores.

    Reuses :data:`~benchmark.reports.scorecard.DIMENSION_WEIGHTS` so the
    formula is identical to the single-model scorecard.

    Args:
        scores: Mapping from dimension name to score in [0, 1].

    Returns:
        Weighted sum in [0, 1].
    """
    return sum(DIMENSION_WEIGHTS.get(dim, 0.0) * scores.get(dim, 0.0) for dim in _DIM_ORDER)


def build_leaderboard(results: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Build a structured leaderboard dict from a list of (model_id, result) pairs.

    Each *result* dict must conform to ``result_schema.json`` and contain at
    minimum ``overall_scores``, ``per_track_scores``, and ``cost_metrics``.
    The ``model_id`` key in each result is used as the canonical identifier;
    the first element of each tuple overrides it when it differs (e.g. when
    the caller wants to use a display slug).

    Leaderboard entry fields
    ------------------------
    Each entry in the returned ``"entries"`` list contains:

    - ``model_id`` (str): Model identifier (from the result or the tuple key).
    - ``composite`` (float | None): Overall weighted composite in [0, 1].
    - ``per_dimension`` (dict[str, float | None]): All six dimension scores.
    - ``per_track_composites`` (dict[str, float | None]): Track-level composites,
      keyed by string track number (``"1"``–``"5"``).
    - ``model_cost_usd`` (float | None): Headline test-model cost.
    - ``cost_efficiency`` (float | None): composite × 100 / model_cost_usd,
      or ``None`` when cost is absent or zero.

    The entries list is sorted by composite descending (best first) by default.

    Args:
        results: List of ``(model_id, result_dict)`` pairs.  The ``model_id``
            string in the tuple may be any display label; it is written into
            the leaderboard entry verbatim.

    Returns:
        A leaderboard dict with the following top-level keys:

        - ``"entries"`` (list[dict]): Per-model rows sorted by composite desc.
        - ``"model_count"`` (int): Number of models included.
        - ``"dimension_weights"`` (dict[str, float]): The locked project weights.

    Example::

        lb = build_leaderboard([("model-a", result_a), ("model-b", result_b)])
        assert lb["entries"][0]["composite"] >= lb["entries"][1]["composite"]
    """
    entries: list[dict[str, Any]] = []

    for model_key, result in results:
        overall_scores: dict[str, Any] = result.get("overall_scores", {})
        overall_composite: float | None = result.get("overall_composite")

        # Fall back to computing composite from overall_scores if not present.
        if overall_composite is None and overall_scores:
            try:
                overall_composite = _compute_composite(
                    {k: float(v) for k, v in overall_scores.items()}
                )
            except (TypeError, ValueError):
                overall_composite = None

        # Per-dimension scores.
        per_dimension: dict[str, float | None] = {}
        for dim in _DIM_ORDER:
            raw = overall_scores.get(dim)
            per_dimension[dim] = float(raw) if raw is not None else None

        # Per-track composites.
        per_track_composites: dict[str, float | None] = {}
        for tk, tv in result.get("per_track_scores", {}).items():
            per_track_composites[str(tk)] = tv.get("composite")

        # Cost metrics.
        cost_metrics: dict[str, Any] = result.get("cost_metrics") or {}
        model_cost_usd: float | None = cost_metrics.get("model_cost_usd") or cost_metrics.get(
            "total_cost_usd"
        )

        # Cost-efficiency (uses model_cost_usd — test model only, not judge overhead).
        cost_efficiency: float | None = None
        if overall_composite is not None and model_cost_usd is not None and model_cost_usd > 0:
            cost_efficiency = (overall_composite * 100.0) / model_cost_usd

        entries.append(
            {
                "model_id": model_key,
                "composite": overall_composite,
                "per_dimension": per_dimension,
                "per_track_composites": per_track_composites,
                "model_cost_usd": model_cost_usd,
                "cost_efficiency": cost_efficiency,
            }
        )

    # Sort by composite descending (None sorts last).
    entries.sort(
        key=lambda e: (e["composite"] is None, -(e["composite"] or 0.0)),
    )

    return {
        "entries": entries,
        "model_count": len(entries),
        "dimension_weights": dict(DIMENSION_WEIGHTS),
    }


def render_leaderboard_markdown(
    leaderboard: dict[str, Any],
    sort_by: str = "composite",
) -> str:
    """Render a leaderboard dict as a ranked Markdown table.

    Table columns
    -------------
    Rank | Model | Composite | GA | IQ | EL | CLH | Con | SU | Model Cost | Cost-Eff.

    Column notes:

    - **Rank**: 1-based position after applying *sort_by*.
    - **Composite**: Overall weighted composite as a percentage.
    - **GA / IQ / EL / CLH / Con / SU**: The six dimension scores as percentages.
    - **Model Cost**: Test-model inference cost in USD (``—`` when not reported).
      This is the *headline leaderboard metric* — judge overhead is excluded.
    - **Cost-Eff.**: composite pts / US$ (``—`` when cost is absent).

    A caption note below the table confirms that cost = test-model cost only.

    Sort keys
    ---------
    *sort_by* accepts any of:

    - ``"composite"`` (default) — overall weighted composite descending.
    - A dimension name or its abbreviation (``"grounding_accuracy"`` / ``"GA"``,
      etc.) — that dimension's score descending.
    - ``"cost_efficiency"`` — cost-efficiency descending (models without cost
      data sort last).

    Args:
        leaderboard: Dict produced by :func:`build_leaderboard`.
        sort_by: Column to sort by.  Defaults to ``"composite"``.

    Returns:
        Multi-line Markdown string with the ranked table and a caption.

    Raises:
        ValueError: If *sort_by* is not a recognised sort key.
    """
    # Normalise abbreviations to full dimension names.
    _abbrev_to_dim: dict[str, str] = {v: k for k, v in _DIM_LABELS.items()}
    sort_key = _abbrev_to_dim.get(sort_by, sort_by)

    if sort_key not in _VALID_SORT_BY and sort_key not in _DIM_ORDER:
        valid = ", ".join(sorted(_VALID_SORT_BY))
        raise ValueError(f"Unknown sort_by={sort_by!r}.  Valid values: {valid}")

    entries: list[dict[str, Any]] = list(leaderboard.get("entries", []))

    # Re-sort if a non-default sort key was requested.
    if sort_key == "composite":
        pass  # already sorted by build_leaderboard
    elif sort_key == "cost_efficiency":
        entries.sort(key=lambda e: (e["cost_efficiency"] is None, -(e["cost_efficiency"] or 0.0)))
    elif sort_key in _DIM_ORDER:
        entries.sort(
            key=lambda e: (
                e["per_dimension"].get(sort_key) is None,
                -(e["per_dimension"].get(sort_key) or 0.0),
            )
        )

    def _pct(v: float | None) -> str:
        if v is None:
            return "—"
        return f"{v * 100:.1f}%"

    def _cost(v: float | None) -> str:
        if v is None:
            return "—"
        return f"${v:.5f}"

    def _eff(v: float | None) -> str:
        if v is None:
            return "—"
        return f"{v:.1f}"

    header = "| Rank | Model | Composite | GA | IQ | EL | CLH | Con | SU | Model Cost | Cost-Eff. |"
    sep = "|------|-------|-----------|----|----|----|----|-----|----|------------|-----------|"
    rows = [header, sep]

    for rank, entry in enumerate(entries, start=1):
        dim = entry.get("per_dimension", {})
        row = (
            f"| {rank}"
            f" | {entry['model_id']}"
            f" | {_pct(entry.get('composite'))}"
            f" | {_pct(dim.get('grounding_accuracy'))}"
            f" | {_pct(dim.get('insight_quality'))}"
            f" | {_pct(dim.get('evidence_linkage'))}"
            f" | {_pct(dim.get('calibration_limitation_handling'))}"
            f" | {_pct(dim.get('consistency'))}"
            f" | {_pct(dim.get('structure_usability'))}"
            f" | {_cost(entry.get('model_cost_usd'))}"
            f" | {_eff(entry.get('cost_efficiency'))} |"
        )
        rows.append(row)

    table = "\n".join(rows)

    sort_note = f"Sorted by: **{sort_by}** (descending)."
    cost_note = (
        "_Note: Model Cost = test-model inference cost only "
        "(judge overhead excluded). "
        "Cost-Eff. = composite pts × 100 / model cost USD._"
    )
    caption = f"\n{sort_note}  {cost_note}"

    return f"## GRADE Leaderboard\n\n{table}\n{caption}\n"
