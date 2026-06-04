r"""C9 — Scorecard report generator for the GRADE benchmark.

This module aggregates per-task dimension scores (produced by C1–C4 scorers)
into per-track and per-pack composites, computes the weighted overall score
using the locked GRADE dimension weights, and renders two output artefacts:

1. A Markdown report rendered from ``templates/scorecard.md.j2`` using Python's
   built-in :meth:`str.format_map` (no external template engine required).
2. A structured ``results.json`` dict that passes :func:`~benchmark.schemas.validate_result`.

Baseline-diff mode
------------------
Pass a *baseline* result to :func:`generate_scorecard_report` (or via the CLI)
to include a delta column in the rendered report showing score changes against a
reference model run.

CLI usage
---------
::

    python -m benchmark.reports.scorecard \
        --results-dir out/<model>/ \
        --baseline out/<other_model>/ \
        --out reports/

Public API
----------
- :data:`DIMENSION_WEIGHTS` — locked per-dimension weights (Grounding 0.35, ...)
- :func:`compute_composite` — apply locked weights to a DimensionScores dict
- :func:`aggregate_dimension_scores` — mean over a list of DimensionScores dicts
- :func:`render_markdown` — render the Markdown scorecard from a result dict
- :func:`generate_scorecard_report` — top-level entry-point
- :class:`ScorecardReport` — dataclass holding both rendered artefacts
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.schemas import validate_result

# ---------------------------------------------------------------------------
# Constants — locked dimension weights (Grounding 35 / Insight 20 / Evidence 15 /
# Calibration 15 / Consistency 10 / Structure 5)
# ---------------------------------------------------------------------------

#: Canonical dimension names in order (matches ``result_schema.json`` / ``DimensionScores``).
DIMENSION_NAMES: tuple[str, ...] = (
    "grounding_accuracy",
    "insight_quality",
    "evidence_linkage",
    "calibration_limitation_handling",
    "consistency",
    "structure_usability",
)

#: Human-readable labels for each dimension, keyed by dimension name.
DIMENSION_LABELS: dict[str, str] = {
    "grounding_accuracy": "Grounding Accuracy",
    "insight_quality": "Insight Quality",
    "evidence_linkage": "Evidence Linkage",
    "calibration_limitation_handling": "Calibration & Limitation Handling",
    "consistency": "Consistency",
    "structure_usability": "Structure & Usability",
}

#: Locked project-level weights (must sum to 1.0).
DIMENSION_WEIGHTS: dict[str, float] = {
    "grounding_accuracy": 0.35,
    "insight_quality": 0.20,
    "evidence_linkage": 0.15,
    "calibration_limitation_handling": 0.15,
    "consistency": 0.10,
    "structure_usability": 0.05,
}

#: Track number → human-readable name mapping.
TRACK_NAMES: dict[int, str] = {
    1: "Grounded Retrieval & Computation",
    2: "Snapshot & Trends Analysis",
    3: "Coaching Recommendations",
    4: "Equity Subgroup Interpretation",
    5: "Effectiveness Research Synthesis",
}

#: Pack identifier → human-readable label.
PACK_LABELS: dict[str, str] = {
    "pack_operations": "Operations Pack",
    "pack_outcomes": "Outcomes Pack",
    "pack_equity_research": "Equity Research Pack",
}

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "scorecard.md.j2"

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScorecardReport:
    """Container for both rendered report artefacts.

    Attributes:
        markdown: The fully rendered Markdown scorecard string.
        result_json: A validated ``result_schema.json``-conformant dict that
            can be serialised to ``results.json`` and consumed by E2/E3.
    """

    markdown: str
    result_json: dict[str, Any]


# ---------------------------------------------------------------------------
# Score computation helpers
# ---------------------------------------------------------------------------


def compute_composite(scores: dict[str, float]) -> float:
    """Apply locked GRADE dimension weights to a set of dimension scores.

    Args:
        scores: Dict mapping each of the six dimension names to a float in
            ``[0, 1]``.  All six dimensions must be present.

    Returns:
        Weighted composite score in ``[0, 1]``.

    Raises:
        KeyError: If any dimension is absent from *scores*.

    Example::

        from benchmark.reports.scorecard import compute_composite, DIMENSION_WEIGHTS
        composite = compute_composite({"grounding_accuracy": 0.9, ...})
    """
    return sum(DIMENSION_WEIGHTS[dim] * scores[dim] for dim in DIMENSION_NAMES)


def aggregate_dimension_scores(
    scores_list: list[dict[str, float]],
) -> dict[str, float]:
    """Compute the mean of each dimension across a list of score dicts.

    Args:
        scores_list: Non-empty list of dimension score dicts, each with all six
            GRADE dimension keys mapping to floats in ``[0, 1]``.

    Returns:
        A single dict with the mean score for each dimension.

    Raises:
        ValueError: If *scores_list* is empty.

    Example::

        means = aggregate_dimension_scores([task["scores"] for task in tasks])
    """
    if not scores_list:
        raise ValueError("scores_list must be non-empty")
    n = len(scores_list)
    return {dim: sum(s[dim] for s in scores_list) / n for dim in DIMENSION_NAMES}


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    """Format a score value as a percentage string, or ``'—'`` if *None*.

    Args:
        value: Float in ``[0, 1]`` or ``None``.

    Returns:
        Formatted string such as ``'85.0%'`` or ``'—'``.
    """
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _delta_str(new: float | None, old: float | None) -> str:
    """Format a score delta for baseline-diff mode.

    Args:
        new: New (current) score or ``None``.
        old: Baseline score or ``None``.

    Returns:
        Delta string such as ``'+3.2 pp'``, ``'-1.5 pp'``, or ``'—'`` when
        either value is absent.
    """
    if new is None or old is None:
        return "—"
    delta = (new - old) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f} pp"


def _build_per_track_table(
    per_track_scores: dict[str, Any],
    baseline_track: dict[str, Any] | None = None,
) -> str:
    """Render the per-track summary Markdown table.

    Args:
        per_track_scores: The ``per_track_scores`` dict from a result object.
        baseline_track: Optional baseline ``per_track_scores`` for diff mode.

    Returns:
        Multi-line Markdown table string.
    """
    diff_mode = baseline_track is not None
    header = (
        "| Track | Name | Tasks | Composite |"
        + (" vs Baseline |" if diff_mode else "")
        + " GA | IQ | EL | CLH | Con | SU |"
    )
    sep = (
        "|-------|------|-------|-----------|"
        + ("-----------:|" if diff_mode else "")
        + "----|----|----|-----|-----|-----|"
    )
    rows = [header, sep]

    for key in sorted(per_track_scores.keys(), key=int):
        tr = per_track_scores[key]
        sc = tr["scores"]
        composite = tr.get("composite")
        name = tr.get("track_name") or TRACK_NAMES.get(tr["track"], f"Track {tr['track']}")

        diff_col = ""
        if diff_mode and baseline_track and key in baseline_track:
            btr = baseline_track[key]
            b_comp = btr.get("composite")
            diff_col = f" {_delta_str(composite, b_comp)} |"

        row = (
            f"| {tr['track']} | {name} | {tr['task_count']} | {_fmt(composite)} |"
            f"{diff_col}"
            f" {_fmt(sc['grounding_accuracy'])} |"
            f" {_fmt(sc['insight_quality'])} |"
            f" {_fmt(sc['evidence_linkage'])} |"
            f" {_fmt(sc['calibration_limitation_handling'])} |"
            f" {_fmt(sc['consistency'])} |"
            f" {_fmt(sc['structure_usability'])} |"
        )
        rows.append(row)

    return "\n".join(rows)


def _build_per_pack_table(
    per_pack_scores: dict[str, Any],
    baseline_pack: dict[str, Any] | None = None,
) -> str:
    """Render the per-pack summary Markdown table.

    Args:
        per_pack_scores: The ``per_pack_scores`` dict from a result object.
        baseline_pack: Optional baseline ``per_pack_scores`` for diff mode.

    Returns:
        Multi-line Markdown table string.
    """
    diff_mode = baseline_pack is not None
    header = (
        "| Pack | Tasks | Composite |"
        + (" vs Baseline |" if diff_mode else "")
        + " GA | IQ | EL | CLH | Con | SU |"
    )
    sep = (
        "|------|-------|-----------|"
        + ("-----------:|" if diff_mode else "")
        + "----|----|----|-----|-----|-----|"
    )
    rows = [header, sep]

    for pack_id in sorted(per_pack_scores.keys()):
        pk = per_pack_scores[pack_id]
        sc = pk["scores"]
        composite = pk.get("composite")
        label = PACK_LABELS.get(pack_id, pack_id)

        diff_col = ""
        if diff_mode and baseline_pack and pack_id in baseline_pack:
            bpk = baseline_pack[pack_id]
            b_comp = bpk.get("composite")
            diff_col = f" {_delta_str(composite, b_comp)} |"

        row = (
            f"| {label} | {pk['task_count']} | {_fmt(composite)} |"
            f"{diff_col}"
            f" {_fmt(sc['grounding_accuracy'])} |"
            f" {_fmt(sc['insight_quality'])} |"
            f" {_fmt(sc['evidence_linkage'])} |"
            f" {_fmt(sc['calibration_limitation_handling'])} |"
            f" {_fmt(sc['consistency'])} |"
            f" {_fmt(sc['structure_usability'])} |"
        )
        rows.append(row)

    return "\n".join(rows)


def _build_cost_speed_section(cost_metrics: dict[str, Any] | None) -> str:
    """Render the Cost & Speed Markdown section from a ``cost_metrics`` dict.

    When *cost_metrics* is ``None`` or empty (e.g. result produced before this
    feature was added), a brief "not available" notice is returned so the
    section is still present and parseable in old reports.

    Layout
    ~~~~~~
    The section **leads with the test-model cost** (``model_cost_usd``) — the
    headline leaderboard figure that reflects the model under test, not
    GRADE's evaluation infrastructure.  Judge overhead (``judge_cost_usd``) and
    the combined ``total_eval_cost_usd`` are shown as a clearly-labeled
    secondary "Evaluation overhead (judge)" block.

    Cost-efficiency formula
    ~~~~~~~~~~~~~~~~~~~~~~~
    When model cost is available:

        cost_efficiency = composite_points_per_dollar
                        = overall_composite * 100 / model_cost_usd

    This answers "how many composite percentage-points did we buy per US
    dollar of test-model cost?".  A higher number is better.  Leaderboard
    comparisons ALWAYS use ``model_cost_usd``, not ``total_eval_cost_usd``.

    When cost is absent but token counts are available (tokens_available=True),
    a token-based proxy is used instead:

        token_efficiency = overall_composite * 100 / (total_tokens / 1000)

    This answers "composite points per 1 k tokens" — useful for comparing
    models when provider pricing is not exposed.

    Both figures are deterministic given the same inputs.

    Args:
        cost_metrics: The ``cost_metrics`` sub-dict from a result object, or
            ``None`` if the result predates cost tracking.

    Returns:
        Multi-line Markdown string describing cost, tokens, latency, and
        efficiency.  Never raises.
    """
    if not cost_metrics:
        return "_Cost & speed data not available for this result._\n"

    lines: list[str] = []

    # --- Test-model cost (HEADLINE) ---
    cost_available: bool = cost_metrics.get("cost_available", False)
    # Prefer the explicit model_cost_usd; fall back to total_cost_usd for
    # scorecards produced by older GRADE versions that lack the new field.
    model_cost: float | None = cost_metrics.get("model_cost_usd") or cost_metrics.get(
        "total_cost_usd"
    )
    cost_partial: bool = cost_metrics.get("cost_partial", False)

    if cost_available and model_cost is not None:
        cost_str = f"${model_cost:.5f}"
        if cost_partial:
            cost_str += " _(partial — some calls did not report cost)_"
        lines.append(f"**Model Cost (test model):** {cost_str}")
    else:
        lines.append("**Model Cost (test model):** not reported")

    # --- Judge / eval overhead (secondary) ---
    judge_cost: float | None = cost_metrics.get("judge_cost_usd")
    total_eval_cost: float | None = cost_metrics.get("total_eval_cost_usd")

    if judge_cost is not None:
        lines.append(
            f"**Evaluation overhead (judge):** ${judge_cost:.5f}"
            f"  _(not used for leaderboard comparisons)_"
        )
        if total_eval_cost is not None:
            lines.append(f"**Total Eval Cost (model + judge):** ${total_eval_cost:.5f}")
    else:
        lines.append(
            "**Evaluation overhead (judge):** not reported"
            "  _(not used for leaderboard comparisons)_"
        )

    lines.append(
        "_Note: leaderboard comparisons use test-model cost only, not evaluation overhead._"
    )

    # --- Tokens ---
    tokens_available: bool = cost_metrics.get("tokens_available", False)
    total_tokens: int = cost_metrics.get("total_tokens", 0)
    prompt_tokens: int = cost_metrics.get("total_prompt_tokens", 0)
    completion_tokens: int = cost_metrics.get("total_completion_tokens", 0)

    if tokens_available:
        lines.append(
            f"**Total Tokens:** {total_tokens:,}"
            f"  _(prompt: {prompt_tokens:,}, completion: {completion_tokens:,})_"
        )
    else:
        lines.append("**Total Tokens:** not reported")

    # --- Latency ---
    mean_lat: float | None = cost_metrics.get("mean_latency_ms")
    p50_lat: float | None = cost_metrics.get("p50_latency_ms")
    max_lat: float | None = cost_metrics.get("max_latency_ms")

    if mean_lat is not None:
        lat_parts = [f"mean {mean_lat:.0f} ms"]
        if p50_lat is not None:
            lat_parts.append(f"p50 {p50_lat:.0f} ms")
        if max_lat is not None:
            lat_parts.append(f"max {max_lat:.0f} ms")
        lines.append(f"**Latency:** {', '.join(lat_parts)}")
    else:
        lines.append("**Latency:** not available")

    return "\n".join(lines) + "\n"


def _build_cost_efficiency_note(
    result: dict[str, Any],
) -> str:
    """Render a cost-efficiency figure for the Cost & Speed section.

    The figure is appended after the main cost/speed table rows.  It is
    separated so callers can include/exclude it independently.

    Formula (see :func:`_build_cost_speed_section` for rationale):

    - **cost_efficiency** (when model cost is available):
      ``overall_composite × 100 / model_cost_usd``

      This uses ``model_cost_usd`` (the test-model headline figure), NOT
      ``total_eval_cost_usd``, so leaderboard comparisons are fair across
      runs with and without a live judge.

    - **token_efficiency** (when tokens available but cost absent):
      ``overall_composite × 100 / (total_tokens / 1000)``

    Args:
        result: Full result dict containing ``overall_composite`` and
            ``cost_metrics``.

    Returns:
        One-line Markdown string with the efficiency figure, or empty string
        if no efficiency figure can be computed.
    """
    composite: float | None = result.get("overall_composite")
    cost_metrics: dict[str, Any] | None = result.get("cost_metrics")

    if composite is None or not cost_metrics:
        return ""

    cost_available: bool = cost_metrics.get("cost_available", False)
    # Use model_cost_usd as the headline; fall back to total_cost_usd for older results.
    model_cost: float | None = cost_metrics.get("model_cost_usd") or cost_metrics.get(
        "total_cost_usd"
    )
    tokens_available: bool = cost_metrics.get("tokens_available", False)
    total_tokens: int = cost_metrics.get("total_tokens", 0)

    if cost_available and model_cost and model_cost > 0:
        efficiency = (composite * 100) / model_cost
        return (
            f"\n**Cost-Efficiency (model cost):** {efficiency:.1f} composite pts / US$"
            f"  _(based on test-model cost; leaderboard metric)_\n"
        )

    if tokens_available and total_tokens > 0:
        efficiency_per_ktok = (composite * 100) / (total_tokens / 1000)
        return (
            f"\n**Token-Efficiency:** {efficiency_per_ktok:.2f} composite pts / 1k tokens"
            f"  _(cost not reported; token proxy used)_\n"
        )

    return ""


def _build_per_task_table(
    per_task_scores: list[dict[str, Any]],
    baseline_tasks: list[dict[str, Any]] | None = None,
) -> str:
    """Render the per-task detail Markdown table.

    Args:
        per_task_scores: The ``per_task_scores`` array from a result object.
        baseline_tasks: Optional baseline ``per_task_scores`` list for diff mode.

    Returns:
        Multi-line Markdown table string.
    """
    diff_mode = baseline_tasks is not None
    # Build a lookup from task_id → composite for baseline diff.
    baseline_map: dict[str, float | None] = {}
    if diff_mode and baseline_tasks:
        baseline_map = {t["task_id"]: t.get("composite") for t in baseline_tasks}

    header = (
        "| Task ID | Track | Pack | Composite |"
        + (" vs Baseline |" if diff_mode else "")
        + " GA | IQ | EL | CLH | Con | SU |"
    )
    sep = (
        "|---------|-------|------|-----------|"
        + ("-----------:|" if diff_mode else "")
        + "----|----|----|-----|-----|-----|"
    )
    rows = [header, sep]

    for task in per_task_scores:
        sc = task["scores"]
        composite = task.get("composite")
        pack = task.get("pack_id") or "—"

        diff_col = ""
        if diff_mode:
            b_comp = baseline_map.get(task["task_id"])
            diff_col = f" {_delta_str(composite, b_comp)} |"

        row = (
            f"| {task['task_id']} | {task['track']} | {pack} | {_fmt(composite)} |"
            f"{diff_col}"
            f" {_fmt(sc['grounding_accuracy'])} |"
            f" {_fmt(sc['insight_quality'])} |"
            f" {_fmt(sc['evidence_linkage'])} |"
            f" {_fmt(sc['calibration_limitation_handling'])} |"
            f" {_fmt(sc['consistency'])} |"
            f" {_fmt(sc['structure_usability'])} |"
        )
        rows.append(row)

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def render_markdown(
    result: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> str:
    """Render a human-readable Markdown scorecard from a validated result dict.

    The template is loaded from ``templates/scorecard.md.j2`` and rendered using
    Python's :meth:`str.format_map` — no external template engine is needed.
    The rendered body contains no wall-clock timestamps, making it deterministic
    and safe to use in snapshot tests.

    Args:
        result: A ``result_schema.json``-conformant dict that has already been
            validated via :func:`~benchmark.schemas.validate_result`.
        baseline: Optional second result dict for baseline-diff mode.  When
            provided, delta columns are added to the per-track, per-pack, and
            per-task tables.

    Returns:
        Fully rendered Markdown string.

    Example::

        from benchmark.reports.scorecard import render_markdown, generate_scorecard_report
        report = generate_scorecard_report(result)
        print(report.markdown)
    """
    overall = result["overall_scores"]
    overall_composite = result.get("overall_composite")

    # Build diff args for tables
    baseline_track = baseline.get("per_track_scores") if baseline else None
    baseline_pack = baseline.get("per_pack_scores") if baseline else None
    baseline_tasks = baseline.get("per_task_scores") if baseline else None

    per_track_table = _build_per_track_table(result["per_track_scores"], baseline_track)
    per_pack_table = _build_per_pack_table(result["per_pack_scores"], baseline_pack)
    per_task_table = _build_per_task_table(result["per_task_scores"], baseline_tasks)

    # Diff header section
    diff_section = ""
    if baseline is not None:
        diff_section = (
            "\n---\n\n"
            f"## Baseline Comparison\n\n"
            f"Baseline model: **{baseline.get('model_id', 'unknown')}**  "
            f"(result `{baseline.get('result_id', 'unknown')}`)\n\n"
            f"Delta = current − baseline.  "
            f"Positive values (pp) indicate improvement.\n"
        )

    cost_speed_section = _build_cost_speed_section(result.get("cost_metrics"))
    cost_speed_section += _build_cost_efficiency_note(result)

    template_text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return template_text.format_map(
        {
            "model_id": result["model_id"],
            "result_id": result["result_id"],
            "grade_version": result["grade_version"],
            "scored_at_utc": result["scored_at_utc"],
            "task_count": result.get("task_count") or len(result["per_task_scores"]),
            "run_count": result.get("run_count") or "—",
            "overall_composite": _fmt(overall_composite),
            "overall_grounding_accuracy": _fmt(overall["grounding_accuracy"]),
            "overall_insight_quality": _fmt(overall["insight_quality"]),
            "overall_evidence_linkage": _fmt(overall["evidence_linkage"]),
            "overall_calibration_limitation_handling": _fmt(
                overall["calibration_limitation_handling"]
            ),
            "overall_consistency": _fmt(overall["consistency"]),
            "overall_structure_usability": _fmt(overall["structure_usability"]),
            "per_track_table": per_track_table,
            "per_pack_table": per_pack_table,
            "per_task_table": per_task_table,
            "cost_speed_section": cost_speed_section,
            "diff_section": diff_section,
        }
    )


# ---------------------------------------------------------------------------
# Top-level entry-point
# ---------------------------------------------------------------------------


def generate_scorecard_report(
    result: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> ScorecardReport:
    """Generate both Markdown and structured JSON scorecard artefacts.

    This is the primary public API for C9.  It validates both *result* and the
    optional *baseline* against ``result_schema.json``, then renders the Markdown
    report and returns the validated result dict (suitable for writing as
    ``results.json``).

    Args:
        result: A ``result_schema.json``-conformant dict produced by aggregating
            C1–C4 scorer outputs.  :func:`~benchmark.schemas.validate_result` is
            called before any rendering; a :exc:`jsonschema.ValidationError` is
            raised if the payload is invalid.
        baseline: Optional baseline result dict for diff mode.  If supplied it
            is also validated against ``result_schema.json``.

    Returns:
        :class:`ScorecardReport` with ``markdown`` (str) and ``result_json``
        (dict) artefacts.

    Raises:
        jsonschema.ValidationError: If *result* or *baseline* fail schema
            validation.

    Example::

        from benchmark.reports.scorecard import generate_scorecard_report

        report = generate_scorecard_report(result_dict)
        print(report.markdown)

        with open("reports/results.json", "w") as fh:
            import json
            json.dump(report.result_json, fh, indent=2)
    """
    validate_result(result)
    if baseline is not None:
        validate_result(baseline)

    markdown = render_markdown(result, baseline)
    return ScorecardReport(markdown=markdown, result_json=result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.reports.scorecard",
        description="GRADE C9 — generate a scorecard Markdown report and results.json.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        metavar="DIR",
        help="Directory containing results.json for the target model run.",
    )
    parser.add_argument(
        "--baseline",
        metavar="DIR",
        default=None,
        help="Directory containing results.json for the baseline model (enables diff mode).",
    )
    parser.add_argument(
        "--out",
        default="reports",
        metavar="DIR",
        help="Output directory for scorecard.md and results.json (default: reports/).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point for the C9 scorecard generator.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).

    Returns:
        Exit code (0 on success, non-zero on error).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out)

    result_path = results_dir / "results.json"
    if not result_path.exists():
        print(f"ERROR: {result_path} not found.", file=sys.stderr)
        return 1

    with result_path.open(encoding="utf-8") as fh:
        result: dict[str, Any] = json.load(fh)

    baseline: dict[str, Any] | None = None
    if args.baseline:
        baseline_path = Path(args.baseline) / "results.json"
        if not baseline_path.exists():
            print(f"ERROR: baseline {baseline_path} not found.", file=sys.stderr)
            return 1
        with baseline_path.open(encoding="utf-8") as fh:
            baseline = json.load(fh)

    report = generate_scorecard_report(result, baseline)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.md").write_text(report.markdown, encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(report.result_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Scorecard written to {out_dir}/scorecard.md")
    print(f"Structured JSON written to {out_dir}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
