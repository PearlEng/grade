"""Tests for benchmark.reports.leaderboard — the multi-model leaderboard renderer.

Coverage:
- build_leaderboard ranks by composite desc; cost-efficiency computed when cost
  present and None when absent.
- sort_by track name, abbreviated track, and cost_efficiency all work.
- render_leaderboard_markdown produces expected column headers.
- Invalid sort_by raises ValueError.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmark.reports.leaderboard import build_leaderboard, render_leaderboard_markdown

# ---------------------------------------------------------------------------
# Helpers — synthetic result dicts
# ---------------------------------------------------------------------------

_DIMENSION_SCORES_TEMPLATE: dict[str, float] = {
    "grounding_accuracy": 0.0,
    "insight_quality": 0.0,
    "evidence_linkage": 0.0,
    "calibration_limitation_handling": 0.0,
    "consistency": 0.0,
    "structure_usability": 0.0,
}

_RUBRIC: dict[str, Any] = {
    "grounding_accuracy": {"weight": 0.35},
    "insight_quality": {"weight": 0.20},
    "evidence_linkage": {"weight": 0.15},
    "calibration_limitation_handling": {"weight": 0.15},
    "consistency": {"weight": 0.10},
    "structure_usability": {"weight": 0.05},
}


def _scores(**kwargs: float) -> dict[str, float]:
    """Return a full DimensionScores dict, defaulting all dims to 0.0."""
    s = dict(_DIMENSION_SCORES_TEMPLATE)
    s.update(kwargs)
    return s


def _make_result(
    model_id: str,
    overall_composite: float,
    scores: dict[str, float] | None = None,
    model_cost_usd: float | None = None,
    track_composites: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a minimal result_schema-shaped dict for leaderboard testing."""
    if scores is None:
        scores = _scores()
    per_track: dict[str, Any] = {}
    if track_composites:
        for tk, comp in track_composites.items():
            per_track[tk] = {"track": int(tk), "task_count": 1, "scores": scores, "composite": comp}

    cost_metrics: dict[str, Any] = {
        "model_cost_usd": model_cost_usd,
        "total_cost_usd": model_cost_usd,
        "judge_cost_usd": None,
        "total_eval_cost_usd": model_cost_usd,
        "cost_available": model_cost_usd is not None,
        "cost_partial": False,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "tokens_available": False,
        "mean_latency_ms": 1.0,
        "p50_latency_ms": 1.0,
        "max_latency_ms": 1.0,
        "per_task_cost": {},
    }

    return {
        "result_id": f"r-{model_id}",
        "model_id": model_id,
        "grade_version": "0.1.0",
        "scored_at_utc": "2026-01-01T00:00:00Z",
        "run_count": 1,
        "task_count": 1,
        "overall_scores": scores,
        "overall_composite": overall_composite,
        "per_track_scores": per_track,
        "per_pack_scores": {},
        "per_task_scores": [
            {
                "task_id": "T-001",
                "track": 1,
                "pack_id": None,
                "run_count": 1,
                "scores": scores,
                "composite": overall_composite,
                "scorer_flags": None,
            }
        ],
        "cost_metrics": cost_metrics,
    }


# ---------------------------------------------------------------------------
# Tests: build_leaderboard
# ---------------------------------------------------------------------------


class TestBuildLeaderboard:
    """build_leaderboard ranks entries correctly."""

    def test_sorted_by_composite_descending(self) -> None:
        """Entries must be sorted by composite descending (best first)."""
        results = [
            ("model-low", _make_result("model-low", 0.50)),
            ("model-high", _make_result("model-high", 0.90)),
            ("model-mid", _make_result("model-mid", 0.70)),
        ]
        lb = build_leaderboard(results)
        composites = [e["composite"] for e in lb["entries"]]
        assert composites == sorted(composites, reverse=True)
        assert lb["entries"][0]["model_id"] == "model-high"

    def test_model_count_correct(self) -> None:
        """model_count must equal the number of input pairs."""
        results = [
            ("a", _make_result("a", 0.8)),
            ("b", _make_result("b", 0.6)),
            ("c", _make_result("c", 0.7)),
        ]
        lb = build_leaderboard(results)
        assert lb["model_count"] == 3
        assert len(lb["entries"]) == 3

    def test_cost_efficiency_computed_when_cost_present(self) -> None:
        """cost_efficiency = composite × 100 / model_cost_usd when cost available."""
        composite = 0.80
        cost = 0.04
        results = [("m", _make_result("m", composite, model_cost_usd=cost))]
        lb = build_leaderboard(results)
        expected = composite * 100.0 / cost
        assert lb["entries"][0]["cost_efficiency"] == pytest.approx(expected)

    def test_cost_efficiency_none_when_no_cost(self) -> None:
        """cost_efficiency must be None when model_cost_usd is absent."""
        results = [("m", _make_result("m", 0.75, model_cost_usd=None))]
        lb = build_leaderboard(results)
        assert lb["entries"][0]["cost_efficiency"] is None

    def test_cost_efficiency_none_when_zero_cost(self) -> None:
        """cost_efficiency must be None when model_cost_usd is zero (avoids div/0)."""
        results = [("m", _make_result("m", 0.75, model_cost_usd=0.0))]
        lb = build_leaderboard(results)
        assert lb["entries"][0]["cost_efficiency"] is None

    def test_per_dimension_scores_present(self) -> None:
        """Each entry must contain per_dimension scores for all six dims."""
        from benchmark.reports.scorecard import DIMENSION_NAMES

        results = [("m", _make_result("m", 0.5))]
        lb = build_leaderboard(results)
        entry = lb["entries"][0]
        assert "per_dimension" in entry
        for dim in DIMENSION_NAMES:
            assert dim in entry["per_dimension"]

    def test_per_track_composites_populated(self) -> None:
        """per_track_composites must reflect the result's per_track_scores."""
        results = [
            (
                "m",
                _make_result("m", 0.6, track_composites={"1": 0.6, "3": 0.55}),
            )
        ]
        lb = build_leaderboard(results)
        ptc = lb["entries"][0]["per_track_composites"]
        assert ptc["1"] == pytest.approx(0.6)
        assert ptc["3"] == pytest.approx(0.55)

    def test_dimension_weights_in_output(self) -> None:
        """dimension_weights must be present and sum to 1.0."""
        lb = build_leaderboard([("m", _make_result("m", 0.5))])
        weights = lb["dimension_weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_empty_results_returns_empty_entries(self) -> None:
        """Empty input produces a leaderboard with zero entries."""
        lb = build_leaderboard([])
        assert lb["entries"] == []
        assert lb["model_count"] == 0

    def test_model_id_from_tuple_key_not_result(self) -> None:
        """The tuple model_id string is used, not result['model_id']."""
        result = _make_result("inner-id", 0.5)
        lb = build_leaderboard([("outer-display-id", result)])
        assert lb["entries"][0]["model_id"] == "outer-display-id"


# ---------------------------------------------------------------------------
# Tests: render_leaderboard_markdown
# ---------------------------------------------------------------------------


class TestRenderLeaderboardMarkdown:
    """render_leaderboard_markdown produces correct Markdown output."""

    def _make_lb(
        self,
        n: int = 3,
        costs: list[float | None] | None = None,
    ) -> dict[str, Any]:
        """Build a leaderboard with *n* synthetic entries."""
        if costs is None:
            costs = [None] * n
        results = [
            (f"model-{i}", _make_result(f"model-{i}", 0.9 - i * 0.1, model_cost_usd=costs[i]))
            for i in range(n)
        ]
        return build_leaderboard(results)

    def test_contains_expected_headers(self) -> None:
        """Markdown table must contain all column headers."""
        lb = self._make_lb(2)
        md = render_leaderboard_markdown(lb)
        assert "Rank" in md
        assert "Model" in md
        assert "Composite" in md
        assert "GA" in md
        assert "IQ" in md
        assert "EL" in md
        assert "CLH" in md
        assert "Con" in md
        assert "SU" in md
        assert "Model Cost" in md
        assert "Cost-Eff." in md

    def test_rank_column_starts_at_1(self) -> None:
        """First data row must have rank 1."""
        lb = self._make_lb(2)
        md = render_leaderboard_markdown(lb)
        lines = [ln for ln in md.splitlines() if "|" in ln and "---" not in ln and "Rank" not in ln]
        first_row = lines[0] if lines else ""
        assert first_row.strip().startswith("| 1")

    def test_cost_dash_when_no_cost(self) -> None:
        """Model Cost and Cost-Eff. columns show '—' when cost is absent."""
        lb = self._make_lb(1, costs=[None])
        md = render_leaderboard_markdown(lb)
        # There should be at least one '—' in the data rows.
        data_rows = [
            ln for ln in md.splitlines() if "|" in ln and "---" not in ln and "Rank" not in ln
        ]
        assert any("—" in row for row in data_rows)

    def test_cost_efficiency_shown_when_cost_present(self) -> None:
        """Cost-Eff. column shows a numeric value when cost is available."""
        lb = self._make_lb(1, costs=[0.05])
        md = render_leaderboard_markdown(lb)
        data_rows = [
            ln for ln in md.splitlines() if "|" in ln and "---" not in ln and "Rank" not in ln
        ]
        # Last cell (before trailing |) must not be '—'.
        last_cells = [row.rstrip("|").split("|")[-1].strip() for row in data_rows]
        assert all(c != "—" for c in last_cells)

    def test_caption_mentions_test_model_cost(self) -> None:
        """The caption must note that cost = test-model cost only."""
        lb = self._make_lb(1)
        md = render_leaderboard_markdown(lb)
        assert "test-model" in md.lower() or "model cost" in md.lower()

    def test_sort_by_composite_default(self) -> None:
        """Default sort (composite) produces highest composite in first row."""
        lb = self._make_lb(3)
        md = render_leaderboard_markdown(lb)
        # The first model-id in the table should be the one with the highest composite.
        top_model = lb["entries"][0]["model_id"]
        data_rows = [
            ln for ln in md.splitlines() if "|" in ln and "---" not in ln and "Rank" not in ln
        ]
        assert top_model in data_rows[0]

    def test_sort_by_cost_efficiency(self) -> None:
        """sort_by='cost_efficiency' re-sorts the table."""
        # Give model-1 higher composite but lower cost (so higher efficiency)
        # vs model-0 which has lower composite but same-ish cost.
        results = [
            ("model-0", _make_result("model-0", 0.80, model_cost_usd=0.10)),
            ("model-1", _make_result("model-1", 0.60, model_cost_usd=0.01)),
        ]
        lb = build_leaderboard(results)
        md = render_leaderboard_markdown(lb, sort_by="cost_efficiency")
        data_rows = [
            ln for ln in md.splitlines() if "|" in ln and "---" not in ln and "Rank" not in ln
        ]
        # model-1 has efficiency = 0.60*100/0.01 = 6000; model-0 = 0.80*100/0.10 = 800
        assert "model-1" in data_rows[0]

    def test_sort_by_dimension_abbreviation(self) -> None:
        """sort_by='GA' (abbreviation) sorts by grounding_accuracy."""
        results = [
            ("low-ga", _make_result("low-ga", 0.7, scores=_scores(grounding_accuracy=0.3))),
            ("high-ga", _make_result("high-ga", 0.5, scores=_scores(grounding_accuracy=0.9))),
        ]
        lb = build_leaderboard(results)
        md = render_leaderboard_markdown(lb, sort_by="GA")
        data_rows = [
            ln for ln in md.splitlines() if "|" in ln and "---" not in ln and "Rank" not in ln
        ]
        assert "high-ga" in data_rows[0]

    def test_sort_by_dimension_full_name(self) -> None:
        """sort_by='grounding_accuracy' (full name) also works."""
        results = [
            ("low-ga", _make_result("low-ga", 0.7, scores=_scores(grounding_accuracy=0.3))),
            ("high-ga", _make_result("high-ga", 0.5, scores=_scores(grounding_accuracy=0.9))),
        ]
        lb = build_leaderboard(results)
        md = render_leaderboard_markdown(lb, sort_by="grounding_accuracy")
        data_rows = [
            ln for ln in md.splitlines() if "|" in ln and "---" not in ln and "Rank" not in ln
        ]
        assert "high-ga" in data_rows[0]

    def test_invalid_sort_by_raises_value_error(self) -> None:
        """An unrecognised sort_by must raise ValueError."""
        lb = self._make_lb(1)
        with pytest.raises(ValueError, match="sort_by"):
            render_leaderboard_markdown(lb, sort_by="not_a_valid_key")
