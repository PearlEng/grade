"""Unit tests for the C9 scorecard report generator.

Coverage:
- :func:`~benchmark.reports.scorecard.compute_composite`: correct weighted sum,
  zero scores, perfect scores, locked weight values.
- :func:`~benchmark.reports.scorecard.aggregate_dimension_scores`: single item,
  multi-item mean, empty raises ValueError.
- :func:`~benchmark.reports.scorecard.render_markdown`: expected sections present,
  values rendered correctly, no wall-clock timestamps in rendered body.
- :func:`~benchmark.reports.scorecard.generate_scorecard_report`: single-result
  generation, baseline-diff mode, schema validation called, invalid input raises.
- Weight constants: verify the six GRADE weights match the locked proposal values.
- :class:`~benchmark.reports.scorecard.ScorecardReport`: result_json passes
  validate_result after round-trip.

No network calls or API keys are required; all tests are pure-function / in-process.
"""

from __future__ import annotations

import copy
import json

import jsonschema
import pytest

from benchmark.reports.scorecard import (
    DIMENSION_NAMES,
    DIMENSION_WEIGHTS,
    ScorecardReport,
    aggregate_dimension_scores,
    compute_composite,
    generate_scorecard_report,
    render_markdown,
)
from benchmark.schemas import validate_result

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_DIM_SCORES_PERFECT: dict[str, float] = {
    "grounding_accuracy": 1.0,
    "insight_quality": 1.0,
    "evidence_linkage": 1.0,
    "calibration_limitation_handling": 1.0,
    "consistency": 1.0,
    "structure_usability": 1.0,
}

_DIM_SCORES_ZERO: dict[str, float] = {
    "grounding_accuracy": 0.0,
    "insight_quality": 0.0,
    "evidence_linkage": 0.0,
    "calibration_limitation_handling": 0.0,
    "consistency": 0.0,
    "structure_usability": 0.0,
}

_DIM_SCORES_PARTIAL: dict[str, float] = {
    "grounding_accuracy": 0.80,
    "insight_quality": 0.60,
    "evidence_linkage": 0.70,
    "calibration_limitation_handling": 0.75,
    "consistency": 0.90,
    "structure_usability": 0.85,
}

#: Minimal valid result dict shared across tests.  Matches result_schema.json.
_VALID_RESULT: dict = {
    "result_id": "test-model-20260601-abc123",
    "model_id": "test/model-a",
    "grade_version": "0.1.0",
    "scored_at_utc": "2026-06-01T14:00:00Z",
    "task_count": 3,
    "run_count": 5,
    "overall_scores": {
        "grounding_accuracy": 0.80,
        "insight_quality": 0.60,
        "evidence_linkage": 0.70,
        "calibration_limitation_handling": 0.75,
        "consistency": 0.90,
        "structure_usability": 0.85,
    },
    "overall_composite": 0.752,
    "per_track_scores": {
        "1": {
            "track": 1,
            "track_name": "Grounded Retrieval & Computation",
            "task_count": 2,
            "scores": {
                "grounding_accuracy": 0.85,
                "insight_quality": 0.65,
                "evidence_linkage": 0.72,
                "calibration_limitation_handling": 0.78,
                "consistency": 0.88,
                "structure_usability": 0.90,
            },
            "composite": 0.787,
        },
        "2": {
            "track": 2,
            "track_name": "Snapshot & Trends Analysis",
            "task_count": 1,
            "scores": {
                "grounding_accuracy": 0.70,
                "insight_quality": 0.50,
                "evidence_linkage": 0.65,
                "calibration_limitation_handling": 0.68,
                "consistency": 0.95,
                "structure_usability": 0.75,
            },
            "composite": 0.691,
        },
    },
    "per_pack_scores": {
        "pack_operations": {
            "pack_id": "pack_operations",
            "task_count": 2,
            "scores": {
                "grounding_accuracy": 0.82,
                "insight_quality": 0.62,
                "evidence_linkage": 0.71,
                "calibration_limitation_handling": 0.76,
                "consistency": 0.91,
                "structure_usability": 0.87,
            },
            "composite": 0.762,
        },
        "pack_outcomes": {
            "pack_id": "pack_outcomes",
            "task_count": 1,
            "scores": {
                "grounding_accuracy": 0.75,
                "insight_quality": 0.55,
                "evidence_linkage": 0.68,
                "calibration_limitation_handling": 0.73,
                "consistency": 0.88,
                "structure_usability": 0.82,
            },
            "composite": 0.725,
        },
    },
    "per_task_scores": [
        {
            "task_id": "T1-OPS-001",
            "track": 1,
            "pack_id": "pack_operations",
            "run_count": 5,
            "scores": {
                "grounding_accuracy": 0.90,
                "insight_quality": 0.70,
                "evidence_linkage": 0.80,
                "calibration_limitation_handling": 0.85,
                "consistency": 0.92,
                "structure_usability": 0.95,
            },
            "composite": 0.856,
            "scorer_flags": None,
        },
        {
            "task_id": "T1-OPS-002",
            "track": 1,
            "pack_id": "pack_operations",
            "run_count": 5,
            "scores": {
                "grounding_accuracy": 0.80,
                "insight_quality": 0.60,
                "evidence_linkage": 0.64,
                "calibration_limitation_handling": 0.71,
                "consistency": 0.84,
                "structure_usability": 0.85,
            },
            "composite": 0.730,
            "scorer_flags": None,
        },
        {
            "task_id": "T2-OUT-001",
            "track": 2,
            "pack_id": "pack_outcomes",
            "run_count": 5,
            "scores": {
                "grounding_accuracy": 0.70,
                "insight_quality": 0.50,
                "evidence_linkage": 0.65,
                "calibration_limitation_handling": 0.68,
                "consistency": 0.95,
                "structure_usability": 0.75,
            },
            "composite": 0.691,
            "scorer_flags": None,
        },
    ],
}

#: A second result used as a baseline for diff-mode tests.
_BASELINE_RESULT: dict = {
    "result_id": "baseline-model-20260601-xyz999",
    "model_id": "test/model-baseline",
    "grade_version": "0.1.0",
    "scored_at_utc": "2026-06-01T10:00:00Z",
    "task_count": 3,
    "run_count": 5,
    "overall_scores": {
        "grounding_accuracy": 0.70,
        "insight_quality": 0.50,
        "evidence_linkage": 0.60,
        "calibration_limitation_handling": 0.65,
        "consistency": 0.80,
        "structure_usability": 0.75,
    },
    "overall_composite": 0.659,
    "per_track_scores": {
        "1": {
            "track": 1,
            "track_name": "Grounded Retrieval & Computation",
            "task_count": 2,
            "scores": {
                "grounding_accuracy": 0.72,
                "insight_quality": 0.52,
                "evidence_linkage": 0.62,
                "calibration_limitation_handling": 0.66,
                "consistency": 0.82,
                "structure_usability": 0.77,
            },
            "composite": 0.674,
        },
        "2": {
            "track": 2,
            "track_name": "Snapshot & Trends Analysis",
            "task_count": 1,
            "scores": {
                "grounding_accuracy": 0.66,
                "insight_quality": 0.46,
                "evidence_linkage": 0.56,
                "calibration_limitation_handling": 0.62,
                "consistency": 0.76,
                "structure_usability": 0.72,
            },
            "composite": 0.638,
        },
    },
    "per_pack_scores": {
        "pack_operations": {
            "pack_id": "pack_operations",
            "task_count": 2,
            "scores": {
                "grounding_accuracy": 0.71,
                "insight_quality": 0.51,
                "evidence_linkage": 0.61,
                "calibration_limitation_handling": 0.65,
                "consistency": 0.81,
                "structure_usability": 0.76,
            },
            "composite": 0.662,
        },
        "pack_outcomes": {
            "pack_id": "pack_outcomes",
            "task_count": 1,
            "scores": {
                "grounding_accuracy": 0.68,
                "insight_quality": 0.48,
                "evidence_linkage": 0.58,
                "calibration_limitation_handling": 0.64,
                "consistency": 0.78,
                "structure_usability": 0.73,
            },
            "composite": 0.648,
        },
    },
    "per_task_scores": [
        {
            "task_id": "T1-OPS-001",
            "track": 1,
            "pack_id": "pack_operations",
            "run_count": 5,
            "scores": {
                "grounding_accuracy": 0.75,
                "insight_quality": 0.55,
                "evidence_linkage": 0.65,
                "calibration_limitation_handling": 0.70,
                "consistency": 0.84,
                "structure_usability": 0.80,
            },
            "composite": 0.714,
            "scorer_flags": None,
        },
        {
            "task_id": "T1-OPS-002",
            "track": 1,
            "pack_id": "pack_operations",
            "run_count": 5,
            "scores": {
                "grounding_accuracy": 0.69,
                "insight_quality": 0.49,
                "evidence_linkage": 0.59,
                "calibration_limitation_handling": 0.62,
                "consistency": 0.80,
                "structure_usability": 0.74,
            },
            "composite": 0.638,
            "scorer_flags": None,
        },
        {
            "task_id": "T2-OUT-001",
            "track": 2,
            "pack_id": "pack_outcomes",
            "run_count": 5,
            "scores": {
                "grounding_accuracy": 0.66,
                "insight_quality": 0.46,
                "evidence_linkage": 0.56,
                "calibration_limitation_handling": 0.62,
                "consistency": 0.76,
                "structure_usability": 0.72,
            },
            "composite": 0.638,
            "scorer_flags": None,
        },
    ],
}


# ---------------------------------------------------------------------------
# DIMENSION_WEIGHTS constant tests
# ---------------------------------------------------------------------------


class TestDimensionWeights:
    """Verify the locked GRADE dimension weights."""

    def test_weights_sum_to_one(self) -> None:
        """The six locked weights must sum exactly to 1.0."""
        total = sum(DIMENSION_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_grounding_accuracy_weight(self) -> None:
        """Grounding Accuracy weight must be 0.35."""
        assert DIMENSION_WEIGHTS["grounding_accuracy"] == pytest.approx(0.35)

    def test_insight_quality_weight(self) -> None:
        """Insight Quality weight must be 0.20."""
        assert DIMENSION_WEIGHTS["insight_quality"] == pytest.approx(0.20)

    def test_evidence_linkage_weight(self) -> None:
        """Evidence Linkage weight must be 0.15."""
        assert DIMENSION_WEIGHTS["evidence_linkage"] == pytest.approx(0.15)

    def test_calibration_weight(self) -> None:
        """Calibration & Limitation Handling weight must be 0.15."""
        assert DIMENSION_WEIGHTS["calibration_limitation_handling"] == pytest.approx(0.15)

    def test_consistency_weight(self) -> None:
        """Consistency weight must be 0.10."""
        assert DIMENSION_WEIGHTS["consistency"] == pytest.approx(0.10)

    def test_structure_usability_weight(self) -> None:
        """Structure & Usability weight must be 0.05."""
        assert DIMENSION_WEIGHTS["structure_usability"] == pytest.approx(0.05)

    def test_all_six_dimensions_present(self) -> None:
        """All six canonical dimension names must be in DIMENSION_WEIGHTS."""
        assert set(DIMENSION_WEIGHTS.keys()) == set(DIMENSION_NAMES)


# ---------------------------------------------------------------------------
# compute_composite tests
# ---------------------------------------------------------------------------


class TestComputeComposite:
    """Tests for :func:`compute_composite`."""

    def test_perfect_scores_give_one(self) -> None:
        """All-1.0 inputs should produce composite == 1.0."""
        result = compute_composite(_DIM_SCORES_PERFECT)
        assert result == pytest.approx(1.0)

    def test_zero_scores_give_zero(self) -> None:
        """All-0.0 inputs should produce composite == 0.0."""
        result = compute_composite(_DIM_SCORES_ZERO)
        assert result == pytest.approx(0.0)

    def test_manual_calculation(self) -> None:
        """Composite must equal the manually computed weighted sum.

        Weights × scores:
          0.35 × 0.80 = 0.280
          0.20 × 0.60 = 0.120
          0.15 × 0.70 = 0.105
          0.15 × 0.75 = 0.1125
          0.10 × 0.90 = 0.090
          0.05 × 0.85 = 0.0425
          Total       = 0.750
        """
        result = compute_composite(_DIM_SCORES_PARTIAL)
        expected = 0.35 * 0.80 + 0.20 * 0.60 + 0.15 * 0.70 + 0.15 * 0.75 + 0.10 * 0.90 + 0.05 * 0.85
        assert result == pytest.approx(expected)

    def test_only_grounding_matters(self) -> None:
        """Score where only grounding_accuracy is 1.0 should give composite == 0.35."""
        scores = dict(_DIM_SCORES_ZERO)
        scores["grounding_accuracy"] = 1.0
        result = compute_composite(scores)
        assert result == pytest.approx(0.35)

    def test_missing_dimension_raises(self) -> None:
        """Missing a dimension should raise KeyError."""
        incomplete = {k: v for k, v in _DIM_SCORES_PERFECT.items() if k != "consistency"}
        with pytest.raises(KeyError):
            compute_composite(incomplete)


# ---------------------------------------------------------------------------
# aggregate_dimension_scores tests
# ---------------------------------------------------------------------------


class TestAggregateDimensionScores:
    """Tests for :func:`aggregate_dimension_scores`."""

    def test_single_item_returns_same(self) -> None:
        """A list with one item should return identical scores."""
        result = aggregate_dimension_scores([_DIM_SCORES_PARTIAL])
        for dim in DIMENSION_NAMES:
            assert result[dim] == pytest.approx(_DIM_SCORES_PARTIAL[dim])

    def test_two_items_mean(self) -> None:
        """Mean of perfect and zero scores should be 0.5 for every dimension."""
        result = aggregate_dimension_scores([_DIM_SCORES_PERFECT, _DIM_SCORES_ZERO])
        for dim in DIMENSION_NAMES:
            assert result[dim] == pytest.approx(0.5)

    def test_multiple_items_mean(self) -> None:
        """Mean of three known sets should match the hand-calculated value."""
        a: dict[str, float] = {**dict.fromkeys(DIMENSION_NAMES, 0.6)}
        b: dict[str, float] = {**dict.fromkeys(DIMENSION_NAMES, 0.8)}
        c: dict[str, float] = {**dict.fromkeys(DIMENSION_NAMES, 1.0)}
        result = aggregate_dimension_scores([a, b, c])
        for dim in DIMENSION_NAMES:
            assert result[dim] == pytest.approx(0.8)

    def test_empty_list_raises(self) -> None:
        """Empty input must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            aggregate_dimension_scores([])

    def test_result_contains_all_dimensions(self) -> None:
        """Result must contain all six dimension keys."""
        result = aggregate_dimension_scores([_DIM_SCORES_PARTIAL])
        assert set(result.keys()) == set(DIMENSION_NAMES)


# ---------------------------------------------------------------------------
# render_markdown tests
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    """Tests for :func:`render_markdown`."""

    def test_contains_model_id(self) -> None:
        """Model ID must appear in the rendered Markdown."""
        md = render_markdown(_VALID_RESULT)
        assert "test/model-a" in md

    def test_contains_result_id(self) -> None:
        """Result ID must appear in the rendered Markdown."""
        md = render_markdown(_VALID_RESULT)
        assert "test-model-20260601-abc123" in md

    def test_contains_grade_version(self) -> None:
        """GRADE version must appear in the rendered Markdown."""
        md = render_markdown(_VALID_RESULT)
        assert "0.1.0" in md

    def test_contains_overall_scores_section(self) -> None:
        """Output must contain an 'Overall Scores' heading."""
        md = render_markdown(_VALID_RESULT)
        assert "## Overall Scores" in md

    def test_contains_per_track_section(self) -> None:
        """Output must contain a 'Per-Track Scores' heading."""
        md = render_markdown(_VALID_RESULT)
        assert "## Per-Track Scores" in md

    def test_contains_per_pack_section(self) -> None:
        """Output must contain a 'Per-Pack Scores' heading."""
        md = render_markdown(_VALID_RESULT)
        assert "## Per-Pack Scores" in md

    def test_contains_per_task_section(self) -> None:
        """Output must contain a 'Per-Task Scores' heading."""
        md = render_markdown(_VALID_RESULT)
        assert "## Per-Task Scores" in md

    def test_overall_grounding_accuracy_rendered(self) -> None:
        """Overall Grounding Accuracy score (80.0%) must appear in the output."""
        md = render_markdown(_VALID_RESULT)
        assert "80.0%" in md

    def test_task_id_in_per_task_table(self) -> None:
        """Task IDs must appear in the per-task table."""
        md = render_markdown(_VALID_RESULT)
        assert "T1-OPS-001" in md
        assert "T2-OUT-001" in md

    def test_no_diff_section_without_baseline(self) -> None:
        """Without a baseline, the Baseline Comparison section must not appear."""
        md = render_markdown(_VALID_RESULT)
        assert "## Baseline Comparison" not in md

    def test_deterministic_output(self) -> None:
        """Two calls with identical input must produce identical output."""
        md1 = render_markdown(_VALID_RESULT)
        md2 = render_markdown(_VALID_RESULT)
        assert md1 == md2

    def test_scored_at_utc_is_from_result_not_wall_clock(self) -> None:
        """The scored_at_utc value in the report must come from the result dict."""
        md = render_markdown(_VALID_RESULT)
        assert "2026-06-01T14:00:00Z" in md

    def test_track_name_in_per_track_table(self) -> None:
        """Track name from the result should appear in the per-track table."""
        md = render_markdown(_VALID_RESULT)
        assert "Grounded Retrieval" in md

    def test_pack_label_in_per_pack_table(self) -> None:
        """Human-readable pack label should appear in the per-pack table."""
        md = render_markdown(_VALID_RESULT)
        assert "Operations Pack" in md


# ---------------------------------------------------------------------------
# baseline-diff mode tests
# ---------------------------------------------------------------------------


class TestRenderMarkdownDiff:
    """Tests for baseline-diff mode in :func:`render_markdown`."""

    def test_diff_section_present_with_baseline(self) -> None:
        """With a baseline, '## Baseline Comparison' must appear."""
        md = render_markdown(_VALID_RESULT, baseline=_BASELINE_RESULT)
        assert "## Baseline Comparison" in md

    def test_diff_section_contains_baseline_model_id(self) -> None:
        """The diff section must name the baseline model."""
        md = render_markdown(_VALID_RESULT, baseline=_BASELINE_RESULT)
        assert "test/model-baseline" in md

    def test_diff_columns_present_in_per_track_table(self) -> None:
        """With a baseline, the per-track table must have a 'vs Baseline' column."""
        md = render_markdown(_VALID_RESULT, baseline=_BASELINE_RESULT)
        assert "vs Baseline" in md

    def test_positive_delta_shown_with_plus_sign(self) -> None:
        """A positive delta (improvement) must be prefixed with '+'."""
        md = render_markdown(_VALID_RESULT, baseline=_BASELINE_RESULT)
        assert "+" in md

    def test_diff_section_absent_when_no_baseline(self) -> None:
        """Without baseline, diff markers must not appear in output."""
        md = render_markdown(_VALID_RESULT)
        assert "vs Baseline" not in md


# ---------------------------------------------------------------------------
# generate_scorecard_report tests
# ---------------------------------------------------------------------------


class TestGenerateScorecardReport:
    """Tests for :func:`generate_scorecard_report`."""

    def test_returns_scorecard_report_instance(self) -> None:
        """Return type must be :class:`ScorecardReport`."""
        report = generate_scorecard_report(_VALID_RESULT)
        assert isinstance(report, ScorecardReport)

    def test_markdown_is_non_empty_string(self) -> None:
        """The markdown field must be a non-empty string."""
        report = generate_scorecard_report(_VALID_RESULT)
        assert isinstance(report.markdown, str)
        assert len(report.markdown) > 0

    def test_result_json_matches_input(self) -> None:
        """The result_json field must be the same object as the input result."""
        report = generate_scorecard_report(_VALID_RESULT)
        assert report.result_json is _VALID_RESULT

    def test_result_json_passes_validate_result(self) -> None:
        """result_json must satisfy validate_result (schema round-trip)."""
        report = generate_scorecard_report(_VALID_RESULT)
        validate_result(report.result_json)  # should not raise

    def test_result_json_serialisable(self) -> None:
        """result_json must be JSON-serialisable without errors."""
        report = generate_scorecard_report(_VALID_RESULT)
        serialised = json.dumps(report.result_json)
        roundtripped = json.loads(serialised)
        assert roundtripped["model_id"] == _VALID_RESULT["model_id"]

    def test_invalid_result_raises_validation_error(self) -> None:
        """An invalid result must raise jsonschema.ValidationError."""
        bad = copy.deepcopy(_VALID_RESULT)
        del bad["overall_scores"]
        with pytest.raises(jsonschema.ValidationError):
            generate_scorecard_report(bad)

    def test_baseline_diff_mode_returns_report(self) -> None:
        """Passing a baseline result must still produce a ScorecardReport."""
        report = generate_scorecard_report(_VALID_RESULT, baseline=_BASELINE_RESULT)
        assert isinstance(report, ScorecardReport)
        assert "## Baseline Comparison" in report.markdown

    def test_invalid_baseline_raises_validation_error(self) -> None:
        """An invalid baseline result must raise jsonschema.ValidationError."""
        bad_baseline = copy.deepcopy(_BASELINE_RESULT)
        del bad_baseline["model_id"]
        with pytest.raises(jsonschema.ValidationError):
            generate_scorecard_report(_VALID_RESULT, baseline=bad_baseline)

    def test_markdown_contains_model_id(self) -> None:
        """Model ID must be present in the generated Markdown."""
        report = generate_scorecard_report(_VALID_RESULT)
        assert _VALID_RESULT["model_id"] in report.markdown

    def test_markdown_contains_overall_scores_heading(self) -> None:
        """'Overall Scores' heading must be in the generated Markdown."""
        report = generate_scorecard_report(_VALID_RESULT)
        assert "## Overall Scores" in report.markdown

    def test_markdown_contains_per_track_heading(self) -> None:
        """'Per-Track Scores' heading must be in the generated Markdown."""
        report = generate_scorecard_report(_VALID_RESULT)
        assert "## Per-Track Scores" in report.markdown

    def test_markdown_contains_per_pack_heading(self) -> None:
        """'Per-Pack Scores' heading must be in the generated Markdown."""
        report = generate_scorecard_report(_VALID_RESULT)
        assert "## Per-Pack Scores" in report.markdown

    def test_markdown_contains_per_task_heading(self) -> None:
        """'Per-Task Scores' heading must be in the generated Markdown."""
        report = generate_scorecard_report(_VALID_RESULT)
        assert "## Per-Task Scores" in report.markdown
