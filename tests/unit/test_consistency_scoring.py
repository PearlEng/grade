"""Unit tests for benchmark.rubrics.consistency_scoring (C4 — Consistency scoring).

Covers finding stability, ranking stability, metric variance, and the
top-level score_consistency aggregator, per the Definition of Done in
ticket #582.

Test matrix
-----------
- :func:`finding_stability`: identical runs (score 1.0), totally different runs
  (score 0.0 or near), partial overlap, empty findings, single run.
- :func:`ranking_stability`: identical order (1.0), reversed order (0.0),
  partial overlap, insufficient common findings, single run.
- :func:`metric_variance_score`: zero variance (1.0), high variance (low score),
  no numeric metrics (1.0), single observation excluded.
- :func:`score_consistency`: identical runs (max), divergent runs (lower),
  partial overlap, missing task_id raises, empty runs raises.
"""

from __future__ import annotations

import pytest

from benchmark.rubrics.consistency_scoring import (
    finding_stability,
    metric_variance_score,
    ranking_stability,
    score_consistency,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_RUNTIME = {
    "adapter_version": "0.1.0",
    "timestamp_utc": "2026-06-01T00:00:00Z",
    "latency_ms": 100.0,
}


def _make_run(
    key_findings: list[str],
    structured_metrics: dict | None = None,
    task_id: str = "task-001",
    model_id: str = "test-model",
    run_index: int = 0,
) -> dict:
    """Build a minimal output dict conforming to output_schema.json."""
    return {
        "task_id": task_id,
        "model_id": model_id,
        "run_index": run_index,
        "structured_metrics": structured_metrics or {},
        "key_findings": key_findings,
        "limitations": [],
        "evidence_citations": [],
        "runtime_metadata": _RUNTIME,
    }


def _make_task(task_id: str = "task-001") -> dict:
    """Build a minimal task dict."""
    return {"task_id": task_id}


def _five_identical_runs(findings: list[str], metrics: dict | None = None) -> list[dict]:
    """Return 5 identical runs with the given findings and metrics."""
    return [_make_run(findings, structured_metrics=metrics, run_index=i) for i in range(5)]


# ===========================================================================
# finding_stability
# ===========================================================================


class TestFindingStability:
    """Tests for finding_stability()."""

    def test_identical_runs_score_one(self) -> None:
        """Five identical runs with the same findings should return 1.0."""
        findings = ["Revenue grew 12% YoY.", "Margin improved to 35%.", "Churn fell by 2 pp."]
        runs = _five_identical_runs(findings)
        score, per_finding = finding_stability(runs)
        assert score == pytest.approx(1.0)
        assert all(v == pytest.approx(1.0) for v in per_finding.values())

    def test_totally_different_runs_score_low(self) -> None:
        """Runs with completely distinct findings should score near 0."""
        # Each string uses a completely different vocabulary so Jaccard < threshold
        distinct_findings = [
            "revenue grew significantly this quarter",
            "churn decreased among enterprise accounts",
            "margin compressed due to headcount additions",
            "pipeline coverage improved in midmarket segment",
            "gross retention remained stable across cohorts",
        ]
        runs = [_make_run([f], run_index=i) for i, f in enumerate(distinct_findings)]
        score, per_finding = finding_stability(runs)
        # Each finding appears in only 1 of 5 runs → appearance rate = 0.2
        assert score == pytest.approx(0.2, abs=1e-9)
        assert all(v == pytest.approx(0.2, abs=1e-9) for v in per_finding.values())

    def test_partial_overlap(self) -> None:
        """When half the runs share the same finding, score should be 0.5."""
        shared = "Retention rate increased to 90%."
        runs = [
            _make_run([shared] if i < 3 else ["Completely different finding xyz abc"], run_index=i)
            for i in range(5)
        ]
        score, per_finding = finding_stability(runs)
        # "shared" appears in runs 0,1,2 → 3/5 = 0.6; unique appears in runs 3,4 → 2/5 = 0.4
        assert score == pytest.approx((0.6 + 0.4) / 2, abs=1e-9)

    def test_empty_findings_all_runs(self) -> None:
        """No findings in any run should return (1.0, {}) — nothing to disagree on."""
        runs = _five_identical_runs([])
        score, per_finding = finding_stability(runs)
        assert score == pytest.approx(1.0)
        assert per_finding == {}

    def test_single_run(self) -> None:
        """A single run should return 1.0 (every finding present in all 1 runs)."""
        runs = [_make_run(["Only one run finding text here."])]
        score, _ = finding_stability(runs)
        assert score == pytest.approx(1.0)

    def test_empty_runs_raises(self) -> None:
        """Empty runs list should raise ValueError."""
        with pytest.raises(ValueError, match="runs must contain"):
            finding_stability([])

    def test_near_duplicate_findings_merged(self) -> None:
        """Near-duplicate findings (high Jaccard) should be treated as the same."""
        # These two strings share most tokens → Jaccard >= 0.5
        f1 = "student retention rate improved significantly this quarter"
        f2 = "student retention rate improved significantly this quarter extra"
        runs = [_make_run([f1], run_index=0), _make_run([f2], run_index=1)]
        score, _ = finding_stability(runs)
        # Both should match the canonical form → appearance rate 1.0
        assert score == pytest.approx(1.0)

    def test_top_k_limits_findings(self) -> None:
        """Only the first top_k findings should be examined per run."""
        # Run 0: 4 findings, but top_k=2 only takes first 2
        # Run 1: same first 2 findings in a different order plus extras
        f_common_1 = "Revenue grew by twelve percent this year."
        f_common_2 = "Margins expanded to thirty five percent."
        f_extra = "Unique extra finding not shared anywhere."
        runs = [
            _make_run([f_common_1, f_common_2, f_extra, f_extra + " again"], run_index=0),
            _make_run([f_common_1, f_common_2, f_extra + " b", f_extra + " c"], run_index=1),
        ]
        score, per_finding = finding_stability(runs, top_k=2)
        # Only first 2 findings per run examined; both are common → score 1.0
        assert score == pytest.approx(1.0)


# ===========================================================================
# ranking_stability
# ===========================================================================


class TestRankingStability:
    """Tests for ranking_stability()."""

    def test_identical_order_score_one(self) -> None:
        """Identical finding order across all runs should return 1.0."""
        findings = ["Alpha finding text here.", "Beta finding text here.", "Gamma finding text."]
        runs = _five_identical_runs(findings)
        score = ranking_stability(runs)
        assert score == pytest.approx(1.0)

    def test_reversed_order_score_zero(self) -> None:
        """Fully reversed finding order between two runs should return 0.0."""
        f1 = "alpha beta gamma delta epsilon zeta"
        f2 = "beta gamma delta epsilon zeta eta"
        f3 = "gamma delta epsilon zeta eta theta"
        runs = [
            _make_run([f1, f2, f3], run_index=0),
            _make_run([f3, f2, f1], run_index=1),
        ]
        score = ranking_stability(runs)
        # Perfect reversal of 3 items → tau = -1 → normalized 0.0
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_partially_different_order(self) -> None:
        """Swapping two adjacent findings reduces ranking stability below 1.0."""
        f1 = "alpha beta gamma delta epsilon one"
        f2 = "beta gamma delta epsilon zeta two"
        f3 = "gamma delta epsilon zeta eta three"
        # Run 0: [f1, f2, f3], Run 1: [f2, f1, f3] — one swap
        runs = [
            _make_run([f1, f2, f3], run_index=0),
            _make_run([f2, f1, f3], run_index=1),
        ]
        score = ranking_stability(runs)
        assert 0.0 < score < 1.0

    def test_single_run_returns_one(self) -> None:
        """A single run should return 1.0 (no pairs to compare)."""
        runs = [_make_run(["Only one run finding."])]
        score = ranking_stability(runs)
        assert score == pytest.approx(1.0)

    def test_empty_runs_raises(self) -> None:
        """Empty runs list should raise ValueError."""
        with pytest.raises(ValueError, match="runs must contain"):
            ranking_stability([])

    def test_no_common_findings_returns_one(self) -> None:
        """If run pairs share no common findings, score defaults to 1.0."""
        runs = [
            _make_run(["unique alpha xyz foo bar"], run_index=0),
            _make_run(["unique beta abc qux baz"], run_index=1),
        ]
        score = ranking_stability(runs)
        assert score == pytest.approx(1.0)

    def test_empty_findings_returns_one(self) -> None:
        """Runs with no findings at all should return 1.0."""
        runs = _five_identical_runs([])
        score = ranking_stability(runs)
        assert score == pytest.approx(1.0)


# ===========================================================================
# metric_variance_score
# ===========================================================================


class TestMetricVarianceScore:
    """Tests for metric_variance_score()."""

    def test_zero_variance_score_one(self) -> None:
        """Identical numeric metrics across all runs should return 1.0."""
        runs = _five_identical_runs([], metrics={"total_students": 500, "avg_score": 82.5})
        score, per_metric = metric_variance_score(runs)
        assert score == pytest.approx(1.0)
        assert per_metric["total_students"] == pytest.approx(1.0)
        assert per_metric["avg_score"] == pytest.approx(1.0)

    def test_high_variance_score_lower(self) -> None:
        """High metric variance should produce a score well below 1.0."""
        # mean ≈ 505, stdev ≈ 712 → CV >> 1 → score = 1/(1+CV) << 0.5
        runs = [
            _make_run([], structured_metrics={"value": 1.0}, run_index=0),
            _make_run([], structured_metrics={"value": 10.0}, run_index=1),
            _make_run([], structured_metrics={"value": 1504.0}, run_index=2),
        ]
        score, per_metric = metric_variance_score(runs)
        assert score < 0.5
        assert per_metric["value"] < 0.5

    def test_no_numeric_metrics_returns_one(self) -> None:
        """No numeric metrics means nothing to vary → return (1.0, {})."""
        runs = _five_identical_runs([], metrics={"label": "good", "flag": True})
        score, per_metric = metric_variance_score(runs)
        assert score == pytest.approx(1.0)
        assert per_metric == {}

    def test_empty_metrics_returns_one(self) -> None:
        """Runs with empty structured_metrics should return (1.0, {})."""
        runs = _five_identical_runs([], metrics={})
        score, per_metric = metric_variance_score(runs)
        assert score == pytest.approx(1.0)
        assert per_metric == {}

    def test_single_observation_excluded(self) -> None:
        """A metric appearing in only one run should not be scored (no variance info)."""
        runs = [
            _make_run([], structured_metrics={"only_in_one": 42.0}, run_index=0),
            _make_run([], structured_metrics={}, run_index=1),
        ]
        score, per_metric = metric_variance_score(runs)
        # "only_in_one" appears only once → excluded; no other numeric metrics → score 1.0
        assert score == pytest.approx(1.0)
        assert "only_in_one" not in per_metric

    def test_zero_mean_metric_excluded(self) -> None:
        """A metric with mean zero should be excluded (CV undefined)."""
        runs = [
            _make_run([], structured_metrics={"zero_mean": 0.0}, run_index=0),
            _make_run([], structured_metrics={"zero_mean": 0.0}, run_index=1),
        ]
        score, per_metric = metric_variance_score(runs)
        assert score == pytest.approx(1.0)
        assert "zero_mean" not in per_metric

    def test_moderate_variance(self) -> None:
        """Moderate variance should produce a score between 0.5 and 1.0."""
        # mean=100, stdev~5 → CV~0.05 → score = 1/(1+0.05) ≈ 0.952
        runs = [
            _make_run([], structured_metrics={"metric": 95.0}, run_index=0),
            _make_run([], structured_metrics={"metric": 100.0}, run_index=1),
            _make_run([], structured_metrics={"metric": 105.0}, run_index=2),
        ]
        score, per_metric = metric_variance_score(runs)
        assert 0.5 < score < 1.0
        assert 0.5 < per_metric["metric"] < 1.0

    def test_empty_runs_raises(self) -> None:
        """Empty runs list should raise ValueError."""
        with pytest.raises(ValueError, match="runs must contain"):
            metric_variance_score([])


# ===========================================================================
# score_consistency (top-level aggregator)
# ===========================================================================


class TestScoreConsistency:
    """Tests for score_consistency()."""

    def test_identical_runs_max_consistency(self) -> None:
        """Five identical runs should return consistency_score == 1.0."""
        findings = ["Revenue grew 12% YoY.", "Margin improved to 35%.", "Churn fell."]
        metrics = {"total_revenue": 1200000.0, "margin_pct": 35.0}
        task = _make_task()
        runs = _five_identical_runs(findings, metrics=metrics)
        result = score_consistency(task, runs)

        assert result["consistency_score"] == pytest.approx(1.0)
        assert result["task_id"] == "task-001"
        assert result["run_count"] == 5

    def test_divergent_runs_lower_consistency(self) -> None:
        """Runs with completely different findings and metrics should score < 1.0."""
        task = _make_task()
        runs = [
            _make_run(
                [f"Unique finding number {i} with distinct tokens abc xyz"],
                structured_metrics={"value": float(i * 100)},
                run_index=i,
            )
            for i in range(5)
        ]
        result = score_consistency(task, runs)
        assert result["consistency_score"] < 1.0

    def test_partial_overlap_between_identical_and_divergent(self) -> None:
        """Partial overlap should produce a score between 0 and 1."""
        task = _make_task()
        shared = "Revenue grew by twelve percent this fiscal year."
        runs = [
            _make_run(
                [shared] if i < 3 else ["Completely unrelated finding number xyz"],
                structured_metrics={"metric": 100.0 if i < 3 else float(i * 50)},
                run_index=i,
            )
            for i in range(5)
        ]
        result = score_consistency(task, runs)
        assert 0.0 < result["consistency_score"] < 1.0

    def test_return_keys_present(self) -> None:
        """score_consistency must always return the expected dict keys."""
        task = _make_task()
        runs = _five_identical_runs(["A finding here."])
        result = score_consistency(task, runs)
        expected_keys = {
            "task_id",
            "run_count",
            "consistency_score",
            "per_finding_stability",
            "ranking_stability",
            "metric_variance",
            "component_scores",
        }
        assert expected_keys.issubset(result.keys())

    def test_component_scores_sum_to_consistency_score(self) -> None:
        """Weighted component scores must reproduce the top-level consistency_score."""
        task = _make_task()
        runs = [
            _make_run(
                ["Finding alpha.", "Finding beta.", "Finding gamma."],
                structured_metrics={"x": 10.0 + i},
                run_index=i,
            )
            for i in range(5)
        ]
        result = score_consistency(task, runs)
        cs = result["component_scores"]
        expected = (
            0.5 * cs["finding_stability"]
            + 0.3 * cs["ranking_stability"]
            + 0.2 * cs["metric_variance"]
        )
        assert result["consistency_score"] == pytest.approx(expected)

    def test_empty_runs_raises(self) -> None:
        """Empty runs list should raise ValueError."""
        with pytest.raises(ValueError, match="runs must contain"):
            score_consistency(_make_task(), [])

    def test_missing_task_id_raises(self) -> None:
        """A task dict without task_id should raise KeyError."""
        with pytest.raises(KeyError):
            score_consistency({}, _five_identical_runs(["Finding."]))

    def test_single_run(self) -> None:
        """A single run should return consistency_score == 1.0 (nothing to disagree on)."""
        task = _make_task()
        runs = [_make_run(["Revenue grew 10%.", "Costs declined.", "Market share up."])]
        result = score_consistency(task, runs)
        assert result["consistency_score"] == pytest.approx(1.0)

    def test_no_findings_no_metrics(self) -> None:
        """Runs with no findings and no metrics should return consistency_score == 1.0."""
        task = _make_task()
        runs = _five_identical_runs([])
        result = score_consistency(task, runs)
        assert result["consistency_score"] == pytest.approx(1.0)

    def test_consistency_score_in_range(self) -> None:
        """consistency_score must always be in [0, 1]."""
        task = _make_task()
        # Use runs with moderate disagreement
        runs = [
            _make_run(
                [f"Finding token set diverges here number {i} alpha beta gamma"],
                structured_metrics={"val": float(i * 10)},
                run_index=i,
            )
            for i in range(5)
        ]
        result = score_consistency(task, runs)
        assert 0.0 <= result["consistency_score"] <= 1.0

    def test_custom_top_k(self) -> None:
        """Custom top_k should be respected by score_consistency."""
        task = _make_task()
        findings = [f"Finding number {i} unique tokens here." for i in range(6)]
        runs = _five_identical_runs(findings)
        result_k1 = score_consistency(task, runs, top_k=1)
        result_k3 = score_consistency(task, runs, top_k=3)
        # Both should be 1.0 when all runs are identical
        assert result_k1["consistency_score"] == pytest.approx(1.0)
        assert result_k3["consistency_score"] == pytest.approx(1.0)
