"""Unit tests for cost & speed metrics (cost_usd, tokens, latency).

Coverage:
- :func:`~runner.aggregator.aggregate_cost_metrics`: totals computed correctly;
  cost absent everywhere → total_cost_usd None and cost_available False;
  partial cost handled; latency stats (mean/p50/max); per_task_cost breakdown.
- :func:`~runner.aggregator.aggregate`: cost_metrics present in output and
  passes result_schema validation.
- :class:`~runner.adapters.openrouter_adapter.OpenRouterAdapter`: cost_usd lives
  in runtime_metadata, NOT in structured_metrics; output validates with and
  without cost.
- :func:`~benchmark.reports.scorecard.render_markdown`: "Cost & Speed" section
  present; cost reported when available; "not reported" shown for stub adapter;
  cost-efficiency figure shown when cost available; token-efficiency proxy shown
  when tokens available but cost absent.

All tests are network-free and do not require any API keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from benchmark.rubrics.rubric_scoring import RUBRIC_DIMENSIONS
from benchmark.schemas import validate_result
from runner.aggregator import aggregate, aggregate_cost_metrics
from runner.dispatcher import TaskRunResult

# ---------------------------------------------------------------------------
# Helpers for building TaskRunResult objects with outputs
# ---------------------------------------------------------------------------

_ZERO_SCORES: dict[str, float] = dict.fromkeys(RUBRIC_DIMENSIONS, 0.0)
_HALF_SCORES: dict[str, float] = dict.fromkeys(RUBRIC_DIMENSIONS, 0.5)


def _make_runtime_meta(
    latency_ms: float = 10.0,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float | None = None,
) -> dict[str, Any]:
    """Build a minimal runtime_metadata dict for test outputs."""
    return {
        "adapter_version": "0.1.0",
        "timestamp_utc": "2026-06-01T12:00:00Z",
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd,
        "model_temperature": 1.0,
        "provider": "test",
        "pack_id": None,
    }


def _make_output(
    task_id: str = "T1-OPS-001",
    run_index: int = 0,
    latency_ms: float = 10.0,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float | None = None,
) -> dict[str, Any]:
    """Build a minimal output dict with the given runtime metadata values."""
    return {
        "task_id": task_id,
        "model_id": "test/model",
        "run_index": run_index,
        "structured_metrics": {},
        "key_findings": ["A finding."],
        "limitations": [],
        "evidence_citations": [],
        "runtime_metadata": _make_runtime_meta(
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        ),
    }


def _make_task_result(
    task_id: str = "T1-OPS-001",
    track: int = 1,
    pack_id: str | None = "pack_operations",
    outputs: list[dict[str, Any]] | None = None,
) -> TaskRunResult:
    """Build a TaskRunResult with the given outputs."""
    return TaskRunResult(
        task_id=task_id,
        track=track,
        pack_id=pack_id,
        run_count=len(outputs) if outputs else 1,
        outputs=outputs or [],
        scores=dict(_HALF_SCORES),
        composite=0.5,
        scorer_flags=[],
    )


# ---------------------------------------------------------------------------
# aggregate_cost_metrics — core logic
# ---------------------------------------------------------------------------


class TestAggregateCostMetricsNoCost:
    """When no output reports cost, cost_available must be False and total_cost None."""

    def test_no_cost_returns_none_total(self) -> None:
        """total_cost_usd must be None when all outputs have cost_usd=None."""
        tr = _make_task_result(
            outputs=[
                _make_output(cost_usd=None, prompt_tokens=100, completion_tokens=50),
                _make_output(run_index=1, cost_usd=None, prompt_tokens=80, completion_tokens=40),
            ]
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["total_cost_usd"] is None

    def test_no_cost_cost_available_false(self) -> None:
        """cost_available must be False when no output has a cost."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=None)])
        metrics = aggregate_cost_metrics([tr])
        assert metrics["cost_available"] is False

    def test_no_cost_cost_partial_false(self) -> None:
        """cost_partial must be False when no output has cost."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=None)])
        metrics = aggregate_cost_metrics([tr])
        assert metrics["cost_partial"] is False

    def test_no_outputs_returns_sensible_defaults(self) -> None:
        """Empty outputs list must not crash and must return None cost."""
        tr = _make_task_result(outputs=[])
        metrics = aggregate_cost_metrics([tr])
        assert metrics["cost_available"] is False
        assert metrics["total_cost_usd"] is None
        assert metrics["mean_latency_ms"] is None


class TestAggregateCostMetricsWithCost:
    """When outputs report cost, totals must be summed correctly."""

    def test_single_output_cost_summed(self) -> None:
        """A single output with cost 0.001 → total_cost_usd == 0.001."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=0.001)])
        metrics = aggregate_cost_metrics([tr])
        assert metrics["total_cost_usd"] == pytest.approx(0.001)
        assert metrics["cost_available"] is True

    def test_multiple_outputs_cost_summed(self) -> None:
        """Two outputs with costs 0.001 and 0.002 → total == 0.003."""
        tr = _make_task_result(
            outputs=[
                _make_output(run_index=0, cost_usd=0.001),
                _make_output(run_index=1, cost_usd=0.002),
            ]
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["total_cost_usd"] == pytest.approx(0.003)

    def test_multiple_tasks_cost_summed(self) -> None:
        """Costs across two tasks must be summed together."""
        tr1 = _make_task_result(
            task_id="T1-OPS-001",
            outputs=[_make_output(task_id="T1-OPS-001", cost_usd=0.001)],
        )
        tr2 = _make_task_result(
            task_id="T1-OPS-002",
            outputs=[_make_output(task_id="T1-OPS-002", cost_usd=0.004)],
        )
        metrics = aggregate_cost_metrics([tr1, tr2])
        assert metrics["total_cost_usd"] == pytest.approx(0.005)

    def test_cost_partial_when_some_missing(self) -> None:
        """cost_partial must be True when some outputs have cost and others don't."""
        tr = _make_task_result(
            outputs=[
                _make_output(run_index=0, cost_usd=0.001),
                _make_output(run_index=1, cost_usd=None),
            ]
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["cost_available"] is True
        assert metrics["cost_partial"] is True
        assert metrics["total_cost_usd"] == pytest.approx(0.001)

    def test_cost_partial_false_when_all_present(self) -> None:
        """cost_partial must be False when every output has cost."""
        tr = _make_task_result(
            outputs=[
                _make_output(run_index=0, cost_usd=0.001),
                _make_output(run_index=1, cost_usd=0.002),
            ]
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["cost_partial"] is False


class TestAggregateCostMetricsTokens:
    """Token count aggregation tests."""

    def test_tokens_summed_correctly(self) -> None:
        """Prompt and completion tokens must be summed across all outputs."""
        tr = _make_task_result(
            outputs=[
                _make_output(run_index=0, prompt_tokens=100, completion_tokens=50),
                _make_output(run_index=1, prompt_tokens=120, completion_tokens=60),
            ]
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["total_prompt_tokens"] == 220
        assert metrics["total_completion_tokens"] == 110
        assert metrics["total_tokens"] == 330

    def test_tokens_available_true_when_reported(self) -> None:
        """tokens_available must be True when at least one output has token counts."""
        tr = _make_task_result(outputs=[_make_output(prompt_tokens=100, completion_tokens=50)])
        metrics = aggregate_cost_metrics([tr])
        assert metrics["tokens_available"] is True

    def test_tokens_available_false_when_none(self) -> None:
        """tokens_available must be False when all outputs have None tokens."""
        tr = _make_task_result(outputs=[_make_output(prompt_tokens=None, completion_tokens=None)])
        metrics = aggregate_cost_metrics([tr])
        assert metrics["tokens_available"] is False

    def test_none_tokens_treated_as_zero_for_sum(self) -> None:
        """None token counts must not crash; they contribute 0 to the sum."""
        tr = _make_task_result(
            outputs=[
                _make_output(run_index=0, prompt_tokens=None, completion_tokens=None),
                _make_output(run_index=1, prompt_tokens=80, completion_tokens=40),
            ]
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["total_prompt_tokens"] == 80
        assert metrics["total_completion_tokens"] == 40


class TestAggregateCostMetricsLatency:
    """Latency statistics tests."""

    def test_mean_latency_correct(self) -> None:
        """Mean latency of 10 ms and 30 ms must be 20 ms."""
        tr = _make_task_result(
            outputs=[
                _make_output(run_index=0, latency_ms=10.0),
                _make_output(run_index=1, latency_ms=30.0),
            ]
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["mean_latency_ms"] == pytest.approx(20.0)

    def test_p50_latency_odd_count(self) -> None:
        """Median of [10, 20, 30] must be 20."""
        tr = _make_task_result(
            outputs=[
                _make_output(run_index=0, latency_ms=10.0),
                _make_output(run_index=1, latency_ms=30.0),
                _make_output(run_index=2, latency_ms=20.0),
            ]
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["p50_latency_ms"] == pytest.approx(20.0)

    def test_p50_latency_even_count(self) -> None:
        """Median of [10, 20] must be 15 (average of the two middle values)."""
        tr = _make_task_result(
            outputs=[
                _make_output(run_index=0, latency_ms=10.0),
                _make_output(run_index=1, latency_ms=20.0),
            ]
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["p50_latency_ms"] == pytest.approx(15.0)

    def test_max_latency_correct(self) -> None:
        """Max latency of [10, 50, 30] must be 50."""
        tr = _make_task_result(
            outputs=[
                _make_output(run_index=0, latency_ms=10.0),
                _make_output(run_index=1, latency_ms=50.0),
                _make_output(run_index=2, latency_ms=30.0),
            ]
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["max_latency_ms"] == pytest.approx(50.0)


class TestAggregateCostMetricsPerTask:
    """Per-task cost breakdown tests."""

    def test_per_task_cost_keyed_by_task_id(self) -> None:
        """per_task_cost must contain an entry for each task_id."""
        tr1 = _make_task_result(
            task_id="T1-OPS-001",
            outputs=[_make_output(task_id="T1-OPS-001", cost_usd=0.001)],
        )
        tr2 = _make_task_result(
            task_id="T1-OPS-002",
            outputs=[_make_output(task_id="T1-OPS-002", cost_usd=0.002)],
        )
        metrics = aggregate_cost_metrics([tr1, tr2])
        assert "T1-OPS-001" in metrics["per_task_cost"]
        assert "T1-OPS-002" in metrics["per_task_cost"]

    def test_per_task_cost_sums_runs(self) -> None:
        """Per-task cost must be the sum of all run costs for that task."""
        tr = _make_task_result(
            task_id="T1-OPS-001",
            outputs=[
                _make_output(task_id="T1-OPS-001", run_index=0, cost_usd=0.001),
                _make_output(task_id="T1-OPS-001", run_index=1, cost_usd=0.002),
            ],
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["per_task_cost"]["T1-OPS-001"]["cost_usd"] == pytest.approx(0.003)

    def test_per_task_cost_none_when_no_cost(self) -> None:
        """Per-task cost_usd must be None when that task's outputs have no cost."""
        tr = _make_task_result(
            task_id="T1-OPS-001",
            outputs=[_make_output(task_id="T1-OPS-001", cost_usd=None)],
        )
        metrics = aggregate_cost_metrics([tr])
        assert metrics["per_task_cost"]["T1-OPS-001"]["cost_usd"] is None

    def test_per_task_tokens_summed(self) -> None:
        """Per-task tokens must be summed from all runs."""
        tr = _make_task_result(
            task_id="T1-OPS-001",
            outputs=[
                _make_output(
                    task_id="T1-OPS-001", run_index=0, prompt_tokens=100, completion_tokens=50
                ),
                _make_output(
                    task_id="T1-OPS-001", run_index=1, prompt_tokens=90, completion_tokens=45
                ),
            ],
        )
        metrics = aggregate_cost_metrics([tr])
        pt = metrics["per_task_cost"]["T1-OPS-001"]
        assert pt["prompt_tokens"] == 190
        assert pt["completion_tokens"] == 95
        assert pt["total_tokens"] == 285


# ---------------------------------------------------------------------------
# aggregate — integration: cost_metrics in result dict
# ---------------------------------------------------------------------------


class TestAggregateWithCostMetrics:
    """Tests that aggregate() includes cost_metrics in the result and validates."""

    def test_aggregate_includes_cost_metrics_key(self) -> None:
        """aggregate() output must contain a 'cost_metrics' key."""
        tr = _make_task_result(
            outputs=[_make_output(cost_usd=0.001, prompt_tokens=100, completion_tokens=50)]
        )
        scorecard = aggregate([tr], model_id="test/model")
        assert "cost_metrics" in scorecard

    def test_aggregate_cost_metrics_validates_schema(self) -> None:
        """A scorecard with cost_metrics must pass result_schema validation."""
        tr = _make_task_result(
            outputs=[_make_output(cost_usd=0.001, prompt_tokens=100, completion_tokens=50)]
        )
        scorecard = aggregate([tr], model_id="test/model")
        validate_result(scorecard)  # must not raise

    def test_aggregate_no_cost_validates_schema(self) -> None:
        """A scorecard with cost_available=False must also pass schema validation."""
        tr = _make_task_result(
            outputs=[_make_output(cost_usd=None, prompt_tokens=None, completion_tokens=None)]
        )
        scorecard = aggregate([tr], model_id="stub/echo-v1")
        validate_result(scorecard)  # must not raise

    def test_aggregate_cost_totals_match_expectations(self) -> None:
        """cost_metrics inside the aggregate result must match manual calculation."""
        tr = _make_task_result(
            task_id="T1-OPS-001",
            outputs=[
                _make_output(run_index=0, cost_usd=0.001, prompt_tokens=100, completion_tokens=50),
                _make_output(run_index=1, cost_usd=0.002, prompt_tokens=120, completion_tokens=60),
            ],
        )
        scorecard = aggregate([tr], model_id="test/model")
        cm = scorecard["cost_metrics"]
        assert cm["total_cost_usd"] == pytest.approx(0.003)
        assert cm["total_prompt_tokens"] == 220
        assert cm["total_completion_tokens"] == 110
        assert cm["total_tokens"] == 330
        assert cm["cost_available"] is True
        assert cm["cost_partial"] is False

    def test_aggregate_stub_outputs_no_cost(self) -> None:
        """Outputs from stub adapter (no cost/tokens) must give cost_available=False."""
        tr = _make_task_result(
            outputs=[
                _make_output(cost_usd=None, prompt_tokens=None, completion_tokens=None),
            ]
        )
        scorecard = aggregate([tr], model_id="stub/echo-v1")
        cm = scorecard["cost_metrics"]
        assert cm["cost_available"] is False
        assert cm["total_cost_usd"] is None
        assert cm["tokens_available"] is False


# ---------------------------------------------------------------------------
# OpenRouter adapter — cost in runtime_metadata, NOT structured_metrics
# ---------------------------------------------------------------------------

_VALID_TASK: dict[str, Any] = {
    "task_id": "T1-OPS-001",
    "track": 1,
    "title": "Test",
    "user_prompt": "How many?",
    "task_type": "retrieval",
    "allowed_inputs": [],
    "gold_facts": [],
    "gold_insights": [],
    "required_limitations": [],
    "forbidden_claims": [],
    "rubric": {
        "grounding_accuracy": {"weight": 0.35},
        "insight_quality": {"weight": 0.20},
        "evidence_linkage": {"weight": 0.15},
        "calibration_limitation_handling": {"weight": 0.15},
        "consistency": {"weight": 0.10},
        "structure_usability": {"weight": 0.05},
    },
}

_MOCK_RESPONSE_WITH_COST: dict[str, Any] = {
    "id": "gen-abc",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "## Key Findings\n- Found 42.\n"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cost": "0.00042",
    },
}

_MOCK_RESPONSE_NO_COST: dict[str, Any] = {
    "id": "gen-xyz",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "## Key Findings\n- Found 42.\n"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 80,
        "completion_tokens": 40,
    },
}


def _inject_mock_httpx(body: dict[str, Any], status_code: int = 200) -> MagicMock:
    """Inject a fake httpx module and return the mock post callable."""
    import sys

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = body
    mock_response.text = json.dumps(body)
    mock_post = MagicMock(return_value=mock_response)

    fake_httpx = ModuleType("httpx")
    fake_httpx.post = mock_post  # type: ignore[attr-defined]
    fake_httpx.Client = MagicMock()  # type: ignore[attr-defined]
    sys.modules["httpx"] = fake_httpx
    return mock_post


def _remove_mock_httpx() -> None:
    import sys

    sys.modules.pop("httpx", None)


class TestAdapterCostPlacement:
    """Verify cost_usd placement in the OpenRouter adapter output."""

    def _run_adapter(self, body: dict[str, Any]) -> dict[str, Any]:
        from runner.adapters.openrouter_adapter import OpenRouterAdapter

        try:
            _inject_mock_httpx(body)
            adapter = OpenRouterAdapter(model="anthropic/claude-sonnet-4-5", api_key="sk-or-test")
            return adapter.run(_VALID_TASK, run_index=0)
        finally:
            _remove_mock_httpx()

    def test_cost_in_runtime_metadata_when_present(self) -> None:
        """When provider returns cost, it must be in runtime_metadata.cost_usd."""
        output = self._run_adapter(_MOCK_RESPONSE_WITH_COST)
        assert output["runtime_metadata"]["cost_usd"] == pytest.approx(0.00042)

    def test_cost_not_in_structured_metrics(self) -> None:
        """cost_usd must NOT appear in structured_metrics (avoids C1 scorer pollution)."""
        output = self._run_adapter(_MOCK_RESPONSE_WITH_COST)
        assert "cost_usd" not in output["structured_metrics"]

    def test_cost_none_when_provider_omits_it(self) -> None:
        """When provider omits cost, runtime_metadata.cost_usd must be None."""
        output = self._run_adapter(_MOCK_RESPONSE_NO_COST)
        assert output["runtime_metadata"]["cost_usd"] is None

    def test_output_schema_valid_with_cost(self) -> None:
        """Output with cost in runtime_metadata must validate against output_schema."""
        import jsonschema

        schema_path = (
            Path(__file__).parent.parent.parent / "benchmark" / "schemas" / "output_schema.json"
        )
        with schema_path.open() as fh:
            schema = json.load(fh)

        output = self._run_adapter(_MOCK_RESPONSE_WITH_COST)
        jsonschema.validate(instance=output, schema=schema)  # must not raise

    def test_output_schema_valid_without_cost(self) -> None:
        """Output with cost_usd=None must also validate against output_schema."""
        import jsonschema

        schema_path = (
            Path(__file__).parent.parent.parent / "benchmark" / "schemas" / "output_schema.json"
        )
        with schema_path.open() as fh:
            schema = json.load(fh)

        output = self._run_adapter(_MOCK_RESPONSE_NO_COST)
        jsonschema.validate(instance=output, schema=schema)  # must not raise


# ---------------------------------------------------------------------------
# Scorecard Cost & Speed section rendering
# ---------------------------------------------------------------------------

# Build a valid result dict that includes cost_metrics.
_DIM_SCORES: dict[str, float] = dict.fromkeys(RUBRIC_DIMENSIONS, 0.5)

_RESULT_WITH_COST: dict[str, Any] = {
    "result_id": "test-with-cost-abc123",
    "model_id": "openrouter/anthropic/claude-sonnet-4-5",
    "grade_version": "0.1.0",
    "scored_at_utc": "2026-06-01T12:00:00Z",
    "task_count": 1,
    "run_count": 2,
    "overall_scores": dict(_DIM_SCORES),
    "overall_composite": 0.5,
    "per_track_scores": {
        "1": {
            "track": 1,
            "track_name": "Grounded Retrieval & Computation",
            "task_count": 1,
            "scores": dict(_DIM_SCORES),
            "composite": 0.5,
        }
    },
    "per_pack_scores": {
        "pack_operations": {
            "pack_id": "pack_operations",
            "task_count": 1,
            "scores": dict(_DIM_SCORES),
            "composite": 0.5,
        }
    },
    "per_task_scores": [
        {
            "task_id": "T1-OPS-001",
            "track": 1,
            "pack_id": "pack_operations",
            "run_count": 2,
            "scores": dict(_DIM_SCORES),
            "composite": 0.5,
            "scorer_flags": None,
        }
    ],
    "cost_metrics": {
        "total_cost_usd": 0.00125,
        "cost_available": True,
        "cost_partial": False,
        "total_prompt_tokens": 220,
        "total_completion_tokens": 110,
        "total_tokens": 330,
        "tokens_available": True,
        "mean_latency_ms": 250.0,
        "p50_latency_ms": 240.0,
        "max_latency_ms": 260.0,
        "per_task_cost": {
            "T1-OPS-001": {
                "cost_usd": 0.00125,
                "prompt_tokens": 220,
                "completion_tokens": 110,
                "total_tokens": 330,
                "mean_latency_ms": 250.0,
            }
        },
    },
}

_RESULT_NO_COST: dict[str, Any] = {
    "result_id": "test-no-cost-xyz999",
    "model_id": "stub/echo-v1",
    "grade_version": "0.1.0",
    "scored_at_utc": "2026-06-01T12:00:00Z",
    "task_count": 1,
    "run_count": 1,
    "overall_scores": dict(_DIM_SCORES),
    "overall_composite": 0.5,
    "per_track_scores": {
        "1": {
            "track": 1,
            "track_name": "Grounded Retrieval & Computation",
            "task_count": 1,
            "scores": dict(_DIM_SCORES),
            "composite": 0.5,
        }
    },
    "per_pack_scores": {
        "pack_operations": {
            "pack_id": "pack_operations",
            "task_count": 1,
            "scores": dict(_DIM_SCORES),
            "composite": 0.5,
        }
    },
    "per_task_scores": [
        {
            "task_id": "T1-OPS-001",
            "track": 1,
            "pack_id": "pack_operations",
            "run_count": 1,
            "scores": dict(_DIM_SCORES),
            "composite": 0.5,
            "scorer_flags": None,
        }
    ],
    "cost_metrics": {
        "total_cost_usd": None,
        "cost_available": False,
        "cost_partial": False,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "tokens_available": False,
        "mean_latency_ms": 1.0,
        "p50_latency_ms": 1.0,
        "max_latency_ms": 1.0,
        "per_task_cost": {
            "T1-OPS-001": {
                "cost_usd": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "mean_latency_ms": 1.0,
            }
        },
    },
}

# Result with tokens but no cost (token-efficiency proxy case).
_RESULT_TOKENS_ONLY: dict[str, Any] = {
    **_RESULT_NO_COST,
    "result_id": "test-tokens-only-aaa111",
    "cost_metrics": {
        "total_cost_usd": None,
        "cost_available": False,
        "cost_partial": False,
        "total_prompt_tokens": 500,
        "total_completion_tokens": 250,
        "total_tokens": 750,
        "tokens_available": True,
        "mean_latency_ms": 50.0,
        "p50_latency_ms": 45.0,
        "max_latency_ms": 60.0,
        "per_task_cost": {
            "T1-OPS-001": {
                "cost_usd": None,
                "prompt_tokens": 500,
                "completion_tokens": 250,
                "total_tokens": 750,
                "mean_latency_ms": 50.0,
            }
        },
    },
}


class TestScorecardCostSpeedSection:
    """Tests for the Cost & Speed section in render_markdown."""

    def test_cost_speed_heading_present(self) -> None:
        """The '## Cost & Speed' heading must appear in the rendered scorecard."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_WITH_COST)
        assert "## Cost & Speed" in md

    def test_cost_speed_heading_present_no_cost(self) -> None:
        """The heading must also appear when cost is not reported (stub adapter)."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_NO_COST)
        assert "## Cost & Speed" in md

    def test_cost_shown_when_available(self) -> None:
        """When cost is available, the dollar amount must appear in the output."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_WITH_COST)
        # $0.00125 should appear somewhere
        assert "$0.00125" in md

    def test_not_reported_when_cost_absent(self) -> None:
        """When cost is not reported (stub), 'not reported' must appear in the section."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_NO_COST)
        assert "not reported" in md.lower()

    def test_token_count_shown_when_available(self) -> None:
        """When tokens are available, the token count must appear in the section."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_WITH_COST)
        assert "330" in md  # total_tokens

    def test_latency_shown_in_section(self) -> None:
        """Mean latency value must appear in the Cost & Speed section."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_WITH_COST)
        assert "250" in md  # mean_latency_ms

    def test_cost_efficiency_shown_when_cost_available(self) -> None:
        """When cost is available, a cost-efficiency figure must appear."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_WITH_COST)
        # composite=0.5, cost=$0.00125 → 0.5*100/0.00125 = 40000.0
        assert "Cost-Efficiency" in md
        assert "40000.0" in md

    def test_token_efficiency_proxy_when_cost_absent_but_tokens_available(self) -> None:
        """When cost is absent but tokens are available, token-efficiency proxy must appear."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_TOKENS_ONLY)
        # composite=0.5, total_tokens=750 → 0.5*100/(750/1000) = 50/0.75 = 66.67
        assert "Token-Efficiency" in md

    def test_no_efficiency_when_neither_cost_nor_tokens(self) -> None:
        """When neither cost nor tokens are reported, no efficiency line must appear."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_NO_COST)
        assert "Efficiency" not in md

    def test_result_with_cost_validates_schema(self) -> None:
        """A result dict containing cost_metrics must pass result_schema validation."""
        validate_result(_RESULT_WITH_COST)  # must not raise

    def test_result_no_cost_validates_schema(self) -> None:
        """A result dict with cost_available=False must also validate."""
        validate_result(_RESULT_NO_COST)  # must not raise

    def test_scorecard_deterministic_with_cost(self) -> None:
        """Two identical renders with cost data must produce identical output."""
        from benchmark.reports.scorecard import render_markdown

        md1 = render_markdown(_RESULT_WITH_COST)
        md2 = render_markdown(_RESULT_WITH_COST)
        assert md1 == md2

    def test_old_result_without_cost_metrics_renders_gracefully(self) -> None:
        """A result dict without a 'cost_metrics' key must render without errors."""
        from benchmark.reports.scorecard import render_markdown
        from tests.unit.test_scorecard import _VALID_RESULT

        # _VALID_RESULT has no cost_metrics — must still render cleanly.
        md = render_markdown(_VALID_RESULT)
        assert "## Cost & Speed" in md
        # Without cost_metrics, it shows the "not available" fallback.
        assert "not available" in md.lower()
