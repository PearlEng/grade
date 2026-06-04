"""Unit tests for the split model/judge cost feature.

Coverage:
- :class:`~benchmark.rubrics.judge_client.JudgeClient`: accumulates
  cost/tokens across multiple mocked judge() calls; None when provider
  reports no cost; judge_call_count increments correctly.
- :func:`~runner.aggregator.aggregate_cost_metrics`: produces distinct
  ``model_cost_usd`` / ``judge_cost_usd`` / ``total_eval_cost_usd`` fields;
  total = sum when both present; judge fields are None without --judge.
- :func:`~runner.aggregator.aggregate`: threads judge_metrics kwarg through;
  resulting scorecard passes result_schema validation.
- :func:`~benchmark.reports.scorecard._build_cost_speed_section` and
  :func:`~benchmark.reports.scorecard._build_cost_efficiency_note`: lead with
  model cost + model-based efficiency; show judge overhead separately; handle
  all-None gracefully.
- :func:`~runner.adapters.openrouter_adapter.post_chat_completion`: the
  ``return_usage=True`` path returns a (text, ChatCompletionUsage) tuple;
  cost still flows to test-model runtime_metadata via the normal adapter path.

All tests are network-free and typed.  No API keys or httpx required.
"""

from __future__ import annotations

import json
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from benchmark.rubrics.rubric_scoring import RUBRIC_DIMENSIONS
from benchmark.schemas import validate_result
from runner.adapters.openrouter_adapter import ChatCompletionUsage
from runner.aggregator import aggregate, aggregate_cost_metrics
from runner.dispatcher import TaskRunResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZERO_SCORES: dict[str, float] = dict.fromkeys(RUBRIC_DIMENSIONS, 0.0)
_HALF_SCORES: dict[str, float] = dict.fromkeys(RUBRIC_DIMENSIONS, 0.5)


def _make_runtime_meta(
    latency_ms: float = 10.0,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float | None = None,
) -> dict[str, Any]:
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
    cost_usd: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "model_id": "test/model",
        "run_index": run_index,
        "structured_metrics": {},
        "key_findings": ["A finding."],
        "limitations": [],
        "evidence_citations": [],
        "runtime_metadata": _make_runtime_meta(
            cost_usd=cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    }


def _make_task_result(
    task_id: str = "T1-OPS-001",
    track: int = 1,
    pack_id: str | None = "pack_operations",
    outputs: list[dict[str, Any]] | None = None,
) -> TaskRunResult:
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
# JudgeClient cost accumulation
# ---------------------------------------------------------------------------


class TestJudgeClientCostAccumulation:
    """JudgeClient.judge() must accumulate cost/tokens into instance state."""

    _TASK: dict[str, Any] = {
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

    _OUTPUT: dict[str, Any] = {
        "task_id": "T1-OPS-001",
        "model_id": "test/model",
        "run_index": 0,
        "structured_metrics": {},
        "key_findings": ["Finding."],
        "limitations": [],
        "evidence_citations": [],
        "runtime_metadata": {
            "adapter_version": "0.1.0",
            "timestamp_utc": "2026-06-01T12:00:00Z",
            "latency_ms": 50.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost_usd": None,
            "model_temperature": 1.0,
            "provider": "test",
            "pack_id": None,
        },
    }

    def _make_usage(
        self,
        cost: float | None,
        pt: int | None = None,
        ct: int | None = None,
    ) -> ChatCompletionUsage:
        return ChatCompletionUsage(
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=cost,
        )

    def _run_judge_calls(
        self,
        usages: list[ChatCompletionUsage],
    ) -> Any:  # returns JudgeClient; typed as Any to avoid a forward-ref issue
        from benchmark.rubrics.judge_client import JudgeClient

        client = JudgeClient(api_key="sk-or-test")
        for i, usage in enumerate(usages):
            with patch(
                "runner.adapters.openrouter_adapter.post_chat_completion",
                return_value=(f"0.{i + 5}", usage),
            ):
                client.judge("grounding_accuracy", "", self._TASK, self._OUTPUT)
        return client

    def test_initial_state_all_zero_and_none(self) -> None:
        """Fresh JudgeClient must start with zero counts and None cost."""
        from benchmark.rubrics.judge_client import JudgeClient

        client = JudgeClient(api_key="sk-or-test")
        assert client.cumulative_cost_usd is None
        assert client.cumulative_prompt_tokens == 0
        assert client.cumulative_completion_tokens == 0
        assert client.judge_call_count == 0

    def test_single_call_with_cost_accumulates(self) -> None:
        """One judge call with cost=0.001 must set cumulative_cost_usd=0.001."""
        client = self._run_judge_calls([self._make_usage(cost=0.001, pt=50, ct=5)])
        assert client.cumulative_cost_usd == pytest.approx(0.001)
        assert client.cumulative_prompt_tokens == 50
        assert client.cumulative_completion_tokens == 5
        assert client.judge_call_count == 1

    def test_multiple_calls_cost_accumulates(self) -> None:
        """Two judge calls with costs 0.001 and 0.002 must sum to 0.003."""
        client = self._run_judge_calls(
            [
                self._make_usage(cost=0.001, pt=40, ct=4),
                self._make_usage(cost=0.002, pt=60, ct=6),
            ]
        )
        assert client.cumulative_cost_usd == pytest.approx(0.003)
        assert client.cumulative_prompt_tokens == 100
        assert client.cumulative_completion_tokens == 10
        assert client.judge_call_count == 2

    def test_cost_remains_none_when_provider_never_reports(self) -> None:
        """cumulative_cost_usd must stay None when every call has cost_usd=None."""
        client = self._run_judge_calls(
            [
                self._make_usage(cost=None, pt=30, ct=3),
                self._make_usage(cost=None, pt=40, ct=4),
            ]
        )
        assert client.cumulative_cost_usd is None
        # Tokens still accumulated.
        assert client.cumulative_prompt_tokens == 70
        assert client.cumulative_completion_tokens == 7
        assert client.judge_call_count == 2

    def test_partial_cost_reports_accumulated(self) -> None:
        """If some calls report cost and others don't, accumulated = sum of reported."""
        client = self._run_judge_calls(
            [
                self._make_usage(cost=0.003),
                self._make_usage(cost=None),
                self._make_usage(cost=0.002),
            ]
        )
        assert client.cumulative_cost_usd == pytest.approx(0.005)
        assert client.judge_call_count == 3

    def test_six_calls_simulates_full_task(self) -> None:
        """Six judge calls (one per dimension) must accumulate correctly."""
        usages = [self._make_usage(cost=0.0005, pt=20, ct=2) for _ in range(6)]
        client = self._run_judge_calls(usages)
        assert client.cumulative_cost_usd == pytest.approx(0.003)
        assert client.cumulative_prompt_tokens == 120
        assert client.cumulative_completion_tokens == 12
        assert client.judge_call_count == 6

    def test_return_usage_true_passed_to_post_chat_completion(self) -> None:
        """JudgeClient must pass return_usage=True to post_chat_completion."""
        from benchmark.rubrics.judge_client import JudgeClient

        client = JudgeClient(api_key="sk-or-test")
        with patch(
            "runner.adapters.openrouter_adapter.post_chat_completion",
            return_value=("0.7", self._make_usage(cost=0.001)),
        ) as mock_pcc:
            client.judge("insight_quality", "", self._TASK, self._OUTPUT)

        _, call_kwargs = mock_pcc.call_args
        assert call_kwargs.get("return_usage") is True


# ---------------------------------------------------------------------------
# aggregate_cost_metrics — new split fields
# ---------------------------------------------------------------------------


class TestAggregateCostMetricsSplitFields:
    """aggregate_cost_metrics must produce model_cost_usd, judge_cost_usd, total_eval_cost_usd."""

    def test_model_cost_usd_present_without_judge(self) -> None:
        """model_cost_usd must equal the test-model cost when no judge is used."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=0.005)])
        metrics = aggregate_cost_metrics([tr])
        assert metrics["model_cost_usd"] == pytest.approx(0.005)

    def test_total_cost_usd_alias_equals_model_cost_usd(self) -> None:
        """total_cost_usd must be an alias for model_cost_usd (backward compat)."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=0.003)])
        metrics = aggregate_cost_metrics([tr])
        assert metrics["total_cost_usd"] == metrics["model_cost_usd"]

    def test_judge_cost_usd_none_without_judge_metrics(self) -> None:
        """judge_cost_usd must be None when judge_metrics is None (no --judge)."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=0.005)])
        metrics = aggregate_cost_metrics([tr], judge_metrics=None)
        assert metrics["judge_cost_usd"] is None

    def test_total_eval_cost_usd_equals_model_when_no_judge(self) -> None:
        """total_eval_cost_usd must equal model_cost_usd when no judge is present."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=0.005)])
        metrics = aggregate_cost_metrics([tr], judge_metrics=None)
        assert metrics["total_eval_cost_usd"] == pytest.approx(0.005)

    def test_judge_cost_from_judge_metrics(self) -> None:
        """judge_cost_usd must come from judge_metrics['cumulative_cost_usd']."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=0.010)])
        judge_metrics = {
            "cumulative_cost_usd": 0.003,
            "cumulative_prompt_tokens": 100,
            "cumulative_completion_tokens": 10,
            "judge_call_count": 6,
        }
        metrics = aggregate_cost_metrics([tr], judge_metrics=judge_metrics)
        assert metrics["judge_cost_usd"] == pytest.approx(0.003)

    def test_total_eval_cost_sums_model_and_judge(self) -> None:
        """total_eval_cost_usd = model_cost_usd + judge_cost_usd when both present."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=0.010)])
        judge_metrics = {"cumulative_cost_usd": 0.003}
        metrics = aggregate_cost_metrics([tr], judge_metrics=judge_metrics)
        assert metrics["total_eval_cost_usd"] == pytest.approx(0.013)

    def test_total_eval_cost_none_when_both_model_and_judge_none(self) -> None:
        """total_eval_cost_usd must be None when neither model nor judge has cost."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=None)])
        judge_metrics = {"cumulative_cost_usd": None}
        metrics = aggregate_cost_metrics([tr], judge_metrics=judge_metrics)
        assert metrics["total_eval_cost_usd"] is None

    def test_total_eval_cost_is_judge_only_when_model_none(self) -> None:
        """total_eval_cost_usd falls back to judge cost when model cost is None."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=None)])
        judge_metrics = {"cumulative_cost_usd": 0.002}
        metrics = aggregate_cost_metrics([tr], judge_metrics=judge_metrics)
        assert metrics["total_eval_cost_usd"] == pytest.approx(0.002)

    def test_all_original_fields_still_present(self) -> None:
        """Existing fields must still be present for backward compatibility."""
        tr = _make_task_result(outputs=[_make_output(cost_usd=0.001)])
        metrics = aggregate_cost_metrics([tr])
        required_keys = {
            "model_cost_usd",
            "total_cost_usd",
            "judge_cost_usd",
            "total_eval_cost_usd",
            "cost_available",
            "cost_partial",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "tokens_available",
            "mean_latency_ms",
            "p50_latency_ms",
            "max_latency_ms",
            "per_task_cost",
        }
        assert required_keys.issubset(set(metrics.keys()))


# ---------------------------------------------------------------------------
# aggregate() — judge_metrics kwarg threaded through
# ---------------------------------------------------------------------------


class TestAggregateWithJudgeMetrics:
    """aggregate() must populate split cost fields and validate against schema."""

    def _scorecard_with_model_cost(
        self, judge_metrics: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        tr = _make_task_result(
            outputs=[_make_output(cost_usd=0.010, prompt_tokens=200, completion_tokens=50)]
        )
        return aggregate([tr], model_id="test/model", judge_metrics=judge_metrics)

    def test_model_cost_in_scorecard(self) -> None:
        """Scorecard cost_metrics must have model_cost_usd == model inference cost."""
        sc = self._scorecard_with_model_cost()
        assert sc["cost_metrics"]["model_cost_usd"] == pytest.approx(0.010)

    def test_judge_cost_none_without_judge(self) -> None:
        """judge_cost_usd must be None when no judge_metrics passed."""
        sc = self._scorecard_with_model_cost(judge_metrics=None)
        assert sc["cost_metrics"]["judge_cost_usd"] is None

    def test_judge_cost_populated_when_judge_metrics_provided(self) -> None:
        """judge_cost_usd must equal cumulative_cost_usd from judge_metrics."""
        sc = self._scorecard_with_model_cost(
            judge_metrics={"cumulative_cost_usd": 0.006, "judge_call_count": 6}
        )
        assert sc["cost_metrics"]["judge_cost_usd"] == pytest.approx(0.006)

    def test_total_eval_cost_sum(self) -> None:
        """total_eval_cost_usd must equal model + judge when both present."""
        sc = self._scorecard_with_model_cost(judge_metrics={"cumulative_cost_usd": 0.006})
        assert sc["cost_metrics"]["total_eval_cost_usd"] == pytest.approx(0.016)

    def test_scorecard_validates_with_judge_cost(self) -> None:
        """Scorecard with judge cost must pass result_schema validation."""
        sc = self._scorecard_with_model_cost(judge_metrics={"cumulative_cost_usd": 0.003})
        validate_result(sc)  # must not raise

    def test_scorecard_validates_without_judge_cost(self) -> None:
        """Scorecard without judge cost must also pass result_schema validation."""
        sc = self._scorecard_with_model_cost(judge_metrics=None)
        validate_result(sc)  # must not raise

    def test_scorecard_validates_all_none_cost(self) -> None:
        """Stub/no-cost scorecard must pass validation even when all costs are None."""
        tr = _make_task_result(
            outputs=[_make_output(cost_usd=None, prompt_tokens=None, completion_tokens=None)]
        )
        sc = aggregate([tr], model_id="stub/echo-v1", judge_metrics=None)
        validate_result(sc)  # must not raise
        assert sc["cost_metrics"]["model_cost_usd"] is None
        assert sc["cost_metrics"]["judge_cost_usd"] is None
        assert sc["cost_metrics"]["total_eval_cost_usd"] is None


# ---------------------------------------------------------------------------
# Scorecard rendering — model cost leads, judge overhead secondary
# ---------------------------------------------------------------------------

_DIM_SCORES: dict[str, float] = dict.fromkeys(RUBRIC_DIMENSIONS, 0.5)


def _make_result(
    model_cost_usd: float | None,
    judge_cost_usd: float | None,
    total_eval_cost_usd: float | None,
    cost_available: bool,
) -> dict[str, Any]:
    """Build a minimal valid result dict with the given cost fields."""
    total_cost_usd = model_cost_usd  # alias
    return {
        "result_id": "test-split-cost-abc123",
        "model_id": "openrouter/test/model",
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
            "model_cost_usd": model_cost_usd,
            "total_cost_usd": total_cost_usd,
            "judge_cost_usd": judge_cost_usd,
            "total_eval_cost_usd": total_eval_cost_usd,
            "cost_available": cost_available,
            "cost_partial": False,
            "total_prompt_tokens": 220 if cost_available else 0,
            "total_completion_tokens": 110 if cost_available else 0,
            "total_tokens": 330 if cost_available else 0,
            "tokens_available": cost_available,
            "mean_latency_ms": 100.0,
            "p50_latency_ms": 100.0,
            "max_latency_ms": 100.0,
            "per_task_cost": {
                "T1-OPS-001": {
                    "cost_usd": model_cost_usd,
                    "prompt_tokens": 220 if cost_available else 0,
                    "completion_tokens": 110 if cost_available else 0,
                    "total_tokens": 330 if cost_available else 0,
                    "mean_latency_ms": 100.0,
                }
            },
        },
    }


_RESULT_WITH_BOTH_COSTS = _make_result(
    model_cost_usd=0.010,
    judge_cost_usd=0.006,
    total_eval_cost_usd=0.016,
    cost_available=True,
)

_RESULT_MODEL_COST_ONLY = _make_result(
    model_cost_usd=0.010,
    judge_cost_usd=None,
    total_eval_cost_usd=0.010,
    cost_available=True,
)

_RESULT_ALL_NONE = _make_result(
    model_cost_usd=None,
    judge_cost_usd=None,
    total_eval_cost_usd=None,
    cost_available=False,
)


class TestScorecardSplitCostRendering:
    """Scorecard Cost & Speed section must lead with model cost."""

    def test_model_cost_leads_in_section(self) -> None:
        """'Model Cost (test model)' must appear before judge overhead in the section."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_WITH_BOTH_COSTS)
        model_pos = md.index("Model Cost (test model)")
        judge_pos = md.index("Evaluation overhead (judge)")
        assert model_pos < judge_pos, (
            "Model cost must appear before judge overhead in the scorecard"
        )

    def test_model_cost_dollar_shown(self) -> None:
        """The model cost dollar amount must appear in the rendered scorecard."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_WITH_BOTH_COSTS)
        assert "$0.01000" in md  # model_cost_usd = 0.010

    def test_judge_overhead_dollar_shown(self) -> None:
        """The judge cost dollar amount must appear in the rendered scorecard."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_WITH_BOTH_COSTS)
        assert "$0.00600" in md  # judge_cost_usd = 0.006

    def test_total_eval_cost_shown(self) -> None:
        """Total eval cost (model + judge) must appear in the rendered scorecard."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_WITH_BOTH_COSTS)
        assert "$0.01600" in md  # total_eval_cost_usd = 0.016

    def test_leaderboard_note_present(self) -> None:
        """A note that leaderboard comparisons use test-model cost must appear."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_WITH_BOTH_COSTS)
        assert "leaderboard" in md.lower()

    def test_model_cost_not_reported_when_none(self) -> None:
        """When model cost is None, 'not reported' must appear."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_ALL_NONE)
        assert "not reported" in md.lower()

    def test_judge_overhead_not_reported_when_none(self) -> None:
        """When judge cost is None, judge overhead line shows 'not reported'."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_MODEL_COST_ONLY)
        assert "Evaluation overhead (judge):" in md
        assert "not reported" in md

    def test_cost_efficiency_based_on_model_cost(self) -> None:
        """Cost-efficiency figure must be computed from model_cost_usd, not total_eval_cost_usd.

        With composite=0.5 and model_cost=0.010:
            efficiency = 0.5 * 100 / 0.010 = 5000.0
        """
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_WITH_BOTH_COSTS)
        assert "Cost-Efficiency" in md
        # model_cost = 0.010 → 0.5 * 100 / 0.010 = 5000.0
        assert "5000.0" in md

    def test_all_none_renders_gracefully(self) -> None:
        """When both model and judge cost are None, no efficiency line must appear."""
        from benchmark.reports.scorecard import render_markdown

        md = render_markdown(_RESULT_ALL_NONE)
        # No crash — section renders with "not reported"
        assert "## Cost & Speed" in md
        assert "not reported" in md.lower()
        # No efficiency when no cost
        assert "Cost-Efficiency" not in md

    def test_result_with_split_costs_validates_schema(self) -> None:
        """A result dict with model_cost_usd and judge_cost_usd must pass validation."""
        validate_result(_RESULT_WITH_BOTH_COSTS)  # must not raise

    def test_result_model_cost_only_validates_schema(self) -> None:
        """A result with model_cost_usd but no judge cost must also validate."""
        validate_result(_RESULT_MODEL_COST_ONLY)  # must not raise

    def test_result_all_none_validates_schema(self) -> None:
        """A result with all-None cost fields must also validate."""
        validate_result(_RESULT_ALL_NONE)  # must not raise


# ---------------------------------------------------------------------------
# post_chat_completion return_usage=True — backward compatibility
# ---------------------------------------------------------------------------


def _inject_mock_httpx(body: dict[str, Any], status_code: int = 200) -> MagicMock:
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


_MOCK_RESPONSE_WITH_COST: dict[str, Any] = {
    "id": "gen-xyz",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "0.85"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 40,
        "completion_tokens": 4,
        "total_tokens": 44,
        "cost": "0.00030",
    },
}

_MOCK_RESPONSE_NO_COST: dict[str, Any] = {
    "id": "gen-abc",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "0.75"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 40,
        "completion_tokens": 4,
    },
}


class TestPostChatCompletionReturnUsage:
    """post_chat_completion with return_usage=True must return (text, usage) tuple."""

    def _call(self, body: dict[str, Any], return_usage: bool = True) -> Any:
        from runner.adapters.openrouter_adapter import post_chat_completion

        try:
            _inject_mock_httpx(body)
            return post_chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="anthropic/claude-opus-4-5",
                api_key="sk-or-test",
                return_usage=return_usage,
            )
        finally:
            _remove_mock_httpx()

    def test_return_usage_false_gives_plain_string(self) -> None:
        """With return_usage=False (default), result is a plain string."""
        result = self._call(_MOCK_RESPONSE_WITH_COST, return_usage=False)
        assert isinstance(result, str)
        assert result == "0.85"

    def test_return_usage_true_gives_tuple(self) -> None:
        """With return_usage=True, result is a (str, ChatCompletionUsage) tuple."""
        result = self._call(_MOCK_RESPONSE_WITH_COST, return_usage=True)
        assert isinstance(result, tuple)
        text, usage = result
        assert text == "0.85"
        assert isinstance(usage, ChatCompletionUsage)

    def test_cost_extracted_when_present(self) -> None:
        """Usage.cost_usd must equal the value from the API response."""
        _, usage = self._call(_MOCK_RESPONSE_WITH_COST, return_usage=True)
        assert usage.cost_usd == pytest.approx(0.00030)

    def test_cost_none_when_absent(self) -> None:
        """Usage.cost_usd must be None when the provider omits 'cost'."""
        _, usage = self._call(_MOCK_RESPONSE_NO_COST, return_usage=True)
        assert usage.cost_usd is None

    def test_tokens_extracted(self) -> None:
        """Usage must contain prompt and completion token counts."""
        _, usage = self._call(_MOCK_RESPONSE_WITH_COST, return_usage=True)
        assert usage.prompt_tokens == 40
        assert usage.completion_tokens == 4

    def test_existing_callers_unaffected(self) -> None:
        """Callers not passing return_usage still get a plain string back."""
        from runner.adapters.openrouter_adapter import post_chat_completion

        try:
            _inject_mock_httpx(_MOCK_RESPONSE_WITH_COST)
            result = post_chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="anthropic/claude-opus-4-5",
                api_key="sk-or-test",
            )
        finally:
            _remove_mock_httpx()

        assert isinstance(result, str)

    def test_model_cost_still_flows_to_adapter_runtime_metadata(self) -> None:
        """The adapter's run() must still place cost_usd in runtime_metadata."""
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
        adapter_response = {
            "id": "gen-adapter",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "## Key Findings\n- Found."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost": "0.00055",
            },
        }
        try:
            _inject_mock_httpx(adapter_response)
            from runner.adapters.openrouter_adapter import OpenRouterAdapter

            adapter = OpenRouterAdapter(model="anthropic/claude-sonnet-4-5", api_key="sk-or-test")
            output = adapter.run(_VALID_TASK, run_index=0)
        finally:
            _remove_mock_httpx()

        assert output["runtime_metadata"]["cost_usd"] == pytest.approx(0.00055)
