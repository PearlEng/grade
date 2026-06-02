"""Unit tests for runner.adapters.stub_adapter.StubAdapter.

Coverage:
- :class:`~runner.adapters.stub_adapter.StubAdapter` satisfies
  :class:`~runner.adapters.base.Adapter` Protocol at runtime.
- Output is schema-valid against ``output_schema.json``.
- Determinism: same (task_id, run_index) always yields the same output.
- run_index is embedded correctly.
- pack_id from task dict is embedded in runtime_metadata.
- Custom model_id is respected.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmark.schemas import validate_output
from runner.adapters.base import Adapter
from runner.adapters.stub_adapter import STUB_MODEL_ID, StubAdapter

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_MINIMAL_TASK: dict[str, Any] = {
    "task_id": "T1-OPS-001",
    "track": 1,
    "title": "Count enrolled students",
    "user_prompt": "Using students.csv, how many students are enrolled?",
    "task_type": "retrieval",
    "allowed_inputs": ["students.csv"],
    "gold_facts": [
        {
            "fact_id": "F1",
            "claim": "The program enrolls 135 students.",
            "source_files": ["students.csv"],
            "numeric_value": 135,
            "tolerance": 0,
        }
    ],
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


# ---------------------------------------------------------------------------
# Protocol / interface tests
# ---------------------------------------------------------------------------


class TestAdapterProtocol:
    """Verify StubAdapter satisfies the Adapter Protocol."""

    def test_stub_is_adapter_instance(self) -> None:
        """Isinstance check against Adapter Protocol must pass."""
        adapter = StubAdapter()
        assert isinstance(adapter, Adapter)

    def test_stub_has_name_attribute(self) -> None:
        """StubAdapter must expose a 'name' string attribute."""
        adapter = StubAdapter()
        assert isinstance(adapter.name, str)
        assert adapter.name == "stub"

    def test_stub_has_run_method(self) -> None:
        """StubAdapter must have a callable 'run' method."""
        adapter = StubAdapter()
        assert callable(adapter.run)


# ---------------------------------------------------------------------------
# Output schema validity
# ---------------------------------------------------------------------------


class TestStubAdapterOutput:
    """Tests for the output produced by StubAdapter.run()."""

    def test_output_validates_against_output_schema(self) -> None:
        """The output of run() must pass output_schema validation."""
        adapter = StubAdapter()
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        validate_output(output)  # should not raise

    def test_task_id_matches(self) -> None:
        """Output task_id must match the input task."""
        adapter = StubAdapter()
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        assert output["task_id"] == "T1-OPS-001"

    def test_model_id_is_stub_default(self) -> None:
        """Default model_id must be STUB_MODEL_ID."""
        adapter = StubAdapter()
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        assert output["model_id"] == STUB_MODEL_ID

    def test_custom_model_id(self) -> None:
        """Custom model_id passed at construction must appear in output."""
        adapter = StubAdapter(model_id="custom/my-model")
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        assert output["model_id"] == "custom/my-model"

    def test_run_index_embedded(self) -> None:
        """run_index argument must be reflected in the output."""
        adapter = StubAdapter()
        for idx in range(3):
            output = adapter.run(_MINIMAL_TASK, run_index=idx)
            assert output["run_index"] == idx

    def test_has_structured_metrics(self) -> None:
        """Output must have a structured_metrics dict."""
        adapter = StubAdapter()
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        assert isinstance(output["structured_metrics"], dict)

    def test_structured_metrics_contains_gold_fact_value(self) -> None:
        """Numeric gold fact value should appear in structured_metrics."""
        adapter = StubAdapter()
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        # F1's numeric_value (135) should be in structured_metrics.
        metrics = output["structured_metrics"]
        assert any(v == 135 for v in metrics.values())

    def test_has_key_findings(self) -> None:
        """Output must have a non-empty key_findings list."""
        adapter = StubAdapter()
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        assert isinstance(output["key_findings"], list)
        assert len(output["key_findings"]) >= 1

    def test_has_limitations(self) -> None:
        """Output must have a non-empty limitations list."""
        adapter = StubAdapter()
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        assert isinstance(output["limitations"], list)
        assert len(output["limitations"]) >= 1

    def test_has_runtime_metadata(self) -> None:
        """Output must have a runtime_metadata dict with required fields."""
        adapter = StubAdapter()
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        meta = output["runtime_metadata"]
        assert "adapter_version" in meta
        assert "timestamp_utc" in meta
        assert "latency_ms" in meta

    def test_pack_id_embedded_in_metadata(self) -> None:
        """If task dict contains pack_id, it should appear in runtime_metadata."""
        task_with_pack = dict(_MINIMAL_TASK, pack_id="pack_operations")
        adapter = StubAdapter()
        output = adapter.run(task_with_pack, run_index=0)
        assert output["runtime_metadata"]["pack_id"] == "pack_operations"

    def test_pack_id_none_when_not_in_task(self) -> None:
        """If task has no pack_id key, runtime_metadata.pack_id should be None."""
        adapter = StubAdapter()
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        assert output["runtime_metadata"]["pack_id"] is None

    def test_latency_matches_configured_value(self) -> None:
        """fixed_latency_ms must appear in runtime_metadata.latency_ms."""
        adapter = StubAdapter(fixed_latency_ms=42.0)
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        assert output["runtime_metadata"]["latency_ms"] == pytest.approx(42.0)

    def test_evidence_citations_is_list(self) -> None:
        """evidence_citations must be a list (may be empty)."""
        adapter = StubAdapter()
        output = adapter.run(_MINIMAL_TASK, run_index=0)
        assert isinstance(output["evidence_citations"], list)

    def test_multiple_runs_all_validate(self) -> None:
        """All runs must produce schema-valid outputs."""
        adapter = StubAdapter()
        for i in range(5):
            output = adapter.run(_MINIMAL_TASK, run_index=i)
            validate_output(output)
