"""Stub adapter for GRADE runner tests and smoke testing.

:class:`StubAdapter` is a deterministic, network-free adapter that returns
schema-valid ``output_schema.json`` dicts without calling any external API.
It is used by the runner's own test suite and is safe to run in CI without
any provider credentials.

The stub response is fully deterministic given (``task_id``, ``run_index``):
it always produces the same output for the same inputs, making test
assertions stable across reruns.

Usage::

    from runner.adapters.stub_adapter import StubAdapter

    adapter = StubAdapter()
    output = adapter.run(task, run_index=0)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from runner.adapters.base import AdapterBase

#: Adapter version string embedded in every stub output's ``runtime_metadata``.
STUB_ADAPTER_VERSION: str = "0.1.0"

#: Model ID embedded in every stub output.
STUB_MODEL_ID: str = "stub/echo-v1"


class StubAdapter(AdapterBase):
    """Deterministic stub adapter that echoes a minimal valid output dict.

    All returned outputs validate against ``output_schema.json``.  Numeric
    values in ``structured_metrics`` are derived from the task's first
    ``gold_facts`` entry (if any) so that C1 fact-scoring can produce a
    non-trivial result.

    Args:
        model_id: Override the model ID embedded in outputs.  Defaults to
            :data:`STUB_MODEL_ID`.
        fixed_latency_ms: Simulated latency value written to
            ``runtime_metadata.latency_ms``.  Defaults to ``1.0``.

    Example::

        adapter = StubAdapter()
        out = adapter.run({"task_id": "T1-OPS-001", ...}, run_index=0)
        assert out["model_id"] == "stub/echo-v1"
    """

    def __init__(
        self,
        model_id: str = STUB_MODEL_ID,
        fixed_latency_ms: float = 1.0,
    ) -> None:
        """Initialise the stub adapter.

        Args:
            model_id: Model ID string written to every output.
            fixed_latency_ms: Simulated latency written to runtime_metadata.
        """
        super().__init__("stub")
        self._model_id = model_id
        self._fixed_latency_ms = fixed_latency_ms

    def run(self, task: dict[str, Any], run_index: int = 0) -> dict[str, Any]:
        """Return a deterministic schema-valid output for *task*.

        The output contains:

        - ``structured_metrics``: one entry per gold fact whose
          ``numeric_value`` is not ``None``, keyed by ``fact_id.lower()``.
          This allows C1 fact scoring to match some facts automatically.
        - ``key_findings``: a single echo sentence derived from the task
          title and ``run_index``.
        - ``limitations``: a single generic limitation string.
        - ``evidence_citations``: an empty list.

        Args:
            task: Task definition dict (``task_schema.json``).
            run_index: 0-based repetition index.

        Returns:
            A dict conforming to ``output_schema.json``.
        """
        task_id: str = task["task_id"]
        pack_id: str | None = task.get("pack_id")

        # Build structured_metrics from gold_facts numeric values so C1 can score.
        structured_metrics: dict[str, Any] = {}
        for fact in task.get("gold_facts", []):
            if fact.get("numeric_value") is not None:
                key = f"{fact['fact_id'].lower()}_value"
                structured_metrics[key] = fact["numeric_value"]

        title: str = task.get("title", "")
        key_findings: list[str] = [f"[stub run {run_index}] {title or task_id}: analysis complete."]
        limitations: list[str] = ["This is a stub response; no real model inference was performed."]

        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "task_id": task_id,
            "model_id": self._model_id,
            "run_index": run_index,
            "structured_metrics": structured_metrics,
            "key_findings": key_findings,
            "limitations": limitations,
            "evidence_citations": [],
            "runtime_metadata": {
                "adapter_version": STUB_ADAPTER_VERSION,
                "timestamp_utc": timestamp,
                "latency_ms": self._fixed_latency_ms,
                "prompt_tokens": None,
                "completion_tokens": None,
                "model_temperature": None,
                "provider": "stub",
                "pack_id": pack_id,
            },
        }
