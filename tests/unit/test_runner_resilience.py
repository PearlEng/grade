"""Tests for runner resilience: per-task failure tolerance and partial scorecard.

Covers:
- Partial run: adapter raises on one task → result.json contains only
  the successful tasks, failures.json lists the failed one, runner returns 0.
- All-failure run: all tasks fail → runner returns 1, no result.json written.
- OpenRouterAdapter default max_tokens is 4096 (raised from 1024, which
  truncated prose analyses — methodology finding H-4); constructor override
  and GRADE_OPENROUTER_MAX_TOKENS env-var override both work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from runner.adapters.base import AdapterBase
from runner.adapters.openrouter_adapter import OpenRouterAdapter
from runner.cli import main

# ---------------------------------------------------------------------------
# Shared task fixtures
# ---------------------------------------------------------------------------

_TASK_TEMPLATE: dict[str, Any] = {
    "track": 1,
    "title": "Count enrolled students",
    "user_prompt": "How many students are enrolled?",
    "task_type": "retrieval",
    "allowed_inputs": ["students.csv"],
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


def _make_task(task_id: str) -> dict[str, Any]:
    """Return a minimal valid task dict with the given task_id."""
    return {**_TASK_TEMPLATE, "task_id": task_id}


def _write_pack(tasks: list[dict[str, Any]], path: Path) -> None:
    """Write a JSONL task pack to *path*."""
    with path.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")


# ---------------------------------------------------------------------------
# Stub adapters for resilience tests
# ---------------------------------------------------------------------------


class _SucceedingAdapter(AdapterBase):
    """Adapter that always returns a minimal schema-valid output."""

    def __init__(self) -> None:
        super().__init__("stub-succeed")

    def run(self, task: dict[str, Any], run_index: int = 0) -> dict[str, Any]:
        from datetime import UTC, datetime

        return {
            "task_id": task["task_id"],
            "model_id": "stub/succeed-v1",
            "run_index": run_index,
            "structured_metrics": {},
            "key_findings": ["Analysis complete."],
            "limitations": ["This is a stub response."],
            "evidence_citations": [],
            "runtime_metadata": {
                "adapter_version": "0.1.0",
                "timestamp_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "latency_ms": 1.0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "model_temperature": None,
                "provider": "stub",
                "pack_id": task.get("pack_id"),
            },
        }


class _FailOnSecondAdapter(AdapterBase):
    """Adapter that raises RuntimeError on the second distinct task_id it sees.

    Task identity is tracked by call order (first unique task_id succeeds,
    the second unique task_id raises, the third succeeds again).
    """

    def __init__(self) -> None:
        super().__init__("stub-fail-second")
        self._seen: list[str] = []

    def run(self, task: dict[str, Any], run_index: int = 0) -> dict[str, Any]:
        task_id: str = task["task_id"]
        if task_id not in self._seen:
            self._seen.append(task_id)
        rank = self._seen.index(task_id) + 1  # 1-based position
        if rank == 2:
            raise RuntimeError(f"HTTP 402: credit limit exceeded for {task_id}")

        from datetime import UTC, datetime

        return {
            "task_id": task_id,
            "model_id": "stub/fail-second-v1",
            "run_index": run_index,
            "structured_metrics": {},
            "key_findings": ["Analysis complete for this task."],
            "limitations": ["This is a stub response; no real inference performed."],
            "evidence_citations": [],
            "runtime_metadata": {
                "adapter_version": "0.1.0",
                "timestamp_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "latency_ms": 1.0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "model_temperature": None,
                "provider": "stub",
                "pack_id": task.get("pack_id"),
            },
        }


class _AlwaysFailAdapter(AdapterBase):
    """Adapter that always raises RuntimeError."""

    def __init__(self) -> None:
        super().__init__("stub-always-fail")

    def run(self, task: dict[str, Any], run_index: int = 0) -> dict[str, Any]:
        raise RuntimeError(f"HTTP 402: credit limit exceeded for {task['task_id']}")


# ---------------------------------------------------------------------------
# Helper: run CLI with a pre-built adapter injected via monkeypatch
# ---------------------------------------------------------------------------


def _run_cli_with_adapter(
    adapter: AdapterBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tasks: list[dict[str, Any]],
) -> int:
    """Write a task pack, patch the CLI adapter registry, and call main().

    Returns the integer exit code from main().
    """
    pack_path = tmp_path / "test_pack.jsonl"
    _write_pack(tasks, pack_path)
    out_dir = tmp_path / "out"

    # Patch _build_adapter so any adapter name returns our stub.
    with patch("runner.cli._build_adapter", return_value=adapter):
        return main(
            [
                "--pack",
                str(pack_path),
                "--adapter",
                "stub",
                "--runs",
                "1",
                "--out",
                str(out_dir),
            ]
        )


# ---------------------------------------------------------------------------
# Test: partial run (one failure out of three tasks)
# ---------------------------------------------------------------------------


class TestPartialRunResilience:
    """Runner survives a per-task failure and writes a partial scorecard."""

    def test_returns_zero_when_some_tasks_succeed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exit code must be 0 when at least one task succeeds."""
        tasks = [_make_task("T-001"), _make_task("T-002"), _make_task("T-003")]
        adapter = _FailOnSecondAdapter()
        rc = _run_cli_with_adapter(adapter, tmp_path, monkeypatch, tasks)
        assert rc == 0

    def test_result_json_contains_only_successful_tasks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """result.json must contain scores only for the tasks that succeeded."""
        tasks = [_make_task("T-001"), _make_task("T-002"), _make_task("T-003")]
        adapter = _FailOnSecondAdapter()
        _run_cli_with_adapter(adapter, tmp_path, monkeypatch, tasks)

        result_path = tmp_path / "out" / "result.json"
        assert result_path.exists(), "result.json must be written on partial success"

        with result_path.open(encoding="utf-8") as fh:
            scorecard = json.load(fh)

        task_ids = [t["task_id"] for t in scorecard["per_task_scores"]]
        assert "T-001" in task_ids
        assert "T-003" in task_ids
        assert "T-002" not in task_ids, "Failed task must not appear in scorecard"
        assert scorecard["task_count"] == 2

    def test_failures_json_lists_failed_task(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """failures.json must list the failed task_id and its error message."""
        tasks = [_make_task("T-001"), _make_task("T-002"), _make_task("T-003")]
        adapter = _FailOnSecondAdapter()
        _run_cli_with_adapter(adapter, tmp_path, monkeypatch, tasks)

        failures_path = tmp_path / "out" / "failures.json"
        assert failures_path.exists(), "failures.json must be written when tasks fail"

        with failures_path.open(encoding="utf-8") as fh:
            failures = json.load(fh)

        assert len(failures) == 1
        assert failures[0]["task_id"] == "T-002"
        assert "error" in failures[0]
        assert failures[0]["error"]  # non-empty error string

    def test_raw_outputs_written(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """raw_outputs.jsonl must exist and contain only successful outputs."""
        tasks = [_make_task("T-001"), _make_task("T-002"), _make_task("T-003")]
        adapter = _FailOnSecondAdapter()
        _run_cli_with_adapter(adapter, tmp_path, monkeypatch, tasks)

        raw_path = tmp_path / "out" / "raw_outputs.jsonl"
        assert raw_path.exists()

        lines = [ln for ln in raw_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        # 2 successful tasks × 1 run each = 2 lines
        assert len(lines) == 2
        task_ids_in_raw = {json.loads(ln)["task_id"] for ln in lines}
        assert "T-002" not in task_ids_in_raw


# ---------------------------------------------------------------------------
# Test: all tasks fail → exit code 1, no result.json
# ---------------------------------------------------------------------------


class TestAllTasksFailure:
    """When every task fails the runner must return 1 and write no scorecard."""

    def test_returns_one_when_all_tasks_fail(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exit code must be 1 when all tasks fail."""
        tasks = [_make_task("T-001"), _make_task("T-002")]
        adapter = _AlwaysFailAdapter()
        rc = _run_cli_with_adapter(adapter, tmp_path, monkeypatch, tasks)
        assert rc == 1

    def test_no_result_json_when_all_fail(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """result.json must NOT be written when all tasks fail."""
        tasks = [_make_task("T-001"), _make_task("T-002")]
        adapter = _AlwaysFailAdapter()
        _run_cli_with_adapter(adapter, tmp_path, monkeypatch, tasks)

        result_path = tmp_path / "out" / "result.json"
        assert not result_path.exists(), "result.json must not be written on total failure"

    def test_no_raw_outputs_when_all_fail(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """raw_outputs.jsonl must NOT be written when all tasks fail."""
        tasks = [_make_task("T-001"), _make_task("T-002")]
        adapter = _AlwaysFailAdapter()
        _run_cli_with_adapter(adapter, tmp_path, monkeypatch, tasks)

        raw_path = tmp_path / "out" / "raw_outputs.jsonl"
        assert not raw_path.exists(), "raw_outputs.jsonl must not be written on total failure"


# ---------------------------------------------------------------------------
# Test: all tasks succeed → no failures.json
# ---------------------------------------------------------------------------


class TestAllTasksSucceed:
    """When all tasks succeed, failures.json is NOT created."""

    def test_no_failures_json_when_all_succeed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """failures.json must NOT be written when every task succeeds."""
        tasks = [_make_task("T-001"), _make_task("T-002")]
        adapter = _SucceedingAdapter()
        rc = _run_cli_with_adapter(adapter, tmp_path, monkeypatch, tasks)

        assert rc == 0
        failures_path = tmp_path / "out" / "failures.json"
        assert not failures_path.exists(), "failures.json must not exist when all tasks succeed"


# ---------------------------------------------------------------------------
# Test: OpenRouterAdapter max_tokens defaults and overrides
# ---------------------------------------------------------------------------


class TestOpenRouterMaxTokens:
    """OpenRouterAdapter default max_tokens is 4096 and can be overridden."""

    def test_default_max_tokens_is_4096(self) -> None:
        """Default max_tokens must be 4096 (1024 truncated prose analyses, H-4)."""
        adapter = OpenRouterAdapter(model="anthropic/claude-sonnet-4.6", api_key="sk-or-test")
        assert adapter._max_tokens == 4096

    def test_default_max_tokens_class_constant(self) -> None:
        """DEFAULT_MAX_TOKENS class attribute must be 4096."""
        assert OpenRouterAdapter.DEFAULT_MAX_TOKENS == 4096

    def test_constructor_override(self) -> None:
        """Explicit max_tokens constructor arg must override the default."""
        adapter = OpenRouterAdapter(
            model="anthropic/claude-sonnet-4-5",
            api_key="sk-or-test",
            max_tokens=512,
        )
        assert adapter._max_tokens == 512

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GRADE_OPENROUTER_MAX_TOKENS env var must override the default."""
        monkeypatch.setenv("GRADE_OPENROUTER_MAX_TOKENS", "768")
        adapter = OpenRouterAdapter(model="anthropic/claude-sonnet-4-5", api_key="sk-or-test")
        assert adapter._max_tokens == 768

    def test_constructor_arg_beats_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit constructor arg must take precedence over the env var."""
        monkeypatch.setenv("GRADE_OPENROUTER_MAX_TOKENS", "768")
        adapter = OpenRouterAdapter(
            model="anthropic/claude-sonnet-4-5",
            api_key="sk-or-test",
            max_tokens=256,
        )
        assert adapter._max_tokens == 256

    def test_env_var_cleared_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When env var is absent and no arg given, must use DEFAULT_MAX_TOKENS."""
        monkeypatch.delenv("GRADE_OPENROUTER_MAX_TOKENS", raising=False)
        adapter = OpenRouterAdapter(model="anthropic/claude-sonnet-4-5", api_key="sk-or-test")
        assert adapter._max_tokens == OpenRouterAdapter.DEFAULT_MAX_TOKENS
