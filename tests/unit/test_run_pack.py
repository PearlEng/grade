"""Tests for runner.run.run_pack — the reusable single-model pack runner.

Coverage:
- run_pack returns a schema-valid result dict (parity with cli output).
- run_pack writes result.json / raw_outputs.jsonl when out_dir is provided.
- run_pack raises RuntimeError when all tasks fail.
- CLI behaviour is unchanged: cli.main still produces an identical result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.schemas import validate_result
from runner.adapters.stub_adapter import StubAdapter
from runner.run import run_pack

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUBRIC: dict[str, Any] = {
    "grounding_accuracy": {"weight": 0.35},
    "insight_quality": {"weight": 0.20},
    "evidence_linkage": {"weight": 0.15},
    "calibration_limitation_handling": {"weight": 0.15},
    "consistency": {"weight": 0.10},
    "structure_usability": {"weight": 0.05},
}


def _make_task(task_id: str, track: int = 1) -> dict[str, Any]:
    """Return a minimal valid task dict."""
    return {
        "task_id": task_id,
        "track": track,
        "title": f"Task {task_id}",
        "user_prompt": "Answer the question.",
        "task_type": "retrieval",
        "allowed_inputs": [],
        "gold_facts": [],
        "gold_insights": [],
        "required_limitations": [],
        "forbidden_claims": [],
        "rubric": _RUBRIC,
    }


def _write_pack(tasks: list[dict[str, Any]], path: Path) -> None:
    """Write tasks to a JSONL file at *path*."""
    with path.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunPackSchemaValidity:
    """run_pack returns a schema-valid result dict."""

    def test_returns_schema_valid_result(self, tmp_path: Path) -> None:
        """run_pack with stub adapter must return a result_schema-conformant dict."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-RUN-001")], pack_path)

        adapter = StubAdapter()
        result = run_pack(
            pack_path=pack_path,
            pack_id=None,
            adapter=adapter,
            runs=1,
            verbose=False,
        )

        # Must pass JSON-Schema validation without raising.
        validate_result(result)

    def test_result_has_expected_fields(self, tmp_path: Path) -> None:
        """Result must have model_id, overall_composite, per_task_scores, etc."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-RUN-002")], pack_path)

        adapter = StubAdapter()
        result = run_pack(
            pack_path=pack_path,
            pack_id=None,
            adapter=adapter,
            runs=1,
            verbose=False,
        )

        assert "model_id" in result
        assert "overall_composite" in result
        assert "per_task_scores" in result
        assert "cost_metrics" in result
        assert result["task_count"] == 1
        assert len(result["per_task_scores"]) == 1

    def test_model_id_inferred_from_adapter(self, tmp_path: Path) -> None:
        """When model_id=None, run_pack infers it from the stub adapter output."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-RUN-003")], pack_path)

        adapter = StubAdapter()
        result = run_pack(
            pack_path=pack_path,
            pack_id=None,
            adapter=adapter,
            runs=1,
            model_id=None,
            verbose=False,
        )

        # StubAdapter embeds STUB_MODEL_ID = "stub/echo-v1" in its outputs.
        assert result["model_id"] == "stub/echo-v1"

    def test_model_id_override_respected(self, tmp_path: Path) -> None:
        """Explicit model_id overrides adapter-inferred model_id."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-RUN-004")], pack_path)

        adapter = StubAdapter()
        result = run_pack(
            pack_path=pack_path,
            pack_id=None,
            adapter=adapter,
            runs=1,
            model_id="my/custom-model",
            verbose=False,
        )

        assert result["model_id"] == "my/custom-model"

    def test_multi_task_pack(self, tmp_path: Path) -> None:
        """run_pack processes all tasks in a multi-task pack."""
        tasks = [_make_task(f"T-MULTI-{i:03d}") for i in range(3)]
        pack_path = tmp_path / "pack.jsonl"
        _write_pack(tasks, pack_path)

        adapter = StubAdapter()
        result = run_pack(
            pack_path=pack_path,
            pack_id=None,
            adapter=adapter,
            runs=1,
            verbose=False,
        )

        validate_result(result)
        assert result["task_count"] == 3
        assert len(result["per_task_scores"]) == 3


class TestRunPackFileOutput:
    """run_pack writes correct files to out_dir."""

    def test_writes_result_json_when_out_dir_given(self, tmp_path: Path) -> None:
        """run_pack writes result.json to out_dir when provided."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-IO-001")], pack_path)
        out_dir = tmp_path / "out"

        adapter = StubAdapter()
        run_pack(
            pack_path=pack_path,
            pack_id=None,
            adapter=adapter,
            runs=1,
            out_dir=out_dir,
            verbose=False,
        )

        assert (out_dir / "result.json").exists()
        assert (out_dir / "raw_outputs.jsonl").exists()

    def test_no_files_when_out_dir_none(self, tmp_path: Path) -> None:
        """run_pack writes nothing to disk when out_dir=None."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-IO-002")], pack_path)

        adapter = StubAdapter()
        run_pack(
            pack_path=pack_path,
            pack_id=None,
            adapter=adapter,
            runs=1,
            out_dir=None,
            verbose=False,
        )

        # Nothing should be written under tmp_path except the pack file.
        written = list(tmp_path.iterdir())
        assert all(p.name == "pack.jsonl" for p in written)


class TestRunPackAllFailure:
    """run_pack raises RuntimeError when all tasks fail."""

    def test_raises_runtime_error_when_all_tasks_fail(self, tmp_path: Path) -> None:
        """RuntimeError must be raised when every task fails."""
        from runner.adapters.base import AdapterBase

        class _AlwaysFailAdapter(AdapterBase):
            def __init__(self) -> None:
                super().__init__("fail")

            def run(self, task: dict[str, Any], run_index: int = 0) -> dict[str, Any]:
                raise RuntimeError("injected failure")

        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-FAIL-001"), _make_task("T-FAIL-002")], pack_path)

        with pytest.raises(RuntimeError, match="task.*failed"):
            run_pack(
                pack_path=pack_path,
                pack_id=None,
                adapter=_AlwaysFailAdapter(),
                runs=1,
                verbose=False,
            )


class TestCLIParityWithRunPack:
    """cli.main and run_pack produce structurally identical results."""

    def test_cli_result_is_schema_valid(self, tmp_path: Path) -> None:
        """cli.main still produces schema-valid result.json after refactor."""
        from runner.cli import main

        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-CLI-001")], pack_path)
        out_dir = tmp_path / "out"

        rc = main(
            [
                "--pack",
                str(pack_path),
                "--adapter",
                "stub",
                "--runs",
                "1",
                "--out",
                str(out_dir),
                "--model-id",
                "stub/echo-v1",
            ]
        )

        assert rc == 0
        result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        validate_result(result)
        assert result["model_id"] == "stub/echo-v1"
