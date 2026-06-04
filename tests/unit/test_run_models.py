"""Tests for scripts.run_models — the multi-model batch runner.

Coverage:
- Batch over 2 stub model names writes 2 per-model result.json files.
- Batch over 2 stub model names writes leaderboard.md and leaderboard.json.
- A model that raises during run_pack is recorded and the batch continues.
- Failed models appear in the final return code (non-zero).
- Successful models still produce artefacts when another model fails.
- _safe_slug sanitises model IDs to filesystem-safe names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.run_models import _safe_slug, main

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


def _make_task(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "track": 1,
        "title": f"Task {task_id}",
        "user_prompt": "Answer.",
        "task_type": "retrieval",
        "allowed_inputs": [],
        "gold_facts": [],
        "gold_insights": [],
        "required_limitations": [],
        "forbidden_claims": [],
        "rubric": _RUBRIC,
    }


def _write_pack(tasks: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")


# ---------------------------------------------------------------------------
# Tests: slug sanitisation
# ---------------------------------------------------------------------------


class TestSafeSlug:
    """_safe_slug converts model IDs to filesystem-safe strings."""

    def test_slashes_become_underscores(self) -> None:
        """Slashes in model IDs are converted to underscores."""
        assert _safe_slug("anthropic/claude-haiku-3") == "anthropic_claude-haiku-3"

    def test_colons_become_underscores(self) -> None:
        """Colons in model IDs are converted to underscores."""
        assert _safe_slug("provider:model:v1") == "provider_model_v1"

    def test_already_safe_unchanged(self) -> None:
        """IDs that are already filesystem-safe pass through unchanged."""
        assert _safe_slug("my-model_v1.0") == "my-model_v1.0"

    def test_consecutive_underscores_collapsed(self) -> None:
        """Multiple consecutive unsafe characters collapse to a single underscore."""
        slug = _safe_slug("a//b")
        assert "__" not in slug
        assert slug == "a_b"


# ---------------------------------------------------------------------------
# Tests: batch over 2 models
# ---------------------------------------------------------------------------


class TestBatchTwoModels:
    """Batch with 2 stub models writes per-model results and leaderboard."""

    def test_writes_two_result_json_files(self, tmp_path: Path) -> None:
        """Each model must have its own result.json under <out>/<slug>/."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-BATCH-001")], pack_path)
        out_dir = tmp_path / "out"

        rc = main(
            [
                "--models",
                "stub/model-a,stub/model-b",
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

        assert rc == 0
        slug_a = _safe_slug("stub/model-a")
        slug_b = _safe_slug("stub/model-b")
        assert (out_dir / slug_a / "result.json").exists()
        assert (out_dir / slug_b / "result.json").exists()

    def test_writes_leaderboard_files(self, tmp_path: Path) -> None:
        """leaderboard.json and leaderboard.md must be written after batch."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-BATCH-002")], pack_path)
        out_dir = tmp_path / "out"

        main(
            [
                "--models",
                "stub/model-a,stub/model-b",
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

        assert (out_dir / "leaderboard.json").exists()
        assert (out_dir / "leaderboard.md").exists()

    def test_leaderboard_json_structure(self, tmp_path: Path) -> None:
        """leaderboard.json must have entries, model_count, dimension_weights."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-BATCH-003")], pack_path)
        out_dir = tmp_path / "out"

        main(
            [
                "--models",
                "stub/model-a,stub/model-b",
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

        lb = json.loads((out_dir / "leaderboard.json").read_text(encoding="utf-8"))
        assert "entries" in lb
        assert "model_count" in lb
        assert "dimension_weights" in lb
        assert lb["model_count"] == 2
        assert len(lb["entries"]) == 2

    def test_leaderboard_sorted_by_composite(self, tmp_path: Path) -> None:
        """Entries in leaderboard.json must be sorted composite descending."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-BATCH-004")], pack_path)
        out_dir = tmp_path / "out"

        main(
            [
                "--models",
                "stub/model-x,stub/model-y",
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

        lb = json.loads((out_dir / "leaderboard.json").read_text(encoding="utf-8"))
        composites = [e["composite"] for e in lb["entries"] if e["composite"] is not None]
        assert composites == sorted(composites, reverse=True)

    def test_per_model_result_schema_valid(self, tmp_path: Path) -> None:
        """Each per-model result.json must pass result_schema validation."""
        from benchmark.schemas import validate_result

        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-BATCH-005")], pack_path)
        out_dir = tmp_path / "out"

        main(
            [
                "--models",
                "stub/m1,stub/m2",
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

        for model_id in ["stub/m1", "stub/m2"]:
            slug = _safe_slug(model_id)
            result = json.loads((out_dir / slug / "result.json").read_text(encoding="utf-8"))
            validate_result(result)


# ---------------------------------------------------------------------------
# Tests: per-model resilience
# ---------------------------------------------------------------------------


class TestBatchModelFailure:
    """A model that raises is recorded; the batch continues with other models."""

    def _make_failing_run_pack(
        self,
        failing_model: str,
    ) -> Any:
        """Return a run_pack replacement that raises for *failing_model*."""
        from runner.run import run_pack as _real_run_pack

        def _patched_run_pack(
            pack_path: Path,
            pack_id: str | None,
            adapter: Any,
            runs: int = 5,
            judge_client: object | None = None,
            grade_version: str = "0.1.0",
            model_id: str | None = None,
            out_dir: Path | None = None,
            verbose: bool = True,
        ) -> dict[str, Any]:
            if model_id == failing_model:
                raise RuntimeError(f"Injected failure for {model_id}")
            return _real_run_pack(
                pack_path=pack_path,
                pack_id=pack_id,
                adapter=adapter,
                runs=runs,
                judge_client=judge_client,
                grade_version=grade_version,
                model_id=model_id,
                out_dir=out_dir,
                verbose=verbose,
            )

        return _patched_run_pack

    def test_returns_nonzero_when_a_model_fails(self, tmp_path: Path) -> None:
        """Exit code must be 1 when at least one model fails."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-FAIL-001")], pack_path)
        out_dir = tmp_path / "out"

        patched = self._make_failing_run_pack("stub/bad-model")

        with patch("scripts.run_models.run_pack", patched):
            rc = main(
                [
                    "--models",
                    "stub/good-model,stub/bad-model",
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

        assert rc == 1

    def test_good_model_result_written_when_another_fails(self, tmp_path: Path) -> None:
        """result.json for the successful model must be written even if another fails."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-FAIL-002")], pack_path)
        out_dir = tmp_path / "out"

        patched = self._make_failing_run_pack("stub/bad-model")

        with patch("scripts.run_models.run_pack", patched):
            main(
                [
                    "--models",
                    "stub/good-model,stub/bad-model",
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

        good_slug = _safe_slug("stub/good-model")
        assert (out_dir / good_slug / "result.json").exists()

    def test_leaderboard_built_from_successful_models(self, tmp_path: Path) -> None:
        """Leaderboard must include only the model that succeeded."""
        pack_path = tmp_path / "pack.jsonl"
        _write_pack([_make_task("T-FAIL-003")], pack_path)
        out_dir = tmp_path / "out"

        patched = self._make_failing_run_pack("stub/bad-model")

        with patch("scripts.run_models.run_pack", patched):
            main(
                [
                    "--models",
                    "stub/good-model,stub/bad-model",
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

        lb = json.loads((out_dir / "leaderboard.json").read_text(encoding="utf-8"))
        model_ids = [e["model_id"] for e in lb["entries"]]
        assert "stub/good-model" in model_ids
        assert "stub/bad-model" not in model_ids
        assert lb["model_count"] == 1
