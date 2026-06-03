"""Tests for the inline-fixture feature (C6 grounding fix).

Covers:
- :func:`~runner.adapters.openrouter_adapter._build_prompt`: fixture contents
  are embedded verbatim in the prompt under ``=== FILE: <name> ===`` headers.
- :func:`~runner.dispatcher._resolve_fixtures`: files are resolved from the
  correct pack directory given a ``pack_id``; operations ``students.csv``
  contains 135-student data (SCH-001=60).
- :func:`~runner.dispatcher.run_task`: the stub adapter still works end-to-end
  when the task dict is enriched with ``fixtures``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.adapters.openrouter_adapter import _build_prompt
from runner.adapters.stub_adapter import StubAdapter
from runner.dispatcher import _resolve_fixtures, run_task

# ---------------------------------------------------------------------------
# Shared minimal task used across multiple test classes
# ---------------------------------------------------------------------------

_MINIMAL_TASK: dict[str, Any] = {
    "task_id": "T1-OPS-001",
    "track": 1,
    "title": "Count enrolled students",
    "user_prompt": "How many students are enrolled?",
    "task_type": "retrieval",
    "allowed_inputs": ["students.csv"],
    "gold_facts": [
        {
            "fact_id": "F1",
            "claim": "135 students are enrolled.",
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
# _build_prompt — fixture content embedding
# ---------------------------------------------------------------------------


class TestBuildPromptFixtureEmbedding:
    """Tests for :func:`_build_prompt` with the ``fixtures`` key populated."""

    def test_fixture_contents_appear_in_prompt(self) -> None:
        """A known value from the fixture must appear verbatim in the prompt."""
        task: dict[str, Any] = {
            **_MINIMAL_TASK,
            "fixtures": {"students.csv": "student_id,name\nSTU-001,Alice\nSTU-002,Bob\n"},
        }
        prompt = _build_prompt(task)
        assert "STU-001" in prompt, "Student ID from fixture must appear in the prompt"
        assert "Alice" in prompt, "Student name from fixture must appear in the prompt"

    def test_file_delimiter_header_present(self) -> None:
        """The ``=== FILE: <name> ===`` delimiter must wrap every fixture block."""
        task: dict[str, Any] = {
            **_MINIMAL_TASK,
            "fixtures": {"students.csv": "col1,col2\n1,2\n"},
        }
        prompt = _build_prompt(task)
        assert "=== FILE: students.csv ===" in prompt
        assert "=== END FILE: students.csv ===" in prompt

    def test_multiple_fixtures_all_embedded(self) -> None:
        """All fixtures referenced in allowed_inputs must appear in the prompt."""
        task: dict[str, Any] = {
            **_MINIMAL_TASK,
            "allowed_inputs": ["students.csv", "groups.csv"],
            "fixtures": {
                "students.csv": "student_id\nSTU-001\n",
                "groups.csv": "group_id\nGRP-001\n",
            },
        }
        prompt = _build_prompt(task)
        assert "=== FILE: students.csv ===" in prompt
        assert "=== END FILE: students.csv ===" in prompt
        assert "=== FILE: groups.csv ===" in prompt
        assert "=== END FILE: groups.csv ===" in prompt
        assert "STU-001" in prompt
        assert "GRP-001" in prompt

    def test_no_fixtures_falls_back_to_filename_list(self) -> None:
        """When ``fixtures`` is absent the prompt should list filenames instead."""
        # No ``fixtures`` key in the task dict.
        prompt = _build_prompt(_MINIMAL_TASK)
        assert "students.csv" in prompt
        # Content delimiters must NOT appear (no data to embed).
        assert "=== FILE:" not in prompt

    def test_empty_fixtures_dict_falls_back_to_filename_list(self) -> None:
        """An empty ``fixtures`` dict should behave like the no-fixtures path."""
        task: dict[str, Any] = {**_MINIMAL_TASK, "fixtures": {}}
        prompt = _build_prompt(task)
        assert "students.csv" in prompt
        assert "=== FILE:" not in prompt

    def test_prompt_contains_user_question(self) -> None:
        """The original user_prompt must still appear in the built prompt."""
        task: dict[str, Any] = {
            **_MINIMAL_TASK,
            "fixtures": {"students.csv": "col\nval\n"},
        }
        prompt = _build_prompt(task)
        assert _MINIMAL_TASK["user_prompt"] in prompt

    def test_prose_first_instructions_present(self) -> None:
        """The prose-first output guidance must appear regardless of fixtures.

        The prompt now instructs the model to write natural analysis rather than
        requiring rigid structured sections.  The old mandatory ``## Key Findings``
        and ``## Structured Metrics`` headers are replaced by a suggestion-based
        structure with prose-first framing.
        """
        task: dict[str, Any] = {
            **_MINIMAL_TASK,
            "fixtures": {"students.csv": "col\nval\n"},
        }
        prompt = _build_prompt(task)
        # Prose-first instruction language must be present.
        assert "natural analysis" in prompt or "plain prose" in prompt
        # The suggested (optional) structure should still reference common topics.
        assert "caveats" in prompt or "limitations" in prompt or "Suggested structure" in prompt

    def test_missing_fixture_file_noted_in_prompt(self) -> None:
        """If a file in allowed_inputs is absent from fixtures, it must be noted."""
        task: dict[str, Any] = {
            **_MINIMAL_TASK,
            "allowed_inputs": ["students.csv", "groups.csv"],
            # Only students.csv provided — groups.csv is missing.
            "fixtures": {"students.csv": "col\nval\n"},
        }
        prompt = _build_prompt(task)
        # groups.csv should be mentioned as unavailable.
        assert "groups.csv" in prompt

    def test_fixture_content_fully_inlined(self, tmp_path: Path) -> None:
        """Every byte of the fixture string must appear in the prompt (no truncation)."""
        # Construct a fixture that is several hundred lines long.
        rows = ["id,value"] + [f"ROW-{i:04d},{i * 7}" for i in range(300)]
        content = "\n".join(rows) + "\n"
        task: dict[str, Any] = {
            **_MINIMAL_TASK,
            "fixtures": {"students.csv": content},
        }
        prompt = _build_prompt(task)
        # The last row must appear verbatim.
        assert "ROW-0299" in prompt


# ---------------------------------------------------------------------------
# _resolve_fixtures — correct pack directory disambiguation
# ---------------------------------------------------------------------------


class TestResolveFixtures:
    """Tests for :func:`_resolve_fixtures` pack-directory resolution."""

    def test_operations_students_contains_135_rows(self) -> None:
        """Operations pack students.csv must have 135 data rows (SCH-001 gold fact)."""
        result = _resolve_fixtures(["students.csv"], "pack_operations")
        assert "students.csv" in result, "students.csv must be resolved from pack_operations"
        contents = result["students.csv"]
        # Count non-header rows (one per student).
        data_rows = [ln for ln in contents.splitlines() if ln.strip() and "student_id" not in ln]
        assert len(data_rows) == 135, f"Expected 135 student rows, got {len(data_rows)}"

    def test_operations_students_contains_sch001_60_rows(self) -> None:
        """Operations pack students.csv must contain exactly 60 rows for SCH-001."""
        result = _resolve_fixtures(["students.csv"], "pack_operations")
        contents = result["students.csv"]
        sch001_rows = [ln for ln in contents.splitlines() if "SCH-001" in ln]
        assert len(sch001_rows) == 60, f"Expected 60 SCH-001 rows, got {len(sch001_rows)}"

    def test_unknown_pack_returns_empty_dict(self) -> None:
        """An unrecognised pack_id must return an empty dict without raising."""
        result = _resolve_fixtures(["students.csv"], "pack_unknown")
        assert result == {}

    def test_none_pack_returns_empty_dict(self) -> None:
        """pack_id=None must return an empty dict without raising."""
        result = _resolve_fixtures(["students.csv"], None)
        assert result == {}

    def test_empty_allowed_inputs_returns_empty_dict(self) -> None:
        """An empty allowed_inputs list must return an empty dict."""
        result = _resolve_fixtures([], "pack_operations")
        assert result == {}

    def test_nonexistent_file_omitted_silently(self) -> None:
        """A filename that does not exist in the pack dir must be silently omitted."""
        result = _resolve_fixtures(["no_such_file.csv"], "pack_operations")
        assert "no_such_file.csv" not in result

    def test_multiple_files_resolved(self) -> None:
        """Multiple valid filenames must all be resolved."""
        result = _resolve_fixtures(["students.csv", "groups.csv"], "pack_operations")
        assert "students.csv" in result
        assert "groups.csv" in result

    def test_returns_string_contents(self) -> None:
        """Each resolved fixture value must be a non-empty string."""
        result = _resolve_fixtures(["students.csv"], "pack_operations")
        assert isinstance(result["students.csv"], str)
        assert len(result["students.csv"]) > 0

    def test_fixture_resolved_from_own_pack_dir(self, tmp_path: Path) -> None:
        """Fixtures must be resolved from the task's own pack dir, not another pack.

        This test creates a minimal temporary fixture tree to make the assertion
        pack-dir-specific without relying on shared file identity across packs.
        The directory layout mirrors the real repo: ``fixtures/<pack_dir>/<file>``.
        """
        from unittest.mock import patch

        # Build two fake pack dirs under fixtures/ with students.csv distinct content.
        fixtures_root = tmp_path / "fixtures"
        pack_a = fixtures_root / "pack_a"
        pack_b = fixtures_root / "pack_b"
        pack_a.mkdir(parents=True)
        pack_b.mkdir(parents=True)
        (pack_a / "students.csv").write_text("id,school\nS1,A-SCHOOL\n", encoding="utf-8")
        (pack_b / "students.csv").write_text("id,school\nS1,B-SCHOOL\n", encoding="utf-8")

        fake_dirs = {"pack_a": "pack_a", "pack_b": "pack_b"}
        with (
            patch("runner.dispatcher._PACK_FIXTURE_DIRS", fake_dirs),
            patch("runner.dispatcher._REPO_ROOT", tmp_path),
        ):
            result_a = _resolve_fixtures(["students.csv"], "pack_a")
            result_b = _resolve_fixtures(["students.csv"], "pack_b")

        assert "A-SCHOOL" in result_a["students.csv"], "pack_a must return pack_a data"
        assert "B-SCHOOL" in result_b["students.csv"], "pack_b must return pack_b data"
        # Cross-contamination check: pack_a must NOT contain pack_b data.
        assert "B-SCHOOL" not in result_a["students.csv"]
        assert "A-SCHOOL" not in result_b["students.csv"]


# ---------------------------------------------------------------------------
# Stub adapter end-to-end — fixtures key does not break the stub
# ---------------------------------------------------------------------------


class TestStubAdapterWithFixtures:
    """Verify the stub adapter still works when the task has a ``fixtures`` key."""

    def test_stub_run_succeeds_with_fixtures_key(self) -> None:
        """StubAdapter.run must succeed when task contains a populated fixtures dict."""
        task: dict[str, Any] = {
            **_MINIMAL_TASK,
            "pack_id": "pack_operations",
            "fixtures": {"students.csv": "student_id,school_id\nSTU-001,SCH-001\n"},
        }
        adapter = StubAdapter()
        output = adapter.run(task, run_index=0)
        assert output["task_id"] == "T1-OPS-001"

    def test_stub_output_schema_valid_with_fixtures(self) -> None:
        """Output from stub must remain schema-valid when fixtures are present."""
        from benchmark.schemas import validate_output

        task: dict[str, Any] = {
            **_MINIMAL_TASK,
            "pack_id": "pack_operations",
            "fixtures": {"students.csv": "student_id,school_id\nSTU-001,SCH-001\n"},
        }
        adapter = StubAdapter()
        output = adapter.run(task, run_index=0)
        validate_output(output)  # must not raise

    def test_run_task_with_pack_id_attaches_fixtures(self) -> None:
        """run_task with pack_id='pack_operations' must attach fixture data to task."""
        # We verify indirectly: the stub doesn't use fixture data, but run_task
        # must complete without error, confirming the enrichment path is correct.
        adapter = StubAdapter()
        result = run_task(_MINIMAL_TASK, adapter, runs=1, pack_id="pack_operations")
        assert result.task_id == "T1-OPS-001"
        assert result.run_count == 1
        assert len(result.outputs) == 1

    def test_run_task_without_pack_id_still_works(self) -> None:
        """run_task with no pack_id must not raise even though fixtures are empty."""
        adapter = StubAdapter()
        result = run_task(_MINIMAL_TASK, adapter, runs=1, pack_id=None)
        assert result.task_id == "T1-OPS-001"
        assert len(result.outputs) == 1

    def test_run_task_unknown_pack_id_still_works(self) -> None:
        """run_task with an unrecognised pack_id must not raise."""
        adapter = StubAdapter()
        result = run_task(_MINIMAL_TASK, adapter, runs=1, pack_id="pack_unknown")
        assert result.task_id == "T1-OPS-001"
        assert len(result.outputs) == 1
