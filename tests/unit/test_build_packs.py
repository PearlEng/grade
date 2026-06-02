"""Unit tests for scripts/build_packs.py — pack assembly and --check mode.

Guarantees tested:

1. ``--check`` exits 0 when the three pack JSONLs are present and consistent.
2. Every per-track source task appears in exactly one pack (no gaps, no duplicates).
3. Task counts add up: sum across packs equals the total task count across all
   five track JSONL files.
4. The ``build()`` helper correctly assigns tracks to their expected packs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_packs import (
    _PACK_PATHS,
    _TRACKS,
    _assign_pack,
    _read_tasks,
    build,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[2]


def _count_source_tasks() -> int:
    """Return total number of tasks across all five track JSONL files."""
    total = 0
    for path in _TRACKS.values():
        total += len(_read_tasks(path))
    return total


def _all_source_task_ids() -> set[str]:
    """Return set of all task_ids from all five track JSONL files."""
    ids: set[str] = set()
    for path in _TRACKS.values():
        for task in _read_tasks(path):
            ids.add(task["task_id"])
    return ids


# ---------------------------------------------------------------------------
# --check mode
# ---------------------------------------------------------------------------


class TestCheckMode:
    """Tests for the --check CLI flag."""

    def test_check_exits_zero(self) -> None:
        """``python -m scripts.build_packs --check`` must exit 0."""
        exit_code = main(["--check"])
        assert exit_code == 0, "build_packs --check returned non-zero exit code"


# ---------------------------------------------------------------------------
# Task count invariants
# ---------------------------------------------------------------------------


class TestTaskCounts:
    """Verify that task counts across the three packs are consistent."""

    def test_total_pack_count_equals_source_count(self) -> None:
        """Sum of tasks across all three packs must equal total source tasks."""
        packs = build(dry_run=True)
        pack_total = sum(len(tasks) for tasks in packs.values())
        source_total = _count_source_tasks()
        assert pack_total == source_total, (
            f"Pack total ({pack_total}) != source total ({source_total})"
        )

    def test_every_source_task_in_exactly_one_pack(self) -> None:
        """Every source task_id must appear in exactly one pack — no gaps or duplicates."""
        packs = build(dry_run=True)
        source_ids = _all_source_task_ids()

        assigned: dict[str, str] = {}  # task_id -> pack_name
        duplicates: list[str] = []

        for pack_name, tasks in packs.items():
            for task in tasks:
                tid = task["task_id"]
                if tid in assigned:
                    duplicates.append(tid)
                else:
                    assigned[tid] = pack_name

        assert not duplicates, f"Tasks appear in multiple packs: {duplicates}"

        missing = source_ids - set(assigned)
        assert not missing, f"Source tasks not assigned to any pack: {missing}"

    def test_pack_task_ids_are_unique_within_each_pack(self) -> None:
        """task_id values within a single pack must be unique."""
        packs = build(dry_run=True)
        for pack_name, tasks in packs.items():
            ids = [t["task_id"] for t in tasks]
            assert len(ids) == len(set(ids)), f"Duplicate task_ids in {pack_name}: {ids}"

    def test_operations_pack_contains_track1_and_track3(self) -> None:
        """Operations pack must contain all Track 1 and Track 3 tasks."""
        packs = build(dry_run=True)
        ops_tracks = {t["track"] for t in packs["operations"]}
        assert 1 in ops_tracks, "Track 1 tasks missing from operations pack"
        assert 3 in ops_tracks, "Track 3 tasks missing from operations pack"

    def test_outcomes_pack_contains_track2_only(self) -> None:
        """Outcomes pack must contain only Track 2 tasks."""
        packs = build(dry_run=True)
        tracks = {t["track"] for t in packs["outcomes"]}
        assert tracks == {2}, f"Unexpected tracks in outcomes pack: {tracks}"

    def test_equity_research_pack_contains_track4_and_track5(self) -> None:
        """Equity & Research pack must contain all Track 4 and Track 5 tasks."""
        packs = build(dry_run=True)
        eq_tracks = {t["track"] for t in packs["equity_research"]}
        assert 4 in eq_tracks, "Track 4 tasks missing from equity_research pack"
        assert 5 in eq_tracks, "Track 5 tasks missing from equity_research pack"


# ---------------------------------------------------------------------------
# Pack assignment logic
# ---------------------------------------------------------------------------


class TestAssignPack:
    """Unit tests for the _assign_pack() helper."""

    def test_track1_maps_to_operations(self) -> None:
        """Track 1 tasks always map to the operations pack."""
        task = {"track": 1, "task_id": "T1-OPS-001"}
        assert _assign_pack(task) == "operations"

    def test_track2_maps_to_outcomes(self) -> None:
        """Track 2 tasks always map to the outcomes pack."""
        task = {"track": 2, "task_id": "T2-OUT-001"}
        assert _assign_pack(task) == "outcomes"

    def test_track3_maps_to_operations_by_default(self) -> None:
        """Track 3 tasks map to operations unless overridden."""
        task = {"track": 3, "task_id": "T3-OPS-001"}
        assert _assign_pack(task) == "operations"

    def test_track4_maps_to_equity_research(self) -> None:
        """Track 4 tasks always map to the equity_research pack."""
        task = {"track": 4, "task_id": "T4-EQU-001"}
        assert _assign_pack(task) == "equity_research"

    def test_track5_maps_to_equity_research(self) -> None:
        """Track 5 tasks always map to the equity_research pack."""
        task = {"track": 5, "task_id": "T5-EFR-001"}
        assert _assign_pack(task) == "equity_research"


# ---------------------------------------------------------------------------
# Written pack file integrity
# ---------------------------------------------------------------------------


class TestWrittenPackFiles:
    """Verify the on-disk pack JSONLs are readable and have correct task counts."""

    @pytest.mark.parametrize("pack_name", list(_PACK_PATHS))
    def test_pack_file_exists(self, pack_name: str) -> None:
        """Each pack file must exist on disk after a build run."""
        assert _PACK_PATHS[pack_name].exists(), f"Pack file not found: {_PACK_PATHS[pack_name]}"

    @pytest.mark.parametrize("pack_name", list(_PACK_PATHS))
    def test_pack_file_is_valid_jsonl(self, pack_name: str) -> None:
        """Each line in a pack file must be valid JSON."""
        path = _PACK_PATHS[pack_name]
        if not path.exists():
            pytest.skip(f"{pack_name} pack not yet built")
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                stripped = line.strip()
                if stripped:
                    try:
                        json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        pytest.fail(f"Invalid JSON on line {i} of {pack_name}: {exc}")

    def test_written_pack_totals_match_source(self) -> None:
        """Sum of written pack file line counts must equal total source tasks."""
        total_written = 0
        for pack_name, path in _PACK_PATHS.items():
            if not path.exists():
                pytest.skip(f"{pack_name} pack not yet built")
            tasks = _read_tasks(path)
            total_written += len(tasks)
        source_total = _count_source_tasks()
        assert total_written == source_total, (
            f"Written total ({total_written}) != source total ({source_total})"
        )
