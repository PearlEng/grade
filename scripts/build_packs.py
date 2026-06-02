"""build_packs — Assemble per-track JSONL tasks into three fixture-pack JSONLs.

Overview
--------
The GRADE benchmark ships five task tracks whose tasks map onto three fixture packs:

  * **Operations pack** (``fixtures/pack_operations/``) — raw operational data:
    students, groups, sessions, attendance, tutors.
  * **Outcomes pack** (``fixtures/pack_outcomes/``) — pre-aggregated monthly
    summaries plus a ground-truth JSON.
  * **Equity & Research pack** (``fixtures/pack_equity_research/``) — subgroup
    summary files plus a research-references JSON.

Pack-assignment rules
---------------------
The assignment is **deterministic** and based on which fixture pack a task's
``allowed_inputs`` primarily draws from:

1. **Track 1 → Operations pack** (all tasks reference ``sessions.csv``,
   ``attendance.csv``, ``students.csv``, ``groups.csv`` — the raw operational
   fixtures in ``pack_operations``).

2. **Track 2 → Outcomes pack** (all tasks reference
   ``monthly_attendance_summary.csv``, ``monthly_satisfaction_summary.csv``, or
   ``ground_truth.json`` — pre-aggregated summaries in ``pack_outcomes``).

3. **Track 3 → Operations pack** (all five coaching-recommendation tasks draw
   their primary evidence from the same raw operational fixtures used by Track 1:
   ``sessions.csv``, ``attendance.csv``, ``students.csv``, ``groups.csv``,
   ``tutors.csv``, ``program_context.json``).  Although Track 3 tasks synthesise
   insights across operational signals, no Track 3 task references any
   Outcomes-pack-exclusive file (e.g. ``monthly_attendance_summary.csv`` or
   ``monthly_satisfaction_summary.csv``).  All five tasks therefore land in the
   Operations pack.  If a future Track 3 task is authored that primarily draws on
   Outcomes-pack fixtures, add its ``task_id`` to ``_B3_OUTCOMES_OVERRIDES``
   below.

4. **Track 4 → Equity & Research pack** (all tasks reference
   ``subgroup_attendance_summary.csv`` or ``subgroup_outcomes_summary.csv`` —
   equity-specific summaries in ``pack_equity_research``).

5. **Track 5 → Equity & Research pack** (all tasks reference
   ``research_refs.json`` — the research evidence file in
   ``pack_equity_research``).

B3 override mechanism
---------------------
``_B3_OUTCOMES_OVERRIDES`` is a ``frozenset`` of Track 3 ``task_id`` values that
should be routed to the *Outcomes* pack rather than the default Operations pack.
Currently empty, because all five authored Track 3 tasks rely on Operations-pack
fixtures.  Add a task_id here if a new Track 3 task primarily references
``monthly_attendance_summary.csv`` or ``monthly_satisfaction_summary.csv``.

CLI usage
---------
::

    python -m scripts.build_packs          # write the three pack JSONLs
    python -m scripts.build_packs --check  # exit 0 iff consistent

``--check`` verifies:

  (a) Every per-track task is accounted for (no task is silently dropped).
  (b) Every task appears in exactly one pack (no duplicates).
  (c) All three packs are schema-valid (``validate_task`` on each line).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.schemas import validate_task

# ---------------------------------------------------------------------------
# Repository layout constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[1]

_TRACKS: dict[int, Path] = {
    1: _REPO_ROOT / "benchmark" / "tracks" / "grounded_retrieval" / "tasks.jsonl",
    2: _REPO_ROOT / "benchmark" / "tracks" / "snapshot_trends" / "tasks.jsonl",
    3: _REPO_ROOT / "benchmark" / "tracks" / "coaching_recommendations" / "tasks.jsonl",
    4: _REPO_ROOT / "benchmark" / "tracks" / "equity_interpretation" / "tasks.jsonl",
    5: _REPO_ROOT / "benchmark" / "tracks" / "effectiveness_research" / "tasks.jsonl",
}

_PACK_DIR = _REPO_ROOT / "benchmark" / "tasks"

_PACK_PATHS: dict[str, Path] = {
    "operations": _PACK_DIR / "operations_pack.jsonl",
    "outcomes": _PACK_DIR / "outcomes_pack.jsonl",
    "equity_research": _PACK_DIR / "equity_research_pack.jsonl",
}

# ---------------------------------------------------------------------------
# Pack-assignment configuration
# ---------------------------------------------------------------------------

# Default pack for each track number.
_TRACK_DEFAULT_PACK: dict[int, str] = {
    1: "operations",
    2: "outcomes",
    3: "operations",  # B3 tasks primarily reference Operations-pack fixtures
    4: "equity_research",
    5: "equity_research",
}

# Per-task overrides for Track 3 tasks that primarily reference Outcomes fixtures.
# Currently empty — no authored T3 task uses Outcomes-pack-exclusive files.
# To override, add the task_id: e.g. "T3-OUT-001": "outcomes"
_B3_OUTCOMES_OVERRIDES: frozenset[str] = frozenset()


def _assign_pack(task: dict) -> str:
    """Return the pack name for *task*.

    Uses the track-level default and applies any per-task override for Track 3.

    Args:
        task: Parsed task dict with at least ``track`` and ``task_id`` keys.

    Returns:
        One of ``"operations"``, ``"outcomes"``, or ``"equity_research"``.
    """
    track: int = task["track"]
    task_id: str = task["task_id"]

    if track == 3 and task_id in _B3_OUTCOMES_OVERRIDES:
        return "outcomes"

    return _TRACK_DEFAULT_PACK[track]


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------


def _read_tasks(path: Path) -> list[dict]:
    """Read all tasks from a JSONL file at *path*.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of parsed task dicts (blank lines are skipped).

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If any non-blank line is not valid JSON.
    """
    tasks: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                tasks.append(json.loads(stripped))
    return tasks


def _write_pack(tasks: list[dict], dest: Path) -> None:
    """Write *tasks* to *dest* as newline-delimited JSON.

    Args:
        tasks: List of task dicts to serialise.
        dest: Destination ``.jsonl`` path (parent must exist).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core assembly logic
# ---------------------------------------------------------------------------


def _collect_all_tasks() -> list[dict]:
    """Load and return all tasks from every track JSONL.

    Returns:
        Flat list of all task dicts across tracks 1–5.
    """
    all_tasks: list[dict] = []
    for track_num in sorted(_TRACKS):
        all_tasks.extend(_read_tasks(_TRACKS[track_num]))
    return all_tasks


def build(*, dry_run: bool = False) -> dict[str, list[dict]]:
    """Assemble all tasks into three packs and optionally write the JSONLs.

    Args:
        dry_run: If ``True``, skip writing files (used by ``--check``).

    Returns:
        Dict mapping pack name → list of task dicts in that pack.
    """
    packs: dict[str, list[dict]] = {name: [] for name in _PACK_PATHS}

    for task in _collect_all_tasks():
        pack = _assign_pack(task)
        packs[pack].append(task)

    if not dry_run:
        for pack_name, tasks in packs.items():
            _write_pack(tasks, _PACK_PATHS[pack_name])

    return packs


# ---------------------------------------------------------------------------
# --check mode
# ---------------------------------------------------------------------------


def check() -> int:
    """Run consistency checks on the three pack JSONLs.

    Checks:
      (a) Every per-track task appears in exactly one pack.
      (b) No task appears in more than one pack (no duplicates across packs).
      (c) Every task in every pack passes ``validate_task``.

    Returns:
        ``0`` if all checks pass, ``1`` otherwise (with diagnostics on stderr).
    """
    errors: list[str] = []

    # --- Load source tasks --------------------------------------------------
    source_tasks: dict[str, dict] = {}
    for _track_num, path in _TRACKS.items():
        for task in _read_tasks(path):
            tid = task["task_id"]
            if tid in source_tasks:
                errors.append(f"Duplicate task_id {tid!r} across source tracks.")
            else:
                source_tasks[tid] = task

    # --- Load pack tasks (from written files) --------------------------------
    pack_task_ids: dict[str, list[str]] = {}  # pack_name → [task_id, ...]
    pack_tasks_all: dict[str, dict] = {}  # task_id → task (for dup check)

    for pack_name, path in _PACK_PATHS.items():
        if not path.exists():
            errors.append(f"Pack file missing: {path}")
            pack_task_ids[pack_name] = []
            continue

        ids: list[str] = []
        for task in _read_tasks(path):
            tid = task["task_id"]
            ids.append(tid)

            # (b) duplicate across packs
            if tid in pack_tasks_all:
                errors.append(
                    f"Task {tid!r} appears in multiple packs "
                    f"(seen in {pack_tasks_all[tid]!r} and {pack_name!r})."
                )
            else:
                pack_tasks_all[tid] = task

            # (c) schema validity
            try:
                validate_task(task)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Schema validation failed for {tid!r} in {pack_name}: {exc}")

        pack_task_ids[pack_name] = ids

    # (a) every source task is accounted for
    for tid in source_tasks:
        if tid not in pack_tasks_all:
            errors.append(f"Source task {tid!r} is not present in any pack.")

    # report
    if errors:
        for err in errors:
            print(f"FAIL: {err}", file=sys.stderr)
        return 1

    # Print summary on success
    total = sum(len(ids) for ids in pack_task_ids.values())
    for pack_name, ids in pack_task_ids.items():
        print(f"  {pack_name}: {len(ids)} tasks")
    print(f"  total: {total} tasks — all checks passed.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.build_packs",
        description=(
            "Assemble GRADE per-track tasks into three fixture-pack JSONLs. "
            "Run without arguments to build; add --check to verify consistency."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify the three pack JSONLs are consistent and schema-valid "
            "without rebuilding them. Exits 0 on success, 1 on failure."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for build_packs.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    args = _parse_args(argv)

    if args.check:
        return check()

    packs = build()
    total = sum(len(tasks) for tasks in packs.values())
    for pack_name, tasks in packs.items():
        print(f"  {pack_name}: {len(tasks)} tasks → {_PACK_PATHS[pack_name]}")
    print(f"  total: {total} tasks written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
