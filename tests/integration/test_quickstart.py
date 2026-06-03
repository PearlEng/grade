"""E5 smoke test — validates the quick-start commands from examples/quickstart.md.

All tests are network-free (stub adapter only) and require no API keys.

Coverage:
- The canonical quick-start command works end-to-end:
      python -m runner.cli --pack operations --adapter stub --runs 1 --out <dir>
- The result.json produced validates against result_schema.json.
- All five sample outputs in examples/sample_outputs/ validate against
  output_schema.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.schemas import validate_output, validate_result
from runner.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_SAMPLE_OUTPUTS_DIR = _REPO_ROOT / "examples" / "sample_outputs"


def _sample_output_files() -> list[Path]:
    """Return all JSON files in examples/sample_outputs/."""
    if not _SAMPLE_OUTPUTS_DIR.exists():
        return []
    return sorted(_SAMPLE_OUTPUTS_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Quick-start smoke test
# ---------------------------------------------------------------------------


class TestQuickStartCommand:
    """Verify the canonical quick-start command from examples/quickstart.md."""

    def test_quickstart_command_succeeds(self, tmp_path: Path) -> None:
        """Running the exact quick-start command produces a valid result.json.

        Equivalent to::

            python -m runner.cli --pack operations --adapter stub --runs 1 --out <dir>
        """
        out_dir = tmp_path / "grade_out"

        exit_code = main(
            [
                "--pack",
                "operations",
                "--adapter",
                "stub",
                "--runs",
                "1",
                "--out",
                str(out_dir),
            ]
        )

        assert exit_code == 0, "Quick-start command must exit with code 0"
        assert (out_dir / "result.json").exists(), "result.json must be written"
        assert (out_dir / "raw_outputs.jsonl").exists(), "raw_outputs.jsonl must be written"

        result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        validate_result(result)  # raises jsonschema.ValidationError if invalid

    def test_outcomes_pack_quickstart(self, tmp_path: Path) -> None:
        """The outcomes pack quick-start command also succeeds end-to-end."""
        out_dir = tmp_path / "grade_out"
        exit_code = main(
            [
                "--pack",
                "outcomes",
                "--adapter",
                "stub",
                "--runs",
                "1",
                "--out",
                str(out_dir),
            ]
        )
        assert exit_code == 0
        validate_result(json.loads((out_dir / "result.json").read_text(encoding="utf-8")))

    def test_equity_research_pack_quickstart(self, tmp_path: Path) -> None:
        """The equity_research pack quick-start command also succeeds end-to-end."""
        out_dir = tmp_path / "grade_out"
        exit_code = main(
            [
                "--pack",
                "equity_research",
                "--adapter",
                "stub",
                "--runs",
                "1",
                "--out",
                str(out_dir),
            ]
        )
        assert exit_code == 0
        validate_result(json.loads((out_dir / "result.json").read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Sample output validation
# ---------------------------------------------------------------------------


class TestSampleOutputs:
    """All files in examples/sample_outputs/ must validate against output_schema.json."""

    @pytest.mark.parametrize(
        "sample_file",
        _sample_output_files(),
        ids=[f.name for f in _sample_output_files()],
    )
    def test_sample_output_validates(self, sample_file: Path) -> None:
        """Each sample output JSON must conform to output_schema.json."""
        output = json.loads(sample_file.read_text(encoding="utf-8"))
        validate_output(output)  # raises jsonschema.ValidationError if invalid

    def test_sample_outputs_cover_all_five_tracks(self) -> None:
        """examples/sample_outputs/ must contain one file per track (5 total)."""
        sample_files = _sample_output_files()
        assert len(sample_files) == 5, (
            f"Expected 5 sample output files (one per track), found {len(sample_files)}: "
            f"{[f.name for f in sample_files]}"
        )

    def test_sample_outputs_cover_distinct_tracks(self) -> None:
        """The five sample outputs must cover all five distinct track numbers."""
        sample_files = _sample_output_files()
        if not sample_files:
            pytest.skip("No sample output files found")

        tracks_seen: set[int] = set()
        for path in sample_files:
            output = json.loads(path.read_text(encoding="utf-8"))
            task_id: str = output["task_id"]
            # Track number is the digit after the first 'T' in the task_id, e.g. T1-OPS-001 → 1
            track_num = int(task_id[1])
            tracks_seen.add(track_num)

        assert tracks_seen == {1, 2, 3, 4, 5}, (
            f"Sample outputs must cover tracks 1–5; got tracks: {sorted(tracks_seen)}"
        )
