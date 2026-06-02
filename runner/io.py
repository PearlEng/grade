"""IO helpers — write raw outputs and scored results to disk.

Two functions are exported:

- :func:`write_raw_outputs` — dumps each per-run output dict as a JSONL file.
- :func:`write_result` — serialises the final ``result_schema.json``-shaped
  scorecard to a JSON file and validates it before writing.

Output paths
------------
Given ``--out /path/to/out_dir`` and ``--pack operations``:

- ``/path/to/out_dir/raw_outputs.jsonl`` — one line per (task × run) output.
- ``/path/to/out_dir/result.json`` — the full scorecard.

The output directory is created if it does not exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.schemas import validate_result


def write_raw_outputs(
    outputs: list[dict[str, Any]],
    out_dir: Path,
    filename: str = "raw_outputs.jsonl",
) -> Path:
    """Write all per-run raw output dicts to a JSONL file.

    Args:
        outputs: List of model output dicts conforming to
            ``output_schema.json``.  One entry per (task × run) pair.
        out_dir: Directory to write the file into.  Created if absent.
        filename: File name for the JSONL output.  Defaults to
            ``"raw_outputs.jsonl"``.

    Returns:
        Absolute path of the written file.

    Example::

        path = write_raw_outputs(all_outputs, Path("/tmp/run_001"))
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as fh:
        for record in outputs:
            fh.write(json.dumps(record, default=str) + "\n")
    return out_path


def write_result(
    result: dict[str, Any],
    out_dir: Path,
    filename: str = "result.json",
    validate: bool = True,
) -> Path:
    """Validate and write the final scorecard result to a JSON file.

    Args:
        result: A dict conforming to ``result_schema.json``.
        out_dir: Directory to write the file into.  Created if absent.
        filename: File name for the result JSON.  Defaults to
            ``"result.json"``.
        validate: If ``True`` (the default), validate *result* against the
            GRADE ``result_schema.json`` before writing.  Set to ``False``
            only in exceptional debugging scenarios.

    Returns:
        Absolute path of the written file.

    Raises:
        jsonschema.ValidationError: If *validate* is ``True`` and *result*
            does not conform to ``result_schema.json``.

    Example::

        path = write_result(scorecard, Path("/tmp/run_001"))
    """
    if validate:
        validate_result(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
        fh.write("\n")
    return out_path
