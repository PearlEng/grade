"""GRADE public schema definitions and validation helpers.

This package exposes the four canonical contracts for the GRADE benchmark:

- ``task_schema.json`` — task definition (question, rubric, gold facts).
- ``output_schema.json`` — normalized model output produced by runner adapters.
- ``fixture_schema.md`` — human-readable column contracts for pack fixture files.
- ``result_schema.json`` — per-task/track/pack scorecard result.

The :func:`validate_task`, :func:`validate_output`, and :func:`validate_result` helpers
load the corresponding JSON Schema (draft 2020-12) and raise
:exc:`jsonschema.ValidationError` on any contract violation.

Design choice — ``additionalProperties``:
    All three JSON schemas use ``"additionalProperties": false`` at the top level.
    This means unknown top-level keys cause validation to fail.  The rationale is
    correctness-by-default: downstream consumers (scorers, the report generator, the
    Arena) rely on well-typed payloads; silently accepting extra keys would allow
    schema drift to go undetected.  If a new field is needed, it must be added to the
    schema explicitly — this is intentional gate-keeping for a keystone contract.
    Nested ``$defs`` also use ``additionalProperties: false`` for the same reason.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import jsonschema
import jsonschema.validators

_SCHEMA_DIR = Path(__file__).parent

_TASK_SCHEMA_PATH = _SCHEMA_DIR / "task_schema.json"
_OUTPUT_SCHEMA_PATH = _SCHEMA_DIR / "output_schema.json"
_RESULT_SCHEMA_PATH = _SCHEMA_DIR / "result_schema.json"


@functools.cache
def _load_schema(path: Path) -> dict:
    """Load and cache a JSON Schema file from *path*.

    Args:
        path: Absolute path to the ``.json`` schema file.

    Returns:
        The parsed schema as a plain ``dict``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _validate(payload: dict, schema_path: Path) -> None:
    """Validate *payload* against the JSON Schema at *schema_path*.

    Uses the ``jsonschema`` registry-based validator so that ``$ref`` within a
    schema resolves correctly without external network access.

    Args:
        payload: The dictionary to validate.
        schema_path: Path to the JSON Schema file.

    Raises:
        jsonschema.ValidationError: If *payload* does not conform to the schema.
        jsonschema.SchemaError: If the schema itself is malformed.
    """
    schema = _load_schema(schema_path)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    validator = validator_cls(schema, format_checker=jsonschema.FormatChecker())
    validator.validate(payload)


def validate_task(payload: dict) -> None:
    """Validate a task definition payload against the GRADE task schema.

    The task schema enforces the contract for ``benchmark/tasks/`` JSON files and
    any programmatically constructed task dicts.  All required top-level fields
    (``task_id``, ``track``, ``title``, ``user_prompt``, ``task_type``,
    ``allowed_inputs``, ``gold_facts``, ``gold_insights``,
    ``required_limitations``, ``forbidden_claims``, ``rubric``) must be present.
    Unknown top-level keys are rejected (``additionalProperties: false``).

    Args:
        payload: Dictionary representing a task definition.

    Raises:
        jsonschema.ValidationError: If *payload* is invalid.

    Example::

        from benchmark.schemas import validate_task
        validate_task(task_dict)  # raises on invalid
    """
    _validate(payload, _TASK_SCHEMA_PATH)


def validate_output(payload: dict) -> None:
    """Validate a normalized model output payload against the GRADE output schema.

    The output schema enforces the contract that runner adapters must satisfy
    before handing responses to scorers.  Required fields: ``task_id``,
    ``model_id``, ``run_index``, ``structured_metrics``, ``key_findings``,
    ``limitations``, ``evidence_citations``, ``runtime_metadata``.

    Args:
        payload: Dictionary representing a normalized model output.

    Raises:
        jsonschema.ValidationError: If *payload* is invalid.

    Example::

        from benchmark.schemas import validate_output
        validate_output(output_dict)  # raises on invalid
    """
    _validate(payload, _OUTPUT_SCHEMA_PATH)


def validate_result(payload: dict) -> None:
    """Validate a scorecard result payload against the GRADE result schema.

    The result schema enforces the contract for scorecard JSON files produced by
    the C9 report generator.  Required fields: ``result_id``, ``model_id``,
    ``grade_version``, ``scored_at_utc``, ``overall_scores``,
    ``per_track_scores``, ``per_pack_scores``, ``per_task_scores``.

    Args:
        payload: Dictionary representing a scorecard result.

    Raises:
        jsonschema.ValidationError: If *payload* is invalid.

    Example::

        from benchmark.schemas import validate_result
        validate_result(result_dict)  # raises on invalid
    """
    _validate(payload, _RESULT_SCHEMA_PATH)


__all__ = [
    "validate_task",
    "validate_output",
    "validate_result",
]
