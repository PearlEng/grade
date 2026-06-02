"""Unit tests for GRADE public schemas and validation helpers.

Design choice — additionalProperties:
    All three JSON schemas use ``"additionalProperties": false`` at the top level
    (and in all ``$defs``).  This means:

    - Valid payloads with ALL required fields pass.
    - Payloads missing required fields raise :exc:`jsonschema.ValidationError`.
    - Payloads with EXTRA unknown fields raise :exc:`jsonschema.ValidationError`.

    This is the intentional contract: the schemas are strict, version-gated
    keystone contracts.  Callers must not inject ad-hoc fields; schema changes
    must go through a deliberate versioning step.
"""

from __future__ import annotations

import copy

import jsonschema
import pytest

from benchmark.schemas import validate_output, validate_result, validate_task

# ---------------------------------------------------------------------------
# Minimal valid fixtures
# ---------------------------------------------------------------------------

_VALID_TASK: dict = {
    "task_id": "T1-OPS-001",
    "track": 1,
    "title": "Count active students in the program",
    "user_prompt": (
        "Using the provided fixture data, how many students are currently active in the program?"
    ),
    "task_type": "retrieval",
    "allowed_inputs": ["students.csv"],
    "gold_facts": [
        {
            "fact_id": "F1",
            "claim": "There are 42 active students.",
            "source_files": ["students.csv"],
            "numeric_value": 42,
            "tolerance": 0,
        }
    ],
    "gold_insights": ["The program has a relatively small active cohort."],
    "required_limitations": ["This count reflects the snapshot reporting period only."],
    "forbidden_claims": ["All students improved their grades."],
    "rubric": {
        "grounding_accuracy": {"weight": 0.35},
        "insight_quality": {"weight": 0.20},
        "evidence_linkage": {"weight": 0.15},
        "calibration_limitation_handling": {"weight": 0.15},
        "consistency": {"weight": 0.10},
        "structure_usability": {"weight": 0.05},
    },
}

_VALID_OUTPUT: dict = {
    "task_id": "T1-OPS-001",
    "model_id": "openrouter/anthropic/claude-sonnet-4-6",
    "run_index": 0,
    "structured_metrics": {"active_student_count": 42},
    "key_findings": ["There are 42 active students enrolled in the reporting period."],
    "limitations": ["Count reflects the snapshot period; students who left earlier are excluded."],
    "evidence_citations": [
        {
            "citation_text": "students.csv: 42 rows with active = true",
            "source_file": "students.csv",
            "source_column": "active",
            "grounded": None,
        }
    ],
    "runtime_metadata": {
        "adapter_version": "0.1.0",
        "timestamp_utc": "2026-06-01T12:00:00Z",
        "latency_ms": 1234.5,
        "prompt_tokens": 800,
        "completion_tokens": 200,
        "model_temperature": 0.0,
        "provider": "openrouter",
        "pack_id": "pack_operations",
    },
}

_VALID_RESULT: dict = {
    "result_id": "claude-sonnet-4-6-20260601-abc123",
    "model_id": "openrouter/anthropic/claude-sonnet-4-6",
    "grade_version": "0.1.0",
    "scored_at_utc": "2026-06-01T14:00:00Z",
    "overall_scores": {
        "grounding_accuracy": 0.85,
        "insight_quality": 0.72,
        "evidence_linkage": 0.68,
        "calibration_limitation_handling": 0.80,
        "consistency": 0.90,
        "structure_usability": 0.95,
    },
    "per_track_scores": {
        "1": {
            "track": 1,
            "track_name": "Grounded Retrieval & Computation",
            "task_count": 5,
            "scores": {
                "grounding_accuracy": 0.90,
                "insight_quality": 0.70,
                "evidence_linkage": 0.65,
                "calibration_limitation_handling": 0.75,
                "consistency": 0.88,
                "structure_usability": 0.92,
            },
            "composite": 0.82,
        }
    },
    "per_pack_scores": {
        "pack_operations": {
            "pack_id": "pack_operations",
            "task_count": 8,
            "scores": {
                "grounding_accuracy": 0.87,
                "insight_quality": 0.73,
                "evidence_linkage": 0.69,
                "calibration_limitation_handling": 0.81,
                "consistency": 0.91,
                "structure_usability": 0.94,
            },
            "composite": 0.83,
        }
    },
    "per_task_scores": [
        {
            "task_id": "T1-OPS-001",
            "track": 1,
            "pack_id": "pack_operations",
            "run_count": 5,
            "scores": {
                "grounding_accuracy": 1.0,
                "insight_quality": 0.75,
                "evidence_linkage": 0.80,
                "calibration_limitation_handling": 0.90,
                "consistency": 0.95,
                "structure_usability": 1.0,
            },
            "composite": 0.91,
            "scorer_flags": None,
        }
    ],
}


# ---------------------------------------------------------------------------
# Task schema tests
# ---------------------------------------------------------------------------


class TestTaskSchema:
    """Tests for validate_task()."""

    def test_valid_task_passes(self) -> None:
        """A fully populated valid task payload should pass without error."""
        validate_task(_VALID_TASK)

    def test_valid_task_with_optional_field_passes(self) -> None:
        """Adding the optional reference_answer_outline field should still pass."""
        payload = copy.deepcopy(_VALID_TASK)
        payload["reference_answer_outline"] = "Begin with active count, then note caveats."
        validate_task(payload)

    def test_valid_task_empty_lists_passes(self) -> None:
        """Tasks with empty gold_facts, gold_insights, forbidden_claims, required_limitations."""
        payload = copy.deepcopy(_VALID_TASK)
        payload["gold_facts"] = []
        payload["gold_insights"] = []
        payload["required_limitations"] = []
        payload["forbidden_claims"] = []
        validate_task(payload)

    def test_missing_task_id_fails(self) -> None:
        """Missing required task_id should raise ValidationError."""
        payload = copy.deepcopy(_VALID_TASK)
        del payload["task_id"]
        with pytest.raises(jsonschema.ValidationError, match="task_id"):
            validate_task(payload)

    def test_missing_track_fails(self) -> None:
        """Missing required track field should raise ValidationError."""
        payload = copy.deepcopy(_VALID_TASK)
        del payload["track"]
        with pytest.raises(jsonschema.ValidationError, match="track"):
            validate_task(payload)

    def test_missing_rubric_fails(self) -> None:
        """Missing required rubric should raise ValidationError."""
        payload = copy.deepcopy(_VALID_TASK)
        del payload["rubric"]
        with pytest.raises(jsonschema.ValidationError, match="rubric"):
            validate_task(payload)

    def test_missing_user_prompt_fails(self) -> None:
        """Missing user_prompt should raise ValidationError."""
        payload = copy.deepcopy(_VALID_TASK)
        del payload["user_prompt"]
        with pytest.raises(jsonschema.ValidationError, match="user_prompt"):
            validate_task(payload)

    def test_extra_top_level_field_fails(self) -> None:
        """Extra unknown top-level field must fail (additionalProperties: false)."""
        payload = copy.deepcopy(_VALID_TASK)
        payload["mystery_field"] = "should not be here"
        with pytest.raises(jsonschema.ValidationError):
            validate_task(payload)

    def test_invalid_track_value_fails(self) -> None:
        """Track value outside 1–5 should fail."""
        payload = copy.deepcopy(_VALID_TASK)
        payload["track"] = 6
        with pytest.raises(jsonschema.ValidationError):
            validate_task(payload)

    def test_invalid_task_type_fails(self) -> None:
        """task_type not in the allowed enum should fail."""
        payload = copy.deepcopy(_VALID_TASK)
        payload["task_type"] = "not_a_valid_type"
        with pytest.raises(jsonschema.ValidationError):
            validate_task(payload)

    def test_rubric_missing_dimension_fails(self) -> None:
        """Rubric missing one of the six required dimensions should fail."""
        payload = copy.deepcopy(_VALID_TASK)
        del payload["rubric"]["consistency"]
        with pytest.raises(jsonschema.ValidationError, match="consistency"):
            validate_task(payload)

    def test_rubric_extra_dimension_fails(self) -> None:
        """Rubric with an unknown extra dimension must fail."""
        payload = copy.deepcopy(_VALID_TASK)
        payload["rubric"]["extra_dimension"] = {"weight": 0.01}
        with pytest.raises(jsonschema.ValidationError):
            validate_task(payload)

    def test_gold_fact_missing_required_field_fails(self) -> None:
        """A gold_fact missing claim should fail."""
        payload = copy.deepcopy(_VALID_TASK)
        del payload["gold_facts"][0]["claim"]
        with pytest.raises(jsonschema.ValidationError, match="claim"):
            validate_task(payload)

    def test_allowed_inputs_must_not_be_empty(self) -> None:
        """allowed_inputs with zero items should fail (minItems: 1)."""
        payload = copy.deepcopy(_VALID_TASK)
        payload["allowed_inputs"] = []
        with pytest.raises(jsonschema.ValidationError):
            validate_task(payload)


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """Tests for validate_output()."""

    def test_valid_output_passes(self) -> None:
        """A fully populated valid output payload should pass without error."""
        validate_output(_VALID_OUTPUT)

    def test_valid_output_empty_lists_passes(self) -> None:
        """Output with empty key_findings, limitations, and citations should pass."""
        payload = copy.deepcopy(_VALID_OUTPUT)
        payload["key_findings"] = []
        payload["limitations"] = []
        payload["evidence_citations"] = []
        validate_output(payload)

    def test_valid_output_with_raw_response_passes(self) -> None:
        """Output with optional raw_response_text included should pass."""
        payload = copy.deepcopy(_VALID_OUTPUT)
        payload["raw_response_text"] = "There are 42 active students."
        validate_output(payload)

    def test_missing_task_id_fails(self) -> None:
        """Missing task_id should raise ValidationError."""
        payload = copy.deepcopy(_VALID_OUTPUT)
        del payload["task_id"]
        with pytest.raises(jsonschema.ValidationError, match="task_id"):
            validate_output(payload)

    def test_missing_runtime_metadata_fails(self) -> None:
        """Missing runtime_metadata should raise ValidationError."""
        payload = copy.deepcopy(_VALID_OUTPUT)
        del payload["runtime_metadata"]
        with pytest.raises(jsonschema.ValidationError, match="runtime_metadata"):
            validate_output(payload)

    def test_missing_model_id_fails(self) -> None:
        """Missing model_id should raise ValidationError."""
        payload = copy.deepcopy(_VALID_OUTPUT)
        del payload["model_id"]
        with pytest.raises(jsonschema.ValidationError, match="model_id"):
            validate_output(payload)

    def test_extra_top_level_field_fails(self) -> None:
        """Extra unknown field must fail (additionalProperties: false)."""
        payload = copy.deepcopy(_VALID_OUTPUT)
        payload["unknown_key"] = "oops"
        with pytest.raises(jsonschema.ValidationError):
            validate_output(payload)

    def test_invalid_run_index_fails(self) -> None:
        """Negative run_index should fail (minimum: 0)."""
        payload = copy.deepcopy(_VALID_OUTPUT)
        payload["run_index"] = -1
        with pytest.raises(jsonschema.ValidationError):
            validate_output(payload)

    def test_runtime_metadata_missing_required_field_fails(self) -> None:
        """runtime_metadata missing latency_ms should fail."""
        payload = copy.deepcopy(_VALID_OUTPUT)
        del payload["runtime_metadata"]["latency_ms"]
        with pytest.raises(jsonschema.ValidationError, match="latency_ms"):
            validate_output(payload)

    def test_runtime_metadata_extra_field_fails(self) -> None:
        """runtime_metadata with an unknown extra field must fail."""
        payload = copy.deepcopy(_VALID_OUTPUT)
        payload["runtime_metadata"]["undocumented"] = "surprise"
        with pytest.raises(jsonschema.ValidationError):
            validate_output(payload)

    def test_citation_missing_citation_text_fails(self) -> None:
        """EvidenceCitation missing citation_text should fail."""
        payload = copy.deepcopy(_VALID_OUTPUT)
        del payload["evidence_citations"][0]["citation_text"]
        with pytest.raises(jsonschema.ValidationError, match="citation_text"):
            validate_output(payload)


# ---------------------------------------------------------------------------
# Result schema tests
# ---------------------------------------------------------------------------


class TestResultSchema:
    """Tests for validate_result()."""

    def test_valid_result_passes(self) -> None:
        """A fully populated valid result payload should pass without error."""
        validate_result(_VALID_RESULT)

    def test_valid_result_minimal_per_track_passes(self) -> None:
        """Result with only one populated track (others absent) should pass."""
        validate_result(_VALID_RESULT)

    def test_missing_result_id_fails(self) -> None:
        """Missing result_id should raise ValidationError."""
        payload = copy.deepcopy(_VALID_RESULT)
        del payload["result_id"]
        with pytest.raises(jsonschema.ValidationError, match="result_id"):
            validate_result(payload)

    def test_missing_overall_scores_fails(self) -> None:
        """Missing overall_scores should raise ValidationError."""
        payload = copy.deepcopy(_VALID_RESULT)
        del payload["overall_scores"]
        with pytest.raises(jsonschema.ValidationError, match="overall_scores"):
            validate_result(payload)

    def test_missing_per_task_scores_fails(self) -> None:
        """Missing per_task_scores should raise ValidationError."""
        payload = copy.deepcopy(_VALID_RESULT)
        del payload["per_task_scores"]
        with pytest.raises(jsonschema.ValidationError, match="per_task_scores"):
            validate_result(payload)

    def test_extra_top_level_field_fails(self) -> None:
        """Extra unknown top-level key must fail (additionalProperties: false)."""
        payload = copy.deepcopy(_VALID_RESULT)
        payload["unlisted_field"] = "not allowed"
        with pytest.raises(jsonschema.ValidationError):
            validate_result(payload)

    def test_dimension_score_out_of_range_fails(self) -> None:
        """A dimension score > 1.0 should fail (maximum: 1.0)."""
        payload = copy.deepcopy(_VALID_RESULT)
        payload["overall_scores"]["grounding_accuracy"] = 1.5
        with pytest.raises(jsonschema.ValidationError):
            validate_result(payload)

    def test_dimension_score_negative_fails(self) -> None:
        """A negative dimension score should fail (minimum: 0.0)."""
        payload = copy.deepcopy(_VALID_RESULT)
        payload["overall_scores"]["insight_quality"] = -0.1
        with pytest.raises(jsonschema.ValidationError):
            validate_result(payload)

    def test_overall_scores_missing_dimension_fails(self) -> None:
        """overall_scores missing one dimension should fail."""
        payload = copy.deepcopy(_VALID_RESULT)
        del payload["overall_scores"]["consistency"]
        with pytest.raises(jsonschema.ValidationError, match="consistency"):
            validate_result(payload)

    def test_per_task_scores_empty_fails(self) -> None:
        """per_task_scores with zero items should fail (minItems: 1)."""
        payload = copy.deepcopy(_VALID_RESULT)
        payload["per_task_scores"] = []
        with pytest.raises(jsonschema.ValidationError):
            validate_result(payload)

    def test_task_result_invalid_track_fails(self) -> None:
        """TaskResult with track > 5 should fail."""
        payload = copy.deepcopy(_VALID_RESULT)
        payload["per_task_scores"][0]["track"] = 6
        with pytest.raises(jsonschema.ValidationError):
            validate_result(payload)

    def test_invalid_pack_id_in_per_pack_fails(self) -> None:
        """PackResult with an unlisted pack_id should fail (enum restriction)."""
        payload = copy.deepcopy(_VALID_RESULT)
        payload["per_pack_scores"]["pack_operations"]["pack_id"] = "pack_unknown"
        with pytest.raises(jsonschema.ValidationError):
            validate_result(payload)

    def test_scored_at_utc_invalid_format_fails(self) -> None:
        """scored_at_utc that is not a valid date-time should fail."""
        payload = copy.deepcopy(_VALID_RESULT)
        payload["scored_at_utc"] = "not-a-date"
        with pytest.raises(jsonschema.ValidationError):
            validate_result(payload)
