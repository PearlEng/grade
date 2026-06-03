"""Unit tests for the OpenRouter adapter (C6).

All HTTP calls are mocked — no real network requests or API key required.
httpx is NOT installed in the CI ``.[dev]`` environment, so tests must not
import httpx directly and must mock it via ``sys.modules`` injection.

Coverage:
- :class:`~runner.adapters.openrouter_adapter.OpenRouterAdapter`: schema-valid
  output, latency/token capture, model slug resolution, missing API key error,
  non-200 API error.
- :func:`~runner.adapters.openrouter_adapter._parse_response`: section parsing
  for findings, limitations, citations, structured metrics.
- :func:`~runner.adapters.openrouter_adapter.post_chat_completion`: shared HTTP
  helper used by the judge client.
- :class:`~benchmark.rubrics.judge_client.JudgeClient`: mocked OpenRouter call
  returns a score; non-float response falls back to 0.0.
- ``runner.adapters.get_adapter("openrouter")`` resolves the correct class.
- CLI ``--adapter openrouter`` resolves the correct adapter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from runner.adapters import DEFAULT_ADAPTER, get_adapter
from runner.adapters.openrouter_adapter import (
    SEED_MODELS,
    OpenRouterAdapter,
    _parse_response,
    _resolve_model,
    post_chat_completion,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VALID_TASK: dict[str, Any] = {
    "task_id": "T1-OPS-001",
    "track": 1,
    "title": "Count active students",
    "user_prompt": "Using the provided data, how many students are currently active?",
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
    "gold_insights": ["The program has a small cohort."],
    "required_limitations": ["Count is a snapshot."],
    "forbidden_claims": ["All students improved."],
    "rubric": {
        "grounding_accuracy": {"weight": 0.35},
        "insight_quality": {"weight": 0.20},
        "evidence_linkage": {"weight": 0.15},
        "calibration_limitation_handling": {"weight": 0.15},
        "consistency": {"weight": 0.10},
        "structure_usability": {"weight": 0.05},
    },
}

#: A plausible OpenRouter chat-completions response body.
_MOCK_OR_RESPONSE: dict[str, Any] = {
    "id": "gen-abc123",
    "object": "chat.completion",
    "model": "anthropic/claude-sonnet-4-5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (
                    "## Key Findings\n"
                    "- There are 42 active students enrolled in the program.\n"
                    "- The cohort size is relatively small compared to district average.\n"
                    "\n"
                    "## Limitations\n"
                    "- Count is a snapshot taken at end of the reporting period.\n"
                    "- Students who enrolled after the snapshot date are not included.\n"
                    "\n"
                    "## Evidence Citations\n"
                    "- students.csv: active_status column\n"
                    "\n"
                    "## Structured Metrics\n"
                    "active_student_count: 42\n"
                    "total_students: 50\n"
                ),
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 80,
        "total_tokens": 200,
    },
}


def _make_mock_httpx_post(body: dict[str, Any], status_code: int = 200) -> MagicMock:
    """Create a mock ``httpx.post`` callable returning the given response.

    Args:
        body: JSON body the mock response returns.
        status_code: HTTP status code.

    Returns:
        A callable mock that behaves like ``httpx.post`` for our usage.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = body
    mock_response.text = json.dumps(body)

    mock_post = MagicMock(return_value=mock_response)
    return mock_post


def _inject_mock_httpx(mock_post: MagicMock) -> ModuleType:
    """Inject a fake httpx module into sys.modules with a mocked ``post``.

    Because httpx is lazy-imported (``import httpx`` inside functions), we
    must place the fake module in ``sys.modules`` *before* the function runs.
    This avoids needing httpx to be installed at all.

    Args:
        mock_post: The mock callable to assign to ``httpx.post``.

    Returns:
        The fake module placed in ``sys.modules["httpx"]``.
    """
    fake_httpx = ModuleType("httpx")
    fake_httpx.post = mock_post  # type: ignore[attr-defined]
    # Provide a minimal Client stub so _make_openrouter_http_client doesn't fail.
    fake_httpx.Client = MagicMock()  # type: ignore[attr-defined]
    sys.modules["httpx"] = fake_httpx
    return fake_httpx


def _remove_mock_httpx() -> None:
    """Remove the fake httpx from sys.modules (cleanup after injection)."""
    sys.modules.pop("httpx", None)


# ---------------------------------------------------------------------------
# Context manager helper for mocked httpx
# ---------------------------------------------------------------------------


class _MockHttpx:
    """Context manager that injects / removes a fake httpx module.

    Usage::

        with _MockHttpx(body=_MOCK_OR_RESPONSE) as mock_post:
            adapter.run(task)
            mock_post.assert_called_once()
    """

    def __init__(self, body: dict[str, Any], status_code: int = 200) -> None:
        self._body = body
        self._status_code = status_code
        self.mock_post: MagicMock | None = None

    def __enter__(self) -> MagicMock:
        self.mock_post = _make_mock_httpx_post(self._body, self._status_code)
        _inject_mock_httpx(self.mock_post)
        return self.mock_post

    def __exit__(self, *_: object) -> None:
        _remove_mock_httpx()


# ---------------------------------------------------------------------------
# _resolve_model tests
# ---------------------------------------------------------------------------


class TestResolveModel:
    """Tests for :func:`_resolve_model`."""

    def test_shorthand_resolves_to_slug(self) -> None:
        """A SEED_MODELS key must map to the documented OpenRouter slug."""
        assert _resolve_model("claude-sonnet-4-6") == SEED_MODELS["claude-sonnet-4-6"]

    def test_raw_slug_passes_through(self) -> None:
        """An unknown string must be returned verbatim."""
        assert _resolve_model("mistralai/mistral-7b-instruct") == "mistralai/mistral-7b-instruct"

    def test_all_seed_models_resolve(self) -> None:
        """Every key in SEED_MODELS must resolve to a non-empty slug."""
        for key, slug in SEED_MODELS.items():
            resolved = _resolve_model(key)
            assert resolved == slug, f"SEED_MODELS[{key!r}] mismatch"
            assert "/" in resolved, f"Slug {resolved!r} lacks provider prefix"


# ---------------------------------------------------------------------------
# _parse_response tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for :func:`_parse_response`."""

    def test_extracts_key_findings(self) -> None:
        """Section '## Key Findings' must populate ``key_findings``."""
        raw = "## Key Findings\n- Finding one.\n- Finding two.\n"
        result = _parse_response(raw, _VALID_TASK)
        assert result["key_findings"] == ["Finding one.", "Finding two."]

    def test_extracts_limitations(self) -> None:
        """Section '## Limitations' must populate ``limitations``."""
        raw = "## Limitations\n- Caveat A.\n- Caveat B.\n"
        result = _parse_response(raw, _VALID_TASK)
        assert result["limitations"] == ["Caveat A.", "Caveat B."]

    def test_extracts_evidence_citations(self) -> None:
        """Section '## Evidence Citations' must populate ``evidence_citations``."""
        raw = "## Evidence Citations\n- students.csv: active_status\n"
        result = _parse_response(raw, _VALID_TASK)
        assert len(result["evidence_citations"]) == 1
        assert result["evidence_citations"][0]["citation_text"] == "students.csv: active_status"

    def test_extracts_structured_metrics_int(self) -> None:
        """Integer metric values must be stored as int."""
        raw = "## Structured Metrics\nactive_count: 42\n"
        result = _parse_response(raw, _VALID_TASK)
        assert result["structured_metrics"]["active_count"] == 42
        assert isinstance(result["structured_metrics"]["active_count"], int)

    def test_extracts_structured_metrics_float(self) -> None:
        """Float metric values must be stored as float."""
        raw = "## Structured Metrics\npass_rate: 0.85\n"
        result = _parse_response(raw, _VALID_TASK)
        assert result["structured_metrics"]["pass_rate"] == pytest.approx(0.85)

    def test_fallback_on_no_sections(self) -> None:
        """When no section headers are present, the whole text becomes one finding."""
        raw = "Students: 42 active."
        result = _parse_response(raw, _VALID_TASK)
        assert len(result["key_findings"]) == 1
        assert "Students" in result["key_findings"][0]

    def test_empty_string_yields_empty_findings(self) -> None:
        """Empty raw text must yield empty key_findings."""
        result = _parse_response("", _VALID_TASK)
        assert result["key_findings"] == []

    def test_prose_response_no_headers_populates_key_findings(self) -> None:
        """A natural-prose response with no section headers must still populate key_findings.

        This is the core prose-first requirement: when the model writes flowing
        paragraphs without any ``## Key Findings`` header, the parser splits the
        text into sentences and uses them as key_findings so the deterministic
        scorer has content to evaluate.
        """
        prose = (
            "Based on the data, there are 42 active students enrolled in the program. "
            "The cohort is relatively small, which allows for personalized instruction. "
            "Data was drawn from students.csv using the active_status column.\n\n"
            "One important caveat is that the count reflects a snapshot at end of term "
            "and may not include late enrolments."
        )
        result = _parse_response(prose, _VALID_TASK)
        # key_findings must be populated from prose (no headers present)
        assert len(result["key_findings"]) >= 1
        # At least one finding should contain a substantive insight
        combined = " ".join(result["key_findings"]).lower()
        assert "42" in combined or "student" in combined or "active" in combined
        # structured_metrics is empty (model didn't emit structured section)
        assert result["structured_metrics"] == {}

    def test_prose_response_schema_valid(self) -> None:
        """A natural-prose response (no structured sections) must parse to a schema-valid output.

        Tests the full adapter pipeline: _parse_response → output dict →
        validate_output.  Verifies that ``key_findings`` is populated from prose,
        ``structured_metrics`` is an empty dict (permitted by the schema), and
        ``limitations`` is an empty list (also permitted).
        """
        from benchmark.schemas import validate_output

        prose = (
            "The program currently serves 42 active students across three schools. "
            "Attendance has been strong this quarter, with most sessions running at "
            "full capacity.  The data comes from students.csv and sessions.csv."
        )

        parsed = _parse_response(prose, _VALID_TASK)

        # Build a full output dict around the parsed result.
        output: dict[str, Any] = {
            "task_id": _VALID_TASK["task_id"],
            "model_id": "anthropic/claude-sonnet-4-5",
            "run_index": 0,
            "structured_metrics": parsed["structured_metrics"],
            "key_findings": parsed["key_findings"],
            "limitations": parsed["limitations"],
            "evidence_citations": parsed["evidence_citations"],
            "runtime_metadata": {
                "adapter_version": "0.1.0",
                "timestamp_utc": "2026-06-01T12:00:00Z",
                "latency_ms": 55.0,
                "prompt_tokens": 120,
                "completion_tokens": 60,
                "model_temperature": 0.0,
                "provider": "anthropic",
                "pack_id": None,
            },
        }

        # Must not raise
        validate_output(output)
        # key_findings populated from prose
        assert len(output["key_findings"]) >= 1
        # structured_metrics may be empty — schema allows {}
        assert isinstance(output["structured_metrics"], dict)


# ---------------------------------------------------------------------------
# OpenRouterAdapter.run tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestOpenRouterAdapterRun:
    """Tests for :class:`OpenRouterAdapter.run` with mocked HTTP."""

    def _run_with_mock(
        self,
        mock_body: dict[str, Any] = _MOCK_OR_RESPONSE,
        status_code: int = 200,
        model: str = "anthropic/claude-sonnet-4-5",
        api_key: str = "sk-or-test",
        run_index: int = 0,
    ) -> dict[str, Any]:
        """Helper: run adapter with a mocked httpx.post."""
        with _MockHttpx(mock_body, status_code):
            adapter = OpenRouterAdapter(model=model, api_key=api_key)
            return adapter.run(_VALID_TASK, run_index=run_index)

    def test_output_has_required_keys(self) -> None:
        """Output dict must contain all output_schema.json required keys."""
        output = self._run_with_mock()
        required = {
            "task_id",
            "model_id",
            "run_index",
            "structured_metrics",
            "key_findings",
            "limitations",
            "evidence_citations",
            "runtime_metadata",
        }
        assert required.issubset(output.keys())

    def test_task_id_matches(self) -> None:
        """task_id must be copied from the input task."""
        output = self._run_with_mock()
        assert output["task_id"] == _VALID_TASK["task_id"]

    def test_run_index_preserved(self) -> None:
        """run_index must match the argument passed to run()."""
        output = self._run_with_mock(run_index=3)
        assert output["run_index"] == 3

    def test_model_id_set(self) -> None:
        """model_id must be the resolved OpenRouter slug."""
        output = self._run_with_mock(model="anthropic/claude-sonnet-4-5")
        assert output["model_id"] == "anthropic/claude-sonnet-4-5"

    def test_model_id_resolves_shorthand(self) -> None:
        """Shorthand model names must be resolved to full slugs."""
        simple_body = {
            **_MOCK_OR_RESPONSE,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "## Key Findings\n- OK.\n"},
                    "finish_reason": "stop",
                }
            ],
        }
        output = self._run_with_mock(mock_body=simple_body, model="claude-sonnet-4-6")
        assert output["model_id"] == SEED_MODELS["claude-sonnet-4-6"]

    def test_runtime_metadata_present(self) -> None:
        """runtime_metadata must be populated with required sub-keys."""
        output = self._run_with_mock()
        meta = output["runtime_metadata"]
        assert "adapter_version" in meta
        assert "timestamp_utc" in meta
        assert "latency_ms" in meta

    def test_latency_non_negative(self) -> None:
        """latency_ms must be a non-negative number."""
        output = self._run_with_mock()
        assert output["runtime_metadata"]["latency_ms"] >= 0.0

    def test_token_counts_from_usage(self) -> None:
        """prompt_tokens and completion_tokens must be populated from usage."""
        output = self._run_with_mock()
        assert output["runtime_metadata"]["prompt_tokens"] == 120
        assert output["runtime_metadata"]["completion_tokens"] == 80

    def test_provider_extracted_from_slug(self) -> None:
        """Provider must be the prefix before the first '/' in the model slug."""
        output = self._run_with_mock(model="anthropic/claude-sonnet-4-5")
        assert output["runtime_metadata"]["provider"] == "anthropic"

    def test_key_findings_populated(self) -> None:
        """key_findings must be a non-empty list from the mock response."""
        output = self._run_with_mock()
        assert isinstance(output["key_findings"], list)
        assert len(output["key_findings"]) > 0

    def test_limitations_populated(self) -> None:
        """Limitations must be a non-empty list from the mock response."""
        output = self._run_with_mock()
        assert isinstance(output["limitations"], list)
        assert len(output["limitations"]) > 0

    def test_evidence_citations_list(self) -> None:
        """evidence_citations must be a list."""
        output = self._run_with_mock()
        assert isinstance(output["evidence_citations"], list)

    def test_raw_response_text_preserved(self) -> None:
        """raw_response_text must be the verbatim model response."""
        output = self._run_with_mock()
        assert output["raw_response_text"] is not None
        assert "42 active students" in output["raw_response_text"]

    def test_non_200_raises_runtime_error(self) -> None:
        """A non-200 HTTP status must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="HTTP 429"):
            self._run_with_mock(
                mock_body={"error": {"message": "Rate limit"}},
                status_code=429,
            )

    def test_missing_api_key_raises_environment_error(self, monkeypatch: Any) -> None:
        """Missing API key (no env var, no constructor arg) must raise EnvironmentError."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        adapter = OpenRouterAdapter(model="anthropic/claude-sonnet-4-5", api_key=None)
        # Must raise before making any HTTP call.
        with _MockHttpx(_MOCK_OR_RESPONSE) as mock_post:
            with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
                adapter.run(_VALID_TASK)
            mock_post.assert_not_called()

    def test_output_valid_against_schema(self) -> None:
        """Output must be valid against output_schema.json."""
        import jsonschema

        schema_path = (
            Path(__file__).parent.parent.parent / "benchmark" / "schemas" / "output_schema.json"
        )
        with schema_path.open() as fh:
            schema = json.load(fh)

        output = self._run_with_mock()
        jsonschema.validate(instance=output, schema=schema)  # raises if invalid

    def test_cost_stored_in_structured_metrics(self) -> None:
        """If OpenRouter returns a cost in usage, it must appear in structured_metrics."""
        mock_body = {
            **_MOCK_OR_RESPONSE,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cost": "0.00042",
            },
        }
        output = self._run_with_mock(mock_body=mock_body)
        assert "cost_usd" in output["structured_metrics"]
        assert output["structured_metrics"]["cost_usd"] == pytest.approx(0.00042)


# ---------------------------------------------------------------------------
# post_chat_completion tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestPostChatCompletion:
    """Tests for :func:`post_chat_completion`."""

    def test_returns_response_text(self) -> None:
        """Must return the first choice's content as a string."""
        mock_body = {"choices": [{"message": {"role": "assistant", "content": "0.87"}}]}
        with _MockHttpx(mock_body):
            result = post_chat_completion(
                messages=[{"role": "user", "content": "Score this."}],
                model="anthropic/claude-opus-4-5",
                api_key="sk-or-test",
            )
        assert result == "0.87"

    def test_non_200_raises_runtime_error(self) -> None:
        """Non-200 response must raise RuntimeError."""
        with _MockHttpx({"error": "bad"}, status_code=500):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                post_chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    model="anthropic/claude-opus-4-5",
                    api_key="sk-or-test",
                )

    def test_missing_api_key_raises_environment_error(self, monkeypatch: Any) -> None:
        """Missing API key must raise EnvironmentError before HTTP call."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
            post_chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="anthropic/claude-opus-4-5",
                api_key=None,
            )


# ---------------------------------------------------------------------------
# JudgeClient tests (mocked OpenRouter HTTP call)
# ---------------------------------------------------------------------------


class TestJudgeClient:
    """Tests for :class:`~benchmark.rubrics.judge_client.JudgeClient`."""

    _VALID_OUTPUT: dict[str, Any] = {
        "task_id": "T1-OPS-001",
        "model_id": "test/model",
        "run_index": 0,
        "structured_metrics": {},
        "key_findings": ["42 active students."],
        "limitations": ["Snapshot count."],
        "evidence_citations": [],
        "runtime_metadata": {
            "adapter_version": "0.1.0",
            "timestamp_utc": "2026-06-01T12:00:00Z",
            "latency_ms": 100.0,
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "model_temperature": 0.0,
            "provider": "test",
            "pack_id": None,
        },
    }

    def test_judge_returns_float_score(self) -> None:
        """A valid float response from OpenRouter must be returned as float."""
        mock_body = {"choices": [{"message": {"role": "assistant", "content": "0.85"}}]}
        with _MockHttpx(mock_body):
            from benchmark.rubrics.judge_client import JudgeClient

            client = JudgeClient(api_key="sk-or-test")
            score = client.judge("grounding_accuracy", "", _VALID_TASK, self._VALID_OUTPUT)
        assert score == pytest.approx(0.85)

    def test_judge_clamps_above_one(self) -> None:
        """Score > 1.0 must be clamped to 1.0."""
        mock_body = {"choices": [{"message": {"role": "assistant", "content": "1.5"}}]}
        with _MockHttpx(mock_body):
            from benchmark.rubrics.judge_client import JudgeClient

            client = JudgeClient(api_key="sk-or-test")
            score = client.judge("grounding_accuracy", "", _VALID_TASK, self._VALID_OUTPUT)
        assert score == pytest.approx(1.0)

    def test_judge_clamps_below_zero(self) -> None:
        """Score < 0.0 must be clamped to 0.0."""
        mock_body = {"choices": [{"message": {"role": "assistant", "content": "-0.3"}}]}
        with _MockHttpx(mock_body):
            from benchmark.rubrics.judge_client import JudgeClient

            client = JudgeClient(api_key="sk-or-test")
            score = client.judge("grounding_accuracy", "", _VALID_TASK, self._VALID_OUTPUT)
        assert score == pytest.approx(0.0)

    def test_judge_non_float_response_falls_back_to_zero(self) -> None:
        """Non-parseable response must fall back to 0.0 without raising."""
        mock_body = {"choices": [{"message": {"role": "assistant", "content": "I am unsure."}}]}
        with _MockHttpx(mock_body):
            from benchmark.rubrics.judge_client import JudgeClient

            client = JudgeClient(api_key="sk-or-test")
            score = client.judge("grounding_accuracy", "", _VALID_TASK, self._VALID_OUTPUT)
        assert score == pytest.approx(0.0)

    def test_judge_uses_openrouter_not_anthropic(self) -> None:
        """JudgeClient must call OpenRouter endpoint, not Anthropic directly."""
        mock_body = {"choices": [{"message": {"role": "assistant", "content": "0.9"}}]}
        with _MockHttpx(mock_body) as mock_post:
            from benchmark.rubrics.judge_client import JudgeClient

            client = JudgeClient(api_key="sk-or-test")
            client.judge("grounding_accuracy", "", _VALID_TASK, self._VALID_OUTPUT)

        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        assert "openrouter.ai" in call_url

    def test_judge_satisfies_protocol(self) -> None:
        """JudgeClient must implement the JudgeClientProtocol interface.

        JudgeClientProtocol is not @runtime_checkable, so we verify
        duck-typing by checking the required method signature directly.
        """
        import inspect

        from benchmark.rubrics.judge_client import JudgeClient

        client = JudgeClient(api_key="sk-or-test")
        assert hasattr(client, "judge"), "JudgeClient must have a 'judge' method"
        assert callable(client.judge)
        # Verify the signature matches the protocol.
        sig = inspect.signature(client.judge)
        params = list(sig.parameters.keys())
        assert "dimension" in params
        assert "guidance" in params
        assert "task" in params
        assert "model_output" in params


# ---------------------------------------------------------------------------
# Adapter registry / CLI resolution tests
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    """Tests for the adapter registry and CLI resolver."""

    def test_default_adapter_is_openrouter(self) -> None:
        """DEFAULT_ADAPTER constant must be 'openrouter'."""
        assert DEFAULT_ADAPTER == "openrouter"

    def test_get_adapter_openrouter_returns_class(self) -> None:
        """get_adapter('openrouter') must return the OpenRouterAdapter class."""
        cls = get_adapter("openrouter")
        assert cls is OpenRouterAdapter

    def test_get_adapter_stub_returns_class(self) -> None:
        """get_adapter('stub') must return the StubAdapter class."""
        from runner.adapters.stub_adapter import StubAdapter

        cls = get_adapter("stub")
        assert cls is StubAdapter

    def test_get_adapter_unknown_raises_key_error(self) -> None:
        """get_adapter with an unknown name must raise KeyError."""
        with pytest.raises(KeyError, match="unknown-adapter"):
            get_adapter("unknown-adapter")

    def test_cli_adapter_arg_resolves_openrouter(self) -> None:
        """CLI --adapter openrouter must resolve to OpenRouterAdapter."""
        from runner.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["--pack", "operations", "--adapter", "openrouter", "--out", "/tmp/grade_out"]
        )
        cls = get_adapter(args.adapter)
        assert cls is OpenRouterAdapter

    def test_cli_default_adapter_is_openrouter(self) -> None:
        """CLI without --adapter must default to 'openrouter'."""
        from runner.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--pack", "operations", "--out", "/tmp/grade_out"])
        assert args.adapter == "openrouter"

    def test_openrouter_adapter_satisfies_adapter_protocol(self) -> None:
        """OpenRouterAdapter must satisfy the Adapter protocol."""
        from runner.adapters.base import Adapter

        adapter = OpenRouterAdapter(model="anthropic/claude-sonnet-4-5", api_key="sk-or-test")
        assert isinstance(adapter, Adapter)

    def test_stub_adapter_satisfies_adapter_protocol(self) -> None:
        """StubAdapter must satisfy the Adapter protocol."""
        from runner.adapters.base import Adapter
        from runner.adapters.stub_adapter import StubAdapter

        adapter = StubAdapter()
        assert isinstance(adapter, Adapter)


# ---------------------------------------------------------------------------
# Live integration test (opt-in, requires OPENROUTER_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("os").environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set — skipping live integration test",
)
class TestOpenRouterAdapterLive:
    """Live integration test — skipped unless OPENROUTER_API_KEY is set.

    Run with::

        OPENROUTER_API_KEY=sk-or-... pytest tests/unit/test_openrouter_adapter.py -k live -s
    """

    def test_live_run_returns_valid_output(self) -> None:
        """Live run must return a schema-valid output dict."""
        import jsonschema

        schema_path = (
            Path(__file__).parent.parent.parent / "benchmark" / "schemas" / "output_schema.json"
        )
        with schema_path.open() as fh:
            schema = json.load(fh)

        # Use the cheapest/fastest model for the live smoke test.
        adapter = OpenRouterAdapter(model="anthropic/claude-haiku-4-5")
        output = adapter.run(_VALID_TASK, run_index=0)
        jsonschema.validate(instance=output, schema=schema)
