"""Tests for the judge-hang fixes (fix/judge-hang).

Fix 1 — fixtures excluded from judge prompt
--------------------------------------------
When the dispatcher enriches a task dict with a ``fixtures`` key (containing
the full inlined CSV data for every allowed_input file), the judge prompt must
NOT include that fixture content.  Including it would balloon the prompt to
tens of thousands of tokens and cause the OpenRouter call to hang.

The test verifies that a known marker value from the fixtures dict is absent
from the built prompt, while the rubric guidance and model output ARE present.

Fix 2 — request timeout on the OpenRouter HTTP call
-----------------------------------------------------
``post_chat_completion`` must pass a ``timeout`` kwarg to ``httpx.post``.
The timeout must be the value of ``GRADE_OPENROUTER_TIMEOUT`` env var when set,
or the module-level default (120 s) when unset.  When ``httpx`` raises a
``TimeoutException``, the function must re-raise it as a :exc:`TimeoutError`
with a descriptive message.

All tests are network-free and use mocked httpx.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from benchmark.rubrics.judge_client import (
    _TASK_FIELDS_EXCLUDED_FROM_JUDGE,
    _build_judge_prompt,
)
from runner.adapters.openrouter_adapter import (
    DEFAULT_OPENROUTER_TIMEOUT,
    _get_openrouter_timeout,
    post_chat_completion,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _inject_mock_httpx(
    mock_post: MagicMock,
    timeout_exc: type | None = None,
) -> ModuleType:
    """Inject a fake httpx module that provides ``post`` and ``TimeoutException``.

    Args:
        mock_post: The mock callable to assign to ``httpx.post``.
        timeout_exc: If not ``None``, configures ``mock_post`` to raise this
            exception class when called.

    Returns:
        The fake module placed in ``sys.modules["httpx"]``.
    """
    fake_httpx = ModuleType("httpx")

    if timeout_exc is not None:
        mock_post.side_effect = timeout_exc("timed out")

    fake_httpx.post = mock_post  # type: ignore[attr-defined]
    fake_httpx.Client = MagicMock()  # type: ignore[attr-defined]
    # TimeoutException must be a real exception class so ``except`` works.
    fake_httpx.TimeoutException = timeout_exc or Exception  # type: ignore[attr-defined]
    sys.modules["httpx"] = fake_httpx
    return fake_httpx


def _remove_mock_httpx() -> None:
    sys.modules.pop("httpx", None)


class _MockHttpxPost:
    """Context manager: inject a fake httpx.post and clean up afterwards."""

    def __init__(
        self,
        response_body: dict[str, Any] | None = None,
        status_code: int = 200,
        timeout_exc: type | None = None,
    ) -> None:
        self._body = response_body or {"choices": [{"message": {"content": "0.75"}}]}
        self._status_code = status_code
        self._timeout_exc = timeout_exc
        self.mock_post: MagicMock = MagicMock()

    def __enter__(self) -> MagicMock:
        if self._timeout_exc is None:
            mock_response = MagicMock()
            mock_response.status_code = self._status_code
            mock_response.json.return_value = self._body
            self.mock_post.return_value = mock_response
        _inject_mock_httpx(self.mock_post, self._timeout_exc)
        return self.mock_post

    def __exit__(self, *_: object) -> None:
        _remove_mock_httpx()


# ---------------------------------------------------------------------------
# Minimal task and output used in judge-prompt tests
# ---------------------------------------------------------------------------

_TASK_WITHOUT_FIXTURES: dict[str, Any] = {
    "task_id": "T2-OUT-001",
    "track": 2,
    "title": "Attendance trend analysis",
    "user_prompt": "Analyse the attendance trend for Q1.",
    "task_type": "trend_analysis",
    "allowed_inputs": ["attendance.csv"],
    "gold_facts": [
        {
            "fact_id": "F1",
            "claim": "Attendance was 78% in September.",
            "source_files": ["attendance.csv"],
            "numeric_value": 78.0,
            "tolerance": 0.1,
        }
    ],
    "gold_insights": ["Attendance improved from September to October."],
    "required_limitations": ["No causal inference is possible."],
    "forbidden_claims": ["Attendance improved steadily each month."],
    "rubric": {
        "grounding_accuracy": {"weight": 0.35},
        "insight_quality": {"weight": 0.20},
        "evidence_linkage": {"weight": 0.15},
        "calibration_limitation_handling": {"weight": 0.15},
        "consistency": {"weight": 0.10},
        "structure_usability": {"weight": 0.05},
    },
}

#: 4 156-row attendance fixture — the actual content that caused the hang.
#: We use a small stand-in with a known unique marker value.
_BIG_FIXTURE_MARKER = "UNIQUE_CSV_MARKER_XYZ_DO_NOT_INCLUDE_IN_JUDGE_PROMPT"
_BIG_FIXTURE_CONTENTS = "\n".join(
    ["student_id,date,present\n"]
    + [f"STU-{i:04d},2025-09-{(i % 28) + 1:02d},{_BIG_FIXTURE_MARKER}" for i in range(4156)]
)

_TASK_WITH_BIG_FIXTURES: dict[str, Any] = {
    **_TASK_WITHOUT_FIXTURES,
    "fixtures": {"attendance.csv": _BIG_FIXTURE_CONTENTS},
    "pack_id": "pack_outcomes",
}

_MODEL_OUTPUT: dict[str, Any] = {
    "task_id": "T2-OUT-001",
    "model_id": "test/model",
    "run_index": 0,
    "structured_metrics": {},
    "key_findings": [
        "Attendance improved from September to October by 3.28 percentage points.",
    ],
    "limitations": [
        "No causal inference is possible from attendance trend data alone.",
    ],
    "evidence_citations": [],
    "runtime_metadata": {
        "adapter_version": "0.1.0",
        "timestamp_utc": "2026-06-01T12:00:00Z",
        "latency_ms": 350.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "model_temperature": 0.0,
        "provider": "test",
        "pack_id": "pack_outcomes",
    },
}


# ---------------------------------------------------------------------------
# Fix 1: fixtures excluded from judge prompt
# ---------------------------------------------------------------------------


class TestJudgePromptExcludesFixtures:
    """Verify that ``_build_judge_prompt`` never includes raw fixture data."""

    def test_fixture_marker_absent_from_prompt(self) -> None:
        """The known fixture marker must NOT appear anywhere in the judge prompt.

        This is the core regression test: a task enriched with a 4 156-row
        fixture dict must produce a prompt that contains no fixture data at all.
        """
        prompt = _build_judge_prompt(
            dimension="insight_quality",
            guidance="Assess whether the model identified the key attendance trend.",
            task=_TASK_WITH_BIG_FIXTURES,
            model_output=_MODEL_OUTPUT,
        )
        assert _BIG_FIXTURE_MARKER not in prompt, (
            "Fixture contents leaked into the judge prompt — "
            "this would cause the OpenRouter call to hang."
        )

    def test_fixtures_key_absent_from_prompt(self) -> None:
        """The literal string 'fixtures' must not appear as a content section header."""
        prompt = _build_judge_prompt(
            dimension="grounding_accuracy",
            guidance="",
            task=_TASK_WITH_BIG_FIXTURES,
            model_output=_MODEL_OUTPUT,
        )
        # The CSV contents marker is the authoritative check; additionally verify
        # the raw filename is not embedded in a data dump.
        assert "attendance.csv" not in prompt or "=== FILE:" not in prompt, (
            "Raw fixture block delimiters must not appear in the judge prompt."
        )

    def test_pack_id_absent_from_prompt(self) -> None:
        """The pack_id value must NOT appear in the judge prompt."""
        prompt = _build_judge_prompt(
            dimension="insight_quality",
            guidance="",
            task=_TASK_WITH_BIG_FIXTURES,
            model_output=_MODEL_OUTPUT,
        )
        assert "pack_outcomes" not in prompt, (
            "pack_id is an internal dispatcher key and must not reach the judge prompt."
        )

    def test_rubric_guidance_present_in_prompt(self) -> None:
        """The rubric guidance text must still appear in the judge prompt."""
        guidance = "Check whether the model identified the 3.28 pp attendance rise."
        prompt = _build_judge_prompt(
            dimension="insight_quality",
            guidance=guidance,
            task=_TASK_WITH_BIG_FIXTURES,
            model_output=_MODEL_OUTPUT,
        )
        assert guidance in prompt, "Rubric guidance must be present in the judge prompt."

    def test_model_output_present_in_prompt(self) -> None:
        """Key findings from the model output must appear in the judge prompt."""
        prompt = _build_judge_prompt(
            dimension="insight_quality",
            guidance="",
            task=_TASK_WITH_BIG_FIXTURES,
            model_output=_MODEL_OUTPUT,
        )
        assert "3.28 percentage points" in prompt, (
            "Model output key findings must appear in the judge prompt."
        )

    def test_gold_facts_present_in_prompt(self) -> None:
        """Gold facts from the task must appear in the judge prompt."""
        prompt = _build_judge_prompt(
            dimension="grounding_accuracy",
            guidance="",
            task=_TASK_WITH_BIG_FIXTURES,
            model_output=_MODEL_OUTPUT,
        )
        assert "78%" in prompt or "78.0" in prompt, (
            "Gold fact numeric value must be present in the judge prompt."
        )

    def test_gold_insights_present_in_prompt(self) -> None:
        """Gold insights from the task must appear in the judge prompt."""
        prompt = _build_judge_prompt(
            dimension="insight_quality",
            guidance="",
            task=_TASK_WITH_BIG_FIXTURES,
            model_output=_MODEL_OUTPUT,
        )
        assert "Attendance improved from September to October" in prompt

    def test_prompt_size_bounded_regardless_of_fixture_size(self) -> None:
        """Prompt length must not grow with fixture size.

        Build prompts with two tasks — one with a tiny fixture and one with the
        full 4 156-row fixture — and assert they are identical (fixtures excluded
        from both).
        """
        task_small_fixture = {
            **_TASK_WITHOUT_FIXTURES,
            "fixtures": {"attendance.csv": "student_id,date,present\nSTU-0001,2025-09-01,yes\n"},
            "pack_id": "pack_outcomes",
        }

        prompt_small = _build_judge_prompt(
            dimension="insight_quality",
            guidance="",
            task=task_small_fixture,
            model_output=_MODEL_OUTPUT,
        )
        prompt_big = _build_judge_prompt(
            dimension="insight_quality",
            guidance="",
            task=_TASK_WITH_BIG_FIXTURES,
            model_output=_MODEL_OUTPUT,
        )
        assert prompt_small == prompt_big, (
            "Judge prompt must be identical regardless of fixture size — "
            "fixtures must be stripped before prompt construction."
        )

    def test_task_without_fixtures_unaffected(self) -> None:
        """A task dict without a fixtures key must still produce a valid prompt."""
        prompt = _build_judge_prompt(
            dimension="insight_quality",
            guidance="Some guidance.",
            task=_TASK_WITHOUT_FIXTURES,
            model_output=_MODEL_OUTPUT,
        )
        assert "Some guidance." in prompt
        assert _TASK_WITHOUT_FIXTURES["user_prompt"] in prompt

    def test_excluded_fields_constant_contains_fixtures(self) -> None:
        """The exclusion constant must include 'fixtures' and 'pack_id'."""
        assert "fixtures" in _TASK_FIELDS_EXCLUDED_FROM_JUDGE
        assert "pack_id" in _TASK_FIELDS_EXCLUDED_FROM_JUDGE


# ---------------------------------------------------------------------------
# Fix 2: request timeout on the OpenRouter HTTP call
# ---------------------------------------------------------------------------


class TestOpenRouterTimeout:
    """Verify that ``post_chat_completion`` passes a timeout to ``httpx.post``."""

    def test_default_timeout_is_120_seconds(self) -> None:
        """The module-level default timeout must be 120 seconds."""
        assert DEFAULT_OPENROUTER_TIMEOUT == pytest.approx(120.0)

    def test_get_openrouter_timeout_returns_default_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GRADE_OPENROUTER_TIMEOUT is not set, default (120 s) is returned."""
        monkeypatch.delenv("GRADE_OPENROUTER_TIMEOUT", raising=False)
        assert _get_openrouter_timeout() == pytest.approx(DEFAULT_OPENROUTER_TIMEOUT)

    def test_get_openrouter_timeout_reads_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When GRADE_OPENROUTER_TIMEOUT is set, its value is returned."""
        monkeypatch.setenv("GRADE_OPENROUTER_TIMEOUT", "30")
        assert _get_openrouter_timeout() == pytest.approx(30.0)

    def test_get_openrouter_timeout_float_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GRADE_OPENROUTER_TIMEOUT may be a float string."""
        monkeypatch.setenv("GRADE_OPENROUTER_TIMEOUT", "45.5")
        assert _get_openrouter_timeout() == pytest.approx(45.5)

    def test_get_openrouter_timeout_invalid_env_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unparseable GRADE_OPENROUTER_TIMEOUT falls back to the default."""
        monkeypatch.setenv("GRADE_OPENROUTER_TIMEOUT", "not-a-number")
        assert _get_openrouter_timeout() == pytest.approx(DEFAULT_OPENROUTER_TIMEOUT)

    def test_post_chat_completion_passes_timeout_kwarg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """httpx.post must be called with a ``timeout`` keyword argument."""
        monkeypatch.delenv("GRADE_OPENROUTER_TIMEOUT", raising=False)
        with _MockHttpxPost() as mock_post:
            post_chat_completion(
                messages=[{"role": "user", "content": "Score this."}],
                model="anthropic/claude-opus-4-5",
                api_key="sk-or-test",
            )
        assert mock_post.called
        _, kwargs = mock_post.call_args
        assert "timeout" in kwargs, (
            "httpx.post must receive a 'timeout' kwarg to prevent indefinite hangs."
        )

    def test_post_chat_completion_uses_default_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When env var is unset, the default timeout (120 s) is passed to httpx."""
        monkeypatch.delenv("GRADE_OPENROUTER_TIMEOUT", raising=False)
        with _MockHttpxPost() as mock_post:
            post_chat_completion(
                messages=[{"role": "user", "content": "Score this."}],
                model="anthropic/claude-opus-4-5",
                api_key="sk-or-test",
            )
        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == pytest.approx(DEFAULT_OPENROUTER_TIMEOUT)

    def test_post_chat_completion_uses_env_override_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GRADE_OPENROUTER_TIMEOUT is set, that value is passed to httpx."""
        monkeypatch.setenv("GRADE_OPENROUTER_TIMEOUT", "30")
        with _MockHttpxPost() as mock_post:
            post_chat_completion(
                messages=[{"role": "user", "content": "Score this."}],
                model="anthropic/claude-opus-4-5",
                api_key="sk-or-test",
            )
        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == pytest.approx(30.0)

    def test_post_chat_completion_raises_timeout_error_on_timeout(self) -> None:
        """When httpx raises TimeoutException, a TimeoutError must be raised.

        The TimeoutError message must name the model so the caller can log it.
        """

        class _FakeTimeoutException(Exception):
            """Minimal stand-in for httpx.TimeoutException."""

        with _MockHttpxPost(timeout_exc=_FakeTimeoutException):
            with pytest.raises(TimeoutError, match="timed out"):
                post_chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    model="anthropic/claude-opus-4-5",
                    api_key="sk-or-test",
                )

    def test_timeout_error_message_contains_model(self) -> None:
        """TimeoutError message must mention the model slug for diagnostics."""

        class _FakeTimeoutException(Exception):
            pass

        model_slug = "anthropic/claude-opus-4-5"
        with _MockHttpxPost(timeout_exc=_FakeTimeoutException):
            with pytest.raises(TimeoutError) as exc_info:
                post_chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    model=model_slug,
                    api_key="sk-or-test",
                )
        assert model_slug in str(exc_info.value), (
            "TimeoutError message must include the model slug for diagnostics."
        )

    def test_timeout_error_message_contains_timeout_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TimeoutError message must mention the configured timeout value."""

        class _FakeTimeoutException(Exception):
            pass

        monkeypatch.setenv("GRADE_OPENROUTER_TIMEOUT", "30")
        with _MockHttpxPost(timeout_exc=_FakeTimeoutException):
            with pytest.raises(TimeoutError) as exc_info:
                post_chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    model="anthropic/claude-opus-4-5",
                    api_key="sk-or-test",
                )
        assert "30" in str(exc_info.value), (
            "TimeoutError message must include the timeout value so the user "
            "knows which env var to adjust."
        )
