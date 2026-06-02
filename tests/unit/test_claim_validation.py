"""Unit tests for the C3 claim validation scorer.

All tests use mock judge clients — no live API calls are made in CI.

Coverage:
- :func:`~benchmark.rubrics.claim_validation.check_insight_present`:
  exact-match present, token-overlap present, paraphrase (mocked judge),
  absent, partial overlap.
- :func:`~benchmark.rubrics.claim_validation.check_forbidden_absent`:
  exact-match present (penalty), paraphrase present via judge (penalty),
  absent (pass), absent confirmed by judge (pass).
- :func:`~benchmark.rubrics.claim_validation.check_limitation_present`:
  exact-match present, token-overlap present, paraphrase (mocked judge),
  absent.
- :func:`~benchmark.rubrics.claim_validation.validate_claims`:
  all-pass, all-fail, mixed, empty lists, judge-cap enforcement,
  scorer_flags, aggregate score computation, raw_response_text search.
- Deterministic helpers: :func:`~benchmark.rubrics.claim_validation._normalise`,
  :func:`~benchmark.rubrics.claim_validation._token_overlap`,
  :func:`~benchmark.rubrics.claim_validation._string_match`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from benchmark.rubrics.claim_validation import (
    MAX_JUDGE_CALLS,
    ClaimCheckDetail,
    ClaimValidationResult,
    _normalise,
    _string_match,
    _token_overlap,
    check_forbidden_absent,
    check_insight_present,
    check_limitation_present,
    validate_claims,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_RUBRIC: dict[str, Any] = {
    "grounding_accuracy": {"weight": 0.35},
    "insight_quality": {"weight": 0.20},
    "evidence_linkage": {"weight": 0.15},
    "calibration_limitation_handling": {"weight": 0.15},
    "consistency": {"weight": 0.10},
    "structure_usability": {"weight": 0.05},
}

_BASE_TASK: dict[str, Any] = {
    "task_id": "T2-OUT-001",
    "track": 2,
    "title": "Attendance trend analysis",
    "user_prompt": "Analyse the attendance trend for Q1.",
    "task_type": "trend_analysis",
    "allowed_inputs": ["attendance.csv"],
    "gold_facts": [],
    "gold_insights": [
        "Attendance improved from September to October by 3.28 pp.",
        "The program ended November above the 80% attendance goal.",
    ],
    "required_limitations": [
        "No causal inference is possible from attendance trend data alone.",
        "The data covers only one program in one district.",
    ],
    "forbidden_claims": [
        "Attendance improved steadily each month.",
        "Attendance improvements were caused by a specific intervention.",
    ],
    "rubric": _RUBRIC,
}

_BASE_OUTPUT: dict[str, Any] = {
    "task_id": "T2-OUT-001",
    "model_id": "test/model",
    "run_index": 0,
    "structured_metrics": {},
    "key_findings": [
        "Attendance improved from September to October by 3.28 percentage points.",
        "By November the program exceeded the 80 percent attendance goal.",
    ],
    "limitations": [
        "No causal inference is possible from attendance trend data alone.",
        "The dataset covers only a single program in one district over three months.",
    ],
    "evidence_citations": [],
    "runtime_metadata": {
        "adapter_version": "0.1.0",
        "timestamp_utc": "2026-06-01T12:00:00Z",
        "latency_ms": 300.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "model_temperature": 0.0,
        "provider": "test",
        "pack_id": "pack_outcomes",
    },
    "raw_response_text": None,
}


def _make_mock_judge(return_value: float = 1.0) -> MagicMock:
    """Return a mock that satisfies JudgeClientProtocol, always returning *return_value*.

    Args:
        return_value: The score to return for every judge call.

    Returns:
        A :class:`~unittest.mock.MagicMock` with a ``judge`` method.
    """
    mock = MagicMock()
    mock.judge.return_value = return_value
    return mock


def _fresh_counter(n: int = MAX_JUDGE_CALLS) -> list[int]:
    """Return a fresh mutable counter for judge calls.

    Args:
        n: Initial value.

    Returns:
        Single-element list used as a shared counter.
    """
    return [n]


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------


class TestNormalise:
    """Tests for :func:`_normalise`."""

    def test_lowercases(self) -> None:
        """Should lowercase all letters."""
        assert _normalise("Hello World") == "hello world"

    def test_strips_punctuation(self) -> None:
        """Should remove punctuation characters."""
        assert _normalise("Hello, world!") == "hello world"

    def test_collapses_whitespace(self) -> None:
        """Should collapse multiple spaces and strip leading/trailing whitespace."""
        assert _normalise("  hello   world  ") == "hello world"

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert _normalise("") == ""


# ---------------------------------------------------------------------------
# _token_overlap
# ---------------------------------------------------------------------------


class TestTokenOverlap:
    """Tests for :func:`_token_overlap`."""

    def test_identical_strings(self) -> None:
        """Identical strings should return 1.0."""
        assert _token_overlap("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Completely different tokens should return 0.0."""
        assert _token_overlap("abc def", "xyz uvw") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Partial overlap should return correct Jaccard value."""
        # {'a', 'b'} ∩ {'a', 'c'} = {'a'} → 1/3
        result = _token_overlap("a b", "a c")
        assert result == pytest.approx(1 / 3)

    def test_empty_strings(self) -> None:
        """Two empty strings should return 0.0."""
        assert _token_overlap("", "") == pytest.approx(0.0)

    def test_one_empty(self) -> None:
        """One empty string should return 0.0."""
        assert _token_overlap("hello", "") == pytest.approx(0.0)
        assert _token_overlap("", "hello") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _string_match
# ---------------------------------------------------------------------------


class TestStringMatch:
    """Tests for :func:`_string_match`."""

    def test_exact_substring_match(self) -> None:
        """Claim appearing verbatim inside a candidate should match via substring."""
        matched, method = _string_match("attendance improved", ["attendance improved a lot"])
        assert matched is True
        assert method == "substring"

    def test_token_overlap_match(self) -> None:
        """Claim with high token overlap should match via token_overlap."""
        # 'attendance improved october' vs 'attendance improved in october by a lot'
        # Tokens: {attendance, improved, october} vs {attendance, improved, in, october, by, a, lot}
        # Intersection: 3, Union: 7 → 3/7 ≈ 0.43
        # Use a very simple case that clears the 0.5 threshold
        claim = "attendance improved october november"  # 4 tokens
        candidate = "attendance improved october november 2025"  # 5 tokens; overlap=4, union=5
        matched, method = _string_match(claim, [candidate])
        assert matched is True
        assert method in ("substring", "token_overlap")

    def test_not_found(self) -> None:
        """Unrelated claim should return not_found."""
        matched, method = _string_match(
            "completely unrelated statement",
            ["attendance data", "program records"],
        )
        assert matched is False
        assert method == "not_found"

    def test_reversed_substring(self) -> None:
        """Short candidate inside long claim should also match via substring."""
        matched, method = _string_match(
            "this is a very long claim that mentions the key phrase",
            ["key phrase"],
        )
        assert matched is True
        assert method == "substring"

    def test_empty_candidates(self) -> None:
        """Empty candidate list should return not_found."""
        matched, method = _string_match("any claim", [])
        assert matched is False
        assert method == "not_found"


# ---------------------------------------------------------------------------
# check_insight_present
# ---------------------------------------------------------------------------


class TestCheckInsightPresent:
    """Tests for :func:`check_insight_present`."""

    def test_exact_match_present(self) -> None:
        """Insight appearing verbatim in key_findings should match without judge."""
        insight = "Attendance improved from September to October."
        findings = ["Attendance improved from September to October.", "Other finding."]
        detail = check_insight_present(
            insight=insight,
            key_findings=findings,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=None,
            judge_calls_remaining=_fresh_counter(),
        )
        assert detail.matched is True
        assert detail.score == pytest.approx(1.0)
        assert detail.method in ("substring", "token_overlap")
        assert detail.claim == insight

    def test_paraphrase_via_judge(self) -> None:
        """Insight not found by Stage 1 but confirmed by judge should match."""
        insight = "The program saw an attendance increase in October."
        # key_findings uses completely different wording
        findings = ["Q1 data shows a rise in student presence during the second month."]
        mock_judge = _make_mock_judge(return_value=0.9)
        counter = _fresh_counter()
        detail = check_insight_present(
            insight=insight,
            key_findings=findings,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=mock_judge,
            judge_calls_remaining=counter,
        )
        assert detail.matched is True
        assert detail.score == pytest.approx(1.0)
        assert detail.method == "judge"
        mock_judge.judge.assert_called_once()
        assert counter[0] == MAX_JUDGE_CALLS - 1

    def test_paraphrase_judge_returns_low_score(self) -> None:
        """Insight not found by Stage 1 and judge returns low score → not matched."""
        insight = "Attendance improved dramatically."
        findings = ["Attendance was mostly flat."]
        mock_judge = _make_mock_judge(return_value=0.1)
        detail = check_insight_present(
            insight=insight,
            key_findings=findings,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=mock_judge,
            judge_calls_remaining=_fresh_counter(),
        )
        assert detail.matched is False
        assert detail.score == pytest.approx(0.0)
        assert detail.method == "judge"

    def test_absent_no_judge(self) -> None:
        """Insight not present and no judge → not matched."""
        insight = "A completely absent insight."
        findings = ["Something else entirely."]
        detail = check_insight_present(
            insight=insight,
            key_findings=findings,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=None,
            judge_calls_remaining=_fresh_counter(),
        )
        assert detail.matched is False
        assert detail.score == pytest.approx(0.0)
        assert detail.method == "not_found"

    def test_judge_not_called_when_stage1_matches(self) -> None:
        """When Stage 1 matches, no judge call should be made."""
        insight = "attendance data"
        findings = ["attendance data covers three months"]
        mock_judge = _make_mock_judge(1.0)
        check_insight_present(
            insight=insight,
            key_findings=findings,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=mock_judge,
            judge_calls_remaining=_fresh_counter(),
        )
        mock_judge.judge.assert_not_called()

    def test_judge_not_called_when_counter_zero(self) -> None:
        """Judge should not be called when budget is exhausted."""
        insight = "A paraphrase insight not in findings."
        findings = ["Completely different text here."]
        mock_judge = _make_mock_judge(1.0)
        counter = _fresh_counter(0)
        detail = check_insight_present(
            insight=insight,
            key_findings=findings,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=mock_judge,
            judge_calls_remaining=counter,
        )
        mock_judge.judge.assert_not_called()
        assert detail.matched is False

    def test_partial_overlap_below_threshold_no_judge_no_match(self) -> None:
        """Token overlap below threshold and no judge → not matched."""
        # Only 1 out of many tokens overlap
        insight = "attendance improved significantly across all three schools"
        findings = ["schools had issues"]
        detail = check_insight_present(
            insight=insight,
            key_findings=findings,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=None,
            judge_calls_remaining=_fresh_counter(),
        )
        # 'schools' overlaps but Jaccard is low:
        # intersection = {schools} (1 token), union = 9 tokens → 1/9 ≈ 0.11 < 0.5
        assert detail.matched is False


# ---------------------------------------------------------------------------
# check_forbidden_absent
# ---------------------------------------------------------------------------


class TestCheckForbiddenAbsent:
    """Tests for :func:`check_forbidden_absent`."""

    def test_forbidden_claim_detected_by_substring(self) -> None:
        """Forbidden claim present in output texts → matched=False (penalty)."""
        fc = "Attendance improved steadily each month."
        output_texts = [
            "Attendance improved steadily each month throughout the quarter.",
        ]
        detail = check_forbidden_absent(
            forbidden_claim=fc,
            all_output_texts=output_texts,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=None,
            judge_calls_remaining=_fresh_counter(),
        )
        assert detail.matched is False
        assert detail.score == pytest.approx(0.0)
        assert detail.method in ("substring", "token_overlap")

    def test_forbidden_claim_absent_no_judge(self) -> None:
        """Forbidden claim absent from output texts and no judge → matched=True."""
        fc = "Attendance improvements were caused by a specific intervention."
        output_texts = [
            "Attendance data was collected monthly.",
            "No causal claims are made here.",
        ]
        detail = check_forbidden_absent(
            forbidden_claim=fc,
            all_output_texts=output_texts,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=None,
            judge_calls_remaining=_fresh_counter(),
        )
        assert detail.matched is True
        assert detail.score == pytest.approx(1.0)
        assert detail.method == "not_found"

    def test_forbidden_claim_absent_confirmed_by_judge(self) -> None:
        """Stage 1 says absent; judge score below threshold → still absent (matched=True)."""
        fc = "Attendance improvements were caused by a specific intervention."
        output_texts = ["Attendance data shows monthly variation."]
        mock_judge = _make_mock_judge(return_value=0.2)  # judge: claim NOT present
        counter = _fresh_counter()
        detail = check_forbidden_absent(
            forbidden_claim=fc,
            all_output_texts=output_texts,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=mock_judge,
            judge_calls_remaining=counter,
        )
        assert detail.matched is True
        assert detail.score == pytest.approx(1.0)
        assert detail.method == "judge"
        mock_judge.judge.assert_called_once()
        assert counter[0] == MAX_JUDGE_CALLS - 1

    def test_forbidden_claim_detected_by_judge_paraphrase(self) -> None:
        """Stage 1 says absent; judge says present (paraphrase) → matched=False (penalty)."""
        fc = "The tutoring program caused the attendance increase."
        # Output uses different wording but same causal claim
        output_texts = [
            "The intervention clearly led to higher attendance rates.",
        ]
        # Stage 1 won't catch this (different wording), so force judge to say it's present.
        mock_judge = _make_mock_judge(return_value=0.9)  # judge: claim IS present
        detail = check_forbidden_absent(
            forbidden_claim=fc,
            all_output_texts=output_texts,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=mock_judge,
            judge_calls_remaining=_fresh_counter(),
        )
        # Judge says forbidden claim IS present → penalty → matched=False
        assert detail.matched is False
        assert detail.score == pytest.approx(0.0)
        assert detail.method == "judge"

    def test_judge_not_called_when_stage1_detects_claim(self) -> None:
        """When Stage 1 detects the forbidden claim, no judge call should be made."""
        fc = "attendance improved steadily"
        output_texts = ["attendance improved steadily each month"]
        mock_judge = _make_mock_judge(0.9)
        check_forbidden_absent(
            forbidden_claim=fc,
            all_output_texts=output_texts,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=mock_judge,
            judge_calls_remaining=_fresh_counter(),
        )
        mock_judge.judge.assert_not_called()

    def test_judge_not_called_when_budget_zero(self) -> None:
        """No judge call when budget is zero, even if Stage 1 says absent."""
        fc = "This forbidden claim is not present."
        output_texts = ["Unrelated finding."]
        mock_judge = _make_mock_judge(0.9)
        detail = check_forbidden_absent(
            forbidden_claim=fc,
            all_output_texts=output_texts,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=mock_judge,
            judge_calls_remaining=_fresh_counter(0),
        )
        mock_judge.judge.assert_not_called()
        # No judge → trust Stage 1 (absent)
        assert detail.matched is True


# ---------------------------------------------------------------------------
# check_limitation_present
# ---------------------------------------------------------------------------


class TestCheckLimitationPresent:
    """Tests for :func:`check_limitation_present`."""

    def test_exact_match_present(self) -> None:
        """Required limitation appearing verbatim in limitations → matched."""
        lim = "No causal inference is possible from attendance trend data alone."
        limitations = [
            "No causal inference is possible from attendance trend data alone.",
            "Only three months of data were available.",
        ]
        detail = check_limitation_present(
            limitation=lim,
            limitations=limitations,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=None,
            judge_calls_remaining=_fresh_counter(),
        )
        assert detail.matched is True
        assert detail.score == pytest.approx(1.0)
        assert detail.method in ("substring", "token_overlap")

    def test_paraphrase_via_judge(self) -> None:
        """Limitation not found by Stage 1 but confirmed by judge → matched."""
        lim = "No causal inference is possible from this data."
        limitations = ["The data does not support drawing causal conclusions."]
        mock_judge = _make_mock_judge(return_value=0.85)
        counter = _fresh_counter()
        detail = check_limitation_present(
            limitation=lim,
            limitations=limitations,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=mock_judge,
            judge_calls_remaining=counter,
        )
        assert detail.matched is True
        assert detail.score == pytest.approx(1.0)
        assert detail.method == "judge"
        mock_judge.judge.assert_called_once()
        assert counter[0] == MAX_JUDGE_CALLS - 1

    def test_absent_no_judge(self) -> None:
        """Required limitation not in limitations and no judge → not matched."""
        lim = "Results should not be generalised."
        limitations = ["Data is limited to three months."]
        detail = check_limitation_present(
            limitation=lim,
            limitations=limitations,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=None,
            judge_calls_remaining=_fresh_counter(),
        )
        assert detail.matched is False
        assert detail.score == pytest.approx(0.0)
        assert detail.method == "not_found"

    def test_paraphrase_judge_low_score_not_matched(self) -> None:
        """Limitation not found and judge gives low score → not matched."""
        lim = "External validity is severely limited."
        limitations = ["Data covers only one quarter."]
        mock_judge = _make_mock_judge(return_value=0.1)
        detail = check_limitation_present(
            limitation=lim,
            limitations=limitations,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=mock_judge,
            judge_calls_remaining=_fresh_counter(),
        )
        assert detail.matched is False
        assert detail.score == pytest.approx(0.0)
        assert detail.method == "judge"

    def test_judge_not_called_when_stage1_matches(self) -> None:
        """When Stage 1 matches, no judge call should be made."""
        lim = "data covers one district"
        limitations = ["data covers one district only"]
        mock_judge = _make_mock_judge(1.0)
        check_limitation_present(
            limitation=lim,
            limitations=limitations,
            task=_BASE_TASK,
            model_output=_BASE_OUTPUT,
            judge_client=mock_judge,
            judge_calls_remaining=_fresh_counter(),
        )
        mock_judge.judge.assert_not_called()


# ---------------------------------------------------------------------------
# validate_claims — full integration
# ---------------------------------------------------------------------------


class TestValidateClaims:
    """Tests for :func:`validate_claims`."""

    def test_all_pass_deterministic(self) -> None:
        """Task with present insights/limitations and absent forbidden claims → score 1.0."""
        task = {
            **_BASE_TASK,
            "gold_insights": ["attendance improved"],
            "required_limitations": ["causal inference not possible"],
            "forbidden_claims": ["attendance improved steadily"],
        }
        output = {
            **_BASE_OUTPUT,
            "key_findings": ["attendance improved from September to October"],
            "limitations": ["causal inference not possible from this data"],
            # Forbidden claim is absent from key_findings and limitations.
        }
        result = validate_claims(task, output, judge_client=None)
        assert isinstance(result, ClaimValidationResult)
        assert len(result.insights_present) == 1
        assert len(result.forbidden_absent) == 1
        assert len(result.limitations_present) == 1
        assert result.insights_present[0].matched is True
        assert result.forbidden_absent[0].matched is True  # absent = good
        assert result.limitations_present[0].matched is True
        assert result.calibration_limitation_handling == pytest.approx(1.0)

    def test_all_fail(self) -> None:
        """Output completely missing insights, asserting forbidden claims → score 0.0."""
        task = {
            **_BASE_TASK,
            "gold_insights": ["a very specific insight not in output"],
            "required_limitations": ["a required limitation not acknowledged"],
            "forbidden_claims": ["attendance improved steadily each month"],
        }
        output = {
            **_BASE_OUTPUT,
            # The forbidden claim IS present in key_findings.
            "key_findings": ["attendance improved steadily each month throughout the quarter"],
            "limitations": [],
        }
        result = validate_claims(task, output, judge_client=None)
        assert result.insights_present[0].matched is False
        assert result.forbidden_absent[0].matched is False  # claim found = bad
        assert result.limitations_present[0].matched is False
        assert result.calibration_limitation_handling == pytest.approx(0.0)

    def test_mixed_results(self) -> None:
        """Mixed results should produce aggregate score between 0 and 1."""
        task = {
            **_BASE_TASK,
            "gold_insights": ["attendance improved"],  # present
            "required_limitations": ["missing limitation"],  # absent
            "forbidden_claims": ["not in output"],  # absent = good
        }
        output = {
            **_BASE_OUTPUT,
            "key_findings": ["attendance improved in October"],
            "limitations": [],  # required limitation missing
        }
        result = validate_claims(task, output, judge_client=None)
        assert result.insights_present[0].matched is True
        assert result.limitations_present[0].matched is False
        assert result.forbidden_absent[0].matched is True
        # sub-scores: 1.0, 1.0, 0.0 → mean = 2/3
        assert result.calibration_limitation_handling == pytest.approx(2.0 / 3.0)

    def test_empty_task_lists(self) -> None:
        """Task with no insights/limitations/forbidden claims → score 1.0 by convention."""
        task = {
            **_BASE_TASK,
            "gold_insights": [],
            "required_limitations": [],
            "forbidden_claims": [],
        }
        result = validate_claims(task, _BASE_OUTPUT, judge_client=None)
        assert result.calibration_limitation_handling == pytest.approx(1.0)
        assert result.insights_present == []
        assert result.forbidden_absent == []
        assert result.limitations_present == []

    def test_empty_task_lists_flags_set(self) -> None:
        """Empty task lists should set appropriate scorer_flags."""
        task = {
            **_BASE_TASK,
            "gold_insights": [],
            "required_limitations": [],
            "forbidden_claims": [],
        }
        result = validate_claims(task, _BASE_OUTPUT, judge_client=None)
        assert "no_gold_insights" in result.scorer_flags
        assert "no_required_limitations" in result.scorer_flags
        assert "no_forbidden_claims" in result.scorer_flags

    def test_judge_calls_are_shared_across_validators(self) -> None:
        """Total judge calls across all three validators must not exceed MAX_JUDGE_CALLS."""
        # Create a task with enough unmatched claims to exhaust the budget.
        n = MAX_JUDGE_CALLS + 5  # more claims than budget
        task = {
            **_BASE_TASK,
            "gold_insights": [f"unique insight {i}" for i in range(n)],
            "required_limitations": [f"unique limitation {i}" for i in range(n)],
            "forbidden_claims": [f"forbidden claim {i}" for i in range(n)],
        }
        output = {
            **_BASE_OUTPUT,
            "key_findings": [],
            "limitations": [],
        }
        mock_judge = _make_mock_judge(return_value=0.0)
        validate_claims(task, output, judge_client=mock_judge)
        assert mock_judge.judge.call_count <= MAX_JUDGE_CALLS

    def test_judge_cap_flag_set(self) -> None:
        """'judge_calls_capped' flag should be set when the budget is exhausted."""
        n = MAX_JUDGE_CALLS + 3
        task = {
            **_BASE_TASK,
            "gold_insights": [f"unique insight phrase {i} xyz" for i in range(n)],
            "required_limitations": [],
            "forbidden_claims": [],
        }
        output = {
            **_BASE_OUTPUT,
            "key_findings": [],
            "limitations": [],
        }
        mock_judge = _make_mock_judge(return_value=0.0)
        result = validate_claims(task, output, judge_client=mock_judge)
        assert "judge_calls_capped" in result.scorer_flags

    def test_forbidden_claim_in_raw_response_text(self) -> None:
        """Forbidden claim appearing only in raw_response_text should be detected."""
        task = {
            **_BASE_TASK,
            "gold_insights": [],
            "required_limitations": [],
            "forbidden_claims": ["attendance improved steadily"],
        }
        output = {
            **_BASE_OUTPUT,
            "key_findings": [],
            "limitations": [],
            # Forbidden claim hidden in raw response text.
            "raw_response_text": ("Overall, attendance improved steadily throughout the quarter."),
        }
        result = validate_claims(task, output, judge_client=None)
        # Forbidden claim was found → matched=False (penalty)
        assert result.forbidden_absent[0].matched is False
        assert result.calibration_limitation_handling == pytest.approx(0.0)

    def test_no_judge_client_skips_stage2(self) -> None:
        """Passing judge_client=None should never trigger a judge call."""
        # Verify that passing None produces a sane result with no exceptions.
        result = validate_claims(_BASE_TASK, _BASE_OUTPUT, judge_client=None)
        assert isinstance(result, ClaimValidationResult)

    def test_multiple_insights_partial_match(self) -> None:
        """Only some insights present → fractional insight sub-score."""
        task = {
            **_BASE_TASK,
            "gold_insights": [
                "attendance improved",  # will match
                "completely absent insight xyz",  # won't match
            ],
            "required_limitations": [],
            "forbidden_claims": [],
        }
        output = {
            **_BASE_OUTPUT,
            "key_findings": ["attendance improved in October"],
            "limitations": [],
        }
        result = validate_claims(task, output, judge_client=None)
        assert result.insights_present[0].matched is True
        assert result.insights_present[1].matched is False
        # insight sub-score = 0.5; no forbidden/limitation lists → score = 0.5
        assert result.calibration_limitation_handling == pytest.approx(0.5)

    def test_result_contains_correct_claim_texts(self) -> None:
        """ClaimCheckDetail.claim should preserve the original claim string."""
        task = {
            **_BASE_TASK,
            "gold_insights": ["My gold insight"],
            "required_limitations": ["My required limitation"],
            "forbidden_claims": ["My forbidden claim"],
        }
        result = validate_claims(task, _BASE_OUTPUT, judge_client=None)
        assert result.insights_present[0].claim == "My gold insight"
        assert result.limitations_present[0].claim == "My required limitation"
        assert result.forbidden_absent[0].claim == "My forbidden claim"

    def test_score_field_is_float_in_range(self) -> None:
        """All score fields in ClaimCheckDetail must be in [0.0, 1.0]."""
        result = validate_claims(_BASE_TASK, _BASE_OUTPUT, judge_client=None)
        all_details: list[ClaimCheckDetail] = (
            result.insights_present + result.forbidden_absent + result.limitations_present
        )
        for detail in all_details:
            assert 0.0 <= detail.score <= 1.0

    def test_calibration_score_in_unit_range(self) -> None:
        """calibration_limitation_handling must always be in [0.0, 1.0]."""
        result = validate_claims(_BASE_TASK, _BASE_OUTPUT, judge_client=None)
        assert 0.0 <= result.calibration_limitation_handling <= 1.0

    def test_judge_dimension_called_correctly_for_insight(self) -> None:
        """Judge should be called with dimension='insight_quality' for insight checks."""
        insight = "a unique insight not in findings xyzabc"
        task = {
            **_BASE_TASK,
            "gold_insights": [insight],
            "required_limitations": [],
            "forbidden_claims": [],
        }
        output = {**_BASE_OUTPUT, "key_findings": [], "limitations": []}
        mock_judge = _make_mock_judge(return_value=0.9)
        validate_claims(task, output, judge_client=mock_judge)
        mock_judge.judge.assert_called_once()
        call_kwargs = mock_judge.judge.call_args.kwargs
        assert call_kwargs["dimension"] == "insight_quality"

    def test_judge_dimension_called_correctly_for_limitation(self) -> None:
        """Judge should be called with calibration_limitation_handling dimension for limitations."""
        lim = "a unique limitation not in output xyzabc"
        task = {
            **_BASE_TASK,
            "gold_insights": [],
            "required_limitations": [lim],
            "forbidden_claims": [],
        }
        output = {**_BASE_OUTPUT, "key_findings": [], "limitations": []}
        mock_judge = _make_mock_judge(return_value=0.9)
        validate_claims(task, output, judge_client=mock_judge)
        mock_judge.judge.assert_called_once()
        call_kwargs = mock_judge.judge.call_args.kwargs
        assert call_kwargs["dimension"] == "calibration_limitation_handling"
