"""C3 — Claim validation scorer for the GRADE benchmark.

This module implements three validators for the C3 (Calibration & Limitation
Handling) rubric dimension:

1. **Gold insight present** — for each ``gold_insight`` in the task, check
   whether it appears (semantically) in ``key_findings`` of the model output.
2. **Forbidden claim absent** — for each ``forbidden_claim``, check that it
   does NOT appear anywhere in the model output (``key_findings``,
   ``limitations``, or ``raw_response_text``).
3. **Required limitation present** — for each ``required_limitation``, check
   whether it appears in ``limitations`` of the model output.

Detection strategy
------------------
All three validators use a two-stage pipeline:

* **Stage 1 — deterministic string matching**: normalise both the claim/insight
  and the candidate text (lower-case, collapse whitespace, strip punctuation)
  and check for substring containment or token-overlap above a configurable
  threshold (:data:`TOKEN_OVERLAP_THRESHOLD`).  This is fast, free of API
  dependencies, and handles exact and near-exact wording.

* **Stage 2 — optional judge fallback**: if Stage 1 is inconclusive *and* a
  ``judge_client`` is provided, the module emits a bounded set of judge calls
  (capped at :data:`MAX_JUDGE_CALLS` across the three claim lists) to handle
  genuine paraphrase cases.  The judge protocol is the same
  :class:`~benchmark.rubrics.rubric_scoring.JudgeClientProtocol` used by C2,
  so any existing mock judge works directly.

If no ``judge_client`` is supplied (``None``), Stage 2 is skipped and only the
deterministic result is returned.

Public API
----------
- :func:`validate_claims` — main entry point; returns a structured dict.
- :func:`check_insight_present` — single gold-insight check.
- :func:`check_forbidden_absent` — single forbidden-claim check.
- :func:`check_limitation_present` — single required-limitation check.
- :class:`ClaimValidationResult` — structured result dataclass.

Usage
-----
Import by full module path::

    from benchmark.rubrics.claim_validation import validate_claims, ClaimValidationResult

Deterministic-only (no judge)::

    result = validate_claims(task, model_output, judge_client=None)

With a mock judge (tests)::

    from unittest.mock import MagicMock
    mock_judge = MagicMock()
    mock_judge.judge.return_value = 0.9
    result = validate_claims(task, model_output, judge_client=mock_judge)

With a live judge (production)::

    from benchmark.rubrics.judge_client import JudgeClient
    client = JudgeClient(api_key="...")
    result = validate_claims(task, model_output, judge_client=client)
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "ClaimCheckDetail",
    "ClaimValidationResult",
    "check_insight_present",
    "check_forbidden_absent",
    "check_limitation_present",
    "validate_claims",
    "TOKEN_OVERLAP_THRESHOLD",
    "JUDGE_THRESHOLD",
    "MAX_JUDGE_CALLS",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum Jaccard token-overlap ratio for Stage 1 to declare a positive match.
TOKEN_OVERLAP_THRESHOLD: float = 0.5

#: Judge score threshold above which a claim is considered present/absent.
JUDGE_THRESHOLD: float = 0.5

#: Maximum total judge calls permitted across all three claim lists per
#: :func:`validate_claims` invocation.
MAX_JUDGE_CALLS: int = 9


# ---------------------------------------------------------------------------
# Judge client protocol (mirrors rubric_scoring.JudgeClientProtocol)
# ---------------------------------------------------------------------------


class JudgeClientProtocol(Protocol):
    """Structural protocol for judge clients consumed by claim validators.

    Any object implementing :meth:`judge` satisfies this protocol, including
    the live :class:`~benchmark.rubrics.judge_client.JudgeClient` and any
    test mock.
    """

    def judge(
        self,
        dimension: str,
        guidance: str,
        task: dict[str, Any],
        model_output: dict[str, Any],
    ) -> float:
        """Evaluate *model_output* for *task* on a single rubric dimension.

        Args:
            dimension: Rubric dimension name passed through from the caller.
            guidance: Free-text guidance for the judge, used here to carry the
                specific claim and target field to evaluate.
            task: Full task definition dict.
            model_output: Normalized model output dict.

        Returns:
            A float in ``[0.0, 1.0]``; values ``> JUDGE_THRESHOLD`` are treated
            as a positive match.
        """
        ...


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClaimCheckDetail:
    """Per-claim scoring detail for one check.

    Attributes:
        claim: The original claim string from the task definition.
        matched: ``True`` if the claim was found (or confirmed absent for
            forbidden-claim checks).
        method: Detection method used: ``'substring'``, ``'token_overlap'``,
            ``'judge'``, or ``'not_found'``.
        score: ``1.0`` if *matched* is ``True``, else ``0.0``.
    """

    claim: str
    matched: bool
    method: str
    score: float


@dataclass
class ClaimValidationResult:
    """Aggregate result of all three claim validators for one (task, output) pair.

    Attributes:
        insights_present: Per-gold-insight check results.  ``True`` means the
            insight was found in ``key_findings``.
        forbidden_absent: Per-forbidden-claim check results.  ``True`` means
            the forbidden claim was NOT found in the output (the desired
            outcome).
        limitations_present: Per-required-limitation check results.  ``True``
            means the limitation was found in ``limitations``.
        calibration_limitation_handling: Aggregate score for the
            ``calibration_limitation_handling`` dimension in ``[0.0, 1.0]``.
            Computed as the mean of all three sub-scores (insights, forbidden
            absence, limitations).
        scorer_flags: List of warning flags, e.g.
            ``['judge_calls_capped', 'no_required_limitations']``.
    """

    insights_present: list[ClaimCheckDetail] = field(default_factory=list)
    forbidden_absent: list[ClaimCheckDetail] = field(default_factory=list)
    limitations_present: list[ClaimCheckDetail] = field(default_factory=list)
    calibration_limitation_handling: float = 1.0
    scorer_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text-normalisation helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase, remove punctuation, and collapse whitespace.

    Args:
        text: Raw input string.

    Returns:
        Normalised string suitable for comparison.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_overlap(a: str, b: str) -> float:
    """Compute the Jaccard token-overlap between two normalised strings.

    Args:
        a: First normalised string.
        b: Second normalised string.

    Returns:
        Jaccard overlap in ``[0.0, 1.0]``: ``|intersection| / |union|`` on
        the token sets.  Returns ``0.0`` if both token sets are empty.
    """
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a and not tokens_b:
        return 0.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _string_match(claim: str, candidates: list[str]) -> tuple[bool, str]:
    """Deterministic Stage 1 match: substring or token-overlap check.

    First tries substring containment (the normalised *claim* appears inside
    any normalised *candidate* or vice-versa).  Falls back to Jaccard token
    overlap against :data:`TOKEN_OVERLAP_THRESHOLD`.

    Args:
        claim: The claim or limitation to look for.
        candidates: List of candidate strings to search in.

    Returns:
        A ``(matched, method)`` tuple where *method* is ``'substring'``,
        ``'token_overlap'``, or ``'not_found'``.
    """
    norm_claim = _normalise(claim)
    for candidate in candidates:
        norm_candidate = _normalise(candidate)
        # Substring: claim inside candidate, or short candidate inside claim
        if norm_claim in norm_candidate or norm_candidate in norm_claim:
            return True, "substring"
    # Token overlap fallback
    for candidate in candidates:
        norm_candidate = _normalise(candidate)
        if _token_overlap(norm_claim, norm_candidate) >= TOKEN_OVERLAP_THRESHOLD:
            return True, "token_overlap"
    return False, "not_found"


# ---------------------------------------------------------------------------
# Individual claim-check helpers
# ---------------------------------------------------------------------------


def check_insight_present(
    insight: str,
    key_findings: list[str],
    task: dict[str, Any],
    model_output: dict[str, Any],
    judge_client: JudgeClientProtocol | None,
    judge_calls_remaining: list[int],
) -> ClaimCheckDetail:
    """Check whether a single gold insight appears in ``key_findings``.

    Uses Stage 1 (deterministic string matching) first.  If inconclusive and
    *judge_client* is not ``None`` and *judge_calls_remaining* is > 0, falls
    back to a judge call for paraphrase tolerance.

    Args:
        insight: The gold insight string from the task definition.
        key_findings: The ``key_findings`` list from the model output.
        task: Full task definition dict (passed through to the judge).
        model_output: Normalized model output dict (passed through to the judge).
        judge_client: Optional judge client.  ``None`` disables Stage 2.
        judge_calls_remaining: Single-element list ``[n]`` used as a mutable
            counter shared across all check helpers within one
            :func:`validate_claims` call.  Decremented by 1 on each judge
            call.

    Returns:
        A :class:`ClaimCheckDetail` for this insight.
    """
    matched, method = _string_match(insight, key_findings)

    if not matched and judge_client is not None and judge_calls_remaining[0] > 0:
        guidance = (
            f"Check whether the following gold insight appears (semantically) "
            f"in the model's key_findings. "
            f"Gold insight: {insight!r}. "
            f"Return a score > {JUDGE_THRESHOLD} if it is present, else <= {JUDGE_THRESHOLD}."
        )
        score_val = judge_client.judge(
            dimension="insight_quality",
            guidance=guidance,
            task=task,
            model_output=model_output,
        )
        judge_calls_remaining[0] -= 1
        matched = score_val > JUDGE_THRESHOLD
        method = "judge"

    return ClaimCheckDetail(
        claim=insight,
        matched=matched,
        method=method,
        score=1.0 if matched else 0.0,
    )


def check_forbidden_absent(
    forbidden_claim: str,
    all_output_texts: list[str],
    task: dict[str, Any],
    model_output: dict[str, Any],
    judge_client: JudgeClientProtocol | None,
    judge_calls_remaining: list[int],
) -> ClaimCheckDetail:
    """Check whether a forbidden claim is absent from all model output text.

    A forbidden claim being **absent** is the desired outcome and yields
    ``matched=True`` (score = 1.0).  If the claim is detected in the output,
    ``matched=False`` (score = 0.0).

    Uses Stage 1 (deterministic string matching) first.  If Stage 1 finds no
    match (i.e., claim appears absent) and *judge_client* is available and
    *judge_calls_remaining* > 0, a judge call confirms the absence with
    paraphrase tolerance — if the judge says the claim IS present, the result
    is flipped to ``matched=False``.

    Args:
        forbidden_claim: The forbidden claim string from the task definition.
        all_output_texts: Combined list of all text fields from the model
            output (``key_findings`` + ``limitations`` + optionally
            ``raw_response_text`` split into sentences).
        task: Full task definition dict (passed through to the judge).
        model_output: Normalized model output dict (passed through to the
            judge).
        judge_client: Optional judge client.  ``None`` disables Stage 2.
        judge_calls_remaining: Shared mutable counter (single-element list).

    Returns:
        A :class:`ClaimCheckDetail` for this forbidden claim.
        ``matched=True`` means the claim is absent (good); ``matched=False``
        means it was detected (penalty).
    """
    found_by_string, method = _string_match(forbidden_claim, all_output_texts)

    if found_by_string:
        # Claim was detected — forbidden claim IS present, penalise.
        return ClaimCheckDetail(
            claim=forbidden_claim,
            matched=False,
            method=method,
            score=0.0,
        )

    # Stage 1 says absent; optionally confirm with judge for paraphrase cases.
    if judge_client is not None and judge_calls_remaining[0] > 0:
        guidance = (
            f"Check whether the following forbidden claim appears (semantically, "
            f"including paraphrase or implication) anywhere in the model output. "
            f"Forbidden claim: {forbidden_claim!r}. "
            f"Return a score > {JUDGE_THRESHOLD} if the claim IS present in the output "
            f"(i.e., the model asserted it), else return <= {JUDGE_THRESHOLD}."
        )
        score_val = judge_client.judge(
            dimension="calibration_limitation_handling",
            guidance=guidance,
            task=task,
            model_output=model_output,
        )
        judge_calls_remaining[0] -= 1
        claim_is_present = score_val > JUDGE_THRESHOLD
        matched = not claim_is_present  # absent == good
        method = "judge"
    else:
        # No judge — accept Stage 1 result: absent
        matched = True

    return ClaimCheckDetail(
        claim=forbidden_claim,
        matched=matched,
        method=method,
        score=1.0 if matched else 0.0,
    )


def check_limitation_present(
    limitation: str,
    limitations: list[str],
    task: dict[str, Any],
    model_output: dict[str, Any],
    judge_client: JudgeClientProtocol | None,
    judge_calls_remaining: list[int],
) -> ClaimCheckDetail:
    """Check whether a required limitation appears in the model's ``limitations``.

    Uses Stage 1 (deterministic string matching) first.  Falls back to a
    judge call when Stage 1 is inconclusive and *judge_client* is available.

    Args:
        limitation: The required limitation string from the task definition.
        limitations: The ``limitations`` list from the model output.
        task: Full task definition dict (passed through to the judge).
        model_output: Normalized model output dict (passed through to the judge).
        judge_client: Optional judge client.  ``None`` disables Stage 2.
        judge_calls_remaining: Shared mutable counter (single-element list).

    Returns:
        A :class:`ClaimCheckDetail` for this required limitation.
    """
    matched, method = _string_match(limitation, limitations)

    if not matched and judge_client is not None and judge_calls_remaining[0] > 0:
        guidance = (
            f"Check whether the following required limitation (caveat) is acknowledged "
            f"(semantically) in the model's limitations section. "
            f"Required limitation: {limitation!r}. "
            f"Return a score > {JUDGE_THRESHOLD} if it is acknowledged, else <= {JUDGE_THRESHOLD}."
        )
        score_val = judge_client.judge(
            dimension="calibration_limitation_handling",
            guidance=guidance,
            task=task,
            model_output=model_output,
        )
        judge_calls_remaining[0] -= 1
        matched = score_val > JUDGE_THRESHOLD
        method = "judge"

    return ClaimCheckDetail(
        claim=limitation,
        matched=matched,
        method=method,
        score=1.0 if matched else 0.0,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate_claims(
    task: dict[str, Any],
    model_output: dict[str, Any],
    judge_client: JudgeClientProtocol | None = None,
) -> ClaimValidationResult:
    """Validate model output against a task's claim lists (insights, forbidden, limitations).

    Returns a :class:`ClaimValidationResult` with per-claim details for each
    of the three claim lists and an aggregate
    ``calibration_limitation_handling`` score.

    The aggregate score is computed as the mean over all non-empty claim
    lists:

    - If *gold_insights* is non-empty: mean insight score.
    - If *forbidden_claims* is non-empty: mean forbidden-absence score.
    - If *required_limitations* is non-empty: mean limitation score.

    If all three lists are empty, the score is ``1.0`` by convention (nothing
    to violate).

    Judge calls across all three validators are shared and capped at
    :data:`MAX_JUDGE_CALLS` to bound API usage.

    Args:
        task: Full task definition dict validated against ``task_schema.json``.
            Uses the ``gold_insights``, ``forbidden_claims``, and
            ``required_limitations`` fields.
        model_output: Normalized model output dict validated against
            ``output_schema.json``.  Uses ``key_findings``, ``limitations``,
            and optionally ``raw_response_text``.
        judge_client: Optional judge client following
            :class:`JudgeClientProtocol`.  If ``None``, only deterministic
            Stage 1 matching is used.

    Returns:
        A :class:`ClaimValidationResult` with:

        - ``insights_present``: one :class:`ClaimCheckDetail` per gold insight.
        - ``forbidden_absent``: one :class:`ClaimCheckDetail` per forbidden
          claim (``matched=True`` means absent, which is the desired state).
        - ``limitations_present``: one :class:`ClaimCheckDetail` per required
          limitation.
        - ``calibration_limitation_handling``: aggregate score in ``[0.0, 1.0]``.
        - ``scorer_flags``: list of warning flags.

    Example::

        from benchmark.rubrics.claim_validation import validate_claims

        result = validate_claims(task, model_output, judge_client=None)
        print(result.calibration_limitation_handling)   # e.g. 0.67
        print(result.insights_present[0].matched)       # True / False
    """
    gold_insights: list[str] = task.get("gold_insights", [])
    forbidden_claims: list[str] = task.get("forbidden_claims", [])
    required_limitations: list[str] = task.get("required_limitations", [])

    key_findings: list[str] = model_output.get("key_findings", [])
    limitations: list[str] = model_output.get("limitations", [])

    # Build full output text corpus for forbidden-claim search.
    all_output_texts: list[str] = list(key_findings) + list(limitations)
    raw: str | None = model_output.get("raw_response_text")
    if raw:
        # Split raw response into sentence-sized chunks for overlap matching.
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", raw) if s.strip()]
        all_output_texts.extend(sentences)

    flags: list[str] = []
    judge_calls_remaining: list[int] = [MAX_JUDGE_CALLS]

    # --- 1. Gold insights ---
    insight_details: list[ClaimCheckDetail] = []
    for insight in gold_insights:
        detail = check_insight_present(
            insight=insight,
            key_findings=key_findings,
            task=task,
            model_output=model_output,
            judge_client=judge_client,
            judge_calls_remaining=judge_calls_remaining,
        )
        insight_details.append(detail)

    # --- 2. Forbidden claims ---
    forbidden_details: list[ClaimCheckDetail] = []
    for fc in forbidden_claims:
        detail = check_forbidden_absent(
            forbidden_claim=fc,
            all_output_texts=all_output_texts,
            task=task,
            model_output=model_output,
            judge_client=judge_client,
            judge_calls_remaining=judge_calls_remaining,
        )
        forbidden_details.append(detail)

    # --- 3. Required limitations ---
    limitation_details: list[ClaimCheckDetail] = []
    for lim in required_limitations:
        detail = check_limitation_present(
            limitation=lim,
            limitations=limitations,
            task=task,
            model_output=model_output,
            judge_client=judge_client,
            judge_calls_remaining=judge_calls_remaining,
        )
        limitation_details.append(detail)

    # Flag if judge budget was exhausted.
    if judge_calls_remaining[0] == 0 and judge_client is not None:
        flags.append("judge_calls_capped")

    # Flag if claim lists were empty.
    if not gold_insights:
        flags.append("no_gold_insights")
    if not required_limitations:
        flags.append("no_required_limitations")
    if not forbidden_claims:
        flags.append("no_forbidden_claims")

    # --- Aggregate calibration_limitation_handling score ---
    sub_scores: list[float] = []

    if insight_details:
        sub_scores.append(sum(d.score for d in insight_details) / len(insight_details))
    if forbidden_details:
        sub_scores.append(sum(d.score for d in forbidden_details) / len(forbidden_details))
    if limitation_details:
        sub_scores.append(sum(d.score for d in limitation_details) / len(limitation_details))

    calibration_score: float = sum(sub_scores) / len(sub_scores) if sub_scores else 1.0

    return ClaimValidationResult(
        insights_present=insight_details,
        forbidden_absent=forbidden_details,
        limitations_present=limitation_details,
        calibration_limitation_handling=calibration_score,
        scorer_flags=flags,
    )
