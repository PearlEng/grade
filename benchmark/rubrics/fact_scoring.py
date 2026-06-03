"""C1 — Exact/tolerant fact scoring primitives for the GRADE benchmark.

This module provides pure-function scoring primitives for the C1 (Grounding
Accuracy) rubric dimension.  Functions operate on the ``structured_metrics``
and ``key_findings`` fields of a normalized model output (``output_schema.json``)
and compare them against the ``gold_facts`` array in a task definition
(``task_schema.json``).

Matching Strategy (value-based)
--------------------------------
A gold fact is credited when its VALUE is actually present in the model output,
**regardless of the key name used**.  Real models invent arbitrary keys (e.g.
``currently_enrolled_students``), so key-name lookup is unreliable.

**Numeric gold facts** (``numeric_value`` is not None):

1. Scan **all** numeric values in the flattened ``structured_metrics`` dict
   (keys ignored entirely).  Credit (score 1.0) if any value is within
   ``tolerance`` of ``gold_fact.numeric_value``.
2. If not found in ``structured_metrics``, parse numbers out of every string in
   ``key_findings`` and ``limitations`` using :func:`_extract_numbers`.  Credit
   if any parsed number is within tolerance.

False-positive guard for numeric matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For ``structured_metrics`` the check is a precise numeric equality-within-
tolerance — no string parsing is involved, so false positives are constrained
to values numerically close to the gold value (controlled by ``tolerance``).
For free-text extraction the same strict ``score_numeric`` check applies; only
values that fall within the specified tolerance window pass.  When
``tolerance == 0`` this is an exact float comparison.

Percent/fraction normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Gold facts for rates/proportions store values as fractions (e.g.
``numeric_value = 0.8236``), but real models often report them as percentages
(e.g. ``"82.4%"`` or a structured value of ``82.4``).  When the gold value
``G`` satisfies ``0 < G <= 1``, the matcher **also** tries the candidate
divided by 100 (``N / 100``) so that a model emitting ``82.4`` (percent form)
is credited against a gold of ``0.8236`` within the stated tolerance.

The ``/100`` normalization is applied **only when ``0 < G <= 1``** (fraction
range) — it is never applied for counts or other values ``> 1``, which
prevents false positives such as crediting ``1.35`` against a gold count of
``135``.  The existing as-is comparison is always tried first; the
percent-normalized path is an additional fallback.

When a match is made via the ``/100`` path the ``method`` label in
:class:`FactDetail` includes the suffix ``+percent_normalized`` to make the
match source traceable (e.g.
``"numeric_absolute(0.0005)[key_findings+percent_normalized]"``).

**Non-numeric gold facts** (``numeric_value`` is None):

Scans ``key_findings`` text (and ``limitations``) for the claim via normalized
substring matching: at least one token from the claim must appear in at least
one finding (token-overlap fallback) or the full claim/finding must be a
substring of the other after lowercasing and whitespace normalization.

Public API
----------
- :func:`score_numeric` — absolute or relative tolerance match for floats.
- :func:`score_exact_match` — case-folded exact equality for strings/scalars.
- :func:`score_date_range` — overlap-based scoring for ISO 8601 date ranges.
- :func:`score_ranking_similarity` — Kendall tau-b normalized to [0, 1].
- :func:`score_fact` — dispatches to the appropriate primitive for one
  :class:`GoldFact`.
- :func:`score_facts` — scores a full list of gold facts and returns the
  aggregate :class:`FactScoreResult`.

Usage
-----
Import by full module path to avoid conflicts with parallel scorer modules::

    from benchmark.rubrics.fact_scoring import score_facts, FactScoreResult
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

__all__ = [
    "GoldFact",
    "FactScoreResult",
    "FactDetail",
    "score_numeric",
    "score_exact_match",
    "score_date_range",
    "score_ranking_similarity",
    "score_fact",
    "score_facts",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldFact:
    """A single verifiable ground-truth claim derived from fixture data.

    Attributes:
        fact_id: Identifier unique within the task, e.g. ``'F1'``.
        claim: The exact natural-language claim.
        source_files: Fixture files that contain the data supporting this fact.
        numeric_value: Expected numeric value for automated comparison, or
            ``None`` for non-numeric facts.
        tolerance: Allowed absolute deviation for numeric comparisons.
            ``None`` means exact numeric match (i.e., tolerance = 0).
    """

    fact_id: str
    claim: str
    source_files: list[str]
    numeric_value: float | None = None
    tolerance: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldFact:
        """Construct a :class:`GoldFact` from a raw task-schema dict.

        Args:
            d: A dict conforming to the ``GoldFact`` $def in
                ``task_schema.json``.

        Returns:
            A :class:`GoldFact` instance.
        """
        return cls(
            fact_id=d["fact_id"],
            claim=d["claim"],
            source_files=d["source_files"],
            numeric_value=d.get("numeric_value"),
            tolerance=d.get("tolerance"),
        )


@dataclass
class FactDetail:
    """Per-fact scoring detail included in :class:`FactScoreResult`.

    Attributes:
        fact_id: Fact identifier from the gold fact.
        score: Score in ``[0.0, 1.0]`` awarded for this fact.
        matched: ``True`` if the predicted value satisfied the scoring
            criterion.
        predicted_value: The value extracted from the model output, or
            ``None`` if no value was found.
        method: The scoring method used (``'numeric_absolute'``,
            ``'numeric_relative'``, ``'exact_match'``, ``'not_found'``).
    """

    fact_id: str
    score: float
    matched: bool
    predicted_value: Any = None
    method: str = "not_found"


@dataclass
class FactScoreResult:
    """Aggregate result for a full set of gold facts scored against one output.

    Attributes:
        grounding_accuracy: Mean per-fact score in ``[0.0, 1.0]``.  This maps
            directly to the ``grounding_accuracy`` dimension in
            ``result_schema.json``.
        facts_total: Total number of gold facts evaluated.
        facts_matched: Number of gold facts that scored 1.0.
        details: Per-fact breakdown as a list of :class:`FactDetail`.
        scorer_flags: List of warning flags, e.g.
            ``['numeric_tolerance_applied']``.
    """

    grounding_accuracy: float
    facts_total: int
    facts_matched: int
    details: list[FactDetail] = field(default_factory=list)
    scorer_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Primitive scoring functions
# ---------------------------------------------------------------------------


def score_numeric(predicted: float, gold: float, tolerance: float | str) -> float:
    """Score a numeric prediction against a gold value with absolute or relative tolerance.

    Supports two tolerance modes:

    - **Absolute**: ``tolerance`` is a non-negative ``float``.  Returns
      ``1.0`` if ``abs(predicted - gold) <= tolerance``, else ``0.0``.
    - **Relative**: ``tolerance`` is a ``str`` ending in ``'%'`` (e.g.
      ``"5%"``).  The percentage is applied to ``abs(gold)`` (or ``1.0`` if
      gold is zero) to derive an absolute threshold.

    Args:
        predicted: The numeric value extracted from the model output.
        gold: The expected numeric value from the gold fact.
        tolerance: Allowed deviation.  Pass ``0.0`` or ``0`` for exact match.

    Returns:
        ``1.0`` if the prediction is within tolerance, ``0.0`` otherwise.

    Raises:
        TypeError: If *tolerance* is neither a ``float``/``int`` nor a
            percentage ``str``.
        ValueError: If *tolerance* is a negative number or a malformed
            percentage string.

    Examples:
        >>> score_numeric(80.9, 80.9, 0.0)
        1.0
        >>> score_numeric(80.85, 80.9, 0.05)
        1.0
        >>> score_numeric(79.0, 80.9, 0.05)
        0.0
        >>> score_numeric(105.0, 100.0, "5%")
        1.0
        >>> score_numeric(106.0, 100.0, "5%")
        0.0
    """
    if isinstance(tolerance, str):
        match = re.fullmatch(r"([0-9]*\.?[0-9]+)%", tolerance.strip())
        if match is None:
            raise ValueError(f"tolerance string must be a percentage like '5%', got {tolerance!r}")
        pct = float(match.group(1))
        if pct < 0:
            raise ValueError(f"tolerance percentage must be non-negative, got {pct}")
        base = abs(gold) if gold != 0 else 1.0
        abs_tolerance = (pct / 100.0) * base
    elif isinstance(tolerance, (int, float)):
        abs_tolerance = float(tolerance)
        if abs_tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {abs_tolerance}")
    else:
        raise TypeError(
            f"tolerance must be float/int or a percentage str, got {type(tolerance).__name__}"
        )

    return 1.0 if abs(predicted - gold) <= abs_tolerance else 0.0


def score_exact_match(predicted: Any, gold: Any) -> float:
    """Score a predicted value using case-folded exact equality.

    For ``str`` values, both sides are stripped of leading/trailing whitespace
    and lower-cased before comparison.  For all other types the raw ``==``
    operator is used.

    Args:
        predicted: The value extracted from the model output.
        gold: The expected value from the gold fact.

    Returns:
        ``1.0`` if *predicted* equals *gold* (after normalization), else ``0.0``.

    Examples:
        >>> score_exact_match("SCH-001", "SCH-001")
        1.0
        >>> score_exact_match("sch-001", "SCH-001")
        1.0
        >>> score_exact_match("SCH-001", "SCH-002")
        0.0
        >>> score_exact_match(42, 42)
        1.0
        >>> score_exact_match(42, 43)
        0.0
    """
    if isinstance(predicted, str) and isinstance(gold, str):
        return 1.0 if predicted.strip().lower() == gold.strip().lower() else 0.0
    return 1.0 if predicted == gold else 0.0


def score_date_range(
    predicted_range: tuple[str, str],
    gold_range: tuple[str, str],
) -> float:
    """Score a predicted date range against a gold date range using overlap ratio.

    Both ranges are expressed as ``(start_iso, end_iso)`` tuples of ISO 8601
    date strings (``'YYYY-MM-DD'``).  The score is the Jaccard overlap of the
    two intervals:

        score = |intersection| / |union|

    where lengths are measured in calendar days (inclusive).  A perfect match
    returns ``1.0``; no overlap returns ``0.0``.

    Args:
        predicted_range: ``(start, end)`` ISO 8601 date strings for the
            predicted range, e.g. ``('2025-09-01', '2025-11-30')``.
        gold_range: ``(start, end)`` ISO 8601 date strings for the gold range.

    Returns:
        Jaccard overlap in ``[0.0, 1.0]``.

    Raises:
        ValueError: If any date string cannot be parsed as ``YYYY-MM-DD`` or
            if start > end for either range.

    Examples:
        >>> score_date_range(("2025-09-01", "2025-11-30"), ("2025-09-01", "2025-11-30"))
        1.0
        >>> score_date_range(("2025-10-01", "2025-10-31"), ("2025-09-01", "2025-11-30"))
        0.3333...
        >>> score_date_range(("2026-01-01", "2026-01-31"), ("2025-09-01", "2025-11-30"))
        0.0
    """

    def _parse(s: str) -> date:
        try:
            return date.fromisoformat(s)
        except ValueError as exc:
            raise ValueError(f"Cannot parse date {s!r} as YYYY-MM-DD: {exc}") from exc

    p_start, p_end = _parse(predicted_range[0]), _parse(predicted_range[1])
    g_start, g_end = _parse(gold_range[0]), _parse(gold_range[1])

    if p_start > p_end:
        raise ValueError(f"predicted_range start {p_start} is after end {p_end}")
    if g_start > g_end:
        raise ValueError(f"gold_range start {g_start} is after end {g_end}")

    # Compute overlap (inclusive day counts)
    overlap_start = max(p_start, g_start)
    overlap_end = min(p_end, g_end)

    if overlap_start > overlap_end:
        return 0.0  # No overlap

    intersection_days = (overlap_end - overlap_start).days + 1
    p_days = (p_end - p_start).days + 1
    g_days = (g_end - g_start).days + 1
    union_days = p_days + g_days - intersection_days

    return intersection_days / union_days


def score_ranking_similarity(
    predicted_ranking: list[Any],
    gold_ranking: list[Any],
) -> float:
    """Score a predicted ranking against a gold ranking using normalized Kendall tau-b.

    Kendall tau-b counts concordant minus discordant pairs in the ordering and
    normalises to ``[−1, 1]``.  This function maps the result to ``[0, 1]``
    via ``(tau_b + 1) / 2`` so that a perfect match returns ``1.0``, a
    perfect reversal returns ``0.0``, and a random ranking returns
    approximately ``0.5``.

    Only items appearing in **both** lists are scored.  Items present in one
    list but absent from the other are ignored.

    Args:
        predicted_ranking: Ordered list of items produced by the model.
            Earlier positions imply higher rank.
        gold_ranking: Ordered list of items as ground truth.

    Returns:
        Normalized Kendall tau-b in ``[0.0, 1.0]``.

    Raises:
        ValueError: If the common item set has fewer than two elements
            (tau is undefined for n < 2).

    Examples:
        >>> score_ranking_similarity(["A", "B", "C"], ["A", "B", "C"])
        1.0
        >>> score_ranking_similarity(["C", "B", "A"], ["A", "B", "C"])
        0.0
        >>> score_ranking_similarity(["A", "C", "B"], ["A", "B", "C"])
        0.6666...
    """
    # Build rank maps for common items
    gold_set = {item: rank for rank, item in enumerate(gold_ranking)}
    common = [item for item in predicted_ranking if item in gold_set]

    n = len(common)
    if n < 2:
        raise ValueError(f"Need at least 2 common items to compute ranking similarity, got {n}")

    predicted_ranks = list(range(n))  # order in predicted (restricted to common)
    gold_ranks = [gold_set[item] for item in common]

    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    tied_pred = 0
    tied_gold = 0

    for i in range(n):
        for j in range(i + 1, n):
            p_diff = predicted_ranks[i] - predicted_ranks[j]
            g_diff = gold_ranks[i] - gold_ranks[j]
            product = p_diff * g_diff
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if p_diff == 0:
                    tied_pred += 1
                if g_diff == 0:
                    tied_gold += 1

    n_pairs = n * (n - 1) / 2
    denominator = math.sqrt((n_pairs - tied_pred) * (n_pairs - tied_gold))

    if denominator == 0.0:
        # All ties — treat as perfect match
        return 1.0

    tau_b = (concordant - discordant) / denominator
    # Map [-1, 1] → [0, 1]
    return (tau_b + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Number extraction helper
# ---------------------------------------------------------------------------

# Matches numbers in free text: handles optional leading $, optional commas as
# thousands separators, optional trailing %, optional decimal part.
# Examples matched: 135, 82.4%, $1,250, 1,234, 0.8092
_NUMBER_PATTERN = re.compile(
    r"(?<![.\d])"  # not preceded by digit or dot (avoid mid-word matches)
    r"\$?"  # optional currency prefix
    r"(-?"  # optional negative sign
    r"\d{1,3}"  # leading digits (1–3 for comma-group start)
    r"(?:,\d{3})*"  # optional comma-grouped thousands
    r"(?:\.\d+)?"  # optional decimal
    r")"
    r"%?"  # optional trailing percent (consumed but not returned)
    r"(?![\d,])"  # not followed by digit or comma (avoid partial matches)
)


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from a free-text string.

    Handles formats like ``135``, ``82.4%``, ``$1,250``, ``1,234``, ``0.8092``.
    Percentage signs are stripped before conversion (the raw numeric value is
    returned, not the fractional equivalent — so ``"82.4%"`` returns ``82.4``).

    Args:
        text: Arbitrary free-text string.

    Returns:
        List of floats extracted from the text, in order of appearance.
        Empty list if no numbers found.
    """
    results: list[float] = []
    for m in _NUMBER_PATTERN.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            results.append(float(raw))
        except ValueError:
            pass
    return results


# ---------------------------------------------------------------------------
# Fact-level dispatcher (value-based matching)
# ---------------------------------------------------------------------------


def _find_predicted_numeric(
    gold_value: float,
    tolerance: float | str,
    structured_metrics: dict[str, Any],
    key_findings: list[str],
    limitations: list[str],
) -> tuple[float | None, str]:
    """Search for a numeric value matching *gold_value* within *tolerance*.

    Search order (stops at first match):

    1. All **numeric** values in ``structured_metrics`` (keys ignored entirely).
    2. Numbers parsed from each string in ``key_findings``.
    3. Numbers parsed from each string in ``limitations``.

    For each candidate value ``N``, two comparisons are attempted:

    - **As-is**: ``score_numeric(N, gold_value, tolerance)``.
    - **Percent-normalized** (only when ``0 < gold_value <= 1``):
      ``score_numeric(N / 100, gold_value, tolerance)``.  This handles the
      common case where the gold fact stores a fraction/rate (e.g. ``0.8236``)
      but the model reports it as a percentage (e.g. ``82.4`` or ``"82.4%"``).
      The guard ``0 < gold_value <= 1`` prevents false positives for counts
      (e.g. gold ``135`` would not match candidate ``1.35``).

    When a match is made via the percent-normalized path the returned source
    label is suffixed with ``'+percent_normalized'`` for traceability.

    Args:
        gold_value: The expected numeric value.
        tolerance: Absolute float tolerance or ``"N%"`` relative string.
        structured_metrics: ``structured_metrics`` dict from the model output.
        key_findings: ``key_findings`` list from the model output.
        limitations: ``limitations`` list from the model output.

    Returns:
        ``(matched_value, source_label)`` where *source_label* is one of
        ``'structured_metrics'``, ``'key_findings'``, ``'limitations'``,
        ``'structured_metrics+percent_normalized'``,
        ``'key_findings+percent_normalized'``,
        ``'limitations+percent_normalized'``, or ``'not_found'``.
    """
    # Whether to attempt the /100 normalization (only safe for fraction/rate gold values)
    try_pct_norm: bool = 0 < gold_value <= 1

    def _matches_candidate(candidate: float) -> str:
        """Return '' if no match, 'as_is' or 'pct_norm' for the matching form."""
        if score_numeric(candidate, gold_value, tolerance) == 1.0:
            return "as_is"
        if try_pct_norm and score_numeric(candidate / 100.0, gold_value, tolerance) == 1.0:
            return "pct_norm"
        return ""

    # 1. Scan all numeric values in structured_metrics (value-based, key-agnostic)
    for val in structured_metrics.values():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            match_kind = _matches_candidate(float(val))
            if match_kind == "as_is":
                return (float(val), "structured_metrics")
            if match_kind == "pct_norm":
                return (float(val), "structured_metrics+percent_normalized")

    # 2. Parse numbers from key_findings free text
    for finding in key_findings:
        for num in _extract_numbers(finding):
            match_kind = _matches_candidate(num)
            if match_kind == "as_is":
                return (num, "key_findings")
            if match_kind == "pct_norm":
                return (num, "key_findings+percent_normalized")

    # 3. Parse numbers from limitations free text
    for lim in limitations:
        for num in _extract_numbers(lim):
            match_kind = _matches_candidate(num)
            if match_kind == "as_is":
                return (num, "limitations")
            if match_kind == "pct_norm":
                return (num, "limitations+percent_normalized")

    return (None, "not_found")


def _find_predicted_text(
    claim: str,
    key_findings: list[str],
    limitations: list[str],
) -> tuple[str | None, str]:
    """Search for a non-numeric claim in ``key_findings`` and ``limitations``.

    Matching uses normalized substring containment (case-insensitive, leading/
    trailing whitespace stripped).  The claim is credited when either the claim
    is a substring of a finding or the finding is a substring of the claim.

    Args:
        claim: The gold fact claim text.
        key_findings: ``key_findings`` list from the model output.
        limitations: ``limitations`` list from the model output.

    Returns:
        ``(matched_text, source_label)`` or ``(None, 'not_found')``.
    """
    claim_lower = claim.strip().lower()

    for text_list, label in [(key_findings, "key_findings"), (limitations, "limitations")]:
        for entry in text_list:
            entry_lower = entry.strip().lower()
            if claim_lower in entry_lower or entry_lower in claim_lower:
                return (entry, label)

    return (None, "not_found")


def score_fact(
    gold_fact: GoldFact,
    structured_metrics: dict[str, Any],
    key_findings: list[str],
    limitations: list[str] | None = None,
) -> FactDetail:
    """Score one gold fact against a model output.

    Dispatches to :func:`score_numeric` when ``gold_fact.numeric_value`` is
    not ``None``, and to :func:`score_exact_match` for non-numeric (string)
    facts.

    For **numeric** facts the effective tolerance is:

    - ``gold_fact.tolerance`` if set (absolute or ``"N%"`` relative string).
    - ``0.0`` (exact match) if ``gold_fact.tolerance`` is ``None``.

    Matching is **value-based**: the key name in ``structured_metrics`` is
    ignored entirely.  Any numeric value in ``structured_metrics`` that falls
    within tolerance of ``gold_fact.numeric_value`` will credit the fact.  If
    no match is found in ``structured_metrics``, numbers are parsed from
    ``key_findings`` (and ``limitations``) free text.

    For **non-numeric** facts the predicted value is taken from the first
    ``key_findings`` (or ``limitations``) entry that contains the claim as a
    substring (or vice-versa); if neither yields a value the fact scores ``0.0``.

    Args:
        gold_fact: The gold fact to evaluate.
        structured_metrics: ``structured_metrics`` dict from the model output.
        key_findings: ``key_findings`` list from the model output.
        limitations: Optional ``limitations`` list from the model output.
            Searched as a fallback after ``key_findings``.

    Returns:
        A :class:`FactDetail` with the per-fact score and diagnostic fields.
    """
    lims: list[str] = limitations if limitations is not None else []

    if gold_fact.numeric_value is not None:
        tolerance: float | str = gold_fact.tolerance if gold_fact.tolerance is not None else 0.0
        method_label = (
            f"numeric_relative({tolerance})"
            if isinstance(tolerance, str)
            else f"numeric_absolute({tolerance})"
        )

        matched_val, source = _find_predicted_numeric(
            gold_fact.numeric_value,
            tolerance,
            structured_metrics,
            key_findings,
            lims,
        )

        if matched_val is None:
            return FactDetail(
                fact_id=gold_fact.fact_id,
                score=0.0,
                matched=False,
                predicted_value=None,
                method="not_found",
            )

        # Score is always 1.0 here (find already verified within tolerance)
        return FactDetail(
            fact_id=gold_fact.fact_id,
            score=1.0,
            matched=True,
            predicted_value=matched_val,
            method=f"{method_label}[{source}]",
        )

    else:
        # Non-numeric: search key_findings / limitations for claim text
        matched_text, source = _find_predicted_text(gold_fact.claim, key_findings, lims)

        if matched_text is None:
            return FactDetail(
                fact_id=gold_fact.fact_id,
                score=0.0,
                matched=False,
                predicted_value=None,
                method="not_found",
            )

        s = score_exact_match(matched_text, gold_fact.claim)
        return FactDetail(
            fact_id=gold_fact.fact_id,
            score=s,
            matched=s == 1.0,
            predicted_value=matched_text,
            method=f"exact_match[{source}]",
        )


def score_facts(
    gold_facts: list[GoldFact] | list[dict[str, Any]],
    structured_metrics: dict[str, Any],
    key_findings: list[str],
    limitations: list[str] | None = None,
) -> FactScoreResult:
    """Score all gold facts for a task against a model output.

    Accepts either a list of :class:`GoldFact` instances or raw dicts
    conforming to the ``GoldFact`` $def in ``task_schema.json``
    (auto-converted via :meth:`GoldFact.from_dict`).

    The ``grounding_accuracy`` field of the returned :class:`FactScoreResult`
    is the arithmetic mean of all per-fact scores and corresponds directly to
    the ``grounding_accuracy`` dimension in ``result_schema.json``.

    If *gold_facts* is empty, ``grounding_accuracy`` is ``1.0`` by convention
    (no facts to violate).

    Matching is **value-based** for numeric facts: the key name in
    ``structured_metrics`` is irrelevant; only the numeric value matters.  See
    :func:`score_fact` for the full matching strategy.

    Args:
        gold_facts: List of gold facts.  May be :class:`GoldFact` instances or
            raw task-schema dicts.
        structured_metrics: ``structured_metrics`` dict from the model output.
        key_findings: ``key_findings`` list from the model output.
        limitations: Optional ``limitations`` list from the model output.
            Searched as a fallback after ``key_findings`` for both numeric and
            non-numeric facts.

    Returns:
        A :class:`FactScoreResult` with the aggregate grounding accuracy score
        and per-fact details.

    Examples:
        >>> from benchmark.rubrics.fact_scoring import GoldFact, score_facts
        >>> facts = [GoldFact("F1", "135 students", ["students.csv"], 135, 0)]
        >>> result = score_facts(facts, {"currently_enrolled_students": 135}, [])
        >>> result.grounding_accuracy
        1.0
    """
    lims: list[str] = limitations if limitations is not None else []

    # Normalise to GoldFact instances
    normalised: list[GoldFact] = []
    for item in gold_facts:
        if isinstance(item, GoldFact):
            normalised.append(item)
        else:
            normalised.append(GoldFact.from_dict(item))

    if not normalised:
        return FactScoreResult(
            grounding_accuracy=1.0,
            facts_total=0,
            facts_matched=0,
            details=[],
            scorer_flags=[],
        )

    details: list[FactDetail] = []
    flags: list[str] = []

    for gf in normalised:
        detail = score_fact(gf, structured_metrics, key_findings, lims)
        details.append(detail)
        if gf.numeric_value is not None:
            tol = gf.tolerance
            if isinstance(tol, str) or (isinstance(tol, (int, float)) and tol > 0):
                if "numeric_tolerance_applied" not in flags:
                    flags.append("numeric_tolerance_applied")

    total = len(details)
    matched = sum(1 for d in details if d.matched)
    accuracy = sum(d.score for d in details) / total

    return FactScoreResult(
        grounding_accuracy=accuracy,
        facts_total=total,
        facts_matched=matched,
        details=details,
        scorer_flags=flags,
    )
