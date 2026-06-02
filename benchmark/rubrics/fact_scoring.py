"""C1 — Exact/tolerant fact scoring primitives for the GRADE benchmark.

This module provides pure-function scoring primitives for the C1 (Grounding
Accuracy) rubric dimension.  Functions operate on the ``structured_metrics``
and ``key_findings`` fields of a normalized model output (``output_schema.json``)
and compare them against the ``gold_facts`` array in a task definition
(``task_schema.json``).

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
# Fact-level dispatcher
# ---------------------------------------------------------------------------


def _find_predicted_value(
    gold_fact: GoldFact,
    structured_metrics: dict[str, Any],
    key_findings: list[str],
) -> tuple[Any, str]:
    """Search structured_metrics and key_findings for a value matching *gold_fact*.

    Strategy:

    1. If *gold_fact* has a ``numeric_value``:

       a. Look for a ``structured_metrics`` key whose name contains the
          fact_id (e.g. ``'f1'`` in ``'f1_total_students'``).
       b. Fall back: among all numeric values in ``structured_metrics``,
          return the one **closest** to ``gold_fact.numeric_value``.  Using
          the closest-value heuristic avoids the ambiguity that arises when
          multiple facts share the same output dict and key names do not
          encode the fact_id.
       c. Try to extract a number from ``key_findings`` text.

    2. For non-numeric facts, look for a ``structured_metrics`` key whose
       name matches the fact_id, or fall back to searching ``key_findings``
       text for the claim.

    This is intentionally simple and permissive — exact matching is done by
    the caller.  Returns ``(value, source)`` where *source* is
    ``'structured_metrics'`` or ``'key_findings'`` or ``'not_found'``.

    Args:
        gold_fact: The gold fact to look up.
        structured_metrics: ``structured_metrics`` dict from the model output.
        key_findings: ``key_findings`` list from the model output.

    Returns:
        A ``(predicted_value, source_label)`` tuple.
    """
    fact_id_lower = gold_fact.fact_id.lower()

    # --- Numeric path ---
    if gold_fact.numeric_value is not None:
        # 1a. Try to find a structured_metrics key that mentions the fact_id
        for key, val in structured_metrics.items():
            if isinstance(val, (int, float)) and fact_id_lower in key.lower():
                return (float(val), "structured_metrics")

        # 1b. Fall back: numeric value in structured_metrics closest to gold
        numeric_candidates = [
            float(v) for v in structured_metrics.values() if isinstance(v, (int, float))
        ]
        if numeric_candidates:
            gold_num: float = gold_fact.numeric_value  # narrowed; not None here
            closest = min(numeric_candidates, key=lambda v: abs(v - gold_num))
            return (closest, "structured_metrics")

        # 1c. Try to extract a number from key_findings text
        for finding in key_findings:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", finding)
            if numbers:
                return (float(numbers[0]), "key_findings")

    # --- Non-numeric path ---
    else:
        # 2a. Look for exact string value in structured_metrics
        for key, val in structured_metrics.items():
            if isinstance(val, str) and fact_id_lower in key.lower():
                return (val, "structured_metrics")

        # 2b. Search key_findings for the claim text
        claim_lower = gold_fact.claim.lower()
        for finding in key_findings:
            if finding.lower() in claim_lower or claim_lower in finding.lower():
                return (finding, "key_findings")

    return (None, "not_found")


def score_fact(
    gold_fact: GoldFact,
    structured_metrics: dict[str, Any],
    key_findings: list[str],
) -> FactDetail:
    """Score one gold fact against a model output.

    Dispatches to :func:`score_numeric` when ``gold_fact.numeric_value`` is
    not ``None``, and to :func:`score_exact_match` for non-numeric (string)
    facts.

    For **numeric** facts the effective tolerance is:

    - ``gold_fact.tolerance`` if set (absolute or ``"N%"`` relative string).
    - ``0.0`` (exact match) if ``gold_fact.tolerance`` is ``None``.

    For **non-numeric** facts the predicted value is taken from
    ``structured_metrics`` first, then from the first ``key_findings`` entry
    that contains the claim; if neither yields a value the fact scores ``0.0``.

    Args:
        gold_fact: The gold fact to evaluate.
        structured_metrics: ``structured_metrics`` dict from the model output.
        key_findings: ``key_findings`` list from the model output.

    Returns:
        A :class:`FactDetail` with the per-fact score and diagnostic fields.
    """
    predicted_value, _source = _find_predicted_value(gold_fact, structured_metrics, key_findings)

    if predicted_value is None:
        return FactDetail(
            fact_id=gold_fact.fact_id,
            score=0.0,
            matched=False,
            predicted_value=None,
            method="not_found",
        )

    if gold_fact.numeric_value is not None:
        tolerance: float | str = gold_fact.tolerance if gold_fact.tolerance is not None else 0.0
        method = (
            f"numeric_relative({tolerance})"
            if isinstance(tolerance, str)
            else f"numeric_absolute({tolerance})"
        )
        s = score_numeric(predicted_value, gold_fact.numeric_value, tolerance)
        return FactDetail(
            fact_id=gold_fact.fact_id,
            score=s,
            matched=s == 1.0,
            predicted_value=predicted_value,
            method=method,
        )
    else:
        s = score_exact_match(predicted_value, gold_fact.claim)
        return FactDetail(
            fact_id=gold_fact.fact_id,
            score=s,
            matched=s == 1.0,
            predicted_value=predicted_value,
            method="exact_match",
        )


def score_facts(
    gold_facts: list[GoldFact] | list[dict[str, Any]],
    structured_metrics: dict[str, Any],
    key_findings: list[str],
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

    Args:
        gold_facts: List of gold facts.  May be :class:`GoldFact` instances or
            raw task-schema dicts.
        structured_metrics: ``structured_metrics`` dict from the model output.
        key_findings: ``key_findings`` list from the model output.

    Returns:
        A :class:`FactScoreResult` with the aggregate grounding accuracy score
        and per-fact details.

    Examples:
        >>> from benchmark.rubrics.fact_scoring import GoldFact, score_facts
        >>> facts = [GoldFact("F1", "135 students", ["students.csv"], 135, 0)]
        >>> result = score_facts(facts, {"total_students": 135}, [])
        >>> result.grounding_accuracy
        1.0
    """
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
        detail = score_fact(gf, structured_metrics, key_findings)
        details.append(detail)
        if (
            detail.method.startswith("numeric_absolute")
            and gf.tolerance is not None
            and gf.tolerance > 0
        ):
            if "numeric_tolerance_applied" not in flags:
                flags.append("numeric_tolerance_applied")
        if detail.method.startswith("numeric_relative"):
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
