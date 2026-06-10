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
   if any parsed number is within tolerance — **but only when the containing
   string shares at least one content token with the gold claim** (the
   free-text context gate; see :func:`_shares_claim_context`).  This prevents
   numerically-close values in unrelated sentences from crediting the fact,
   which would otherwise reward number-dense outputs regardless of relevance.

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

Leniency floor for prose-first grading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When models write natural prose (rather than rigid structured output) they
naturally express rates and counts with conversational rounding — e.g.
"about 82%" for a gold value of 0.8236, or "around 133 students" for a gold
count of 135.  A gold fact's authored ``tolerance`` may be too tight for this
realistic variation.

Two named floor constants define the **minimum** tolerance that is always
applied, regardless of the authored value:

- :data:`RATE_TOLERANCE_FLOOR` (``0.01``, i.e. 1 percentage point): applied
  when the gold value ``G`` satisfies ``0 < G < 1`` (a rate/fraction).
  Effective tolerance = ``max(authored_tolerance, RATE_TOLERANCE_FLOOR)``.

- :data:`COUNT_RELATIVE_FLOOR` (``0.02``, i.e. 2%): applied when ``G >= 1``
  (a count or magnitude).  The floor is ``COUNT_RELATIVE_FLOOR * abs(G)``,
  so for a gold count of 135 the minimum window is ±2.7 (i.e. values 132–138
  are always credited).  Effective tolerance =
  ``max(authored_absolute_tolerance, COUNT_RELATIVE_FLOOR * abs(G))``.

The floor is **only a floor** — it loosens tight tolerances but never
tightens a generous one.  It is applied inside :func:`score_fact` before
delegating to :func:`_find_predicted_numeric`, which means all match paths
(structured_metrics, key_findings, limitations, and the /100 percent-
normalized path) benefit from the lenient tolerance automatically.

False-positive guard
''''''''''''''''''''
The floor values are deliberately conservative so that clearly wrong answers
are still rejected:

- A rate of 0.70 (70%) is **not** credited for a gold of 0.8236 (82.4%)
  because |0.70 − 0.8236| = 0.1236 >> 0.01.
- A count of 200 is **not** credited for a gold of 135 because
  |200 − 135| = 65 >> 0.02 × 135 ≈ 2.7.

**Non-numeric gold facts** (``numeric_value`` is None):

Scans ``key_findings`` text (and ``limitations``) for the claim in two
stages: (1) normalized substring containment (either direction, after
lowercasing and whitespace stripping), then (2) token overlap — at least
:data:`TEXT_FACT_OVERLAP_THRESHOLD` (50%) of the claim's content tokens must
appear in the entry.  A matching entry scores 1.0; paraphrases are credited
without requiring a verbatim echo of the claim.

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
    "RATE_TOLERANCE_FLOOR",
    "COUNT_RELATIVE_FLOOR",
    "score_numeric",
    "score_exact_match",
    "score_date_range",
    "score_ranking_similarity",
    "score_fact",
    "score_facts",
]

# ---------------------------------------------------------------------------
# Leniency-floor constants
# ---------------------------------------------------------------------------

#: Minimum absolute tolerance for **rate/fraction** gold facts (``0 < G < 1``).
#:
#: Natural-prose responses round rates conversationally — "about 82%" vs a gold
#: of 0.8236.  A floor of 1 percentage point (0.01 on the fraction scale) ensures
#: such phrasing is credited without accepting clearly wrong answers (e.g. 70%
#: for a true 82% is still rejected since |0.70 − 0.8236| = 0.1236 >> 0.01).
RATE_TOLERANCE_FLOOR: float = 0.01

#: Minimum relative tolerance for **count/magnitude** gold facts (``G >= 1``),
#: expressed as a fraction of the gold value.
#:
#: Applied as ``COUNT_RELATIVE_FLOOR * abs(gold)`` to derive an absolute floor.
#: A 2% window on a gold count of 135 allows values 132–138, accommodating
#: phrasing like "approximately 133 students" without crediting clearly wrong
#: answers (e.g. 200 for a true 135 is still rejected since |200 − 135| = 65
#: >> 0.02 × 135 ≈ 2.7).
COUNT_RELATIVE_FLOOR: float = 0.02

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

#: Common function words excluded when extracting a claim's content tokens for
#: the free-text context gate.  Deliberately small — only words that carry no
#: topical signal in benchmark claims.
_CONTEXT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "per",
        "than",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-]*")


def _content_tokens(text: str) -> set[str]:
    """Extract normalized content tokens from *text* for the context gate.

    Lowercases, splits on non-alphanumerics (keeping in-word hyphens, so IDs
    like ``sch-001`` survive), drops stopwords, pure numbers, and 1-2 char
    fragments, and normalizes a trailing plural ``s`` so ``student`` matches
    ``students``.

    Args:
        text: Arbitrary text (a gold claim or a model sentence).

    Returns:
        A set of normalized content tokens (possibly empty).
    """
    tokens: set[str] = set()
    for tok in _TOKEN_PATTERN.findall(text.lower()):
        if tok in _CONTEXT_STOPWORDS or len(tok) < 3:
            continue
        if tok.replace("-", "").replace(".", "").isdigit():
            continue
        tokens.add(tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok)
    return tokens


def _shares_claim_context(claim_tokens: set[str], sentence: str) -> bool:
    """Return True when *sentence* shares at least one content token with the claim.

    This is the free-text false-positive gate: a number found in prose is only
    credited when its containing sentence is topically related to the gold
    claim.  When the claim yields no content tokens at all (pathological), the
    gate is open — gating on nothing would reject everything.

    Args:
        claim_tokens: Output of :func:`_content_tokens` for the gold claim.
        sentence: The candidate finding/limitation string.

    Returns:
        ``True`` if the sentence passes the context gate.
    """
    if not claim_tokens:
        return True
    return bool(claim_tokens & _content_tokens(sentence))


def _find_predicted_numeric(
    gold_value: float,
    tolerance: float | str,
    structured_metrics: dict[str, Any],
    key_findings: list[str],
    limitations: list[str],
    claim: str = "",
) -> tuple[float | None, str]:
    """Search for a numeric value matching *gold_value* within *tolerance*.

    Search order (stops at first match):

    1. All **numeric** values in ``structured_metrics`` (keys ignored entirely).
    2. Numbers parsed from each string in ``key_findings``.
    3. Numbers parsed from each string in ``limitations``.

    Free-text context gate (false-positive guard)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Numbers parsed from free text (``key_findings`` / ``limitations``) are
    only credited when the containing string shares at least one content
    token with the gold *claim* (see :func:`_content_tokens`).  Without the
    gate, any numerically-close value anywhere in a long prose response —
    up to 50 sentences — credits the fact, which rewards number-dense
    outputs regardless of relevance.  ``structured_metrics`` matching stays
    fully key-agnostic: structured values are deliberate model assertions
    (and key-name matching was previously found too brittle against real
    model outputs), so the spam guard targets prose only.

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
        claim: The gold fact's claim text, used for the free-text context
            gate.  An empty claim disables the gate (matches old behavior).

    Returns:
        ``(matched_value, source_label)`` where *source_label* is one of
        ``'structured_metrics'``, ``'key_findings'``, ``'limitations'``,
        ``'structured_metrics+percent_normalized'``,
        ``'key_findings+percent_normalized'``,
        ``'limitations+percent_normalized'``, or ``'not_found'``.
    """
    # Whether to attempt the /100 normalization (only safe for fraction/rate gold values)
    try_pct_norm: bool = 0 < gold_value <= 1
    claim_tokens: set[str] = _content_tokens(claim)

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

    # 2./3. Parse numbers from free text, gated on claim-context overlap.
    for text_list, label in ((key_findings, "key_findings"), (limitations, "limitations")):
        for entry in text_list:
            if not _shares_claim_context(claim_tokens, entry):
                continue
            for num in _extract_numbers(entry):
                match_kind = _matches_candidate(num)
                if match_kind == "as_is":
                    return (num, label)
                if match_kind == "pct_norm":
                    return (num, f"{label}+percent_normalized")

    return (None, "not_found")


#: Minimum share of the claim's content tokens that must appear in a finding
#: for a non-numeric fact to be credited via token overlap.  Mirrors C3's
#: ``TOKEN_OVERLAP_THRESHOLD`` in :mod:`benchmark.rubrics.claim_validation`.
TEXT_FACT_OVERLAP_THRESHOLD: float = 0.5


def _find_predicted_text(
    claim: str,
    key_findings: list[str],
    limitations: list[str],
) -> tuple[str | None, str]:
    """Search for a non-numeric claim in ``key_findings`` and ``limitations``.

    Two matching stages, applied per entry (substring takes precedence):

    1. **Substring containment** (case-insensitive, whitespace-stripped):
       the claim is contained in the entry or vice-versa.
    2. **Token overlap**: at least :data:`TEXT_FACT_OVERLAP_THRESHOLD` of the
       claim's content tokens (see :func:`_content_tokens`) appear in the
       entry.  This credits paraphrases — the dominant way real models state
       non-numeric facts — without requiring a verbatim echo of the claim.

    Args:
        claim: The gold fact claim text.
        key_findings: ``key_findings`` list from the model output.
        limitations: ``limitations`` list from the model output.

    Returns:
        ``(matched_text, source_label)`` or ``(None, 'not_found')`` where
        *source_label* is ``'key_findings'``, ``'limitations'``, or their
        ``'+token_overlap'``-suffixed variants.
    """
    claim_lower = claim.strip().lower()
    claim_tokens = _content_tokens(claim)

    for text_list, label in [(key_findings, "key_findings"), (limitations, "limitations")]:
        for entry in text_list:
            entry_lower = entry.strip().lower()
            if claim_lower in entry_lower or entry_lower in claim_lower:
                return (entry, label)
            if claim_tokens:
                overlap = len(claim_tokens & _content_tokens(entry)) / len(claim_tokens)
                if overlap >= TEXT_FACT_OVERLAP_THRESHOLD:
                    return (entry, f"{label}+token_overlap")

    return (None, "not_found")


def score_fact(
    gold_fact: GoldFact,
    structured_metrics: dict[str, Any],
    key_findings: list[str],
    limitations: list[str] | None = None,
) -> FactDetail:
    """Score one gold fact against a model output.

    Dispatches to :func:`score_numeric` when ``gold_fact.numeric_value`` is
    not ``None``, and to :func:`_find_predicted_text` (substring or token
    overlap) for non-numeric (string) facts.

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
    substring (or vice-versa) or shares >= 50% of the claim's content tokens;
    if neither yields a value the fact scores ``0.0``.

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
        authored: float | str = gold_fact.tolerance if gold_fact.tolerance is not None else 0.0
        gold_val: float = gold_fact.numeric_value

        # Apply leniency floor: loosen tight authored tolerances so that
        # prose rounding ("about 82%", "around 133 students") is credited.
        # The floor is a *minimum* — it never tightens a generous tolerance.
        if isinstance(authored, str):
            # Relative string tolerance (e.g. "5%"): leave unchanged; the
            # relative form already expresses a proportional window.
            tolerance: float | str = authored
        else:
            authored_abs: float = float(authored)
            if 0 < gold_val < 1:
                # Rate/fraction: floor is 1 percentage point on the fraction scale.
                effective: float = max(authored_abs, RATE_TOLERANCE_FLOOR)
            elif gold_val >= 1:
                # Count/magnitude: floor is 2% of the gold value.
                effective = max(authored_abs, COUNT_RELATIVE_FLOOR * abs(gold_val))
            else:
                # gold_val <= 0: no floor adjustment (unusual; keep authored).
                effective = authored_abs
            tolerance = effective

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
            claim=gold_fact.claim,
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

        # The find already verified the match (substring or token overlap) —
        # score it directly.  Re-checking with exact string equality here
        # (the old behavior) made non-numeric facts near-impossible to credit
        # unless the model echoed the claim verbatim.
        return FactDetail(
            fact_id=gold_fact.fact_id,
            score=1.0,
            matched=True,
            predicted_value=matched_text,
            method=f"text_match[{source}]",
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
