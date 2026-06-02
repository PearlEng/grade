"""C4 — Consistency scoring across repeated runs for the GRADE benchmark.

This module provides pure-function scoring primitives for the C4 (Consistency)
rubric dimension.  Given the N (default: 5) repeated outputs for one
(model × task) pair, it measures agreement and variance across runs and
produces a single consistency score in [0, 1] feeding the ``consistency``
dimension of ``result_schema.json``.

Metric definition
-----------------
The overall ``consistency_score`` is a weighted average of three sub-metrics,
each in [0, 1]:

1. **Finding stability** (weight 0.5) — fraction of the top-K findings (K = 3
   by default) that appear verbatim or near-verbatim in *all* runs, aggregated
   as the mean per-finding appearance rate across the union of unique findings
   seen across runs.  A finding is considered a "match" between two runs using
   a token-level Jaccard similarity with threshold 0.5.

2. **Ranking stability** (weight 0.3) — mean pairwise Kendall tau-b (mapped to
   [0, 1]) across all ``C(N, 2)`` run pairs, computed over the top-K findings
   shared between each pair.  Penalises reordering of the same findings across
   runs.  Defaults to 1.0 when fewer than 2 findings are present.

3. **Metric variance** (weight 0.2) — inverse coefficient of variation (CV)
   for each numeric key in ``structured_metrics``.  CV = stdev / mean; the
   per-metric score is ``1 / (1 + CV)``.  The sub-score is the mean over all
   numeric metrics.  Non-numeric metrics and metrics with a zero mean are
   excluded from the average (they do not penalise or reward).  Defaults to 1.0
   when no numeric metrics are present.

The three component weights sum to 1.0.  To obtain the final score, call
:func:`score_consistency`.

Public API
----------
- :func:`finding_stability` — fraction of findings stable across all runs.
- :func:`ranking_stability` — mean pairwise Kendall tau-b over top-K findings.
- :func:`metric_variance_score` — inverse-CV score for numeric structured_metrics.
- :func:`score_consistency` — top-level aggregator; the primary entry point.

Usage::

    from benchmark.rubrics.consistency_scoring import score_consistency

    result = score_consistency(task, runs)
    print(result["consistency_score"])        # e.g. 0.87
    print(result["per_finding_stability"])    # {"finding_text": 0.8, ...}
    print(result["metric_variance"])          # {"total_students": 0.97, ...}
"""

from __future__ import annotations

import math
import statistics
from typing import Any

__all__ = [
    "finding_stability",
    "ranking_stability",
    "metric_variance_score",
    "score_consistency",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default number of top findings to consider for stability and ranking metrics.
TOP_K_FINDINGS: int = 3

#: Jaccard similarity threshold for two finding strings to be considered a match.
FINDING_MATCH_THRESHOLD: float = 0.5

#: Component weights for the aggregate consistency score.  Must sum to 1.0.
_WEIGHT_FINDING_STABILITY: float = 0.5
_WEIGHT_RANKING_STABILITY: float = 0.3
_WEIGHT_METRIC_VARIANCE: float = 0.2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Lowercase-split *text* into a token set for Jaccard similarity.

    Args:
        text: A finding string.

    Returns:
        A set of lowercase whitespace-separated tokens.
    """
    return set(text.lower().split())


def _jaccard(a: str, b: str) -> float:
    """Compute token-level Jaccard similarity between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Jaccard similarity in [0.0, 1.0].  Returns 0.0 when both strings are
        empty.
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def _best_match_score(candidate: str, findings: list[str]) -> float:
    """Return the maximum Jaccard similarity between *candidate* and *findings*.

    Args:
        candidate: A finding string to match against.
        findings: List of finding strings to compare against.

    Returns:
        Maximum Jaccard similarity in [0.0, 1.0].  Returns 0.0 for an empty
        *findings* list.
    """
    if not findings:
        return 0.0
    return max(_jaccard(candidate, f) for f in findings)


def _finding_present(finding: str, run_findings: list[str]) -> bool:
    """Decide whether *finding* is present in *run_findings* by Jaccard threshold.

    Args:
        finding: The finding text to look for.
        run_findings: The list of findings in a particular run.

    Returns:
        ``True`` if any finding in *run_findings* has Jaccard similarity >=
        :data:`FINDING_MATCH_THRESHOLD` with *finding*.
    """
    return _best_match_score(finding, run_findings) >= FINDING_MATCH_THRESHOLD


def _kendall_tau_b_normalized(list_a: list[int], list_b: list[int]) -> float:
    """Compute Kendall tau-b between two rank lists and map the result to [0, 1].

    Only concordant/discordant pairs among shared positions are counted.
    The raw tau-b is in [-1, 1]; this function returns ``(tau_b + 1) / 2``.

    Args:
        list_a: Rank sequence for items in run A.
        list_b: Rank sequence for items in run B (paired with *list_a*).

    Returns:
        Normalized Kendall tau-b in [0.0, 1.0].  Returns 1.0 when ``n < 2``
        (no pairs to compare — convention: perfect agreement).
    """
    n = len(list_a)
    if n < 2:
        return 1.0

    concordant = 0
    discordant = 0
    tied_a = 0
    tied_b = 0

    for i in range(n):
        for j in range(i + 1, n):
            diff_a = list_a[i] - list_a[j]
            diff_b = list_b[i] - list_b[j]
            product = diff_a * diff_b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if diff_a == 0:
                    tied_a += 1
                if diff_b == 0:
                    tied_b += 1

    n_pairs = n * (n - 1) / 2
    denominator = math.sqrt((n_pairs - tied_a) * (n_pairs - tied_b))
    if denominator == 0.0:
        return 1.0

    tau_b = (concordant - discordant) / denominator
    return (tau_b + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Sub-metric scorers
# ---------------------------------------------------------------------------


def finding_stability(
    runs: list[dict[str, Any]],
    top_k: int = TOP_K_FINDINGS,
) -> tuple[float, dict[str, float]]:
    """Measure the stability of top-K findings across runs.

    For each unique finding (across all runs, using Jaccard-based deduplication
    within the top-K slice of each run), computes the fraction of runs in which
    that finding appears (using :data:`FINDING_MATCH_THRESHOLD`).  The aggregate
    score is the mean of those per-finding appearance rates.

    Args:
        runs: List of normalized model output dicts (each conforming to
            ``output_schema.json``).  Must contain at least one entry.
        top_k: Number of top findings to examine per run.  Defaults to
            :data:`TOP_K_FINDINGS`.

    Returns:
        A ``(score, per_finding_stability)`` tuple where:

        - *score* is the mean appearance rate in ``[0.0, 1.0]``.
        - *per_finding_stability* is a ``dict[str, float]`` mapping each unique
          finding's canonical text to its appearance rate.

    Raises:
        ValueError: If *runs* is empty.

    Example::

        score, details = finding_stability(runs)
        # score == 1.0 when every finding appears in every run
    """
    if not runs:
        raise ValueError("runs must contain at least one output dict")

    n_runs = len(runs)
    # Collect top-K findings per run
    run_findings: list[list[str]] = [r.get("key_findings", [])[:top_k] for r in runs]

    # Build a deduplicated set of canonical findings (representative text)
    canonical: list[str] = []
    for findings in run_findings:
        for f in findings:
            # Add f to canonical if it does not already have a near-duplicate
            if not any(_jaccard(f, c) >= FINDING_MATCH_THRESHOLD for c in canonical):
                canonical.append(f)

    if not canonical:
        # No findings in any run — treat as perfectly consistent (nothing to disagree on)
        return 1.0, {}

    per_finding: dict[str, float] = {}
    for c in canonical:
        appearances = sum(1 for rf in run_findings if _finding_present(c, rf))
        per_finding[c] = appearances / n_runs

    score = statistics.mean(per_finding.values())
    return score, per_finding


def ranking_stability(
    runs: list[dict[str, Any]],
    top_k: int = TOP_K_FINDINGS,
) -> float:
    """Measure stability of finding order across all pairs of runs.

    For each pair of runs, determines the set of findings common to both (by
    Jaccard matching) and computes normalized Kendall tau-b on their relative
    positions.  The final score is the mean pairwise tau-b.

    Args:
        runs: List of normalized model output dicts.  Must contain at least one
            entry; a single run returns 1.0 (no pairs to compare).
        top_k: Number of top findings to examine per run.

    Returns:
        Mean pairwise normalized Kendall tau-b in ``[0.0, 1.0]``.  Returns
        ``1.0`` when there are fewer than two runs, or when no run pair has at
        least two common findings.

    Raises:
        ValueError: If *runs* is empty.

    Example::

        score = ranking_stability(runs)
        # score == 1.0 when finding order is identical across all runs
    """
    if not runs:
        raise ValueError("runs must contain at least one output dict")

    n_runs = len(runs)
    if n_runs < 2:
        return 1.0

    run_findings: list[list[str]] = [r.get("key_findings", [])[:top_k] for r in runs]

    pair_scores: list[float] = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            findings_i = run_findings[i]
            findings_j = run_findings[j]
            if not findings_i or not findings_j:
                # One run has no findings — treat as perfect to avoid penalising
                pair_scores.append(1.0)
                continue

            # Find common findings: for each finding in i, find the best match in j
            matched_pairs: list[tuple[int, int]] = []
            used_j: set[int] = set()
            for pos_i, f_i in enumerate(findings_i):
                best_pos_j: int | None = None
                best_sim = FINDING_MATCH_THRESHOLD - 1e-9
                for pos_j, f_j in enumerate(findings_j):
                    if pos_j in used_j:
                        continue
                    sim = _jaccard(f_i, f_j)
                    if sim >= FINDING_MATCH_THRESHOLD and sim > best_sim:
                        best_sim = sim
                        best_pos_j = pos_j
                if best_pos_j is not None:
                    matched_pairs.append((pos_i, best_pos_j))
                    used_j.add(best_pos_j)

            if len(matched_pairs) < 2:
                # Not enough common findings for tau — treat as perfectly consistent
                pair_scores.append(1.0)
                continue

            ranks_i = [p[0] for p in matched_pairs]
            ranks_j = [p[1] for p in matched_pairs]
            pair_scores.append(_kendall_tau_b_normalized(ranks_i, ranks_j))

    if not pair_scores:
        return 1.0
    return statistics.mean(pair_scores)


def metric_variance_score(runs: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    """Score the variance of numeric ``structured_metrics`` across runs.

    For each numeric metric key that appears in at least two runs, computes the
    coefficient of variation (CV = stdev / |mean|).  The per-metric score is
    ``1 / (1 + CV)``, which is 1.0 for zero variance and approaches 0.0 for
    high variance.  The aggregate score is the mean over all qualifying metrics.

    Metrics with a mean of zero are excluded from the aggregate to avoid
    division-by-zero artefacts (their CV is undefined and they typically
    represent binary flags rather than continuous quantities).

    Args:
        runs: List of normalized model output dicts.  Each must have a
            ``structured_metrics`` key.

    Returns:
        A ``(score, per_metric_scores)`` tuple where:

        - *score* is the mean per-metric inverse-CV score in ``[0.0, 1.0]``.
          Returns ``1.0`` when no qualifying numeric metrics are present.
        - *per_metric_scores* is a ``dict[str, float]`` mapping metric name to
          its per-metric inverse-CV score.

    Raises:
        ValueError: If *runs* is empty.

    Example::

        score, details = metric_variance_score(runs)
        # score == 1.0 when all numeric metrics are identical across runs
    """
    if not runs:
        raise ValueError("runs must contain at least one output dict")

    # Collect numeric values per metric key across all runs
    metric_values: dict[str, list[float]] = {}
    for run in runs:
        for key, val in run.get("structured_metrics", {}).items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                metric_values.setdefault(key, []).append(float(val))

    per_metric: dict[str, float] = {}
    for key, values in metric_values.items():
        if len(values) < 2:
            # Only one observation — no variance info, skip
            continue
        mean_val = statistics.mean(values)
        if mean_val == 0.0:
            # CV undefined for zero mean; exclude silently
            continue
        stdev_val = statistics.stdev(values)
        cv = stdev_val / abs(mean_val)
        per_metric[key] = 1.0 / (1.0 + cv)

    if not per_metric:
        return 1.0, {}

    score = statistics.mean(per_metric.values())
    return score, per_metric


# ---------------------------------------------------------------------------
# Top-level aggregator
# ---------------------------------------------------------------------------


def score_consistency(
    task: dict[str, Any],
    runs: list[dict[str, Any]],
    top_k: int = TOP_K_FINDINGS,
) -> dict[str, Any]:
    """Score the consistency of repeated model outputs for one (model × task).

    Aggregates three sub-metrics — finding stability, ranking stability, and
    metric variance — into a single ``consistency_score`` in [0, 1].

    The component weights are:

    - Finding stability: 0.5
    - Ranking stability: 0.3
    - Metric variance:   0.2

    Args:
        task: The full task definition dict (validated against
            ``task_schema.json``).  Only ``task["task_id"]`` is read; the
            remaining fields are accepted for forward-compatibility.
        runs: List of N (typically 5) normalized model output dicts for the
            same (model × task) pair.  Each dict must conform to
            ``output_schema.json``.  Must contain at least one entry.
        top_k: Number of top findings to examine per run when computing
            finding stability and ranking stability.  Defaults to
            :data:`TOP_K_FINDINGS` (3).

    Returns:
        A dict with the following keys:

        - ``"task_id"`` (str): Copied from *task*.
        - ``"run_count"`` (int): Number of runs provided.
        - ``"consistency_score"`` (float): Weighted aggregate in [0.0, 1.0].
        - ``"per_finding_stability"`` (dict[str, float]): Per-finding appearance
          rate keyed by canonical finding text.
        - ``"ranking_stability"`` (float): Mean pairwise Kendall tau-b score.
        - ``"metric_variance"`` (dict[str, float]): Per-metric inverse-CV score.
        - ``"component_scores"`` (dict[str, float]): The three component scores
          before weighting (keys: ``finding_stability``, ``ranking_stability``,
          ``metric_variance``).

    Raises:
        ValueError: If *runs* is empty.
        KeyError: If *task* does not have a ``"task_id"`` key.

    Example::

        from benchmark.rubrics.consistency_scoring import score_consistency

        result = score_consistency(task, runs)
        print(result["consistency_score"])   # e.g. 0.87
        print(result["per_finding_stability"])
        print(result["metric_variance"])
    """
    if not runs:
        raise ValueError("runs must contain at least one output dict")

    task_id: str = task["task_id"]

    fs_score, per_finding = finding_stability(runs, top_k=top_k)
    rs_score = ranking_stability(runs, top_k=top_k)
    mv_score, per_metric = metric_variance_score(runs)

    consistency_score = (
        _WEIGHT_FINDING_STABILITY * fs_score
        + _WEIGHT_RANKING_STABILITY * rs_score
        + _WEIGHT_METRIC_VARIANCE * mv_score
    )

    return {
        "task_id": task_id,
        "run_count": len(runs),
        "consistency_score": consistency_score,
        "per_finding_stability": per_finding,
        "ranking_stability": rs_score,
        "metric_variance": per_metric,
        "component_scores": {
            "finding_stability": fs_score,
            "ranking_stability": rs_score,
            "metric_variance": mv_score,
        },
    }
