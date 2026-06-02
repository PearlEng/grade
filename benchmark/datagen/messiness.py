"""Deterministic helpers to inject missingness, low-N subgroups, and MoM fluctuations.

All functions accept a seeded :class:`random.Random` instance and produce
controlled noise — **not** free randomness.  The caller is responsible for
threading the same :class:`~random.Random` instance through the entire
generation pipeline so that the output is byte-for-byte reproducible given a
fixed seed.

Design philosophy
-----------------
- **No global RNG.**  Every function takes an explicit ``rng`` argument.
- **Deterministic ordering.**  Functions that iterate over collections always
  operate on sorted sequences so that dictionary iteration order cannot
  influence output.
- **Rate parameters from** ``spec.py``.  Callers pass explicit rate floats
  (not looked up globally here) so that this module stays pure and testable.
"""

from __future__ import annotations

import random
from typing import Any


def apply_nullable_rate(
    records: list[dict[str, Any]],
    field: str,
    rate: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Set *field* to ``None`` in a random fraction of *records*.

    Operates in-place and returns the modified list for convenience.

    Args:
        records: List of row dictionaries.  Modified in-place.
        field: Key to nullify.
        rate: Probability [0, 1] of nullifying each row's value.
        rng: Seeded :class:`random.Random` instance.

    Returns:
        The same list, modified in-place.

    Raises:
        ValueError: If *rate* is outside [0, 1].
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1]; got {rate!r}")
    for record in records:
        if rng.random() < rate:
            record[field] = None
    return records


def sample_without_replacement(
    population: list[Any],
    k: int,
    rng: random.Random,
) -> list[Any]:
    """Return *k* items sampled without replacement from *population*.

    A deterministic wrapper around :meth:`random.Random.sample` that sorts the
    population first to eliminate any ambient ordering dependency.

    Args:
        population: Source collection (will be sorted if sortable).
        k: Number of items to draw.
        rng: Seeded :class:`random.Random` instance.

    Returns:
        List of *k* sampled items (original order within the sample preserved
        from the sorted population).

    Raises:
        ValueError: If *k* > len(*population*).
    """
    sorted_pop = sorted(population, key=str)
    return rng.sample(sorted_pop, k)


def weighted_choice(
    choices: dict[str, float],
    rng: random.Random,
) -> str:
    """Choose a key from *choices* according to normalized weights.

    Args:
        choices: Mapping of label → relative weight (need not sum to 1).
        rng: Seeded :class:`random.Random` instance.

    Returns:
        One key chosen according to the weights.

    Raises:
        ValueError: If *choices* is empty.
    """
    if not choices:
        raise ValueError("choices must be non-empty")
    keys = sorted(choices.keys())
    weights = [choices[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def jitter_float(
    base: float,
    max_delta: float,
    lo: float,
    hi: float,
    rng: random.Random,
) -> float:
    """Return *base* ± a uniform random delta, clamped to [*lo*, *hi*].

    Used to add per-school/per-month noise to attendance rates and dosage
    figures without drifting outside valid ranges.

    Args:
        base: Central value.
        max_delta: Maximum absolute deviation.
        lo: Minimum allowed return value.
        hi: Maximum allowed return value.
        rng: Seeded :class:`random.Random` instance.

    Returns:
        Jittered value in [*lo*, *hi*].
    """
    delta = rng.uniform(-max_delta, max_delta)
    return max(lo, min(hi, base + delta))


def mom_attendance_rate(
    month_index: int,
    start: float,
    delta: float,
    trend_months: int,
    school_jitter: float,
    rng: random.Random,
) -> float:
    """Compute the attendance rate for a given month with the MoM trend signal.

    Attendance rises by *delta* per month for *trend_months* months, then
    plateaus.  A small per-school jitter (≤ *school_jitter*) is added.

    Args:
        month_index: 0-based month index.
        start: Baseline attendance rate (month 0).
        delta: Per-month increase during the trend period.
        trend_months: Number of months with rising trend.
        school_jitter: Maximum absolute per-school random deviation.
        rng: Seeded :class:`random.Random` instance.

    Returns:
        Attendance rate in [0, 1].
    """
    trend_step = min(month_index, trend_months - 1)
    base = start + delta * trend_step
    return jitter_float(base, school_jitter, 0.0, 1.0, rng)


def satisfaction_score(
    month_index: int,
    baseline: float,
    dip_month: int,
    dip_magnitude: float,
    recovery: float,
    rng: random.Random,
) -> float:
    """Compute the mean satisfaction score for a given month.

    Implements a dip-and-recovery pattern:

    - Before *dip_month*: ``baseline``
    - At *dip_month*: ``baseline - dip_magnitude``
    - After *dip_month*: ``baseline`` (full recovery)

    A small uniform jitter (≤ 0.1) is added.

    Args:
        month_index: 0-based month index.
        baseline: Normal mean satisfaction score (Likert 1–5).
        dip_month: Index of the dip month.
        dip_magnitude: How much the score drops in the dip month.
        recovery: How much the score recovers after the dip.
        rng: Seeded :class:`random.Random` instance.

    Returns:
        Satisfaction score in [1.0, 5.0].
    """
    if month_index < dip_month:
        base = baseline
    elif month_index == dip_month:
        base = baseline - dip_magnitude
    else:
        base = baseline - dip_magnitude + recovery
    return jitter_float(base, 0.1, 1.0, 5.0, rng)


def assign_race_ethnicity(
    n_students: int,
    dist: dict[str, float],
    aian_target_n: int,
    rng: random.Random,
) -> list[str]:
    """Generate a list of race/ethnicity codes for *n_students* students.

    AIAN students are capped at exactly *aian_target_n* to guarantee the low-N
    signal; the remaining students are distributed according to *dist* (AIAN
    weight is ignored).

    Args:
        n_students: Total number of students.
        dist: Desired distribution mapping code → relative weight.
        aian_target_n: Exact number of AIAN students to produce.
        rng: Seeded :class:`random.Random` instance.

    Returns:
        List of length *n_students* with shuffled race/ethnicity codes.
    """
    aian_n = min(aian_target_n, n_students)
    non_aian_n = n_students - aian_n

    non_aian_dist = {k: v for k, v in dist.items() if k != "AIAN" and v > 0}
    codes: list[str] = []
    for _ in range(non_aian_n):
        codes.append(weighted_choice(non_aian_dist, rng))
    codes += ["AIAN"] * aian_n

    rng.shuffle(codes)
    return codes


def suppress_if_low_n(
    record: dict[str, Any],
    n_field: str,
    metric_fields: list[str],
    threshold: int,
) -> dict[str, Any]:
    """Apply low-N suppression to *record* in-place and return it.

    If the value of *n_field* < *threshold*, sets all *metric_fields* to
    ``None`` and sets ``suppressed = True`` / ``suppression_reason``.

    Args:
        record: Row dictionary to check and possibly suppress.
        n_field: Name of the field holding the N count.
        metric_fields: Fields to null out when suppressed.
        threshold: Suppression threshold (e.g. 10).

    Returns:
        The modified record (same object, modified in-place).
    """
    n = record.get(n_field, 0) or 0
    if n < threshold:
        for f in metric_fields:
            record[f] = None
        record["suppressed"] = True
        record["suppression_reason"] = f"n < {threshold}"
    else:
        record["suppressed"] = False
        record["suppression_reason"] = None
    return record
