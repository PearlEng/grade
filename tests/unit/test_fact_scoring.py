"""Unit tests for benchmark.rubrics.fact_scoring (C1 — Exact/tolerant fact scoring).

Covers positive and negative cases for each scoring primitive and the
aggregate scorer, per the Definition of Done in ticket #575.

Test matrix
-----------
- :func:`score_numeric`: exact match, within absolute tolerance, outside absolute tolerance,
  within relative tolerance, outside relative tolerance, edge cases.
- :func:`score_exact_match`: string exact, string case-insensitive, string mismatch,
  non-string types.
- :func:`score_date_range`: perfect overlap, partial overlap, no overlap, single-day ranges.
- :func:`score_ranking_similarity`: perfect order, reversed order, partial match, ties.
- :func:`score_fact`: numeric dispatch, non-numeric dispatch, missing value.
- :func:`score_facts`: all-correct, all-wrong, mixed, empty facts list, raw dict input.
- Value-based matching: arbitrary key names, key_findings text extraction, false positives.
"""

from __future__ import annotations

import pytest

from benchmark.rubrics.fact_scoring import (
    FactScoreResult,
    GoldFact,
    _extract_numbers,
    score_date_range,
    score_exact_match,
    score_fact,
    score_facts,
    score_numeric,
    score_ranking_similarity,
)

# ===========================================================================
# score_numeric
# ===========================================================================


class TestScoreNumeric:
    """Tests for score_numeric()."""

    # --- Exact match (tolerance = 0) ---

    def test_exact_match_zero_tolerance(self) -> None:
        """Identical values with zero tolerance should return 1.0."""
        assert score_numeric(80.9, 80.9, 0.0) == 1.0

    def test_exact_match_integer_tolerance(self) -> None:
        """Integer tolerance = 0 should work the same as float 0.0."""
        assert score_numeric(135, 135, 0) == 1.0

    def test_exact_match_fails_with_zero_tolerance(self) -> None:
        """Any deviation with zero tolerance must return 0.0."""
        assert score_numeric(80.95, 80.9, 0.0) == 0.0

    # --- Absolute tolerance ---

    def test_within_absolute_tolerance(self) -> None:
        """Value clearly within absolute tolerance bound should return 1.0."""
        # 80.87 is 0.03 away from 80.9, well within 0.05
        assert score_numeric(80.87, 80.9, 0.05) == 1.0

    def test_on_boundary_of_absolute_tolerance(self) -> None:
        """Value exactly at the tolerance boundary should return 1.0 (use exact repr).

        Using integer arithmetic to avoid floating-point boundary ambiguity:
        abs(10 - 15) == 5 <= 5 is unambiguously True.
        """
        assert score_numeric(10.0, 15.0, 5.0) == 1.0

    def test_just_outside_absolute_tolerance(self) -> None:
        """Value just beyond the absolute tolerance bound should return 0.0."""
        # 80.83 is 0.07 away from 80.9, outside tolerance of 0.05
        assert score_numeric(80.83, 80.9, 0.05) == 0.0

    def test_far_outside_absolute_tolerance(self) -> None:
        """Value far from gold should return 0.0."""
        assert score_numeric(79.0, 80.9, 0.05) == 0.0

    def test_negative_values_within_tolerance(self) -> None:
        """Negative values within tolerance should return 1.0."""
        assert score_numeric(-10.02, -10.0, 0.05) == 1.0

    def test_large_absolute_tolerance_always_matches(self) -> None:
        """Large tolerance should match wildly different values."""
        assert score_numeric(0.0, 100.0, 1000.0) == 1.0

    # --- Relative tolerance ---

    def test_within_relative_tolerance_5pct(self) -> None:
        """Value within 5% of gold should return 1.0."""
        assert score_numeric(105.0, 100.0, "5%") == 1.0

    def test_on_boundary_relative_tolerance(self) -> None:
        """Value exactly at the relative tolerance boundary should return 1.0."""
        assert score_numeric(105.0, 100.0, "5%") == 1.0

    def test_just_outside_relative_tolerance(self) -> None:
        """Value just beyond relative tolerance should return 0.0."""
        assert score_numeric(106.0, 100.0, "5%") == 0.0

    def test_relative_tolerance_with_decimal_pct(self) -> None:
        """Decimal percentage tolerance string should be parsed correctly."""
        assert score_numeric(0.8095, 0.8092, "0.1%") == 1.0

    def test_relative_tolerance_zero_gold(self) -> None:
        """When gold is zero the base defaults to 1.0 to avoid division by zero."""
        assert score_numeric(0.0, 0.0, "5%") == 1.0
        assert score_numeric(0.04, 0.0, "5%") == 1.0  # 5% of 1.0 = 0.05

    # --- Error cases ---

    def test_negative_absolute_tolerance_raises(self) -> None:
        """Negative absolute tolerance must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            score_numeric(1.0, 1.0, -0.1)

    def test_malformed_pct_string_raises(self) -> None:
        """Malformed percentage string must raise ValueError."""
        with pytest.raises(ValueError, match="percentage"):
            score_numeric(1.0, 1.0, "5 percent")

    def test_invalid_tolerance_type_raises(self) -> None:
        """Non-numeric non-str tolerance must raise TypeError."""
        with pytest.raises(TypeError):
            score_numeric(1.0, 1.0, [0.05])  # type: ignore[arg-type]


# ===========================================================================
# score_exact_match
# ===========================================================================


class TestScoreExactMatch:
    """Tests for score_exact_match()."""

    def test_identical_strings(self) -> None:
        """Identical strings should return 1.0."""
        assert score_exact_match("SCH-001", "SCH-001") == 1.0

    def test_case_insensitive_string_match(self) -> None:
        """Strings differing only in case should return 1.0."""
        assert score_exact_match("sch-001", "SCH-001") == 1.0
        assert score_exact_match("SCH-001", "sch-001") == 1.0

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace should be stripped before comparison."""
        assert score_exact_match("  SCH-001  ", "SCH-001") == 1.0

    def test_string_mismatch(self) -> None:
        """Different string values must return 0.0."""
        assert score_exact_match("SCH-001", "SCH-002") == 0.0

    def test_integer_exact_match(self) -> None:
        """Identical integers should return 1.0."""
        assert score_exact_match(42, 42) == 1.0

    def test_integer_mismatch(self) -> None:
        """Different integers must return 0.0."""
        assert score_exact_match(42, 43) == 0.0

    def test_float_exact_match(self) -> None:
        """Identical floats should return 1.0."""
        assert score_exact_match(3.14, 3.14) == 1.0

    def test_float_mismatch(self) -> None:
        """Different floats must return 0.0."""
        assert score_exact_match(3.14, 3.15) == 0.0

    def test_bool_match(self) -> None:
        """Boolean True == True should return 1.0."""
        assert score_exact_match(True, True) == 1.0

    def test_bool_mismatch(self) -> None:
        """Boolean True vs False must return 0.0."""
        assert score_exact_match(True, False) == 0.0

    def test_none_match(self) -> None:
        """None == None should return 1.0."""
        assert score_exact_match(None, None) == 1.0

    def test_none_vs_string_mismatch(self) -> None:
        """None vs a string must return 0.0."""
        assert score_exact_match(None, "value") == 0.0


# ===========================================================================
# score_date_range
# ===========================================================================


class TestScoreDateRange:
    """Tests for score_date_range()."""

    def test_perfect_overlap(self) -> None:
        """Identical date ranges should return 1.0."""
        assert score_date_range(
            ("2025-09-01", "2025-11-30"),
            ("2025-09-01", "2025-11-30"),
        ) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Non-overlapping ranges should return 0.0."""
        result = score_date_range(
            ("2026-01-01", "2026-01-31"),
            ("2025-09-01", "2025-11-30"),
        )
        assert result == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Partial overlap should return a Jaccard score between 0 and 1."""
        # predicted: Oct only (31 days), gold: Sep-Nov (91 days)
        # intersection = 31 days, union = 91 days
        # Jaccard = 31/91
        result = score_date_range(
            ("2025-10-01", "2025-10-31"),
            ("2025-09-01", "2025-11-30"),
        )
        expected = 31 / 91
        assert result == pytest.approx(expected, abs=1e-6)

    def test_single_day_exact_match(self) -> None:
        """Single-day ranges that match should return 1.0."""
        assert score_date_range(
            ("2025-10-15", "2025-10-15"),
            ("2025-10-15", "2025-10-15"),
        ) == pytest.approx(1.0)

    def test_single_day_no_overlap(self) -> None:
        """Non-overlapping single-day ranges should return 0.0."""
        assert score_date_range(
            ("2025-10-15", "2025-10-15"),
            ("2025-10-16", "2025-10-16"),
        ) == pytest.approx(0.0)

    def test_predicted_contains_gold(self) -> None:
        """Predicted range that contains gold should give Jaccard < 1."""
        # predicted: Sep-Nov (91 days), gold: Oct only (31 days)
        result = score_date_range(
            ("2025-09-01", "2025-11-30"),
            ("2025-10-01", "2025-10-31"),
        )
        expected = 31 / 91
        assert result == pytest.approx(expected, abs=1e-6)

    def test_invalid_date_string_raises(self) -> None:
        """Non-ISO date string should raise ValueError."""
        with pytest.raises(ValueError):
            score_date_range(("01/09/2025", "30/11/2025"), ("2025-09-01", "2025-11-30"))

    def test_start_after_end_raises(self) -> None:
        """Range where start > end should raise ValueError."""
        with pytest.raises(ValueError, match="after end"):
            score_date_range(("2025-12-01", "2025-09-01"), ("2025-09-01", "2025-11-30"))


# ===========================================================================
# score_ranking_similarity
# ===========================================================================


class TestScoreRankingSimilarity:
    """Tests for score_ranking_similarity()."""

    def test_perfect_match(self) -> None:
        """Identical ranking should return 1.0."""
        assert score_ranking_similarity(["A", "B", "C"], ["A", "B", "C"]) == pytest.approx(1.0)

    def test_reversed_order(self) -> None:
        """Completely reversed ranking should return 0.0."""
        assert score_ranking_similarity(["C", "B", "A"], ["A", "B", "C"]) == pytest.approx(0.0)

    def test_one_swap(self) -> None:
        """Ranking with one adjacent swap should return close to 0.67."""
        # tau_b = (concordant - discordant) / sqrt(...) for ["A","C","B"] vs ["A","B","C"]
        # Pairs: (A,C)=concordant, (A,B)=concordant, (C,B)=discordant
        # tau_b = (2-1)/3 = 1/3 → normalized = (1/3+1)/2 = 2/3
        result = score_ranking_similarity(["A", "C", "B"], ["A", "B", "C"])
        assert result == pytest.approx(2 / 3, abs=1e-6)

    def test_extra_items_in_predicted_ignored(self) -> None:
        """Extra items in predicted not in gold should be ignored."""
        # Only A, B, C appear in both; D is ignored
        assert score_ranking_similarity(["A", "B", "C", "D"], ["A", "B", "C"]) == pytest.approx(1.0)

    def test_extra_items_in_gold_ignored(self) -> None:
        """Extra items in gold not in predicted should be ignored."""
        assert score_ranking_similarity(["A", "B", "C"], ["A", "B", "C", "D"]) == pytest.approx(1.0)

    def test_fewer_than_two_common_items_raises(self) -> None:
        """Fewer than two common items should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            score_ranking_similarity(["A"], ["A", "B", "C"])

    def test_no_common_items_raises(self) -> None:
        """No common items should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            score_ranking_similarity(["X", "Y"], ["A", "B", "C"])

    def test_four_item_partial_order(self) -> None:
        """Four-item ranking with two swaps should be between 0 and 1."""
        result = score_ranking_similarity(["B", "A", "D", "C"], ["A", "B", "C", "D"])
        assert 0.0 < result < 1.0

    def test_numeric_rankings(self) -> None:
        """Numeric items should work the same as strings."""
        assert score_ranking_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
        assert score_ranking_similarity([3, 2, 1], [1, 2, 3]) == pytest.approx(0.0)


# ===========================================================================
# _extract_numbers  (helper)
# ===========================================================================


class TestExtractNumbers:
    """Tests for _extract_numbers() number extraction helper."""

    def test_plain_integer(self) -> None:
        """Plain integer string should be extracted."""
        assert _extract_numbers("135 students") == [135.0]

    def test_decimal_number(self) -> None:
        """Decimal number should be extracted."""
        result = _extract_numbers("rate is 82.4%")
        assert 82.4 in result

    def test_comma_thousands(self) -> None:
        """Comma-formatted thousands should be parsed as a single number."""
        result = _extract_numbers("1,234 records")
        assert 1234.0 in result

    def test_currency_prefix(self) -> None:
        """Dollar sign prefix should be stripped."""
        result = _extract_numbers("cost $1,250")
        assert 1250.0 in result

    def test_multiple_numbers(self) -> None:
        """Multiple numbers in one string should all be extracted."""
        result = _extract_numbers("60 students in SCH-001 and 45 in SCH-002")
        assert 60.0 in result
        assert 45.0 in result

    def test_empty_string(self) -> None:
        """Empty string should return empty list."""
        assert _extract_numbers("") == []

    def test_no_numbers(self) -> None:
        """Text without numbers should return empty list."""
        assert _extract_numbers("no numbers here") == []


# ===========================================================================
# score_fact  (dispatcher)
# ===========================================================================


class TestScoreFact:
    """Tests for score_fact()."""

    def _make_fact(
        self,
        fact_id: str = "F1",
        claim: str = "135 students",
        numeric_value: float | None = 135.0,
        tolerance: float | None = 0.0,
    ) -> GoldFact:
        return GoldFact(
            fact_id=fact_id,
            claim=claim,
            source_files=["students.csv"],
            numeric_value=numeric_value,
            tolerance=tolerance,
        )

    # --- Numeric dispatch ---

    def test_numeric_exact_match(self) -> None:
        """Numeric fact with exact match should score 1.0."""
        fact = self._make_fact(numeric_value=135.0, tolerance=0.0)
        detail = score_fact(fact, {"total_students": 135}, [])
        assert detail.score == 1.0
        assert detail.matched is True

    def test_numeric_within_tolerance(self) -> None:
        """Numeric fact within tolerance should score 1.0."""
        fact = self._make_fact(numeric_value=0.8092, tolerance=0.0005)
        detail = score_fact(fact, {"attendance_rate": 0.8090}, [])
        assert detail.score == 1.0
        assert detail.matched is True

    def test_numeric_outside_tolerance(self) -> None:
        """Numeric fact outside tolerance should score 0.0."""
        fact = self._make_fact(numeric_value=0.8092, tolerance=0.0005)
        detail = score_fact(fact, {"attendance_rate": 0.80}, [])
        assert detail.score == 0.0
        assert detail.matched is False

    def test_numeric_fact_with_none_tolerance_uses_exact(self) -> None:
        """None tolerance on numeric fact should default to exact match (tolerance=0)."""
        fact = self._make_fact(numeric_value=135.0, tolerance=None)
        detail = score_fact(fact, {"total_students": 135}, [])
        assert detail.score == 1.0

    def test_numeric_from_key_findings_fallback(self) -> None:
        """When structured_metrics is empty, numeric value should be sought in key_findings."""
        fact = self._make_fact(numeric_value=135.0, tolerance=0.0)
        detail = score_fact(fact, {}, ["The program enrolls 135 students in total."])
        assert detail.score == 1.0

    # --- Value-based matching: arbitrary key names ---

    def test_numeric_arbitrary_key_in_structured_metrics(self) -> None:
        """Gold numeric fact should be credited when value appears under ANY key.

        This is the core benchmark-validity fix: real models use arbitrary key names
        like 'currently_enrolled_students' rather than fact IDs.
        """
        fact = self._make_fact(numeric_value=135.0, tolerance=0.0)
        detail = score_fact(fact, {"currently_enrolled_students": 135}, [])
        assert detail.score == 1.0
        assert detail.matched is True

    def test_numeric_value_in_key_findings_text(self) -> None:
        """Gold numeric fact should be credited when value appears only in key_findings text."""
        fact = self._make_fact(numeric_value=135.0, tolerance=0.0)
        detail = score_fact(fact, {}, ["135 students are enrolled in the program."])
        assert detail.score == 1.0
        assert detail.matched is True

    def test_numeric_value_absent_scores_zero(self) -> None:
        """Gold numeric fact must NOT be credited when value is absent from all sources."""
        fact = self._make_fact(numeric_value=135.0, tolerance=0.0)
        detail = score_fact(fact, {"total_students": 200}, ["Program has 200 students."])
        assert detail.score == 0.0
        assert detail.matched is False

    def test_numeric_near_tolerance_boundary_credited(self) -> None:
        """Value just inside absolute tolerance should be credited (score 1.0)."""
        # gold=135, tolerance=2 → window [133, 137]; predicted=134 is inside
        fact = self._make_fact(numeric_value=135.0, tolerance=2.0)
        detail = score_fact(fact, {"num_students": 134}, [])
        assert detail.score == 1.0
        assert detail.matched is True

    def test_numeric_just_outside_tolerance_not_credited(self) -> None:
        """Value just outside absolute tolerance should NOT be credited."""
        # gold=135, tolerance=2 → window [133, 137]; predicted=132 is outside
        fact = self._make_fact(numeric_value=135.0, tolerance=2.0)
        detail = score_fact(fact, {"num_students": 132}, [])
        assert detail.score == 0.0
        assert detail.matched is False

    def test_numeric_relative_tolerance_credited(self) -> None:
        """Value within relative-% tolerance should be credited via value-based matching."""
        # gold=100, 5% tolerance → window [95, 105]; predicted=103 is inside
        # score_facts supports string tolerances on GoldFact instances
        from benchmark.rubrics.fact_scoring import score_numeric as _sn

        assert _sn(103.0, 100.0, "5%") == 1.0
        assert _sn(106.0, 100.0, "5%") == 0.0

    def test_numeric_value_in_limitations_text(self) -> None:
        """Gold numeric fact should be credited when value appears only in limitations text."""
        fact = self._make_fact(numeric_value=135.0, tolerance=0.0)
        detail = score_fact(fact, {}, [], limitations=["Note: 135 records were processed."])
        assert detail.score == 1.0
        assert detail.matched is True

    # --- Non-numeric dispatch ---

    def test_non_numeric_exact_match_in_key_findings(self) -> None:
        """Non-numeric fact found verbatim in key_findings should score 1.0."""
        claim = "SCH-001 has 6 tutoring groups"
        fact = GoldFact("F6", claim, ["groups.csv"], None, None)
        # key_findings entry that is a substring of (or contains) the claim
        detail = score_fact(fact, {}, [claim])
        assert detail.score == 1.0

    def test_non_numeric_fact_not_found_scores_zero(self) -> None:
        """Non-numeric fact with no matching value should score 0.0."""
        fact = GoldFact("F6", "Some unique claim", ["groups.csv"], None, None)
        detail = score_fact(fact, {}, ["Completely different finding."])
        assert detail.score == 0.0
        assert detail.method == "not_found"

    # --- Missing value ---

    def test_missing_value_scores_zero(self) -> None:
        """When predicted value cannot be found anywhere, score must be 0.0."""
        fact = self._make_fact(numeric_value=135.0, tolerance=0.0)
        detail = score_fact(fact, {}, [])
        assert detail.score == 0.0
        assert detail.matched is False
        assert detail.predicted_value is None


# ===========================================================================
# score_facts  (aggregate)
# ===========================================================================


class TestScoreFacts:
    """Tests for score_facts()."""

    def _ops001_facts(self) -> list[GoldFact]:
        """Return a subset of gold facts mirroring T1-OPS-001."""
        return [
            GoldFact("F1", "135 students", ["students.csv"], 135.0, 0.0),
            GoldFact("F5", "14 tutoring groups", ["groups.csv"], 14.0, 0.0),
        ]

    # --- Perfect score ---

    def test_all_correct_returns_1(self) -> None:
        """All correctly predicted facts should return grounding_accuracy = 1.0."""
        result = score_facts(
            self._ops001_facts(),
            {"total_students": 135, "total_groups": 14},
            [],
        )
        assert result.grounding_accuracy == pytest.approx(1.0)
        assert result.facts_matched == 2
        assert result.facts_total == 2

    # --- All wrong ---

    def test_all_wrong_returns_0(self) -> None:
        """All wrong predictions should return grounding_accuracy = 0.0."""
        result = score_facts(
            self._ops001_facts(),
            {"total_students": 999, "total_groups": 999},
            [],
        )
        assert result.grounding_accuracy == pytest.approx(0.0)
        assert result.facts_matched == 0

    # --- Mixed ---

    def test_mixed_returns_partial_score(self) -> None:
        """One correct and one wrong should return grounding_accuracy = 0.5."""
        result = score_facts(
            self._ops001_facts(),
            {"total_students": 135, "total_groups": 999},
            [],
        )
        assert result.grounding_accuracy == pytest.approx(0.5)
        assert result.facts_matched == 1

    # --- Empty gold facts ---

    def test_empty_gold_facts_returns_1(self) -> None:
        """Empty gold_facts list should return grounding_accuracy = 1.0 by convention."""
        result = score_facts([], {}, [])
        assert result.grounding_accuracy == pytest.approx(1.0)
        assert result.facts_total == 0
        assert result.facts_matched == 0

    # --- Raw dict input ---

    def test_accepts_raw_dict_facts(self) -> None:
        """score_facts should accept raw dicts and auto-convert via GoldFact.from_dict."""
        raw_facts = [
            {
                "fact_id": "F1",
                "claim": "135 students",
                "source_files": ["students.csv"],
                "numeric_value": 135.0,
                "tolerance": 0.0,
            }
        ]
        result = score_facts(raw_facts, {"total_students": 135}, [])
        assert result.grounding_accuracy == pytest.approx(1.0)

    # --- Tolerance flag ---

    def test_tolerance_flag_set_when_nonzero_tolerance(self) -> None:
        """scorer_flags should contain 'numeric_tolerance_applied' when tolerance > 0."""
        facts = [GoldFact("F4", "80.9% attendance", ["attendance.csv"], 0.8092, 0.0005)]
        result = score_facts(facts, {"attendance_rate": 0.8090}, [])
        assert "numeric_tolerance_applied" in result.scorer_flags

    def test_no_tolerance_flag_with_zero_tolerance(self) -> None:
        """scorer_flags should NOT contain tolerance flag for zero-tolerance numeric facts."""
        facts = [GoldFact("F1", "135 students", ["students.csv"], 135.0, 0.0)]
        result = score_facts(facts, {"total_students": 135}, [])
        assert "numeric_tolerance_applied" not in result.scorer_flags

    # --- Detail completeness ---

    def test_details_length_matches_facts(self) -> None:
        """Length of details list must match the number of gold facts."""
        facts = self._ops001_facts()
        result = score_facts(facts, {"total_students": 135, "total_groups": 14}, [])
        assert len(result.details) == len(facts)

    def test_detail_fact_ids_match(self) -> None:
        """FactDetail.fact_id values should match the gold fact IDs."""
        facts = self._ops001_facts()
        result = score_facts(facts, {"total_students": 135, "total_groups": 14}, [])
        detail_ids = [d.fact_id for d in result.details]
        assert detail_ids == ["F1", "F5"]

    # --- GoldFact.from_dict ---

    def test_gold_fact_from_dict_numeric(self) -> None:
        """GoldFact.from_dict should correctly parse a numeric gold fact dict."""
        d = {
            "fact_id": "F1",
            "claim": "135 students",
            "source_files": ["students.csv"],
            "numeric_value": 135,
            "tolerance": 0,
        }
        gf = GoldFact.from_dict(d)
        assert gf.fact_id == "F1"
        assert gf.numeric_value == 135
        assert gf.tolerance == 0

    def test_gold_fact_from_dict_non_numeric(self) -> None:
        """GoldFact.from_dict should handle a non-numeric fact (numeric_value=None)."""
        d = {
            "fact_id": "F6",
            "claim": "SCH-001 has 6 tutoring groups",
            "source_files": ["groups.csv"],
            "numeric_value": None,
            "tolerance": None,
        }
        gf = GoldFact.from_dict(d)
        assert gf.numeric_value is None
        assert gf.tolerance is None

    # --- Real task data shape ---

    def test_real_t1_ops002_fact4_within_tolerance(self) -> None:
        """Reproduces T1-OPS-002 F4: attendance 0.8092 with tolerance 0.0005."""
        fact = GoldFact(
            "F4",
            "The program-wide student attendance rate for the quarter is approximately 80.9%",
            ["attendance.csv", "sessions.csv"],
            0.8092,
            0.0005,
        )
        # Model output has 80.91% — within tolerance
        result = score_facts([fact], {"attendance_rate": 0.8091}, [])
        assert result.grounding_accuracy == pytest.approx(1.0)

    def test_real_t1_ops002_fact4_outside_tolerance(self) -> None:
        """Model output with attendance 0.79 is outside tolerance 0.0005 of 0.8092."""
        fact = GoldFact(
            "F4",
            "The program-wide student attendance rate for the quarter is approximately 80.9%",
            ["attendance.csv", "sessions.csv"],
            0.8092,
            0.0005,
        )
        result = score_facts([fact], {"attendance_rate": 0.79}, [])
        assert result.grounding_accuracy == pytest.approx(0.0)

    def test_relative_tolerance_on_score_facts(self) -> None:
        """score_facts supports GoldFact instances with string relative tolerance."""
        # We construct this manually as task schema only stores float tolerances,
        # but score_numeric itself supports string tolerances.
        from benchmark.rubrics.fact_scoring import score_numeric

        assert score_numeric(105.0, 100.0, "5%") == pytest.approx(1.0)
        assert score_numeric(106.0, 100.0, "5%") == pytest.approx(0.0)

    # --- Value-based matching in score_facts ---

    def test_arbitrary_key_names_full_credit(self) -> None:
        """score_facts must give full credit when all correct values appear under arbitrary keys.

        This reproduces the live benchmark failure: model reports correct values
        (135 enrolled, 14 groups) under invented key names and was previously
        scored ~3% because key-name matching failed.
        """
        facts = self._ops001_facts()
        result = score_facts(
            facts,
            {
                "currently_enrolled_students": 135,
                "number_of_active_tutoring_groups": 14,
            },
            [],
        )
        assert result.grounding_accuracy == pytest.approx(1.0)
        assert result.facts_matched == 2

    def test_value_in_key_findings_full_credit(self) -> None:
        """score_facts must give full credit when correct values appear only in key_findings."""
        facts = self._ops001_facts()
        result = score_facts(
            facts,
            {},
            [
                "A total of 135 students are currently enrolled across all schools.",
                "The program operates 14 tutoring groups in total.",
            ],
        )
        assert result.grounding_accuracy == pytest.approx(1.0)
        assert result.facts_matched == 2

    def test_value_absent_no_false_positive(self) -> None:
        """score_facts must NOT credit a fact when the gold value is absent."""
        facts = [GoldFact("F1", "135 students", ["students.csv"], 135.0, 0.0)]
        result = score_facts(
            facts,
            {"enrollment_count": 200},
            ["The program has approximately 200 students enrolled."],
        )
        assert result.grounding_accuracy == pytest.approx(0.0)
        assert result.facts_matched == 0


# ===========================================================================
# FactScoreResult  (dataclass sanity)
# ===========================================================================


class TestFactScoreResult:
    """Sanity checks for FactScoreResult dataclass."""

    def test_default_fields(self) -> None:
        """FactScoreResult should initialize with empty lists by default."""
        r = FactScoreResult(grounding_accuracy=0.5, facts_total=2, facts_matched=1)
        assert r.details == []
        assert r.scorer_flags == []

    def test_grounding_accuracy_bounds(self) -> None:
        """grounding_accuracy returned by score_facts is always in [0, 1]."""
        facts = [GoldFact("F1", "claim", ["f.csv"], 10.0, 0.0)]
        # exact match
        r1 = score_facts(facts, {"f1": 10.0}, [])
        assert 0.0 <= r1.grounding_accuracy <= 1.0
        # miss
        r2 = score_facts(facts, {"f1": 999.0}, [])
        assert 0.0 <= r2.grounding_accuracy <= 1.0


# ===========================================================================
# Integration smoke test using real task shape
# ===========================================================================


class TestIntegrationRealTaskShape:
    """Smoke test using a mock of real T1-OPS-001 gold facts and a model output."""

    _GOLD_FACTS: list[dict[str, object]] = [
        {
            "fact_id": "F1",
            "claim": "The program enrolls 135 students in total.",
            "source_files": ["students.csv"],
            "numeric_value": 135,
            "tolerance": 0,
        },
        {
            "fact_id": "F2",
            "claim": "School SCH-001 has 60 enrolled students.",
            "source_files": ["students.csv"],
            "numeric_value": 60,
            "tolerance": 0,
        },
        {
            "fact_id": "F3",
            "claim": "School SCH-002 has 45 enrolled students.",
            "source_files": ["students.csv"],
            "numeric_value": 45,
            "tolerance": 0,
        },
        {
            "fact_id": "F5",
            "claim": "The program operates 14 tutoring groups across all three schools.",
            "source_files": ["groups.csv"],
            "numeric_value": 14,
            "tolerance": 0,
        },
    ]

    def test_perfect_model_output(self) -> None:
        """A model that reports all correct values should score 1.0."""
        structured_metrics = {
            "f1_total_students": 135,
            "f2_sch001_students": 60,
            "f3_sch002_students": 45,
            "f5_total_groups": 14,
        }
        result = score_facts(self._GOLD_FACTS, structured_metrics, [])
        assert result.grounding_accuracy == pytest.approx(1.0)
        assert result.facts_matched == 4

    def test_perfect_model_output_arbitrary_keys(self) -> None:
        """A model that reports all correct values under arbitrary key names should score 1.0.

        This is the primary regression test for the value-based matching fix.
        """
        structured_metrics = {
            "currently_enrolled_students": 135,
            "sch001_student_count": 60,
            "sch002_students": 45,
            "active_tutoring_groups": 14,
        }
        result = score_facts(self._GOLD_FACTS, structured_metrics, [])
        assert result.grounding_accuracy == pytest.approx(1.0)
        assert result.facts_matched == 4

    def test_partially_correct_model_output(self) -> None:
        """Model that gets 2 out of 4 facts right should score 0.5.

        Values 135 and 45 are correct; 999 does not match any gold fact value
        (60 or 14), so grounding accuracy = 2/4 = 0.5.
        """
        structured_metrics = {
            "f1_total_students": 135,  # correct (matches F1=135)
            "f2_sch001_students": 999,  # wrong
            "f3_sch002_students": 45,  # correct (matches F3=45)
            "f5_total_groups": 999,  # wrong (999 != 60, 14 — note: 999 is deduplicated in scan)
        }
        result = score_facts(self._GOLD_FACTS, structured_metrics, [])
        assert result.grounding_accuracy == pytest.approx(0.5)
        assert result.facts_matched == 2

    def test_completely_wrong_model_output(self) -> None:
        """Model that gets all facts wrong should score 0.0."""
        structured_metrics = {
            "f1_total_students": 100,
            "f2_sch001_students": 50,
            "f3_sch002_students": 40,
            "f5_total_groups": 10,
        }
        result = score_facts(self._GOLD_FACTS, structured_metrics, [])
        assert result.grounding_accuracy == pytest.approx(0.0)
        assert result.facts_matched == 0
