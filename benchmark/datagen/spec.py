"""Source-of-truth specification for the GRADE synthetic data generator.

This module defines the *known, computable truths* that are baked into the
generated fixtures and that downstream B-task authors can reference as
``gold_facts``/``gold_insights`` and that C1/C3 scorers can verify.

Intentional signals embedded in every generated pack
------------------------------------------------------
1.  **Month-over-month attendance trend (Outcomes pack)**
    Attendance rate starts at roughly ``MOM_ATTENDANCE_START`` and rises by
    ``MOM_ATTENDANCE_DELTA`` each month through ``MOM_TREND_MONTHS`` months,
    then plateaus.  B-task authors: the net rise is
    ``MOM_ATTENDANCE_DELTA * (MOM_TREND_MONTHS - 1)`` percentage points.

2.  **Low-N subgroup (Equity & Research pack)**
    The ``AIAN`` (American Indian / Alaska Native) race-ethnicity subgroup is
    intentionally kept below ``SUPPRESSION_THRESHOLD`` students program-wide.
    All subgroup tables for this cohort carry ``suppressed = true`` and null
    metric columns.  B-task authors: gold tasks for Track 4 must require the
    model to acknowledge low-N suppression for ``AIAN``.

3.  **Subgroup attendance disparity (Equity & Research pack)**
    The IEP subgroup (students with an Individualized Education Program) attends
    at ``IEP_ATTENDANCE_PENALTY`` below the non-IEP baseline.  This disparity is
    reproducible given a fixed seed.  B-task authors: the gap is detectable in
    ``subgroup_attendance_summary.csv`` when comparing ``iep=true`` vs
    ``iep=false`` rows.

4.  **Satisfaction score dip and recovery (Outcomes pack)**
    Student satisfaction scores dip in the middle month by
    ``SATISFACTION_DIP_MAGNITUDE`` Likert points and recover in the final month.
    B-task authors: the dip month is month index ``SATISFACTION_DIP_MONTH``
    (0-based).

All of these values are constants; changing them constitutes a schema-breaking
change that requires updating all downstream B/C tasks and regenerating fixtures.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Program structure
# ---------------------------------------------------------------------------

PROGRAM_ID = "PROG-001"
PROGRAM_NAME = "Pearl Academic Support Initiative"
PROGRAM_TYPE = "tutoring"
DISTRICT_ID = "DIST-001"

# Reporting period: one quarter (13 weeks ≈ 3 full months of data)
REPORTING_PERIOD_START = "2025-09-01"
REPORTING_PERIOD_END = "2025-11-30"

# Months covered (used for MoM tables)
MONTHS: list[str] = ["2025-09", "2025-10", "2025-11"]

# Schools
SCHOOL_COUNT = 3
SCHOOL_IDS = [f"SCH-{i:03d}" for i in range(1, SCHOOL_COUNT + 1)]
SCHOOL_NAMES = [
    "Lincoln Elementary",
    "Roosevelt Middle School",
    "Jefferson Academy",
]
SCHOOL_LOCALES = ["urban", "suburban", "rural"]
SCHOOL_TITLE_I = [True, True, False]
SCHOOL_GRADE_SPANS = ["K-5", "6-8", "K-8"]
SCHOOL_ENROLLMENTS = [420, 310, 185]

# Tutors per school
TUTORS_PER_SCHOOL = [4, 3, 2]  # total 9 tutors

# Students per school
STUDENTS_PER_SCHOOL = [60, 45, 30]  # total 135

# Groups per school (small-group tutoring: 3–5 students per group)
GROUPS_PER_SCHOOL = [6, 5, 3]  # total 14 groups

# Sessions per group per week
SESSIONS_PER_WEEK = 3
SESSION_DURATION_MINUTES = 45
DOSAGE_TARGET_MINUTES_PER_WEEK = 135  # 3 × 45

# Surveys administered (one per type per reporting period)
SURVEY_TYPES = ["student_satisfaction", "tutor_self_eval", "parent_feedback"]

# ---------------------------------------------------------------------------
# Signal 1: Month-over-month attendance trend (Outcomes pack)
# ---------------------------------------------------------------------------

# Baseline attendance rate in the first month (September)
MOM_ATTENDANCE_START: float = 0.72

# Attendance rises by this amount each month for the trend period, then
# plateaus.  Net change across the 3-month window: +0.10 (i.e., 72% → 82%).
MOM_ATTENDANCE_DELTA: float = 0.05

# Number of months with the rising trend (remaining months plateau)
MOM_TREND_MONTHS: int = 3  # all three months show upward trend

# B-task gold fact: "Attendance rate rose from 72% in September to 82% in
# November — a +10 percentage-point improvement across the reporting period."

# ---------------------------------------------------------------------------
# Signal 2: Low-N subgroup (Equity & Research pack)
# ---------------------------------------------------------------------------

# Target N for the AIAN subgroup program-wide (must be < suppression threshold)
AIAN_TARGET_N: int = 4  # well below n < 10 threshold

# Suppression threshold (must match fixture_schema.md §4)
SUPPRESSION_THRESHOLD: int = 10

# B-task gold fact: "The AIAN subgroup has n < 10 students program-wide;
# attendance and dosage metrics are suppressed to protect student privacy."

# ---------------------------------------------------------------------------
# Signal 3: IEP subgroup attendance disparity (Equity & Research pack)
# ---------------------------------------------------------------------------

# Fraction of students with an IEP
IEP_PREVALENCE: float = 0.12

# IEP students attend at this many percentage points below the non-IEP baseline
IEP_ATTENDANCE_PENALTY: float = 0.11

# B-task gold insight: "Students with IEPs attended at approximately 11
# percentage points below their non-IEP peers — a meaningful equity gap that
# warrants programmatic attention."

# ---------------------------------------------------------------------------
# Signal 4: Satisfaction score dip and recovery (Outcomes pack)
# ---------------------------------------------------------------------------

# Mean student satisfaction score in a "normal" month (Likert 1–5 scale)
SATISFACTION_BASELINE: float = 3.8

# The month index (0-based, so index 1 = October) where the dip occurs
SATISFACTION_DIP_MONTH: int = 1

# How much the average score drops in the dip month
SATISFACTION_DIP_MAGNITUDE: float = 0.4

# How much the score recovers in the final month (full recovery)
SATISFACTION_RECOVERY: float = 0.4

# B-task gold insight: "Student satisfaction dipped to 3.4 in October before
# recovering to 3.8 in November.  The dip correlates with a period of
# elevated session cancellations."

# ---------------------------------------------------------------------------
# Race/ethnicity distribution (must sum to 1.0 after AIAN override)
# ---------------------------------------------------------------------------

# Approximate desired distribution BEFORE ensuring AIAN low-N.
# AIAN will be assigned to exactly AIAN_TARGET_N students regardless.
RACE_ETHNICITY_DIST: dict[str, float] = {
    "Hispanic": 0.38,
    "Black": 0.22,
    "White": 0.18,
    "Asian": 0.10,
    "TwoOrMore": 0.06,
    "NHPI": 0.02,
    "Unknown": 0.02,
    "AIAN": 0.00,  # overridden to AIAN_TARGET_N absolute students
}

# ---------------------------------------------------------------------------
# Missingness profiles (passed to messiness.py)
# ---------------------------------------------------------------------------

# Fraction of survey responses that are "sparse" (question skipped)
SURVEY_SPARSITY_RATE: float = 0.18

# Fraction of attendance records with a null minutes_attended (full session or
# absent — schema allows null for full-session attendance)
ATTENDANCE_PARTIAL_RATE: float = 0.08

# Fraction of students whose tutoring_group_id is null at snapshot time
# (newly enrolled, not yet assigned)
UNASSIGNED_STUDENT_RATE: float = 0.06

# Fraction of tutors with null avg_session_rating (no surveys yet)
TUTOR_NO_RATING_RATE: float = 0.22

# ---------------------------------------------------------------------------
# Grade levels served
# ---------------------------------------------------------------------------

TARGET_GRADE_LEVELS = [3, 4, 5, 6, 7, 8]
SUBJECT_AREAS = ["math", "reading"]

# ---------------------------------------------------------------------------
# Certification level distribution for tutors
# ---------------------------------------------------------------------------

CERT_LEVEL_DIST: dict[str, float] = {
    "none": 0.10,
    "paraprofessional": 0.40,
    "certified_teacher": 0.40,
    "specialist": 0.10,
}

# ---------------------------------------------------------------------------
# Session cancellation profile
# ---------------------------------------------------------------------------

# In the dip month (October), cancellation rate is elevated
CANCELLATION_RATE_NORMAL: float = 0.08
CANCELLATION_RATE_DIP_MONTH: float = 0.18  # aligns with satisfaction dip

# Session status weights outside the dip month
SESSION_STATUS_WEIGHTS_NORMAL: dict[str, float] = {
    "completed": 0.88,
    "cancelled_tutor": 0.03,
    "cancelled_student": 0.03,
    "cancelled_school": 0.02,
    "no_show": 0.04,
}

SESSION_STATUS_WEIGHTS_DIP: dict[str, float] = {
    "completed": 0.78,
    "cancelled_tutor": 0.06,
    "cancelled_student": 0.06,
    "cancelled_school": 0.04,
    "no_show": 0.06,
}

# ---------------------------------------------------------------------------
# Research references (Equity & Research pack)
# ---------------------------------------------------------------------------

RESEARCH_REFS: list[dict[str, str | None]] = [
    {
        "ref_id": "R1",
        "citation": "Nickow, A., Oreopoulos, P., & Quan, V. (2020). The impressive effects of "
        "tutoring on PreK-12 learning. NBER Working Paper No. 27476.",
        "summary": (
            "A meta-analysis of 96 randomized studies finds that tutoring programs reliably "
            "improve student learning, with effect sizes ranging from 0.1 to 0.5 SD. "
            "Small-group and one-on-one formats both show significant gains. "
            "Dosage (total instructional minutes) is among the strongest moderators of effect "
            "size."
        ),
        "key_finding": (
            "Higher dosage tutoring (≥90 min/week) produces effect sizes roughly twice as "
            "large as lower-dosage programs."
        ),
        "relevance_to_program": (
            "The fixture program targets 135 min/week — above the high-dosage threshold — "
            "lending external validity to expected gains if dosage targets are met."
        ),
        "supports_claim": (
            "Meeting the 135 min/week dosage target is associated with meaningfully larger "
            "learning gains based on the evidence base."
        ),
        "contradicts_claim": None,
        "caution": (
            "Meta-analytic effect sizes aggregate across many program types and contexts; "
            "generalizability to this specific program requires local evaluation data."
        ),
    },
    {
        "ref_id": "R2",
        "citation": (
            "Robinson, C. D., Kraft, M. A., & Loeb, S. (2021). Accelerating student learning "
            "with high-dosage tutoring. EdResearch for Recovery, Brief No. 5."
        ),
        "summary": (
            "Reviews implementation evidence from pandemic-era high-dosage tutoring scale-ups. "
            "Finds that attendance and student engagement are the primary drivers of realized "
            "dosage, not session scheduling. Programs that tracked and responded to attendance "
            "weekly outperformed those with monthly monitoring cycles."
        ),
        "key_finding": (
            "Attendance monitoring at weekly rather than monthly cadence is associated with "
            "8–12 percentage-point higher realized attendance rates."
        ),
        "relevance_to_program": (
            "The fixture program shows an attendance rate below dosage targets in September, "
            "suggesting the program may benefit from tighter monitoring cycles per R2's finding."
        ),
        "supports_claim": None,
        "contradicts_claim": (
            "The program's monthly attendance monitoring cadence may be insufficient given "
            "the September–October attendance dip documented in the fixture data."
        ),
        "caution": (
            "R2's findings derive from high-scale pandemic recovery programs; resource and "
            "context differences limit direct comparison to smaller, ongoing programs."
        ),
    },
    {
        "ref_id": "R3",
        "citation": (
            "Morgan, P. L., Farkas, G., & Hibel, J. (2008). Matthew effects for whom? "
            "Learning Disability Quarterly, 31(4), 187–198."
        ),
        "summary": (
            "Examines differential learning trajectories for students with disabilities "
            "versus without, finding that without targeted intervention, students with IEPs "
            "show slower growth over time relative to peers (Matthew effect). "
            "Intensive tutoring can close — but does not automatically eliminate — this gap."
        ),
        "key_finding": (
            "Students with IEPs require explicitly equity-targeted program design to achieve "
            "attendance and outcome parity with non-IEP peers."
        ),
        "relevance_to_program": (
            "The fixture data shows an IEP attendance gap of approximately 11 percentage "
            "points, consistent with this literature's warning about passive program designs."
        ),
        "supports_claim": (
            "The observed IEP attendance gap in this program is consistent with known "
            "patterns and warrants explicit programmatic response."
        ),
        "contradicts_claim": None,
        "caution": (
            "Morgan et al. studied elementary reading trajectories; caution is warranted "
            "when applying findings to middle-grade math tutoring contexts."
        ),
    },
]
