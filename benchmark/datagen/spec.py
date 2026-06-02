"""Source-of-truth specification for the GRADE synthetic data generator.

This module defines the *input constants* that configure the intentional signals
baked into the generated fixtures.  Downstream B-task and C-task authors should
use the *realized* values from ``ground_truth.json`` (emitted alongside each pack)
for assertions, not the constants here — the generator applies noise and
per-school jitter so realized aggregates differ from the input targets.

Intentional signals embedded in every generated pack
------------------------------------------------------
1.  **Month-over-month attendance trend (Outcomes pack)**
    Attendance rate is seeded at ``MOM_ATTENDANCE_START`` and rises by
    approximately ``MOM_ATTENDANCE_DELTA`` each month, but per-school noise
    makes the realized program-wide aggregate noisy and non-monotonic.
    B-task authors: use ``ground_truth.json > monthly_attendance_rate`` for
    the actual numbers; do NOT assume a clean +10 pp monotonic rise.

2.  **Low-N subgroup (Equity & Research pack)**
    The ``AIAN`` (American Indian / Alaska Native) race-ethnicity subgroup is
    intentionally kept below ``SUPPRESSION_THRESHOLD`` students program-wide.
    All subgroup tables for this cohort carry ``suppressed = true`` and null
    metric columns.  B-task authors: gold tasks for Track 4 must require the
    model to acknowledge low-N suppression for ``AIAN``.

3.  **Subgroup attendance disparity (Equity & Research pack)**
    The IEP subgroup (students with an Individualized Education Program) attends
    at approximately ``IEP_ATTENDANCE_PENALTY`` below the non-IEP baseline.
    The realized gap may differ due to stochastic attendance draws; see
    ``ground_truth.json`` for the actual gap in percentage points.

4.  **Satisfaction score dip and recovery (Outcomes pack)**
    ``monthly_satisfaction_summary.csv`` contains a genuine three-month series
    (September / October / November) reflecting baseline → October dip →
    November recovery.  The dip month is ``SATISFACTION_DIP_MONTH`` (0-based
    index into ``MONTHS``).  Realized scores have small jitter applied; see
    ``ground_truth.json > monthly_satisfaction`` and
    ``ground_truth.json > satisfaction_dip_realized`` for actual values.

All of these constants are inputs; changing them constitutes a schema-breaking
change requiring fixture regeneration and downstream B/C task updates.
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

# Number of months with the rising trend (remaining months plateau).
# Note: per-school noise means the realized program-wide aggregate is
# non-monotonic (~79% → ~82% → ~82%); see ground_truth.json for actual values.
MOM_TREND_MONTHS: int = 3

# B-task authors: the realized attendance values are noisy/non-monotonic due
# to per-school jitter.  Use ground_truth.json > monthly_attendance_rate for
# assertions, not the spec target constants above.

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

# B-task authors: the monthly_satisfaction_summary.csv now contains a real
# three-month series.  Realized scores have ±0.1 jitter; see
# ground_truth.json > monthly_satisfaction and > satisfaction_dip_realized
# for the actual per-month avg_score values.

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
    {
        "ref_id": "R4",
        "citation": (
            "Fryer, R. G. (2014). Injecting successful charter school strategies into "
            "traditional public schools: Evidence from field experiments. "
            "The Quarterly Journal of Economics, 129(3), 1355–1407."
        ),
        "summary": (
            "A randomized field experiment testing whether importing five key practices from "
            "high-performing charter schools (including intensive small-group tutoring) into "
            "traditional public schools improves math and reading achievement. "
            "Schools that adopted the full bundle of practices, including high-dosage tutoring, "
            "saw significant gains in math achievement (0.15–0.23 SD) within one year."
        ),
        "key_finding": (
            "High-dosage tutoring is among the most impactful individual components when "
            "implemented within a coherent instructional model; schools adopting tutoring alone "
            "without broader instructional alignment showed smaller effects."
        ),
        "relevance_to_program": (
            "The fixture program operates as a stand-alone tutoring add-on; Fryer's finding "
            "suggests that alignment between tutoring content and classroom instruction may "
            "be critical to realizing the full effect-size potential."
        ),
        "supports_claim": None,
        "contradicts_claim": (
            "Stand-alone high-dosage tutoring may produce smaller gains than projected if "
            "it is not coordinated with classroom instruction and school culture."
        ),
        "caution": (
            "Fryer's study was conducted in Houston public schools; implementation quality "
            "and school context may differ substantially from other districts."
        ),
    },
    {
        "ref_id": "R5",
        "citation": (
            "Cook, P. J., Dodge, K., Farkas, G., Fryer, R. G., Guryan, J., Ludwig, J., "
            "Mayer, S., Pollack, H., & Steinberg, L. (2015). Not too late: Improving academic "
            "outcomes for disadvantaged youth. IPR Working Paper WP-15-01, Northwestern "
            "University Institute for Policy Research."
        ),
        "summary": (
            "Evaluates a randomized trial of intensive in-school mentoring and tutoring for "
            "high-school students in Chicago. The program combined one-on-one tutoring during "
            "the school day with social-emotional learning components. "
            "Participants showed a 0.5 SD improvement in math achievement and a 46% reduction "
            "in violent crime arrests over two years."
        ),
        "key_finding": (
            "Intensive tutoring embedded within the school day for high-risk youth produces "
            "large academic and behavioral co-benefits, suggesting attendance and engagement "
            "barriers can be overcome through in-school scheduling."
        ),
        "relevance_to_program": (
            "The fixture program's elevated absenteeism among IEP students may be partially "
            "addressable by shifting to in-school session scheduling, which Cook et al. found "
            "substantially reduced engagement barriers for high-need students."
        ),
        "supports_claim": (
            "Embedding tutoring sessions within the school day is a viable strategy to improve "
            "attendance among high-need subgroups, including students with IEPs."
        ),
        "contradicts_claim": None,
        "caution": (
            "Cook et al. studied a high-school population in an urban district; effects may "
            "not transfer directly to elementary or middle-grade contexts."
        ),
    },
    {
        "ref_id": "R6",
        "citation": (
            "Kraft, M. A., & Falken, G. T. (2021). A blueprint for scaling tutoring and "
            "mentoring programs in public schools. AERA Open, 7, 1–21."
        ),
        "summary": (
            "Provides a practical framework for scaling high-dosage tutoring programs, "
            "identifying six key design levers: tutor selection, training, scheduling, "
            "data use, alignment with curriculum, and family engagement. "
            "Programs that scored high on all six levers maintained effect sizes above 0.2 SD "
            "at scale, while those that cut corners on data use showed the sharpest declines."
        ),
        "key_finding": (
            "Data use — defined as weekly monitoring of attendance and learning progress at "
            "the student level — is the single design lever most predictive of sustained "
            "effect sizes when programs scale to hundreds of sites."
        ),
        "relevance_to_program": (
            "The fixture program's satisfaction dip and IEP attendance gap are exactly the "
            "kind of signals that Kraft & Falken argue should trigger immediate mid-quarter "
            "program adjustments via a weekly data-review cadence."
        ),
        "supports_claim": (
            "The fixture program's data — including the IEP attendance gap and October "
            "satisfaction dip — provide actionable signals that, per Kraft & Falken, warrant "
            "a program adjustment before the end of the reporting period."
        ),
        "contradicts_claim": None,
        "caution": (
            "Kraft & Falken's framework is prescriptive; compliance with individual levers "
            "does not guarantee outcome improvement without implementation fidelity."
        ),
    },
    {
        "ref_id": "R7",
        "citation": (
            "Guryan, J., Ludwig, J., Bhatt, M. P., Cook, P. J., Davis, J. M. V., "
            "Dodge, K., Farkas, G., Mayer, S. E., Pollack, H., Steinberg, L., & "
            "Tessler-Lavine, I. (2023). Not too late: A randomized controlled trial of "
            "a high-dosage tutoring program. American Economic Review, 113(3), 738–765."
        ),
        "summary": (
            "A large-scale randomized controlled trial of in-school high-dosage math tutoring "
            "for ninth-grade students in Chicago public schools. Tutoring was delivered during "
            "the school day in a dedicated elective period. "
            "The program produced 0.10 SD gains in math achievement per semester of treatment, "
            "with larger gains for students who received more sessions."
        ),
        "key_finding": (
            "Dosage-response relationship is strongly positive: each additional 20 sessions "
            "attended corresponds to an estimated 0.04 SD additional math gain, reinforcing "
            "the importance of maximizing realized attendance."
        ),
        "relevance_to_program": (
            "The fixture program's attendance tracking is directly relevant here — the "
            "IEP attendance gap of ~11 pp translates, per Guryan et al.'s dosage-response "
            "estimate, into a meaningful projected outcome shortfall for this subgroup."
        ),
        "supports_claim": (
            "Every percentage-point improvement in attendance for IEP students corresponds "
            "to measurable expected gains in academic outcomes based on the dosage-response "
            "evidence."
        ),
        "contradicts_claim": None,
        "caution": (
            "Guryan et al.'s study covers ninth-grade math in Chicago; dosage-response "
            "slopes may differ for younger students or reading-focused programs."
        ),
    },
    {
        "ref_id": "R8",
        "citation": (
            "Pellegrini, M., Lake, C., Inns, A., & Slavin, R. E. (2018). Effective programs "
            "in elementary mathematics: A best-evidence synthesis. "
            "Best Evidence Encyclopedia, Johns Hopkins University School of Education."
        ),
        "summary": (
            "A best-evidence synthesis of 78 studies of elementary mathematics programs, "
            "finding that tutoring approaches (both one-on-one and small-group) consistently "
            "outperform technology-only and curriculum-only interventions. "
            "Small-group tutoring (2–6 students) showed mean effect sizes of 0.31 SD, "
            "comparable to one-on-one tutoring at substantially lower cost per student."
        ),
        "key_finding": (
            "Small-group tutoring of 3–6 students is nearly as effective as one-on-one "
            "tutoring for elementary math, making it the most cost-efficient format for "
            "programs serving large numbers of students."
        ),
        "relevance_to_program": (
            "The fixture program uses groups of 3–5 students — within the range Pellegrini "
            "et al. find to be effective — providing support for the program's group-size "
            "design choice."
        ),
        "supports_claim": (
            "The 3–5 student group sizes used in this program are well-supported by the "
            "evidence base for elementary math tutoring effectiveness."
        ),
        "contradicts_claim": None,
        "caution": (
            "Best-evidence syntheses weight higher-quality studies; studies of small-group "
            "tutoring for ELL students and students with IEPs are underrepresented in the "
            "evidence base reviewed."
        ),
    },
    {
        "ref_id": "R9",
        "citation": (
            "Dietrichson, J., Bøg, M., Filges, T., & Klint Jørgensen, A.-M. (2017). "
            "Academic interventions for elementary and middle school students with low "
            "socioeconomic status: A systematic review and meta-analysis. "
            "Review of Educational Research, 87(2), 243–282."
        ),
        "summary": (
            "A systematic review of 101 studies examining academic interventions for "
            "low-socioeconomic-status (low-SES) students in grades K–8. "
            "Tutoring and small-group instruction showed the largest effect sizes (mean 0.36 SD) "
            "among all intervention categories, substantially outperforming feedback-only or "
            "curriculum-alignment approaches. Effects were largest for students in grades 1–5."
        ),
        "key_finding": (
            "For low-SES students (proxied by free/reduced lunch eligibility), tutoring "
            "interventions yield effect sizes approximately twice as large as non-tutoring "
            "interventions, underscoring the equity value of high-dosage tutoring for "
            "Title I school populations."
        ),
        "relevance_to_program": (
            "Two of the three fixture schools are Title I, and 60% of students in the "
            "fixture data are FRL-eligible — precisely the population Dietrichson et al. "
            "find to benefit most from tutoring interventions."
        ),
        "supports_claim": (
            "The program's Title I school focus and high FRL-eligible population align with "
            "the subgroups for whom the tutoring evidence base is strongest."
        ),
        "contradicts_claim": None,
        "caution": (
            "Many studies in the synthesis were conducted outside the United States; "
            "SES measurement and school context vary considerably across the studies reviewed."
        ),
    },
    {
        "ref_id": "R10",
        "citation": (
            "Vaughn, S., Wanzek, J., Murray, C. S., & Roberts, G. (2012). Intensive "
            "interventions for students struggling in reading and mathematics: A practice "
            "guide. Portsmouth, NH: RMC Research Corporation, Center on Instruction."
        ),
        "summary": (
            "A practitioner-focused synthesis of Tier 3 intensive intervention research for "
            "students with persistent learning difficulties, including those with IEPs. "
            "Recommends small-group instruction (1–3 students) with explicit, systematic "
            "delivery and frequent progress monitoring (at minimum bi-weekly) for students "
            "with IEPs or significant skill gaps."
        ),
        "key_finding": (
            "Students with IEPs receiving intensive small-group interventions with bi-weekly "
            "progress monitoring show 0.40–0.60 SD improvements over typical school-year "
            "instruction alone, but only when session attendance exceeds 80%."
        ),
        "relevance_to_program": (
            "The fixture data shows IEP students attending at approximately 71% — below the "
            "80% threshold Vaughn et al. identify as the minimum for realizing the full "
            "intervention effect. This is a critical program risk to flag."
        ),
        "supports_claim": None,
        "contradicts_claim": (
            "At the observed ~71% IEP attendance rate in the fixture data, students with "
            "IEPs are below the minimum attendance threshold Vaughn et al. associate with "
            "effective intensive intervention, putting their expected outcomes at risk."
        ),
        "caution": (
            "Vaughn et al.'s synthesis focuses on Tier 3 intervention and may overstate "
            "the attendance threshold for Tier 2 programs like the fixture program."
        ),
    },
]
