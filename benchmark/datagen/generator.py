"""Top-level seeded synthetic data generator for the three GRADE fixture packs.

CLI usage::

    python -m benchmark.datagen.generator --seed 42 --out /tmp/fixtures
    python -m benchmark.datagen.generator --seed 42 --out /tmp/fixtures --pack operations
    python -m benchmark.datagen.generator --seed 42 --out /tmp/fixtures --pack outcomes
    python -m benchmark.datagen.generator --seed 42 --out /tmp/fixtures --pack equity_research

Writes::

    <out>/pack_operations/        programs, schools, students, tutors, groups,
                                   sessions, attendance, surveys, survey_responses,
                                   program_context.json, manifest.json
    <out>/pack_outcomes/          all of the above PLUS monthly_attendance_summary,
                                   monthly_satisfaction_summary, manifest.json
    <out>/pack_equity_research/   all of the above PLUS subgroup_attendance_summary,
                                   subgroup_outcomes_summary, research_refs.json,
                                   manifest.json

Determinism guarantee
---------------------
A single :class:`random.Random` instance is seeded once from ``--seed``.  Every
random decision in every helper is drawn from that one instance in a fixed call
order.  No wall-clock, no ``os.urandom``, no global/unseeded RNG, no unordered
set/dict iteration affecting output.  Re-running with the same ``--seed`` and
the same generator source code produces byte-for-byte identical files.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import os
import random
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from benchmark.datagen import messiness as mx
from benchmark.datagen import spec

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

Row = dict[str, Any]
Table = list[Row]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _date_range(start: str, end: str) -> list[date]:
    """Return every calendar date in [*start*, *end*] inclusive.

    Args:
        start: ISO date string ``YYYY-MM-DD``.
        end: ISO date string ``YYYY-MM-DD``.

    Returns:
        Sorted list of :class:`datetime.date` objects.
    """
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out: list[date] = []
    cur = s
    while cur <= e:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _month_bounds(year_month: str) -> tuple[date, date]:
    """Return (first, last) dates for an ISO year-month string.

    Args:
        year_month: ``YYYY-MM`` string.

    Returns:
        Tuple of ``(month_start, month_end)`` as :class:`datetime.date`.
    """
    year, month = int(year_month[:4]), int(year_month[5:7])
    first = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    last = date(year, month, last_day)
    return first, last


def _iso_ts(d: date, hour: int = 12, minute: int = 0) -> str:
    """Format a date + time as an ISO 8601 UTC timestamp string.

    Args:
        d: Date.
        hour: Hour component (default 12).
        minute: Minute component (default 0).

    Returns:
        String like ``"2025-09-15T12:00:00Z"``.
    """
    return f"{d.isoformat()}T{hour:02d}:{minute:02d}:00Z"


# ---------------------------------------------------------------------------
# Table generators — Operations pack tables
# ---------------------------------------------------------------------------


def _gen_programs() -> Table:
    """Generate ``programs.csv`` — one row for the single program.

    Returns:
        Single-element list of program row dicts.
    """
    return [
        {
            "program_id": spec.PROGRAM_ID,
            "program_name": spec.PROGRAM_NAME,
            "program_type": spec.PROGRAM_TYPE,
            "start_date": "2025-09-01",
            "end_date": "",
            "target_grade_levels": ",".join(str(g) for g in spec.TARGET_GRADE_LEVELS),
            "subject_areas": ",".join(spec.SUBJECT_AREAS),
            "district_id": spec.DISTRICT_ID,
            "school_count": spec.SCHOOL_COUNT,
            "dosage_target_minutes_per_week": spec.DOSAGE_TARGET_MINUTES_PER_WEEK,
            "reporting_period_start": spec.REPORTING_PERIOD_START,
            "reporting_period_end": spec.REPORTING_PERIOD_END,
        }
    ]


def _gen_schools() -> Table:
    """Generate ``schools.csv`` — one row per school.

    Returns:
        List of school row dicts.
    """
    rows: Table = []
    for i, sid in enumerate(spec.SCHOOL_IDS):
        rows.append(
            {
                "school_id": sid,
                "school_name": spec.SCHOOL_NAMES[i],
                "program_id": spec.PROGRAM_ID,
                "district_id": spec.DISTRICT_ID,
                "enrollment": spec.SCHOOL_ENROLLMENTS[i],
                "locale": spec.SCHOOL_LOCALES[i],
                "title_i": str(spec.SCHOOL_TITLE_I[i]).lower(),
                "grade_span": spec.SCHOOL_GRADE_SPANS[i],
            }
        )
    return rows


def _gen_tutors(rng: random.Random) -> Table:
    """Generate ``tutors.csv`` — one row per tutor.

    Args:
        rng: Seeded RNG.

    Returns:
        List of tutor row dicts.
    """
    rows: Table = []
    tutor_idx = 1
    for school_idx, sid in enumerate(spec.SCHOOL_IDS):
        for _ in range(spec.TUTORS_PER_SCHOOL[school_idx]):
            tid = f"TUT-{tutor_idx:03d}"
            hire_delta = rng.randint(0, 180)
            hire_date = date.fromisoformat("2025-03-01") + timedelta(days=hire_delta)
            cert = mx.weighted_choice(spec.CERT_LEVEL_DIST, rng)
            subject = rng.choice(["math", "reading", "both"])
            sessions_ytd = rng.randint(30, 90)
            has_rating = rng.random() >= spec.TUTOR_NO_RATING_RATE
            avg_rating: float | None = round(rng.uniform(3.0, 5.0), 2) if has_rating else None
            rows.append(
                {
                    "tutor_id": tid,
                    "program_id": spec.PROGRAM_ID,
                    "school_id": sid,
                    "hire_date": hire_date.isoformat(),
                    "certification_level": cert,
                    "subject_specialization": subject,
                    "active": "true",
                    "sessions_delivered_ytd": sessions_ytd,
                    "avg_session_rating": avg_rating if avg_rating is not None else "",
                }
            )
            tutor_idx += 1
    return rows


def _gen_groups(tutors: Table, rng: random.Random) -> Table:
    """Generate ``groups.csv`` — one row per tutoring group.

    Args:
        tutors: Generated tutor rows (to look up tutor assignments).
        rng: Seeded RNG.

    Returns:
        List of group row dicts.
    """
    # Build a mapping school → list of tutor_ids
    school_tutors: dict[str, list[str]] = {sid: [] for sid in spec.SCHOOL_IDS}
    for t in tutors:
        school_tutors[str(t["school_id"])].append(str(t["tutor_id"]))

    rows: Table = []
    group_idx = 1
    subjects = sorted(spec.SUBJECT_AREAS)
    grade_levels = sorted(spec.TARGET_GRADE_LEVELS)
    for school_idx, sid in enumerate(spec.SCHOOL_IDS):
        tutors_here = sorted(school_tutors[sid])
        for gi in range(spec.GROUPS_PER_SCHOOL[school_idx]):
            gid = f"GRP-{group_idx:03d}"
            t_id = tutors_here[gi % len(tutors_here)]
            subject = subjects[gi % len(subjects)]
            grade = grade_levels[gi % len(grade_levels)]
            group_size = rng.randint(3, 5)
            rows.append(
                {
                    "group_id": gid,
                    "program_id": spec.PROGRAM_ID,
                    "school_id": sid,
                    "tutor_id": t_id,
                    "group_size": group_size,
                    "subject": subject,
                    "grade_level": grade,
                    "sessions_per_week": spec.SESSIONS_PER_WEEK,
                    "session_duration_minutes": spec.SESSION_DURATION_MINUTES,
                    "start_date": "2025-09-08",
                }
            )
            group_idx += 1
    return rows


def _gen_students(groups: Table, rng: random.Random) -> Table:
    """Generate ``students.csv`` — one row per enrolled student.

    Race/ethnicity assignment uses :func:`~messiness.assign_race_ethnicity` to
    guarantee AIAN_TARGET_N students across the program (low-N signal).

    Args:
        groups: Generated group rows (for group assignment).
        rng: Seeded RNG.

    Returns:
        List of student row dicts.
    """
    # Pre-generate race codes for all students
    total_students = sum(spec.STUDENTS_PER_SCHOOL)
    all_race_codes = mx.assign_race_ethnicity(
        total_students, spec.RACE_ETHNICITY_DIST, spec.AIAN_TARGET_N, rng
    )

    # Build school → groups mapping
    school_groups: dict[str, list[str]] = {sid: [] for sid in spec.SCHOOL_IDS}
    for g in groups:
        school_groups[str(g["school_id"])].append(str(g["group_id"]))

    rows: Table = []
    student_idx = 1
    race_cursor = 0
    genders = ["M", "F", "NB", "U"]
    gender_weights = [0.46, 0.46, 0.04, 0.04]

    for school_idx, sid in enumerate(spec.SCHOOL_IDS):
        n = spec.STUDENTS_PER_SCHOOL[school_idx]
        grps = sorted(school_groups[sid])
        for i in range(n):
            stuid = f"STU-{student_idx:05d}"
            race = all_race_codes[race_cursor]
            race_cursor += 1

            gender = rng.choices(genders, weights=gender_weights, k=1)[0]
            iep = rng.random() < spec.IEP_PREVALENCE
            ell = rng.random() < 0.15
            frl = rng.random() < 0.60
            grade = rng.choice(spec.TARGET_GRADE_LEVELS)

            # Enrollment date: random day in the first two weeks of September
            enroll_date = date(2025, 9, 1) + timedelta(days=rng.randint(0, 13))

            # Group assignment: most students assigned; a few unassigned
            unassigned = rng.random() < spec.UNASSIGNED_STUDENT_RATE
            grp_id: str | None = None if unassigned else grps[i % len(grps)]

            rows.append(
                {
                    "student_id": stuid,
                    "program_id": spec.PROGRAM_ID,
                    "school_id": sid,
                    "grade_level": grade,
                    "gender": gender,
                    "race_ethnicity": race,
                    "iep": str(iep).lower(),
                    "ell": str(ell).lower(),
                    "free_reduced_lunch": str(frl).lower(),
                    "enrollment_date": enroll_date.isoformat(),
                    "active": "true",
                    "tutoring_group_id": grp_id if grp_id is not None else "",
                }
            )
            student_idx += 1
    return rows


def _gen_sessions(groups: Table, rng: random.Random) -> Table:
    """Generate ``sessions.csv`` — one row per scheduled session slot.

    Cancellation rate is elevated in October (dip month) per ``spec.py``
    signal #4 (satisfaction dip correlates with elevated cancellations).

    Args:
        groups: Generated group rows.
        rng: Seeded RNG.

    Returns:
        List of session row dicts.
    """
    rows: Table = []
    session_idx = 1

    all_dates = _date_range(spec.REPORTING_PERIOD_START, spec.REPORTING_PERIOD_END)
    # Only schedule sessions Monday–Friday
    weekdays = [d for d in all_dates if d.weekday() < 5]

    for g in groups:
        gid = str(g["group_id"])
        t_id = str(g["tutor_id"])
        sessions_per_week = int(str(g["sessions_per_week"]))
        duration = int(str(g["session_duration_minutes"]))

        # Assign fixed days of week for this group (deterministic from group idx)
        all_weekday_nums = [0, 1, 2, 3, 4]  # Mon=0 … Fri=4
        rng.shuffle(all_weekday_nums)
        assigned_weekdays = sorted(all_weekday_nums[:sessions_per_week])

        # Fixed start time per group
        hour = rng.choice([9, 10, 11, 13, 14])
        minute = rng.choice([0, 30])

        for d in weekdays:
            if d.weekday() not in assigned_weekdays:
                continue

            month_label = f"{d.year}-{d.month:02d}"
            dip = month_label == spec.MONTHS[spec.SATISFACTION_DIP_MONTH]
            status_weights = (
                spec.SESSION_STATUS_WEIGHTS_DIP if dip else spec.SESSION_STATUS_WEIGHTS_NORMAL
            )
            status = mx.weighted_choice(status_weights, rng)

            actual_dur: int | None
            on_time: bool | None
            if status == "completed":
                actual_dur = duration + rng.randint(-5, 5)
                actual_dur = max(10, actual_dur)
                on_time = rng.random() < 0.92
            else:
                actual_dur = None
                on_time = None

            platform = rng.choice(["in_person", "virtual", "hybrid"])

            sid = f"SES-{session_idx:05d}"
            rows.append(
                {
                    "session_id": sid,
                    "group_id": gid,
                    "tutor_id": t_id,
                    "scheduled_date": d.isoformat(),
                    "scheduled_start_time": f"{hour:02d}:{minute:02d}",
                    "duration_minutes": duration,
                    "status": status,
                    "actual_duration_minutes": actual_dur if actual_dur is not None else "",
                    "tutor_on_time": (str(on_time).lower() if on_time is not None else ""),
                    "platform": platform,
                }
            )
            session_idx += 1
    return rows


def _gen_attendance(sessions: Table, students: Table, rng: random.Random) -> Table:
    """Generate ``attendance.csv`` — one row per student per completed session.

    Only generates attendance records for ``completed`` sessions.  IEP students
    attend at a lower rate (signal #3: IEP attendance disparity).

    Args:
        sessions: Generated session rows.
        students: Generated student rows.
        rng: Seeded RNG.

    Returns:
        List of attendance row dicts.
    """
    # Build group → list of student_ids mapping
    group_students: dict[str, list[str]] = {}
    for st in students:
        grp = str(st.get("tutoring_group_id") or "")
        if grp:
            group_students.setdefault(grp, [])
            group_students[grp].append(str(st["student_id"]))

    # Build student → iep lookup
    iep_lookup: dict[str, bool] = {str(st["student_id"]): st["iep"] == "true" for st in students}

    rows: Table = []
    att_idx = 1
    absence_reasons = ["illness", "school_conflict", "family", "unknown", "no_show"]
    absence_weights = [0.30, 0.25, 0.15, 0.20, 0.10]

    completed_sessions = [s for s in sessions if s["status"] == "completed"]

    for sess in completed_sessions:
        gid = str(sess["group_id"])
        sdate = date.fromisoformat(str(sess["scheduled_date"]))
        dur = int(str(sess["duration_minutes"]))

        student_ids = sorted(group_students.get(gid, []))
        for stuid in student_ids:
            is_iep = iep_lookup.get(stuid, False)
            # Base attendance probability, penalized for IEP
            base_att_prob = 0.82
            att_prob = base_att_prob - (spec.IEP_ATTENDANCE_PENALTY if is_iep else 0.0)
            attended = rng.random() < att_prob

            # Partial attendance (minutes_attended only for partial)
            minutes_attended: int | None = None
            if attended and rng.random() < spec.ATTENDANCE_PARTIAL_RATE:
                minutes_attended = rng.randint(max(5, dur - 15), dur - 1)

            absence_reason: str | None = None
            if not attended:
                absence_reason = rng.choices(absence_reasons, weights=absence_weights, k=1)[0]

            recorded_hour = rng.randint(14, 18)
            recorded_min = rng.randint(0, 59)

            rows.append(
                {
                    "attendance_id": f"ATT-{att_idx:06d}",
                    "session_id": str(sess["session_id"]),
                    "student_id": stuid,
                    "attended": str(attended).lower(),
                    "minutes_attended": minutes_attended if minutes_attended is not None else "",
                    "absence_reason": absence_reason if absence_reason is not None else "",
                    "recorded_at": _iso_ts(sdate, recorded_hour, recorded_min),
                }
            )
            att_idx += 1
    return rows


def _gen_surveys(rng: random.Random) -> Table:
    """Generate ``surveys.csv`` — one row per survey instrument.

    Args:
        rng: Seeded RNG.

    Returns:
        List of survey row dicts.
    """
    rows: Table = []
    total_students = sum(spec.STUDENTS_PER_SCHOOL)
    total_tutors = sum(spec.TUTORS_PER_SCHOOL)

    survey_configs = [
        (
            "SRV-001",
            "student_satisfaction",
            "Student Satisfaction Survey",
            total_students,
            "2025-11-15",
        ),
        ("SRV-002", "tutor_self_eval", "Tutor Self-Evaluation", total_tutors, "2025-11-20"),
        ("SRV-003", "parent_feedback", "Parent Feedback Survey", total_students, "2025-11-10"),
    ]

    for survey_id, stype, sname, total_invited, admin_date in survey_configs:
        response_rate = rng.uniform(0.55, 0.80)
        total_responded = round(total_invited * response_rate)
        rows.append(
            {
                "survey_id": survey_id,
                "program_id": spec.PROGRAM_ID,
                "survey_name": sname,
                "survey_type": stype,
                "administered_date": admin_date,
                "total_invited": total_invited,
                "total_responded": total_responded,
                "response_rate": round(total_responded / total_invited, 4),
            }
        )
    return rows


def _gen_survey_responses(surveys: Table, students: Table, rng: random.Random) -> Table:
    """Generate ``survey_responses.csv`` — one row per respondent per question.

    Survey responses are sparse per ``spec.SURVEY_SPARSITY_RATE``.

    Args:
        surveys: Generated survey rows.
        students: Generated student rows.
        rng: Seeded RNG.

    Returns:
        List of survey response row dicts.
    """
    student_ids = sorted(str(s["student_id"]) for s in students)

    questions: dict[str, list[tuple[str, str, str]]] = {
        "student_satisfaction": [
            ("Q1", "I enjoy my tutoring sessions.", "likert_5"),
            ("Q2", "My tutor explains things clearly.", "likert_5"),
            ("Q3", "I feel more confident in this subject.", "likert_5"),
            ("NPS_SCORE", "How likely are you to recommend this program?", "nps"),
        ],
        "tutor_self_eval": [
            ("TE1", "I feel prepared for each session.", "likert_5"),
            ("TE2", "I receive adequate support from program staff.", "likert_4"),
            ("TE3", "Overall program quality rating.", "likert_5"),
        ],
        "parent_feedback": [
            ("PF1", "My child looks forward to tutoring.", "likert_5"),
            ("PF2", "I have seen improvement in my child's skills.", "binary"),
            ("PF3", "Overall satisfaction with the program.", "likert_5"),
        ],
    }

    rows: Table = []
    resp_idx = 1

    for survey in surveys:
        sid = str(survey["survey_id"])
        stype = str(survey["survey_type"])
        admin_date = date.fromisoformat(str(survey["administered_date"]))
        n_responded = int(str(survey["total_responded"]))

        qs = questions.get(stype, [])

        # Sample respondents
        n_sample = min(n_responded, len(student_ids))
        respondents = mx.sample_without_replacement(student_ids, n_sample, rng)

        for respondent_id in respondents:
            for q_code, q_text, q_type in qs:
                # Sparsity: skip this question with some probability
                if rng.random() < spec.SURVEY_SPARSITY_RATE:
                    continue

                # Generate response value based on type
                if q_type == "likert_5":
                    val = str(rng.randint(1, 5))
                elif q_type == "likert_4":
                    val = str(rng.randint(1, 4))
                elif q_type == "nps":
                    val = str(rng.randint(0, 10))
                elif q_type == "binary":
                    val = rng.choice(["yes", "no"])
                else:
                    val = str(rng.randint(1, 5))

                resp_hour = rng.randint(8, 20)
                resp_min = rng.randint(0, 59)
                rows.append(
                    {
                        "response_id": f"RSP-{resp_idx:06d}",
                        "survey_id": sid,
                        "student_id": respondent_id,
                        "question_code": q_code,
                        "question_text": q_text,
                        "response_value": val,
                        "response_type": q_type,
                        "responded_at": _iso_ts(admin_date, resp_hour, resp_min),
                    }
                )
                resp_idx += 1
    return rows


def _gen_program_context(seed: int) -> dict[str, Any]:
    """Generate ``program_context.json``.

    Args:
        seed: Integer seed used for this pack generation.

    Returns:
        Dictionary conforming to the ``program_context.json`` schema.
    """
    return {
        "program_id": spec.PROGRAM_ID,
        "narrative_description": (
            "Pearl Academic Support Initiative is a high-dosage tutoring program serving "
            f"{sum(spec.STUDENTS_PER_SCHOOL)} students across {spec.SCHOOL_COUNT} schools in "
            "the district.  The program targets grades 3–8 in math and reading, with small-group "
            "sessions (3–5 students) delivered at 135 minutes per student per week.  The program "
            "launched in September 2025 and this pack covers the first reporting quarter "
            "(September–November 2025)."
        ),
        "stated_goals": [
            "Improve math and reading proficiency for students in grades 3–8.",
            "Achieve ≥80% attendance rate across all schools by the end of the quarter.",
            "Reduce the IEP attendance gap to ≤5 percentage points.",
            "Achieve a student satisfaction mean score of ≥4.0 by end of quarter.",
        ],
        "primary_contact": {
            "name": "Alex Rivera",
            "role": "Program Director",
        },
        "data_collection_notes": (
            "Attendance is recorded by tutors within 30 minutes of each session. "
            "Survey data was collected in the final two weeks of the reporting period. "
            "Parent survey response rate is lower than targets due to translation delays "
            "for non-English-speaking households."
        ),
        "known_data_gaps": [
            "AIAN subgroup has n < 10 students program-wide; all AIAN metrics are suppressed.",
            "Parent feedback survey administered late in the period; full response data "
            "may not reflect early-quarter satisfaction.",
            f"{round(spec.UNASSIGNED_STUDENT_RATE * 100)}% of students not yet assigned to a "
            "tutoring group at time of snapshot.",
            "Tutor self-evaluation survey has partial coverage; "
            f"{round(spec.TUTOR_NO_RATING_RATE * 100)}% of tutors have no session rating data.",
        ],
        "implementation_challenges": [
            "October session cancellation rate was elevated (18%) due to schedule conflicts.",
            "Student satisfaction dipped in October, correlating with the cancellation spike.",
            "IEP students are attending ~11 percentage points below non-IEP peers.",
        ],
        "reporting_period": {
            "start": spec.REPORTING_PERIOD_START,
            "end": spec.REPORTING_PERIOD_END,
            "label": "Q1 FY2025-26 (September–November 2025)",
        },
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Outcomes pack additional tables
# ---------------------------------------------------------------------------


def _compute_monthly_attendance(
    sessions: Table,
    attendance: Table,
    groups: Table,
    rng: random.Random,
) -> Table:
    """Generate ``monthly_attendance_summary.csv``.

    Computes per-school x per-month aggregates from actual session and
    attendance data, then applies the MoM trend signal (sessions are
    already generated with that signal embedded via cancellation rates).

    Args:
        sessions: Generated session rows.
        attendance: Generated attendance rows.
        groups: Generated group rows (used to resolve group_id to school_id).
        rng: Seeded RNG.

    Returns:
        List of monthly summary row dicts.
    """
    # Build group -> school_id lookup (sessions only have group_id)
    group_school: dict[str, str] = {str(g["group_id"]): str(g["school_id"]) for g in groups}

    # Build lookup maps
    session_by_id: dict[str, Row] = {str(s["session_id"]): s for s in sessions}

    # Accumulate by (school_id, month_label)
    Agg = dict[str, Any]
    agg: dict[tuple[str, str], Agg] = {}

    for sess in sessions:
        sid = group_school.get(str(sess["group_id"]), spec.SCHOOL_IDS[0])
        d = date.fromisoformat(str(sess["scheduled_date"]))
        ml = f"{d.year}-{d.month:02d}"
        key = (sid, ml)
        if key not in agg:
            agg[key] = {
                "sessions_scheduled": 0,
                "sessions_completed": 0,
                "sessions_cancelled": 0,
                "student_session_slots": 0,
                "student_sessions_attended": 0,
                "total_dosage_minutes": 0.0,
                "active_students": set(),
            }
        a = agg[key]
        a["sessions_scheduled"] += 1
        if sess["status"] == "completed":
            a["sessions_completed"] += 1
        elif str(sess["status"]).startswith("cancelled") or sess["status"] == "no_show":
            a["sessions_cancelled"] += 1

    # Accumulate attendance into the agg
    for att in attendance:
        att_sess: Row | None = session_by_id.get(str(att["session_id"]))
        if att_sess is None:
            continue
        d = date.fromisoformat(str(att_sess["scheduled_date"]))
        ml = f"{d.year}-{d.month:02d}"
        att_sid = group_school.get(str(att_sess["group_id"]), spec.SCHOOL_IDS[0])
        key = (att_sid, ml)
        att_agg = agg[key]
        att_agg["student_session_slots"] += 1
        if att["attended"] == "true":
            att_agg["student_sessions_attended"] += 1
            dur = int(str(att_sess["duration_minutes"]))
            mins = int(str(att["minutes_attended"])) if att["minutes_attended"] != "" else dur
            att_agg["total_dosage_minutes"] += mins
            stuid = str(att["student_id"])
            att_agg["active_students"].add(stuid)

    rows: Table = []
    prev_att_rate: dict[str, float] = {}
    prev_dosage: dict[str, float] = {}
    idx = 1

    for month_label in sorted(spec.MONTHS):
        first, last = _month_bounds(month_label)
        for sid in spec.SCHOOL_IDS:
            key = (sid, month_label)
            month_agg: dict[str, Any] = agg.get(key) or {}
            sched = int(month_agg.get("sessions_scheduled", 0))
            comp = int(month_agg.get("sessions_completed", 0))
            canc = int(month_agg.get("sessions_cancelled", 0))
            slots = int(month_agg.get("student_session_slots", 0))
            attended_slots = int(month_agg.get("student_sessions_attended", 0))
            total_dosage = float(month_agg.get("total_dosage_minutes", 0.0))
            active_set: set[str] = month_agg.get("active_students") or set()
            active_n = len(active_set)

            att_rate = (attended_slots / slots) if slots > 0 else 0.0
            # Number of enrolled students for this school for avg dosage
            school_idx = spec.SCHOOL_IDS.index(sid)
            n_enrolled = spec.STUDENTS_PER_SCHOOL[school_idx]
            avg_dosage = total_dosage / n_enrolled if n_enrolled > 0 else 0.0

            prev_k = sid
            mom_att: float | None = (
                (att_rate - prev_att_rate[prev_k]) if prev_k in prev_att_rate else None
            )
            mom_dos: float | None = (
                (avg_dosage - prev_dosage[prev_k]) if prev_k in prev_dosage else None
            )

            prev_att_rate[prev_k] = att_rate
            prev_dosage[prev_k] = avg_dosage

            rows.append(
                {
                    "summary_id": f"MAS-{idx:04d}",
                    "program_id": spec.PROGRAM_ID,
                    "school_id": sid,
                    "month_label": month_label,
                    "month_start": first.isoformat(),
                    "month_end": last.isoformat(),
                    "sessions_scheduled": sched,
                    "sessions_completed": comp,
                    "sessions_cancelled": canc,
                    "attendance_rate": round(att_rate, 4),
                    "avg_dosage_minutes": round(avg_dosage, 2),
                    "active_students": active_n,
                    "mom_attendance_delta": round(mom_att, 4) if mom_att is not None else "",
                    "mom_dosage_delta": round(mom_dos, 2) if mom_dos is not None else "",
                }
            )
            idx += 1
    return rows


def _compute_monthly_satisfaction(surveys: Table, survey_responses: Table) -> Table:
    """Generate ``monthly_satisfaction_summary.csv``.

    Aggregates survey responses by (school_id=program-wide, survey_type, month).
    Uses the administered_date month as the month_label.

    Args:
        surveys: Generated survey rows.
        survey_responses: Generated survey response rows.

    Returns:
        List of monthly satisfaction summary row dicts.
    """
    # Map survey_id → (survey_type, administered_month, school_id=None for program-level)
    survey_meta: dict[str, tuple[str, str]] = {}
    for s in surveys:
        ml = str(s["administered_date"])[:7]
        survey_meta[str(s["survey_id"])] = (str(s["survey_type"]), ml)

    # Accumulate scores
    agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    for resp in survey_responses:
        sv_id = str(resp["survey_id"])
        if sv_id not in survey_meta:
            continue
        stype, ml = survey_meta[sv_id]
        key = (spec.SCHOOL_IDS[0], stype, ml)  # program-level (first school as proxy)
        if key not in agg:
            agg[key] = {"scores": [], "responded": 0, "invited": 0}
        raw = str(resp["response_value"])
        try:
            val = float(raw)
            agg[key]["scores"].append(val)
        except ValueError:
            pass
        agg[key]["responded"] += 1

    # Add invited counts from surveys
    for s in surveys:
        sv_id = str(s["survey_id"])
        if sv_id not in survey_meta:
            continue
        stype, ml = survey_meta[sv_id]
        key = (spec.SCHOOL_IDS[0], stype, ml)
        if key in agg:
            agg[key]["invited"] = int(str(s["total_invited"]))

    rows: Table = []
    prev_score: dict[tuple[str, str], float | None] = {}
    idx = 1

    for (school_id, stype, ml), data in sorted(agg.items()):
        scores: list[float] = data["scores"]
        avg_sc: float | None = round(sum(scores) / len(scores), 4) if scores else None
        responded = data["responded"]
        invited = data.get("invited", 0)
        resp_rate: float | None = round(responded / invited, 4) if invited > 0 else None

        prev_k2 = (stype, school_id)
        mom_delta: float | None = None
        if prev_k2 in prev_score and prev_score[prev_k2] is not None and avg_sc is not None:
            mom_delta = round(avg_sc - prev_score[prev_k2], 4)  # type: ignore[operator]
        prev_score[prev_k2] = avg_sc

        rows.append(
            {
                "summary_id": f"MSS-{idx:04d}",
                "program_id": spec.PROGRAM_ID,
                "school_id": school_id,
                "survey_type": stype,
                "month_label": ml,
                "responses_count": responded,
                "avg_score": avg_sc if avg_sc is not None else "",
                "response_rate": resp_rate if resp_rate is not None else "",
                "mom_score_delta": mom_delta if mom_delta is not None else "",
            }
        )
        idx += 1
    return rows


# ---------------------------------------------------------------------------
# Equity & Research pack additional tables
# ---------------------------------------------------------------------------


def _compute_subgroup_attendance(sessions: Table, attendance: Table, students: Table) -> Table:
    """Generate ``subgroup_attendance_summary.csv``.

    One row per (subgroup_dimension × subgroup_value × month).  Low-N
    suppression is applied per ``spec.SUPPRESSION_THRESHOLD``.

    Args:
        sessions: Generated session rows.
        attendance: Generated attendance rows.
        students: Generated student rows.

    Returns:
        List of subgroup attendance summary row dicts.
    """
    session_by_id: dict[str, Row] = {str(s["session_id"]): s for s in sessions}

    # Build student attribute lookup
    student_attrs: dict[str, dict[str, str]] = {}
    for st in students:
        student_attrs[str(st["student_id"])] = {
            "race_ethnicity": str(st.get("race_ethnicity") or "Unknown"),
            "gender": str(st.get("gender") or "U"),
            "iep": str(st["iep"]),
            "ell": str(st["ell"]),
            "free_reduced_lunch": str(st["free_reduced_lunch"]),
            "grade_level": str(st["grade_level"]),
        }

    # Accumulate: (dim, val, month_label) → {n_students, attended_sessions, total_sessions}
    agg: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _inc(dim: str, val: str, ml: str, attended: bool) -> None:
        k = (dim, val, ml)
        if k not in agg:
            agg[k] = {"students": set(), "attended": 0, "total": 0, "dosage": 0.0}
        agg[k]["total"] += 1
        if attended:
            agg[k]["attended"] += 1

    for att in attendance:
        stuid = str(att["student_id"])
        sess = session_by_id.get(str(att["session_id"]))
        if sess is None:
            continue
        d = date.fromisoformat(str(sess["scheduled_date"]))
        ml = f"{d.year}-{d.month:02d}"
        attrs = student_attrs.get(stuid, {})
        attended = att["attended"] == "true"

        for dim, val in sorted(attrs.items()):
            k = (dim, val, ml)
            if k not in agg:
                agg[k] = {"students": set(), "attended": 0, "total": 0, "dosage": 0.0}
            agg[k]["students"].add(stuid)
            agg[k]["total"] += 1
            if attended:
                agg[k]["attended"] += 1
                dur = int(str(sess["duration_minutes"]))
                mins = int(str(att["minutes_attended"])) if att["minutes_attended"] != "" else dur
                agg[k]["dosage"] += mins

    rows: Table = []
    idx = 1
    for (dim, val, ml), data in sorted(agg.items()):
        n = len(data["students"])
        total = data["total"]
        attended = data["attended"]
        dosage = data["dosage"]

        att_rate: float = (attended / total) if total > 0 else 0.0
        avg_dosage: float = (dosage / n) if n > 0 else 0.0

        row: Row = {
            "summary_id": f"SAS-{idx:05d}",
            "program_id": spec.PROGRAM_ID,
            "school_id": "",  # program-wide aggregate
            "month_label": ml,
            "subgroup_dimension": dim,
            "subgroup_value": val,
            "n_students": n,
            "n_sessions_attended": attended,
            "attendance_rate": round(att_rate, 4),
            "avg_dosage_minutes": round(avg_dosage, 2),
            "suppressed": False,
            "suppression_reason": "",
        }
        mx.suppress_if_low_n(
            row,
            "n_students",
            ["attendance_rate", "avg_dosage_minutes"],
            spec.SUPPRESSION_THRESHOLD,
        )
        # Normalize booleans and suppressed metrics to CSV-friendly strings
        if row["suppressed"]:
            row["attendance_rate"] = ""
            row["avg_dosage_minutes"] = ""
        row["suppressed"] = str(row["suppressed"]).lower()
        rows.append(row)
        idx += 1
    return rows


def _compute_subgroup_outcomes(students: Table, rng: random.Random) -> Table:
    """Generate ``subgroup_outcomes_summary.csv``.

    Generates synthetic assessment scores for two periods (fall/spring) with
    IEP disparity baked in.

    Args:
        students: Generated student rows.
        rng: Seeded RNG.

    Returns:
        List of subgroup outcomes summary row dicts.
    """
    # Build subgroup student sets
    SubgroupKey = tuple[str, str]
    student_subgroups: dict[SubgroupKey, list[str]] = {}
    for st in students:
        stuid = str(st["student_id"])
        dims = {
            "race_ethnicity": str(st.get("race_ethnicity") or "Unknown"),
            "gender": str(st.get("gender") or "U"),
            "iep": str(st["iep"]),
            "ell": str(st["ell"]),
            "free_reduced_lunch": str(st["free_reduced_lunch"]),
            "grade_level": str(st["grade_level"]),
        }
        for dim, val in sorted(dims.items()):
            k: SubgroupKey = (dim, val)
            student_subgroups.setdefault(k, [])
            student_subgroups[k].append(stuid)

    rows: Table = []
    idx = 1
    periods = ["fall_2025", "spring_2026"]

    for period in periods:
        is_spring = period == "spring_2026"
        for (dim, val), studs in sorted(student_subgroups.items()):
            n = len(studs)
            is_iep_group = dim == "iep" and val == "true"

            # Base scale score around 400 (like a scaled score)
            base_score = 398.0 if is_iep_group else 412.0
            avg_scale = round(mx.jitter_float(base_score, 8.0, 300.0, 500.0, rng), 1)

            # Gains are larger in spring
            base_gain = (6.5 if is_iep_group else 9.0) if is_spring else 0.0
            avg_gain: float | None = (
                round(mx.jitter_float(base_gain, 2.0, -5.0, 30.0, rng), 2) if is_spring else None
            )

            # Proficiency rate
            base_prof = 0.41 if is_iep_group else 0.58
            prof_rate: float | None = round(mx.jitter_float(base_prof, 0.05, 0.0, 1.0, rng), 4)

            row: Row = {
                "summary_id": f"SOS-{idx:05d}",
                "program_id": spec.PROGRAM_ID,
                "school_id": "",
                "assessment_period": period,
                "subgroup_dimension": dim,
                "subgroup_value": val,
                "n_students": n,
                "avg_scale_score": avg_scale,
                "avg_score_gain": avg_gain if avg_gain is not None else "",
                "benchmark_proficiency_rate": prof_rate,
                "suppressed": False,
                "suppression_reason": "",
            }
            mx.suppress_if_low_n(
                row,
                "n_students",
                ["avg_scale_score", "avg_score_gain", "benchmark_proficiency_rate"],
                spec.SUPPRESSION_THRESHOLD,
            )
            if row["suppressed"]:
                row["avg_scale_score"] = ""
                row["avg_score_gain"] = ""
                row["benchmark_proficiency_rate"] = ""
            row["suppressed"] = str(row["suppressed"]).lower()
            rows.append(row)
            idx += 1
    return rows


def _gen_research_refs(program_id: str) -> dict[str, Any]:
    """Generate ``research_refs.json``.

    Args:
        program_id: Program identifier.

    Returns:
        Dictionary conforming to the ``research_refs.json`` schema.
    """
    return {
        "program_id": program_id,
        "references": spec.RESEARCH_REFS,
    }


# ---------------------------------------------------------------------------
# CSV / JSON I/O
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: Table) -> None:
    """Write *rows* to a CSV file at *path*.

    Uses ``csv.DictWriter`` with a deterministic column order taken from the
    first row's keys.  Empty string values are written as empty fields (not
    ``None``).

    Args:
        path: Destination file path.
        rows: List of row dicts with consistent keys.
    """
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Normalize: None → empty string for CSV output
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _write_json(path: Path, data: Any) -> None:
    """Write *data* as pretty-printed JSON to *path*.

    Args:
        path: Destination file path.
        data: JSON-serializable object.
    """
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False, ensure_ascii=False)
        fh.write("\n")


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of *path*.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex string of the SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_count(path: Path) -> int | None:
    """Return the number of data rows in a CSV file (excluding the header).

    Returns ``None`` for non-CSV files.

    Args:
        path: File to count.

    Returns:
        Row count integer or ``None``.
    """
    if path.suffix != ".csv":
        return None
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = sum(1 for _ in reader)
    return max(0, rows - 1)  # exclude header


def _write_manifest(
    pack_dir: Path,
    pack_id: str,
    seed: int,
    generated_at_utc: str,
) -> None:
    """Write ``manifest.json`` for a pack directory.

    Enumerates all files in *pack_dir* (excluding ``manifest.json`` itself),
    computes SHA-256 hashes, and records row counts for CSV files.

    Args:
        pack_dir: Pack output directory.
        pack_id: Pack identifier string.
        seed: Integer seed.
        generated_at_utc: ISO 8601 UTC timestamp string.
    """
    files_list = []
    for fpath in sorted(pack_dir.iterdir()):
        if fpath.name == "manifest.json":
            continue
        files_list.append(
            {
                "filename": fpath.name,
                "sha256": _sha256_file(fpath),
                "row_count": _row_count(fpath),
            }
        )
    manifest = {
        "pack_id": pack_id,
        "grade_version": "0.1.0.dev0",
        "seed": seed,
        "generated_at_utc": generated_at_utc,
        "files": files_list,
    }
    _write_json(pack_dir / "manifest.json", manifest)


# ---------------------------------------------------------------------------
# Pack generation entrypoints
# ---------------------------------------------------------------------------


def generate_operations(out_dir: Path, seed: int) -> Path:
    """Generate the Operations pack into ``out_dir/pack_operations/``.

    Args:
        out_dir: Root output directory.
        seed: Integer RNG seed.

    Returns:
        Path to the ``pack_operations/`` directory.
    """
    rng = random.Random(seed)
    pack_dir = out_dir / "pack_operations"
    pack_dir.mkdir(parents=True, exist_ok=True)

    programs = _gen_programs()
    schools = _gen_schools()
    tutors = _gen_tutors(rng)
    groups = _gen_groups(tutors, rng)
    students = _gen_students(groups, rng)
    sessions = _gen_sessions(groups, rng)
    attendance = _gen_attendance(sessions, students, rng)
    surveys = _gen_surveys(rng)
    survey_responses = _gen_survey_responses(surveys, students, rng)

    _write_csv(pack_dir / "programs.csv", programs)
    _write_csv(pack_dir / "schools.csv", schools)
    _write_csv(pack_dir / "tutors.csv", tutors)
    _write_csv(pack_dir / "groups.csv", groups)
    _write_csv(pack_dir / "students.csv", students)
    _write_csv(pack_dir / "sessions.csv", sessions)
    _write_csv(pack_dir / "attendance.csv", attendance)
    _write_csv(pack_dir / "surveys.csv", surveys)
    _write_csv(pack_dir / "survey_responses.csv", survey_responses)

    ctx = _gen_program_context(seed)
    _write_json(pack_dir / "program_context.json", ctx)

    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_manifest(pack_dir, "pack_operations", seed, generated_at)
    return pack_dir


def generate_outcomes(out_dir: Path, seed: int) -> Path:
    """Generate the Outcomes pack into ``out_dir/pack_outcomes/``.

    Reuses all Operations tables (generated with the same seed), then adds
    month-over-month summary tables.

    Args:
        out_dir: Root output directory.
        seed: Integer RNG seed.

    Returns:
        Path to the ``pack_outcomes/`` directory.
    """
    rng = random.Random(seed)
    pack_dir = out_dir / "pack_outcomes"
    pack_dir.mkdir(parents=True, exist_ok=True)

    programs = _gen_programs()
    schools = _gen_schools()
    tutors = _gen_tutors(rng)
    groups = _gen_groups(tutors, rng)
    students = _gen_students(groups, rng)
    sessions = _gen_sessions(groups, rng)
    attendance = _gen_attendance(sessions, students, rng)
    surveys = _gen_surveys(rng)
    survey_responses = _gen_survey_responses(surveys, students, rng)

    # MoM tables (extra RNG for jitter)
    monthly_att = _compute_monthly_attendance(sessions, attendance, groups, rng)
    monthly_sat = _compute_monthly_satisfaction(surveys, survey_responses)

    _write_csv(pack_dir / "programs.csv", programs)
    _write_csv(pack_dir / "schools.csv", schools)
    _write_csv(pack_dir / "tutors.csv", tutors)
    _write_csv(pack_dir / "groups.csv", groups)
    _write_csv(pack_dir / "students.csv", students)
    _write_csv(pack_dir / "sessions.csv", sessions)
    _write_csv(pack_dir / "attendance.csv", attendance)
    _write_csv(pack_dir / "surveys.csv", surveys)
    _write_csv(pack_dir / "survey_responses.csv", survey_responses)
    _write_csv(pack_dir / "monthly_attendance_summary.csv", monthly_att)
    _write_csv(pack_dir / "monthly_satisfaction_summary.csv", monthly_sat)

    ctx = _gen_program_context(seed)
    _write_json(pack_dir / "program_context.json", ctx)

    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_manifest(pack_dir, "pack_outcomes", seed, generated_at)
    return pack_dir


def generate_equity_research(out_dir: Path, seed: int) -> Path:
    """Generate the Equity & Research pack into ``out_dir/pack_equity_research/``.

    Reuses all Operations tables, then adds subgroup summaries and
    research references.

    Args:
        out_dir: Root output directory.
        seed: Integer RNG seed.

    Returns:
        Path to the ``pack_equity_research/`` directory.
    """
    rng = random.Random(seed)
    pack_dir = out_dir / "pack_equity_research"
    pack_dir.mkdir(parents=True, exist_ok=True)

    programs = _gen_programs()
    schools = _gen_schools()
    tutors = _gen_tutors(rng)
    groups = _gen_groups(tutors, rng)
    students = _gen_students(groups, rng)
    sessions = _gen_sessions(groups, rng)
    attendance = _gen_attendance(sessions, students, rng)
    surveys = _gen_surveys(rng)
    survey_responses = _gen_survey_responses(surveys, students, rng)

    subgroup_att = _compute_subgroup_attendance(sessions, attendance, students)
    subgroup_out = _compute_subgroup_outcomes(students, rng)
    research_refs = _gen_research_refs(spec.PROGRAM_ID)

    _write_csv(pack_dir / "programs.csv", programs)
    _write_csv(pack_dir / "schools.csv", schools)
    _write_csv(pack_dir / "tutors.csv", tutors)
    _write_csv(pack_dir / "groups.csv", groups)
    _write_csv(pack_dir / "students.csv", students)
    _write_csv(pack_dir / "sessions.csv", sessions)
    _write_csv(pack_dir / "attendance.csv", attendance)
    _write_csv(pack_dir / "surveys.csv", surveys)
    _write_csv(pack_dir / "survey_responses.csv", survey_responses)
    _write_csv(pack_dir / "subgroup_attendance_summary.csv", subgroup_att)
    _write_csv(pack_dir / "subgroup_outcomes_summary.csv", subgroup_out)
    _write_json(pack_dir / "research_refs.json", research_refs)

    ctx = _gen_program_context(seed)
    _write_json(pack_dir / "program_context.json", ctx)

    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_manifest(pack_dir, "pack_equity_research", seed, generated_at)
    return pack_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Generate GRADE benchmark fixture packs.",
        prog="python -m benchmark.datagen.generator",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Integer random seed (default: 42).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output root directory.  Pack subdirectories are created inside.",
    )
    parser.add_argument(
        "--pack",
        choices=["operations", "outcomes", "equity_research"],
        default=None,
        help="Generate only one pack.  Omit to generate all three.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the GRADE synthetic data generator.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 = success).
    """
    args = _parse_args(argv)
    out = args.out
    seed: int = args.seed
    pack: str | None = args.pack

    os.makedirs(out, exist_ok=True)

    packs_to_run = [pack] if pack else ["operations", "outcomes", "equity_research"]

    generators = {
        "operations": generate_operations,
        "outcomes": generate_outcomes,
        "equity_research": generate_equity_research,
    }

    for p in packs_to_run:
        dest = generators[p](out, seed)
        print(f"[grade] {p} → {dest}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
