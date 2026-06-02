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


def _gen_monthly_satisfaction_real(seed: int) -> Table:
    """Generate ``monthly_satisfaction_summary.csv`` with the realized dip signal.

    Produces a genuine three-month series (Sep / Oct / Nov) for each survey type,
    reflecting the baseline → October dip → November recovery pattern defined by
    ``spec.SATISFACTION_*`` constants.

    Uses a **dedicated, independently-seeded** RNG (``seed + 9999``) that is
    created after all other generation is complete, so it cannot affect the draw
    order of any other table and preserves byte-identity for the Operations and
    Equity & Research packs.

    Response counts are synthesized as plausible monthly cohort sizes.  Invited
    counts are approximate program-wide totals per survey type.

    Args:
        seed: Pack-level integer seed (same value passed to ``generate_outcomes``).

    Returns:
        List of monthly satisfaction summary row dicts — 9 rows total
        (3 survey types × 3 months), sorted by (survey_type, month_label).
    """
    # Dedicated RNG — isolated from the main generation pipeline.
    sat_rng = random.Random(seed + 9999)

    survey_types = sorted(spec.SURVEY_TYPES)  # deterministic ordering

    # Approximate total invited per survey type
    total_students = sum(spec.STUDENTS_PER_SCHOOL)
    total_tutors = sum(spec.TUTORS_PER_SCHOOL)
    invited_by_type: dict[str, int] = {
        "parent_feedback": total_students,
        "student_satisfaction": total_students,
        "tutor_self_eval": total_tutors,
    }

    rows: Table = []
    idx = 1

    for stype in survey_types:
        n_invited = invited_by_type.get(stype, total_students)
        prev_score: float | None = None

        for mi, month_label in enumerate(sorted(spec.MONTHS)):
            # Realized avg_score from spec signal with small jitter
            avg_sc = round(
                mx.satisfaction_score(
                    month_index=mi,
                    baseline=spec.SATISFACTION_BASELINE,
                    dip_month=spec.SATISFACTION_DIP_MONTH,
                    dip_magnitude=spec.SATISFACTION_DIP_MAGNITUDE,
                    recovery=spec.SATISFACTION_RECOVERY,
                    rng=sat_rng,
                ),
                4,
            )

            # Plausible response count: 40–70% of invited, with per-month noise
            response_rate_frac = round(sat_rng.uniform(0.40, 0.70), 4)
            responses_count = round(n_invited * response_rate_frac)

            resp_rate = round(responses_count / n_invited, 4) if n_invited > 0 else None

            mom_delta: float | None = None
            if prev_score is not None:
                mom_delta = round(avg_sc - prev_score, 4)
            prev_score = avg_sc

            rows.append(
                {
                    "summary_id": f"MSS-{idx:04d}",
                    "program_id": spec.PROGRAM_ID,
                    "school_id": spec.SCHOOL_IDS[0],
                    "survey_type": stype,
                    "month_label": month_label,
                    "responses_count": responses_count,
                    "avg_score": avg_sc,
                    "response_rate": resp_rate if resp_rate is not None else "",
                    "mom_score_delta": mom_delta if mom_delta is not None else "",
                }
            )
            idx += 1

    return rows


def _compute_ground_truth_operations(
    sessions: Table,
    students: Table,
    attendance: Table,
    seed: int,
) -> dict[str, Any]:
    """Compute the ground-truth manifest for the Operations pack.

    Derives realized signal values from sessions, students, and attendance —
    the tables available in pack_operations (no MoM summary tables).  Includes
    monthly cancellation rates, IEP attendance gap, and suppressed low-N
    subgroups.

    Args:
        sessions: Generated ``sessions`` rows.
        students: Generated ``students`` rows.
        attendance: Generated ``attendance`` rows.
        seed: Integer seed used for this generation run.

    Returns:
        Dictionary suitable for ``ground_truth.json`` in pack_operations.
    """
    # -----------------------------------------------------------------------
    # Monthly attendance rate (per month, program-wide)
    # -----------------------------------------------------------------------
    att_slots_by_month: dict[str, dict[str, int]] = {}
    # Build session lookup for date resolution
    session_by_id: dict[str, Row] = {str(s["session_id"]): s for s in sessions}

    for att in attendance:
        sess = session_by_id.get(str(att["session_id"]))
        if sess is None:
            continue
        d = date.fromisoformat(str(sess["scheduled_date"]))
        ml = f"{d.year}-{d.month:02d}"
        if ml not in att_slots_by_month:
            att_slots_by_month[ml] = {"slots": 0, "attended": 0}
        att_slots_by_month[ml]["slots"] += 1
        if str(att["attended"]) == "true":
            att_slots_by_month[ml]["attended"] += 1

    monthly_attendance_rate: dict[str, float] = {
        ml: round(data["attended"] / data["slots"], 4)
        for ml, data in sorted(att_slots_by_month.items())
        if data["slots"] > 0
    }

    # Program-wide rate: total attended / total slots
    total_slots = sum(d["slots"] for d in att_slots_by_month.values())
    total_attended = sum(d["attended"] for d in att_slots_by_month.values())
    program_wide_attendance_rate = (
        round(total_attended / total_slots, 4) if total_slots > 0 else 0.0
    )

    # -----------------------------------------------------------------------
    # Monthly cancellation rate
    # -----------------------------------------------------------------------
    canc_by_month: dict[str, dict[str, int]] = {}
    for sess in sessions:
        d = date.fromisoformat(str(sess["scheduled_date"]))
        ml = f"{d.year}-{d.month:02d}"
        canc_by_month.setdefault(ml, {"scheduled": 0, "cancelled": 0})
        canc_by_month[ml]["scheduled"] += 1
        status = str(sess["status"])
        if status.startswith("cancelled") or status == "no_show":
            canc_by_month[ml]["cancelled"] += 1

    monthly_cancellation_rate: dict[str, float] = {
        ml: round(data["cancelled"] / data["scheduled"], 4)
        for ml, data in sorted(canc_by_month.items())
        if data["scheduled"] > 0
    }

    # -----------------------------------------------------------------------
    # IEP attendance gap
    # -----------------------------------------------------------------------
    iep_attended = 0
    iep_total = 0
    non_iep_attended = 0
    non_iep_total = 0

    iep_lookup: dict[str, bool] = {
        str(st["student_id"]): str(st["iep"]) == "true" for st in students
    }
    for att in attendance:
        stuid = str(att["student_id"])
        is_iep = iep_lookup.get(stuid, False)
        if is_iep:
            iep_total += 1
            if str(att["attended"]) == "true":
                iep_attended += 1
        else:
            non_iep_total += 1
            if str(att["attended"]) == "true":
                non_iep_attended += 1

    iep_rate = round(iep_attended / iep_total, 4) if iep_total > 0 else None
    non_iep_rate = round(non_iep_attended / non_iep_total, 4) if non_iep_total > 0 else None
    iep_gap_pp: float | None = (
        round((non_iep_rate - iep_rate) * 100, 2)
        if iep_rate is not None and non_iep_rate is not None
        else None
    )

    # -----------------------------------------------------------------------
    # Suppressed low-N subgroups (from students table)
    # -----------------------------------------------------------------------
    race_counts: dict[str, int] = {}
    for st in students:
        rc = str(st.get("race_ethnicity") or "Unknown")
        race_counts[rc] = race_counts.get(rc, 0) + 1

    suppressed_subgroups = [
        {"subgroup_dimension": "race_ethnicity", "subgroup_value": rc, "n": n}
        for rc, n in sorted(race_counts.items())
        if n < spec.SUPPRESSION_THRESHOLD
    ]

    return {
        "pack_id": "pack_operations",
        "seed": seed,
        "note": (
            "All values computed from the generated data — not from spec.py input constants. "
            "B-task authors should assert these realized numbers, not the spec targets."
        ),
        "monthly_attendance_rate": monthly_attendance_rate,
        "program_wide_attendance_rate": program_wide_attendance_rate,
        "monthly_cancellation_rate": monthly_cancellation_rate,
        "iep_attendance_rate": iep_rate,
        "non_iep_attendance_rate": non_iep_rate,
        "iep_attendance_gap_pp": iep_gap_pp,
        "suppressed_low_n_subgroups": suppressed_subgroups,
    }


def _compute_ground_truth_equity(
    sessions: Table,
    students: Table,
    attendance: Table,
    subgroup_att: Table,
    subgroup_out: Table,
    seed: int,
) -> dict[str, Any]:
    """Compute the ground-truth manifest for the Equity & Research pack.

    Derives realized signal values from sessions, students, attendance, and the
    subgroup summary tables.  Includes the IEP attendance gap, suppressed low-N
    subgroups, and subgroup outcome disparities (IEP vs non-IEP proficiency gap).

    Args:
        sessions: Generated ``sessions`` rows.
        students: Generated ``students`` rows.
        attendance: Generated ``attendance`` rows.
        subgroup_att: Generated ``subgroup_attendance_summary`` rows.
        subgroup_out: Generated ``subgroup_outcomes_summary`` rows.
        seed: Integer seed used for this generation run.

    Returns:
        Dictionary suitable for ``ground_truth.json`` in pack_equity_research.
    """
    # -----------------------------------------------------------------------
    # Monthly attendance rate (per month, program-wide, computed from raw data)
    # -----------------------------------------------------------------------
    session_by_id: dict[str, Row] = {str(s["session_id"]): s for s in sessions}
    att_slots_by_month: dict[str, dict[str, int]] = {}

    for att in attendance:
        sess = session_by_id.get(str(att["session_id"]))
        if sess is None:
            continue
        d = date.fromisoformat(str(sess["scheduled_date"]))
        ml = f"{d.year}-{d.month:02d}"
        if ml not in att_slots_by_month:
            att_slots_by_month[ml] = {"slots": 0, "attended": 0}
        att_slots_by_month[ml]["slots"] += 1
        if str(att["attended"]) == "true":
            att_slots_by_month[ml]["attended"] += 1

    monthly_attendance_rate: dict[str, float] = {
        ml: round(data["attended"] / data["slots"], 4)
        for ml, data in sorted(att_slots_by_month.items())
        if data["slots"] > 0
    }

    total_slots = sum(d["slots"] for d in att_slots_by_month.values())
    total_attended = sum(d["attended"] for d in att_slots_by_month.values())
    program_wide_attendance_rate = (
        round(total_attended / total_slots, 4) if total_slots > 0 else 0.0
    )

    # -----------------------------------------------------------------------
    # IEP attendance gap (from raw attendance + students)
    # -----------------------------------------------------------------------
    iep_attended = 0
    iep_total = 0
    non_iep_attended = 0
    non_iep_total = 0

    iep_lookup: dict[str, bool] = {
        str(st["student_id"]): str(st["iep"]) == "true" for st in students
    }
    for att in attendance:
        stuid = str(att["student_id"])
        is_iep = iep_lookup.get(stuid, False)
        if is_iep:
            iep_total += 1
            if str(att["attended"]) == "true":
                iep_attended += 1
        else:
            non_iep_total += 1
            if str(att["attended"]) == "true":
                non_iep_attended += 1

    iep_rate = round(iep_attended / iep_total, 4) if iep_total > 0 else None
    non_iep_rate = round(non_iep_attended / non_iep_total, 4) if non_iep_total > 0 else None
    iep_gap_pp: float | None = (
        round((non_iep_rate - iep_rate) * 100, 2)
        if iep_rate is not None and non_iep_rate is not None
        else None
    )

    # -----------------------------------------------------------------------
    # Suppressed low-N subgroups (from students table)
    # -----------------------------------------------------------------------
    race_counts: dict[str, int] = {}
    for st in students:
        rc = str(st.get("race_ethnicity") or "Unknown")
        race_counts[rc] = race_counts.get(rc, 0) + 1

    suppressed_subgroups = [
        {"subgroup_dimension": "race_ethnicity", "subgroup_value": rc, "n": n}
        for rc, n in sorted(race_counts.items())
        if n < spec.SUPPRESSION_THRESHOLD
    ]

    # -----------------------------------------------------------------------
    # Subgroup outcome disparities from subgroup_outcomes_summary
    # IEP vs non-IEP proficiency rate gap (spring_2026)
    # -----------------------------------------------------------------------
    iep_outcomes: dict[str, Any] = {}
    for row in subgroup_out:
        dim_match = str(row["subgroup_dimension"]) == "iep"
        period_match = str(row["assessment_period"]) == "spring_2026"
        if dim_match and period_match:
            val = row["benchmark_proficiency_rate"]
            iep_outcomes[str(row["subgroup_value"])] = (
                float(str(val)) if val != "" and val is not None else None
            )

    iep_prof = iep_outcomes.get("true")
    non_iep_prof = iep_outcomes.get("false")
    iep_proficiency_gap_pp: float | None = (
        round((non_iep_prof - iep_prof) * 100, 2)
        if iep_prof is not None and non_iep_prof is not None
        else None
    )

    # Collect all subgroup-outcome disparities where suppressed=false, dim=iep, spring
    subgroup_outcome_disparities: list[dict[str, Any]] = []
    for row in subgroup_out:
        if (
            str(row["subgroup_dimension"]) == "iep"
            and str(row["assessment_period"]) == "spring_2026"
            and str(row["suppressed"]) == "false"
        ):
            val = row["benchmark_proficiency_rate"]
            subgroup_outcome_disparities.append(
                {
                    "assessment_period": str(row["assessment_period"]),
                    "subgroup_dimension": str(row["subgroup_dimension"]),
                    "subgroup_value": str(row["subgroup_value"]),
                    "n_students": int(str(row["n_students"])),
                    "benchmark_proficiency_rate": (
                        float(str(val)) if val != "" and val is not None else None
                    ),
                }
            )

    return {
        "pack_id": "pack_equity_research",
        "seed": seed,
        "note": (
            "All values computed from the generated data — not from spec.py input constants. "
            "B-task authors should assert these realized numbers, not the spec targets."
        ),
        "monthly_attendance_rate": monthly_attendance_rate,
        "program_wide_attendance_rate": program_wide_attendance_rate,
        "iep_attendance_rate": iep_rate,
        "non_iep_attendance_rate": non_iep_rate,
        "iep_attendance_gap_pp": iep_gap_pp,
        "suppressed_low_n_subgroups": suppressed_subgroups,
        "iep_proficiency_gap_pp_spring": iep_proficiency_gap_pp,
        "subgroup_outcome_disparities": subgroup_outcome_disparities,
    }


def _compute_ground_truth(
    monthly_att: Table,
    monthly_sat: Table,
    sessions: Table,
    students: Table,
    attendance: Table,
    seed: int,
) -> dict[str, Any]:
    """Compute the ground-truth manifest from actual generated data.

    Derives realized signal values from the generated tables so that B-task
    authors can assert exact numbers.  All values are computed from the data,
    not from ``spec.py`` input constants.

    Args:
        monthly_att: Generated ``monthly_attendance_summary`` rows.
        monthly_sat: Generated ``monthly_satisfaction_summary`` rows.
        sessions: Generated ``sessions`` rows.
        students: Generated ``students`` rows.
        attendance: Generated ``attendance`` rows.
        seed: Integer seed used for this generation run.

    Returns:
        Dictionary suitable for ``ground_truth.json``.
    """
    # -----------------------------------------------------------------------
    # Attendance: per-month program-wide rates
    # -----------------------------------------------------------------------
    att_by_month: dict[str, list[float]] = {}
    for row in monthly_att:
        ml = str(row["month_label"])
        val = row["attendance_rate"]
        if val != "" and val is not None:
            att_by_month.setdefault(ml, []).append(float(str(val)))

    monthly_attendance_rate: dict[str, float] = {}
    for ml in sorted(att_by_month):
        rates = att_by_month[ml]
        # Weighted by school (equal weight here — schools differ in session count)
        monthly_attendance_rate[ml] = round(sum(rates) / len(rates), 4)

    # True program-wide rate: total attended / total slots across all schools/months
    total_slots = 0
    total_attended = 0
    for att in attendance:
        total_slots += 1
        if str(att["attended"]) == "true":
            total_attended += 1
    program_wide_attendance_rate = (
        round(total_attended / total_slots, 4) if total_slots > 0 else 0.0
    )

    # -----------------------------------------------------------------------
    # Monthly satisfaction: per survey_type per month
    # -----------------------------------------------------------------------
    monthly_satisfaction: dict[str, dict[str, Any]] = {}
    for row in monthly_sat:
        stype = str(row["survey_type"])
        ml = str(row["month_label"])
        avg_sc = row["avg_score"]
        monthly_satisfaction.setdefault(stype, {})[ml] = (
            float(str(avg_sc)) if avg_sc != "" and avg_sc is not None else None
        )

    # Summarize the student_satisfaction signal for the dip
    sat_months = sorted(spec.MONTHS)
    dip_month_label = sat_months[spec.SATISFACTION_DIP_MONTH]
    pre_dip_months = [m for i, m in enumerate(sat_months) if i < spec.SATISFACTION_DIP_MONTH]
    post_dip_months = [m for i, m in enumerate(sat_months) if i > spec.SATISFACTION_DIP_MONTH]

    student_sat = monthly_satisfaction.get("student_satisfaction", {})
    satisfaction_dip_realized: dict[str, Any] = {
        "dip_month": dip_month_label,
        "scores_by_month": {ml: student_sat.get(ml) for ml in sat_months},
        "dip_confirmed": (
            student_sat.get(dip_month_label) is not None
            and all(
                (student_sat.get(m) or 0) > (student_sat.get(dip_month_label) or 99)
                for m in pre_dip_months + post_dip_months
                if student_sat.get(m) is not None
            )
        ),
    }

    # -----------------------------------------------------------------------
    # Monthly cancellation rate
    # -----------------------------------------------------------------------
    canc_by_month: dict[str, dict[str, int]] = {}
    for sess in sessions:
        d = date.fromisoformat(str(sess["scheduled_date"]))
        ml = f"{d.year}-{d.month:02d}"
        canc_by_month.setdefault(ml, {"scheduled": 0, "cancelled": 0})
        canc_by_month[ml]["scheduled"] += 1
        status = str(sess["status"])
        if status.startswith("cancelled") or status == "no_show":
            canc_by_month[ml]["cancelled"] += 1

    monthly_cancellation_rate: dict[str, float] = {
        ml: round(data["cancelled"] / data["scheduled"], 4)
        for ml, data in sorted(canc_by_month.items())
        if data["scheduled"] > 0
    }

    # -----------------------------------------------------------------------
    # IEP attendance gap
    # -----------------------------------------------------------------------
    iep_attended = 0
    iep_total = 0
    non_iep_attended = 0
    non_iep_total = 0

    iep_lookup: dict[str, bool] = {
        str(st["student_id"]): str(st["iep"]) == "true" for st in students
    }
    for att in attendance:
        stuid = str(att["student_id"])
        is_iep = iep_lookup.get(stuid, False)
        if is_iep:
            iep_total += 1
            if str(att["attended"]) == "true":
                iep_attended += 1
        else:
            non_iep_total += 1
            if str(att["attended"]) == "true":
                non_iep_attended += 1

    iep_rate = round(iep_attended / iep_total, 4) if iep_total > 0 else None
    non_iep_rate = round(non_iep_attended / non_iep_total, 4) if non_iep_total > 0 else None
    iep_gap_pp: float | None = (
        round((non_iep_rate - iep_rate) * 100, 2)
        if iep_rate is not None and non_iep_rate is not None
        else None
    )

    # -----------------------------------------------------------------------
    # Suppressed low-N subgroups (from students table — AIAN specifically)
    # -----------------------------------------------------------------------
    race_counts: dict[str, int] = {}
    for st in students:
        rc = str(st.get("race_ethnicity") or "Unknown")
        race_counts[rc] = race_counts.get(rc, 0) + 1

    suppressed_subgroups = [
        {"subgroup_dimension": "race_ethnicity", "subgroup_value": rc, "n": n}
        for rc, n in sorted(race_counts.items())
        if n < spec.SUPPRESSION_THRESHOLD
    ]

    return {
        "pack_id": "pack_outcomes",
        "seed": seed,
        "note": (
            "All values computed from the generated data — not from spec.py input constants. "
            "B-task authors should assert these realized numbers, not the spec targets."
        ),
        "monthly_attendance_rate": monthly_attendance_rate,
        "program_wide_attendance_rate": program_wide_attendance_rate,
        "monthly_satisfaction": monthly_satisfaction,
        "satisfaction_dip_realized": satisfaction_dip_realized,
        "monthly_cancellation_rate": monthly_cancellation_rate,
        "iep_attendance_rate": iep_rate,
        "non_iep_attendance_rate": non_iep_rate,
        "iep_attendance_gap_pp": iep_gap_pp,
        "suppressed_low_n_subgroups": suppressed_subgroups,
    }


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

    # Ground-truth manifest: computed from actual generated data (not spec constants).
    ground_truth = _compute_ground_truth_operations(sessions, students, attendance, seed)

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
    _write_json(pack_dir / "ground_truth.json", ground_truth)

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

    # MoM tables: monthly_attendance uses the main RNG; monthly_satisfaction
    # uses a dedicated isolated RNG to preserve byte-identity of other packs.
    monthly_att = _compute_monthly_attendance(sessions, attendance, groups, rng)
    # NOTE: _gen_monthly_satisfaction_real uses random.Random(seed + 9999) internally —
    # it does NOT consume draws from the shared `rng` so operations/equity output is
    # unaffected.
    monthly_sat = _gen_monthly_satisfaction_real(seed)

    # Ground-truth manifest: computed from actual generated data (not spec constants).
    ground_truth = _compute_ground_truth(
        monthly_att, monthly_sat, sessions, students, attendance, seed
    )

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
    _write_json(pack_dir / "ground_truth.json", ground_truth)

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

    # Ground-truth manifest: computed from actual generated data (not spec constants).
    ground_truth = _compute_ground_truth_equity(
        sessions, students, attendance, subgroup_att, subgroup_out, seed
    )

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
    _write_json(pack_dir / "ground_truth.json", ground_truth)

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
