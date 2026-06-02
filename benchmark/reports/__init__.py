"""GRADE benchmark report generators (C9).

This package provides the scorecard report generator that aggregates per-task
dimension scores from C1–C4 scorers into per-track and per-pack aggregates,
computes the weighted overall composite, and renders human-readable Markdown and
structured JSON summaries.

Public API::

    from benchmark.reports.scorecard import generate_scorecard_report, ScorecardReport

The :class:`~benchmark.reports.scorecard.ScorecardReport` dataclass holds both
rendered artefacts; :func:`~benchmark.reports.scorecard.generate_scorecard_report`
is the single entry-point for callers.
"""
