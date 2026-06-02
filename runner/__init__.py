"""GRADE generic runner package (C5).

The runner is responsible for loading task packs, dispatching to model
adapters, scoring outputs, and writing results to disk.

Sub-modules
-----------
- :mod:`runner.adapters` — adapter interface (Protocol) and stub adapter.
- :mod:`runner.dispatcher` — task execution loop and C1/C2/C4 scoring.
- :mod:`runner.aggregator` — per-track/pack/overall result aggregation.
- :mod:`runner.io` — disk I/O helpers for raw outputs and result JSON.
- :mod:`runner.cli` — CLI entry point (``python -m runner.cli``).
"""
