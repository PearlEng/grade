"""Adapter interface and registry for GRADE runner model adapters.

Each adapter wraps one model provider (or a stub) and converts raw API
responses into the normalized ``output_schema.json`` shape consumed by the
GRADE scorers (C1–C4).

Exported names
--------------
- :class:`Adapter` — structural ``Protocol`` that every adapter must satisfy.
- :class:`AdapterBase` — optional ABC helpers (override ``name`` + ``run``).
"""

from __future__ import annotations

from runner.adapters.base import Adapter, AdapterBase

__all__ = ["Adapter", "AdapterBase"]
