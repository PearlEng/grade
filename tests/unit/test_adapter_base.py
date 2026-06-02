"""Unit tests for runner.adapters.base — Adapter Protocol and AdapterBase.

Coverage:
- :class:`~runner.adapters.base.Adapter` is a ``runtime_checkable`` Protocol.
- :class:`~runner.adapters.base.AdapterBase` exposes ``name`` and raises
  ``NotImplementedError`` for the default ``run`` implementation.
- A custom concrete class satisfies the Protocol.
"""

from __future__ import annotations

from typing import Any

import pytest

from runner.adapters.base import Adapter, AdapterBase


class _ConcreteAdapter(AdapterBase):
    """Minimal concrete adapter for testing."""

    def __init__(self) -> None:
        super().__init__("test_adapter")

    def run(self, task: dict[str, Any], run_index: int = 0) -> dict[str, Any]:
        """Return a trivially valid output stub.

        Args:
            task: Task dict (unused).
            run_index: Repetition index (unused).

        Returns:
            An empty dict (not schema-valid; just for interface testing).
        """
        return {}


class TestAdapterBase:
    """Tests for :class:`AdapterBase`."""

    def test_name_set_correctly(self) -> None:
        """AdapterBase should expose the name passed to __init__."""
        base = _ConcreteAdapter()
        assert base.name == "test_adapter"

    def test_concrete_satisfies_protocol(self) -> None:
        """A class that overrides run() must satisfy the Adapter Protocol."""
        adapter = _ConcreteAdapter()
        assert isinstance(adapter, Adapter)

    def test_base_run_raises_not_implemented(self) -> None:
        """The default AdapterBase.run() must raise NotImplementedError."""

        class _IncompleteAdapter(AdapterBase):
            pass

        adapter = _IncompleteAdapter("incomplete")
        with pytest.raises(NotImplementedError):
            adapter.run({})

    def test_protocol_is_runtime_checkable(self) -> None:
        """Adapter must be @runtime_checkable so isinstance() works."""
        adapter = _ConcreteAdapter()
        assert isinstance(adapter, Adapter)

    def test_arbitrary_object_without_run_fails_protocol(self) -> None:
        """An object without run() must not satisfy the Protocol."""

        class _NoRun:
            name = "no_run"

        obj = _NoRun()
        assert not isinstance(obj, Adapter)
