"""Runner adapter registry for the GRADE benchmark CLI.

Adapters are resolved by *name* string (the ``--adapter`` CLI flag).  The
default adapter is ``"openrouter"``.

Registered adapters
-------------------
- ``"openrouter"`` *(default)* — :class:`~runner.adapters.openrouter_adapter.OpenRouterAdapter`:
  calls OpenRouter's unified chat-completions API; requires ``OPENROUTER_API_KEY``.
- ``"stub"`` — :class:`~runner.adapters.stub_adapter.StubAdapter`:
  returns minimal schema-valid output without any API call; useful for CI and
  pipeline smoke tests.

Exported names
--------------
- :class:`Adapter` — structural ``Protocol`` that every adapter must satisfy.
- :class:`AdapterBase` — optional ABC helpers (override ``name`` + ``run``).

Usage
-----
::

    from runner.adapters import get_adapter, ADAPTER_REGISTRY, DEFAULT_ADAPTER

    # Resolve by name (returns a class, not an instance):
    AdapterCls = get_adapter("openrouter")
    adapter = AdapterCls(model="anthropic/claude-sonnet-4.6")

    # Use the default adapter:
    DefaultCls = get_adapter(DEFAULT_ADAPTER)
"""

from __future__ import annotations

from runner.adapters.base import Adapter, AdapterBase
from runner.adapters.openrouter_adapter import OpenRouterAdapter
from runner.adapters.stub_adapter import StubAdapter

#: Mapping from adapter name string to adapter class.
ADAPTER_REGISTRY: dict[str, type] = {
    "openrouter": OpenRouterAdapter,
    "stub": StubAdapter,
}

#: Default adapter name used by the CLI when ``--adapter`` is not specified.
DEFAULT_ADAPTER: str = "openrouter"


def get_adapter(name: str) -> type:
    """Resolve an adapter class by name.

    Args:
        name: Adapter name string (e.g. ``"openrouter"``, ``"stub"``).

    Returns:
        The adapter class (not an instance) registered under *name*.

    Raises:
        KeyError: If *name* is not in :data:`ADAPTER_REGISTRY`.

    Example::

        AdapterCls = get_adapter("openrouter")
        adapter = AdapterCls(model="openai/gpt-5.5")
    """
    if name not in ADAPTER_REGISTRY:
        known = ", ".join(sorted(ADAPTER_REGISTRY))
        raise KeyError(f"Unknown adapter {name!r}.  Known adapters: {known}")
    return ADAPTER_REGISTRY[name]


__all__ = [
    "ADAPTER_REGISTRY",
    "DEFAULT_ADAPTER",
    "Adapter",
    "AdapterBase",
    "OpenRouterAdapter",
    "StubAdapter",
    "get_adapter",
]
