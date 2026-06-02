"""Generic adapter interface (Protocol) for GRADE model providers.

Any object that satisfies :class:`Adapter` can be plugged into the runner
without changes to the runner core.  Concrete providers (OpenRouter C6,
Anthropic C7, …) implement this interface; tests use
:mod:`runner.adapters.stub_adapter`.

Design note
-----------
The interface is intentionally minimal: one method (``run``) and one
attribute (``name``).  All provider-specific concerns — authentication,
rate-limiting, retry logic, token counting — live inside the concrete
adapter, not here.

Example::

    from runner.adapters.base import Adapter

    def run_once(adapter: Adapter, task: dict) -> dict:
        return adapter.run(task)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Adapter(Protocol):
    """Structural protocol that every GRADE model adapter must satisfy.

    An adapter is responsible for:

    1. Receiving a task definition dict (``task_schema.json``).
    2. Calling the underlying model provider with the task's ``user_prompt``
       (and any fixture context the adapter chooses to inject).
    3. Returning a dict that validates against ``output_schema.json``.

    The runner calls :meth:`run` once per (task × repetition) pair and passes
    the result directly to the scorers.  The adapter must set ``run_index``
    correctly for each call.

    Attributes:
        name: Stable, human-readable identifier for this adapter, e.g.
            ``"openrouter"`` or ``"stub"``.  Used in result metadata and
            CLI ``--adapter`` flag lookup.
    """

    name: str

    def run(self, task: dict[str, Any], run_index: int = 0) -> dict[str, Any]:
        """Execute the model for *task* and return a normalized output dict.

        Args:
            task: A task definition dict validated against
                ``benchmark/schemas/task_schema.json``.  The adapter may
                read any field but must not mutate the dict.
            run_index: 0-based repetition index within the current
                (model × task) run set.  The adapter must embed this value
                in the returned output's ``run_index`` field.

        Returns:
            A dict conforming to ``benchmark/schemas/output_schema.json``.
            The runner validates the returned dict before passing it to
            scorers; returning an invalid dict will raise
            :exc:`jsonschema.ValidationError`.

        Raises:
            NotImplementedError: If the adapter subclass does not override
                this method (only relevant for :class:`AdapterBase`).
            Exception: Concrete adapters may raise provider-specific errors
                (network timeouts, auth failures, etc.); the runner will
                propagate them to the caller.
        """
        ...


class AdapterBase:
    """Optional ABC-style base class providing default ``__init__`` scaffolding.

    Concrete adapters may inherit from :class:`AdapterBase` or simply
    implement the :class:`Adapter` ``Protocol`` directly — both work with the
    runner.

    Subclasses must:

    - Set ``self.name`` to a stable, unique identifier string.
    - Override :meth:`run` to call the model provider and return a valid
      ``output_schema.json`` dict.

    Args:
        name: Adapter name string.

    Example::

        class MyAdapter(AdapterBase):
            def __init__(self) -> None:
                super().__init__("my_provider")

            def run(self, task: dict, run_index: int = 0) -> dict:
                ...
    """

    def __init__(self, name: str) -> None:
        """Initialise with an adapter *name*.

        Args:
            name: Stable identifier for this adapter, e.g. ``"openrouter"``.
        """
        self.name: str = name

    def run(self, task: dict[str, Any], run_index: int = 0) -> dict[str, Any]:
        """Execute the model for *task*.

        Must be overridden by concrete subclasses.

        Args:
            task: Task definition dict.
            run_index: 0-based repetition index.

        Returns:
            A dict conforming to ``output_schema.json``.

        Raises:
            NotImplementedError: Always — subclasses must override.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement run()")
