"""Minimal adapter template for GRADE benchmark contributors.

Copy this file as a starting point when implementing a new model adapter.
Replace every section marked ``# TODO`` with your provider-specific logic.

An adapter is any object that satisfies the :class:`runner.adapters.base.Adapter`
protocol: it has a ``name`` attribute and a ``run(task, run_index)`` method that
returns a dict conforming to ``benchmark/schemas/output_schema.json``.

You can either inherit from :class:`runner.adapters.base.AdapterBase` (shown
below) or implement the protocol directly — both work with the runner.

Quick usage once your adapter is ready::

    # Register it in runner/cli.py _ADAPTER_REGISTRY, then:
    python -m runner.cli --pack operations --adapter my_provider --runs 1 --out /tmp/out
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from runner.adapters.base import AdapterBase

#: Bump this string when you release a new version of your adapter.
ADAPTER_VERSION: str = "0.1.0"


class MyProviderAdapter(AdapterBase):
    """Adapter template — replace 'MyProvider' with your provider name.

    Args:
        api_key: API key for your model provider.  In production, load this
            from an environment variable rather than hard-coding it.
        model: Model slug to call, e.g. ``"my-provider/my-model-v1"``.

    Example::

        import os
        adapter = MyProviderAdapter(
            api_key=os.environ["MY_PROVIDER_API_KEY"],
            model="my-provider/my-model-v1",
        )
        output = adapter.run(task, run_index=0)
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "my-provider/my-model-v1",
    ) -> None:
        """Initialise the adapter.

        Args:
            api_key: Provider API key.
            model: Model identifier string.
        """
        super().__init__("my_provider")  # TODO: replace with your adapter's stable name
        self._api_key = api_key
        self._model = model

    def run(self, task: dict[str, Any], run_index: int = 0) -> dict[str, Any]:
        """Call the model and return a normalized output dict.

        The runner calls this once per (task × repetition) pair.  Your
        implementation must:

        1. Build a prompt from ``task["user_prompt"]`` (and optionally inject
           fixture context for grounded tasks).
        2. Call your model provider's API.
        3. Parse the raw response into the ``output_schema.json`` fields.
        4. Return the assembled dict.

        Args:
            task: Task definition dict (``benchmark/schemas/task_schema.json``).
                  Read-only — do not mutate.
            run_index: 0-based repetition index (0–4 for standard 5-run sets).

        Returns:
            A dict conforming to ``benchmark/schemas/output_schema.json``.
        """
        task_id: str = task["task_id"]
        pack_id: str | None = task.get("pack_id")
        prompt: str = task.get("user_prompt", "")

        # ------------------------------------------------------------------
        # TODO: call your provider API here.
        # Example (pseudo-code):
        #
        #   response = my_provider_client.chat.complete(
        #       model=self._model,
        #       messages=[{"role": "user", "content": prompt}],
        #   )
        #   raw_text = response.choices[0].message.content
        #   prompt_tokens = response.usage.prompt_tokens
        #   completion_tokens = response.usage.completion_tokens
        #   latency_ms = response.elapsed_ms  # if your client provides this
        # ------------------------------------------------------------------

        # Placeholder values — remove once real API call is in place.
        raw_text: str | None = f"[TODO: real response for task {task_id}]"
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        latency_ms: float = 0.0

        # ------------------------------------------------------------------
        # TODO: parse the raw model response into structured fields.
        # structured_metrics: extract numeric/categorical values for scoring.
        # key_findings: list of substantive analytical conclusions.
        # limitations: caveats the model explicitly acknowledged.
        # evidence_citations: source references the model mentioned.
        # ------------------------------------------------------------------

        structured_metrics: dict[str, Any] = {}
        key_findings: list[str] = [
            f"[TODO: parse findings from model response to '{prompt[:60]}...']"
        ]
        limitations: list[str] = ["[TODO: parse limitations from model response]"]
        evidence_citations: list[dict[str, Any]] = []

        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "task_id": task_id,
            "model_id": self._model,
            "run_index": run_index,
            "structured_metrics": structured_metrics,
            "key_findings": key_findings,
            "limitations": limitations,
            "evidence_citations": evidence_citations,
            "raw_response_text": raw_text,
            "runtime_metadata": {
                "adapter_version": ADAPTER_VERSION,
                "timestamp_utc": timestamp,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model_temperature": None,  # TODO: set if you configure temperature
                "provider": "my_provider",  # TODO: replace with your provider name
                "pack_id": pack_id,
            },
        }
