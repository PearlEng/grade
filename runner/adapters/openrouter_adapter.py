"""OpenRouter adapter for the GRADE benchmark runner (C6).

This adapter calls the OpenRouter chat-completions API (OpenAI-compatible,
base URL ``https://openrouter.ai/api/v1``) and maps the response into the
normalized ``output_schema.json`` contract.

A single ``OPENROUTER_API_KEY`` covers GPT, Claude, Gemini, Llama, Mistral,
and 100+ other models — the whole point of OpenRouter is to eliminate per-
provider API key management.

Seed model identifiers
----------------------
The mapping below lists the five seed models used in the GRADE arena (F4).
Pass the *key* as ``--model`` on the CLI; the adapter will resolve it to the
correct OpenRouter slug automatically.

.. code-block:: python

    from runner.adapters.openrouter_adapter import SEED_MODELS
    print(SEED_MODELS)
    # {
    #   "claude-opus-4-7":    "anthropic/claude-opus-4-5",
    #   "claude-sonnet-4-6":  "anthropic/claude-sonnet-4-5",
    #   "claude-haiku-4-5":   "anthropic/claude-haiku-4-5",
    #   "gpt-5":              "openai/gpt-4o",
    #   "gemini-2.5-pro":     "google/gemini-pro-1.5",
    # }

Usage
-----
Minimal::

    import os
    from runner.adapters.openrouter_adapter import OpenRouterAdapter

    adapter = OpenRouterAdapter(model="anthropic/claude-sonnet-4-5")
    output = adapter.run(task, run_index=0)

The adapter reads ``OPENROUTER_API_KEY`` from the environment if *api_key* is
not supplied explicitly.

HTTP client
-----------
Uses :mod:`httpx` (lazy-imported inside methods) for typed, sync HTTP without
pulling in any vendor SDK.  Install the ``openrouter`` optional dependency
group to get it::

    pip install -e ".[openrouter]"

If ``httpx`` is absent (e.g. in CI running ``.[dev]`` only), importing this
module still works — the ``ImportError`` is deferred until the first
:meth:`run` call.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import time
from typing import Any

#: Adapter semantic version — recorded in every output's ``runtime_metadata``.
ADAPTER_VERSION: str = "0.1.0"

#: OpenRouter API base URL.
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

#: HTTP-Referer header value sent with every request (OpenRouter attribution).
HTTP_REFERER: str = "https://github.com/PearlEng/grade"

#: X-Title header value sent with every request (OpenRouter attribution).
X_TITLE: str = "GRADE Benchmark"

#: Seed model shorthand → OpenRouter model slug mapping.
#:
#: These are the five models seeded in the GRADE arena (ticket F4).  Pass a
#: shorthand string as the ``model`` constructor argument and the adapter
#: resolves it automatically; you may also supply any raw OpenRouter slug
#: directly (e.g. ``"anthropic/claude-3-5-sonnet"``) and it will be used
#: verbatim.
SEED_MODELS: dict[str, str] = {
    # Claude family (via Anthropic on OpenRouter)
    "claude-opus-4-7": "anthropic/claude-opus-4-5",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-5",
    "claude-haiku-4-5": "anthropic/claude-haiku-4-5",
    # OpenAI
    "gpt-5": "openai/gpt-4o",
    # Google
    "gemini-2.5-pro": "google/gemini-pro-1.5",
}


def _resolve_model(model: str) -> str:
    """Resolve a model shorthand or slug to the full OpenRouter model string.

    If *model* is a key in :data:`SEED_MODELS`, returns the corresponding
    slug.  Otherwise returns *model* unchanged (allows raw slugs like
    ``"mistralai/mistral-7b-instruct"``).

    Args:
        model: Shorthand key or raw OpenRouter model slug.

    Returns:
        The fully qualified OpenRouter model string.
    """
    return SEED_MODELS.get(model, model)


def _build_prompt(task: dict[str, Any]) -> str:
    """Construct the user prompt to send to the model.

    Includes the task's ``user_prompt`` plus a description of the allowed
    fixture inputs and a structured-output instruction so the model returns
    a JSON block that the adapter can parse.

    Args:
        task: Validated task definition dict.

    Returns:
        A fully formatted prompt string.
    """
    user_prompt = task.get("user_prompt", "")
    allowed_inputs = task.get("allowed_inputs", [])
    inputs_list = ", ".join(allowed_inputs) if allowed_inputs else "(none)"

    lines = [
        user_prompt,
        "",
        f"Available data sources: {inputs_list}",
        "",
        "---",
        "Please structure your response as follows:",
        "",
        "## Key Findings",
        "List your main analytical conclusions as bullet points.",
        "",
        "## Limitations",
        "List any caveats, uncertainties, or data limitations.",
        "",
        "## Evidence Citations",
        "List any explicit references to source data (file names, columns, etc.).",
        "",
        "## Structured Metrics",
        "If the task involves specific numeric or categorical values, list them as:",
        "metric_name: value",
    ]
    return "\n".join(lines)


def _parse_response(raw_text: str, task: dict[str, Any]) -> dict[str, Any]:
    """Parse the model's raw response text into structured output fields.

    Extracts key findings, limitations, evidence citations, and structured
    metrics from the model's response using section-based parsing.

    Args:
        raw_text: Verbatim text returned by the model.
        task: Task definition dict (used as fallback for ``task_id``).

    Returns:
        A dict with keys ``key_findings``, ``limitations``,
        ``evidence_citations``, and ``structured_metrics``.
    """
    key_findings: list[str] = []
    limitations: list[str] = []
    evidence_citations: list[dict[str, Any]] = []
    structured_metrics: dict[str, Any] = {}

    # Split on section headers (## Heading or **Heading**).
    section_pattern = re.compile(
        r"(?:^|\n)(?:#{1,3}\s*|(?:\*\*))([^\n*]+?)(?:\*\*)?(?:\n|$)",
        re.IGNORECASE,
    )

    sections: dict[str, str] = {}
    matches = list(section_pattern.finditer(raw_text))

    for i, match in enumerate(matches):
        section_name = match.group(1).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        sections[section_name] = raw_text[start:end].strip()

    def _extract_bullets(text: str) -> list[str]:
        """Extract non-empty lines, stripping leading bullet characters."""
        items: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip common bullet markers
            line = re.sub(r"^[-*•·]\s*", "", line)
            line = re.sub(r"^\d+[.)]\s*", "", line)
            if len(line) >= 5:
                items.append(line)
        return items

    # Parse Key Findings section.
    for header in ("key findings", "findings", "main findings", "key finding"):
        if header in sections:
            key_findings = _extract_bullets(sections[header])
            break

    # Parse Limitations section.
    for header in ("limitations", "caveats", "limitation", "caveat"):
        if header in sections:
            limitations = _extract_bullets(sections[header])
            break

    # Parse Evidence Citations section.
    for header in ("evidence citations", "citations", "evidence", "sources"):
        if header in sections:
            for line in _extract_bullets(sections[header]):
                evidence_citations.append(
                    {
                        "citation_text": line,
                        "source_file": None,
                        "source_column": None,
                        "grounded": None,
                    }
                )
            break

    # Parse Structured Metrics section.
    for header in ("structured metrics", "metrics", "structured metric"):
        if header in sections:
            for line in sections[header].splitlines():
                line = line.strip()
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip().lower().replace(" ", "_")
                    val = val.strip()
                    if key and val:
                        # Attempt numeric coercion.
                        try:
                            structured_metrics[key] = int(val)
                        except ValueError:
                            try:
                                structured_metrics[key] = float(val)
                            except ValueError:
                                structured_metrics[key] = val
            break

    # Fallback: if no sections were found, treat entire response as one finding.
    if not key_findings and raw_text.strip():
        # Use the first non-empty line / paragraph as a single finding.
        first_para = next((p.strip() for p in raw_text.split("\n\n") if p.strip()), "")
        if len(first_para) >= 5:
            key_findings = [first_para[:500]]

    return {
        "key_findings": key_findings,
        "limitations": limitations,
        "evidence_citations": evidence_citations,
        "structured_metrics": structured_metrics,
    }


class OpenRouterAdapter:
    """OpenRouter-backed adapter for the GRADE benchmark runner.

    Calls the OpenRouter chat-completions endpoint (OpenAI-compatible API)
    and maps the response into the normalized ``output_schema.json`` shape.

    The adapter is registered as ``"openrouter"`` in the runner CLI — pass
    ``--adapter openrouter`` to select it.

    Args:
        model: OpenRouter model slug or :data:`SEED_MODELS` shorthand.
            Defaults to ``"anthropic/claude-sonnet-4-5"`` (Sonnet 4.6 on
            OpenRouter).
        api_key: OpenRouter API key.  If ``None``, falls back to the
            ``OPENROUTER_API_KEY`` environment variable.  Raises
            :exc:`EnvironmentError` at run time if neither is set.
        temperature: Sampling temperature.  Defaults to ``0.0`` for
            reproducibility.
        max_tokens: Maximum tokens to request from the model.  When ``None``
            (the default), falls back to the ``GRADE_OPENROUTER_MAX_TOKENS``
            environment variable if set, otherwise uses
            :attr:`DEFAULT_MAX_TOKENS` (1024).

    Example::

        adapter = OpenRouterAdapter(model="claude-sonnet-4-6")
        output = adapter.run(task, run_index=0)
    """

    #: CLI id — ``--adapter openrouter``.
    name: str = "openrouter"

    #: Default maximum tokens — lower than the original 2048 to reduce cost.
    #: Override via ``GRADE_OPENROUTER_MAX_TOKENS`` env var or the constructor arg.
    DEFAULT_MAX_TOKENS: int = 1024

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-5",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        """Initialise the adapter.

        Args:
            model: OpenRouter model slug or shorthand key.
            api_key: API key; falls back to ``OPENROUTER_API_KEY`` env var.
            temperature: Sampling temperature for the model call.
            max_tokens: Maximum tokens in the model response.  When ``None``
                (the default), the value is resolved from the
                ``GRADE_OPENROUTER_MAX_TOKENS`` environment variable if set,
                otherwise :attr:`DEFAULT_MAX_TOKENS` (1024) is used.
        """
        self._model_input = model
        self._model = _resolve_model(model)
        self._api_key = api_key  # resolved lazily in run() to support env var
        self._temperature = temperature
        if max_tokens is not None:
            self._max_tokens = max_tokens
        else:
            env_val = os.environ.get("GRADE_OPENROUTER_MAX_TOKENS")
            self._max_tokens = int(env_val) if env_val else self.DEFAULT_MAX_TOKENS

    def _get_api_key(self) -> str:
        """Resolve and return the API key.

        Returns:
            The API key string.

        Raises:
            EnvironmentError: If no API key is set via constructor or env var.
        """
        key = self._api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise OSError(
                "OpenRouter API key is required.  Set OPENROUTER_API_KEY in your "
                "environment or pass api_key= to OpenRouterAdapter()."
            )
        return key

    def run(self, task: dict[str, Any], run_index: int = 0) -> dict[str, Any]:
        """Run *task* against the OpenRouter model and return normalized output.

        Makes a single synchronous HTTP POST to the OpenRouter chat-completions
        endpoint, captures latency and token counts from the response, then
        maps the raw text into the ``output_schema.json`` shape.

        Args:
            task: Validated task definition dict (``task_schema.json``).
            run_index: 0-based repetition index (0–4 for GRADE's 5-run
                protocol).

        Returns:
            A dict valid against ``benchmark/schemas/output_schema.json``.

        Raises:
            EnvironmentError: If no ``OPENROUTER_API_KEY`` is available.
            ImportError: If ``httpx`` is not installed (install
                ``pip install -e ".[openrouter]"``).
            RuntimeError: If the OpenRouter API returns a non-2xx status.
        """
        try:
            import httpx  # lazy import — absent in CI without openrouter extra
        except ImportError as exc:
            raise ImportError(
                "httpx is required to use OpenRouterAdapter.  "
                "Install it with: pip install -e '.[openrouter]'"
            ) from exc

        api_key = self._get_api_key()
        prompt = _build_prompt(task)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": X_TITLE,
        }

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        t_start = time.monotonic()
        response = httpx.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120.0,
        )
        latency_ms = (time.monotonic() - t_start) * 1000.0

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API returned HTTP {response.status_code}: {response.text[:500]}"
            )

        now = datetime.datetime.now(datetime.UTC)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        body: dict[str, Any] = response.json()
        choice = body["choices"][0]
        raw_text: str = choice["message"]["content"] or ""

        usage: dict[str, Any] = body.get("usage", {})
        prompt_tokens: int | None = usage.get("prompt_tokens")
        completion_tokens: int | None = usage.get("completion_tokens")

        # OpenRouter may include cost in the response body.
        cost_usd: float | None = None
        if "usage" in body and "cost" in body["usage"]:
            try:
                cost_usd = float(body["usage"]["cost"])
            except (TypeError, ValueError):
                cost_usd = None

        parsed = _parse_response(raw_text, task)

        # Derive provider from the model slug (prefix before first '/').
        provider = self._model.split("/")[0] if "/" in self._model else "openrouter"

        runtime_metadata: dict[str, Any] = {
            "adapter_version": ADAPTER_VERSION,
            "timestamp_utc": timestamp,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model_temperature": self._temperature,
            "provider": provider,
            "pack_id": None,
        }
        if cost_usd is not None:
            # Store cost in structured_metrics since output_schema has no cost field.
            parsed["structured_metrics"]["cost_usd"] = cost_usd

        return {
            "task_id": task["task_id"],
            "model_id": self._model,
            "run_index": run_index,
            "structured_metrics": parsed["structured_metrics"],
            "key_findings": parsed["key_findings"],
            "limitations": parsed["limitations"],
            "evidence_citations": parsed["evidence_citations"],
            "runtime_metadata": runtime_metadata,
            "raw_response_text": raw_text,
        }


def _make_openrouter_http_client(api_key: str | None = None) -> Any:
    """Create a pre-configured httpx client for the OpenRouter API.

    This is a shared factory used by both the adapter and :mod:`judge_client`
    to avoid duplicating header configuration.

    Args:
        api_key: OpenRouter API key.  Falls back to ``OPENROUTER_API_KEY``
            env var if ``None``.

    Returns:
        An ``httpx.Client`` configured with the correct base URL and headers.

    Raises:
        EnvironmentError: If no API key is available.
        ImportError: If ``httpx`` is not installed.
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required.  Install it with: pip install -e '.[openrouter]'"
        ) from exc

    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise OSError("OPENROUTER_API_KEY is required but not set.")

    return httpx.Client(
        base_url=OPENROUTER_BASE_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": X_TITLE,
        },
        timeout=60.0,
    )


def post_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 16,
    api_key: str | None = None,
) -> str:
    """Send a chat-completion request to OpenRouter and return the response text.

    Convenience function used by :mod:`benchmark.rubrics.judge_client` so that
    the judge can share the same OpenRouter HTTP logic as the adapter without
    importing the whole :class:`OpenRouterAdapter`.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        model: OpenRouter model slug.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        api_key: Optional API key; falls back to ``OPENROUTER_API_KEY`` env var.

    Returns:
        The model's response text (first choice content).

    Raises:
        EnvironmentError: If no API key is available.
        ImportError: If ``httpx`` is not installed.
        RuntimeError: If the API returns a non-2xx status.
    """
    # Validate API key before attempting the HTTP import so the error message
    # is clear when the key is missing (even if httpx is also absent).
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise OSError("OPENROUTER_API_KEY is required but not set.")

    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required.  Install it with: pip install -e '.[openrouter]'"
        ) from exc

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = httpx.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": X_TITLE,
        },
        json=payload,
        timeout=60.0,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API returned HTTP {response.status_code}: {response.text[:500]}"
        )

    body: dict[str, Any] = response.json()
    return str(body["choices"][0]["message"]["content"] or "")


# ---------------------------------------------------------------------------
# JSON serialization helper (used in tests / CLI)
# ---------------------------------------------------------------------------


def _default_json_serializer(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default.

    Args:
        obj: Object to serialize.

    Returns:
        A JSON-serializable representation.
    """
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def output_to_json(output: dict[str, Any]) -> str:
    """Serialize a normalized output dict to a JSON string.

    Args:
        output: Normalized output dict valid against ``output_schema.json``.

    Returns:
        Compact JSON string.
    """
    return json.dumps(output, default=_default_json_serializer, ensure_ascii=False)
