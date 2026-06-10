"""OpenRouter adapter for the GRADE benchmark runner (C6).

This adapter calls the OpenRouter chat-completions API (OpenAI-compatible,
base URL ``https://openrouter.ai/api/v1``) and maps the response into the
normalized ``output_schema.json`` contract.

A single ``OPENROUTER_API_KEY`` covers GPT, Claude, Gemini, Llama, Mistral,
and 100+ other models — the whole point of OpenRouter is to eliminate per-
provider API key management.

Seed model identifiers
----------------------
The mapping below lists the launch leaderboard models.  Pass the *key* as
``--model`` on the CLI; the adapter will resolve it to the correct OpenRouter
slug automatically.  Every slug is verified against the live OpenRouter
``/api/v1/models`` listing — see ``docs/methodology_review_findings.md`` for
the verification date and full lineup rationale.

.. code-block:: python

    from runner.adapters.openrouter_adapter import SEED_MODELS
    print(SEED_MODELS["claude-sonnet-4-6"])
    # "anthropic/claude-sonnet-4.6"

Usage
-----
Minimal::

    import os
    from runner.adapters.openrouter_adapter import OpenRouterAdapter

    adapter = OpenRouterAdapter(model="anthropic/claude-sonnet-4.6")
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
from typing import Any, Literal, overload

#: Adapter semantic version — recorded in every output's ``runtime_metadata``.
ADAPTER_VERSION: str = "0.1.0"

#: OpenRouter API base URL.
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

#: HTTP-Referer header value sent with every request (OpenRouter attribution).
HTTP_REFERER: str = "https://github.com/PearlEng/grade"

#: X-Title header value sent with every request (OpenRouter attribution).
X_TITLE: str = "GRADE Benchmark"

#: Default request timeout in seconds for all OpenRouter HTTP calls.
#: Override via the ``GRADE_OPENROUTER_TIMEOUT`` environment variable.
#: Set to a value that is long enough for large completions but short enough
#: to fail-fast rather than block an entire benchmark run indefinitely.
DEFAULT_OPENROUTER_TIMEOUT: float = 120.0


def _get_openrouter_timeout() -> float:
    """Return the configured OpenRouter request timeout in seconds.

    Reads the ``GRADE_OPENROUTER_TIMEOUT`` environment variable.  If unset or
    unparseable, falls back to :data:`DEFAULT_OPENROUTER_TIMEOUT` (120 s).

    Returns:
        Timeout in seconds as a :class:`float`.
    """
    raw = os.environ.get("GRADE_OPENROUTER_TIMEOUT", "")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_OPENROUTER_TIMEOUT


#: Seed model shorthand → OpenRouter model slug mapping.
#:
#: These are the launch leaderboard models.  Pass a shorthand string as the
#: ``model`` constructor argument and the adapter resolves it automatically;
#: you may also supply any raw OpenRouter slug directly (e.g.
#: ``"anthropic/claude-sonnet-4.6"``) and it will be used verbatim.
#:
#: Every slug below was verified against the live OpenRouter
#: ``/api/v1/models`` listing on 2026-06-10.  Note that current OpenRouter
#: slugs use dots in version numbers (``claude-opus-4.8``), not dashes.
SEED_MODELS: dict[str, str] = {
    # Anthropic
    "claude-opus-4-8": "anthropic/claude-opus-4.8",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
    # OpenAI
    "gpt-5-5": "openai/gpt-5.5",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    # Google
    "gemini-3-5-flash": "google/gemini-3.5-flash",
    "gemini-3-1-pro": "google/gemini-3.1-pro-preview",
    # NVIDIA
    "nemotron-3-ultra": "nvidia/nemotron-3-ultra-550b-a55b",
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

    Includes the task's ``user_prompt``, the full contents of every fixture
    file referenced by ``allowed_inputs`` (embedded verbatim under clearly
    delimited file headers), and a prose-first instruction that encourages the
    model to write a clear, natural analysis rather than forcing rigid
    structured output.

    The model is asked to write as a knowledgeable program analyst would —
    natural prose with supporting numbers — and structured sections are offered
    as *optional* scaffolding rather than hard requirements.  The response
    parser can extract key findings from any prose response, so structural
    compliance is not required for a high score.

    When the task dict contains a ``fixtures`` key (a ``{filename: contents}``
    mapping populated by the dispatcher), each file's full contents are
    embedded in the prompt so the model has the actual data to reason over
    rather than just filenames.

    Args:
        task: Validated task definition dict, optionally enriched with a
            ``fixtures`` key by the dispatcher.

    Returns:
        A fully formatted prompt string with fixture data inlined.
    """
    user_prompt = task.get("user_prompt", "")
    allowed_inputs = task.get("allowed_inputs", [])
    fixtures: dict[str, str] = task.get("fixtures", {})

    lines: list[str] = [user_prompt, ""]

    if fixtures:
        lines += [
            "The following data files are provided in full for you to analyse.",
            "Base your answer exclusively on the data below — do not assume or",
            "fabricate any values.",
            "",
        ]
        for filename in allowed_inputs:
            if filename in fixtures:
                lines += [
                    f"=== FILE: {filename} ===",
                    fixtures[filename],
                    f"=== END FILE: {filename} ===",
                    "",
                ]
        # List any allowed_inputs that had no fixture data (graceful fallback).
        missing = [f for f in allowed_inputs if f not in fixtures]
        if missing:
            missing_list = ", ".join(missing)
            lines += [
                f"Note: the following referenced files were not available: {missing_list}",
                "",
            ]
    else:
        # No fixture data — fall back to listing filenames only.
        inputs_list = ", ".join(allowed_inputs) if allowed_inputs else "(none)"
        lines += [f"Available data sources: {inputs_list}", ""]

    lines += [
        "---",
        "Please write a clear, natural analysis as a knowledgeable program analyst would.",
        "Use plain prose — complete sentences and paragraphs — to explain what the data",
        "shows.  Embed specific numbers and rates directly in your narrative wherever they",
        "support your points.",
        "",
        "You may optionally use the section headings below as a loose guide, but you are",
        "not required to follow them rigidly.  What matters is that your response is",
        "accurate, grounded in the data, and easy for a non-technical reader to follow.",
        "",
        "Suggested structure (optional):",
        "  - A short summary of the main findings, written as prose.",
        "  - Any important caveats or limitations of the data.",
        "  - References to the specific files or columns that support your conclusions.",
        "  - If it is helpful, a brief list of key numeric values (e.g. rates, counts).",
    ]
    return "\n".join(lines)


def _parse_response(raw_text: str, task: dict[str, Any]) -> dict[str, Any]:
    """Parse the model's raw response text into structured output fields.

    Designed for prose-first responses: the parser can extract ``key_findings``
    from plain paragraphs even when the model emits no section headers at all.
    Structured sections (``## Key Findings``, ``## Limitations``, etc.) are
    parsed when present but are not required.

    Parsing strategy
    ~~~~~~~~~~~~~~~~
    1. Section detection: split on ``## Heading`` or ``**Heading**`` markers.
    2. If recognised section headers are found, extract their content.
    3. **Prose fallback for key_findings**: if no findings were extracted via
       sections, split the response into paragraphs/sentences and use every
       substantive block (>= 10 characters) as a finding.  This ensures that a
       model that writes flowing prose without any ``## Key Findings`` header
       still produces a non-empty ``key_findings`` list for the deterministic
       scorer to evaluate.
    4. ``structured_metrics``: extracted only from an explicit
       ``## Structured Metrics`` / ``## Metrics`` section (best-effort).  An
       empty dict ``{}`` is a valid value per ``output_schema.json``.
    5. ``limitations``: extracted from the explicit section.  Falls back to an
       empty list (schema permits ``minItems: 0``).

    Args:
        raw_text: Verbatim text returned by the model.
        task: Task definition dict (unused currently; reserved for future use).

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
    for header in ("key findings", "findings", "main findings", "key finding", "summary"):
        if header in sections:
            key_findings = _extract_bullets(sections[header])
            break

    # Parse Limitations section.
    for header in ("limitations", "caveats", "limitation", "caveat"):
        if header in sections:
            limitations = _extract_bullets(sections[header])
            break

    # Parse Evidence Citations section.
    for header in (
        "evidence citations",
        "citations",
        "evidence",
        "sources",
        "references",
        "data sources",
    ):
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

    # Parse Structured Metrics section (best-effort; empty dict is valid).
    for header in (
        "structured metrics",
        "metrics",
        "structured metric",
        "key numeric values",
        "numeric values",
    ):
        if header in sections:
            for line in sections[header].splitlines():
                line = line.strip()
                # Strip bullet markers before key:value parsing.
                line = re.sub(r"^[-*•·]\s*", "", line)
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

    # ------------------------------------------------------------------
    # Prose fallback for key_findings
    #
    # When the model writes flowing prose without structured headers, split
    # the response into paragraphs (double-newline separated) and then into
    # individual sentences so the deterministic scorer has text to scan.
    # This is the core change for prose-first support: a clean prose response
    # with no section headers will still produce a populated key_findings list.
    # ------------------------------------------------------------------
    if not key_findings and raw_text.strip():
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        for para in paragraphs:
            # Split paragraph into sentences on ". ", "! ", or "? " boundaries.
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) >= 10:
                    key_findings.append(sentence)
        # Hard cap to keep the list manageable.
        key_findings = key_findings[:50]

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
            Defaults to ``"anthropic/claude-sonnet-4.6"`` (Sonnet 4.6 on
            OpenRouter).
        api_key: OpenRouter API key.  If ``None``, falls back to the
            ``OPENROUTER_API_KEY`` environment variable.  Raises
            :exc:`EnvironmentError` at run time if neither is set.
        temperature: Sampling temperature.  Defaults to ``1.0`` so that
            repeated runs vary, making the C4 consistency dimension
            meaningful.  Pass ``0.0`` explicitly for fully deterministic
            inference (e.g. debugging or reproducing a specific output).
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

    #: Default maximum tokens.  The prompt asks for a multi-section prose
    #: analysis of full CSV fixtures; 1024 (the old default) routinely
    #: truncated responses — and limitations sections come last, so the cut
    #: silently deflated calibration scores.  Override via the
    #: ``GRADE_OPENROUTER_MAX_TOKENS`` env var or the constructor arg.
    DEFAULT_MAX_TOKENS: int = 4096

    #: Default maximum tokens when a reasoning effort is set.  Reasoning
    #: tokens and the visible analysis share this budget, so reasoning models
    #: need substantially more headroom than :attr:`DEFAULT_MAX_TOKENS` — at
    #: high effort the model can otherwise burn the whole budget thinking and
    #: return an empty visible response.
    DEFAULT_REASONING_MAX_TOKENS: int = 16384

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4.6",
        api_key: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Initialise the adapter.

        Args:
            model: OpenRouter model slug or shorthand key.  May carry an
                ``@<effort>`` suffix (e.g. ``"openai/gpt-5.5@xhigh"``) to set
                the reasoning effort inline — the suffix is stripped from the
                API slug but kept in the reported ``model_id`` so leaderboard
                rows for different effort levels don't collide.
            api_key: API key; falls back to ``OPENROUTER_API_KEY`` env var.
            temperature: Sampling temperature for the model call.  Defaults
                to ``1.0`` so that repeated benchmark runs vary, making the
                C4 consistency dimension meaningful.
            max_tokens: Maximum tokens in the model response.  When ``None``
                (the default), the value is resolved from the
                ``GRADE_OPENROUTER_MAX_TOKENS`` environment variable if set,
                otherwise :attr:`DEFAULT_MAX_TOKENS` (4096) — or
                :attr:`DEFAULT_REASONING_MAX_TOKENS` (16384) when a reasoning
                effort is in play, since reasoning tokens share the budget.
            reasoning_effort: Optional reasoning effort level (``"low"``,
                ``"medium"``, ``"high"``, ``"xhigh"``) forwarded to
                OpenRouter as ``{"reasoning": {"effort": ...}}``.  Takes
                precedence over an ``@<effort>`` suffix in *model*.
        """
        self._model_input = model
        base_model, _, suffix_effort = model.partition("@")
        self._reasoning_effort = reasoning_effort or (suffix_effort or None)
        self._model = _resolve_model(base_model)
        #: model_id reported in outputs — includes the effort label so results
        #: for the same slug at different efforts stay distinct.
        self._model_label = (
            f"{self._model}@{self._reasoning_effort}" if self._reasoning_effort else self._model
        )
        self._api_key = api_key  # resolved lazily in run() to support env var
        self._temperature = temperature
        if max_tokens is not None:
            self._max_tokens = max_tokens
        else:
            env_val = os.environ.get("GRADE_OPENROUTER_MAX_TOKENS")
            if env_val:
                self._max_tokens = int(env_val)
            elif self._reasoning_effort is not None:
                self._max_tokens = self.DEFAULT_REASONING_MAX_TOKENS
            else:
                self._max_tokens = self.DEFAULT_MAX_TOKENS

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
        if self._reasoning_effort is not None:
            payload["reasoning"] = {"effort": self._reasoning_effort}

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
        finish_reason: str | None = choice.get("finish_reason")

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
            "cost_usd": cost_usd,
            "model_temperature": self._temperature,
            "provider": provider,
            "pack_id": None,
            # "length" means the response hit max_tokens and was truncated —
            # the dispatcher stamps a 'truncated_output' scorer flag from this.
            "finish_reason": finish_reason,
        }

        return {
            "task_id": task["task_id"],
            "model_id": self._model_label,
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


class ChatCompletionUsage:
    """Lightweight container for usage/cost data from a single chat-completion call.

    Attributes:
        prompt_tokens: Number of prompt tokens, or ``None`` if not reported.
        completion_tokens: Number of completion tokens, or ``None`` if not
            reported.
        cost_usd: Call cost in USD, or ``None`` if not reported by the provider.
    """

    __slots__ = ("prompt_tokens", "completion_tokens", "cost_usd")

    def __init__(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost_usd = cost_usd


@overload
def post_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = ...,
    max_tokens: int = ...,
    api_key: str | None = ...,
    return_usage: Literal[False] = ...,
    reasoning_effort: str | None = ...,
) -> str: ...


@overload
def post_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = ...,
    max_tokens: int = ...,
    api_key: str | None = ...,
    *,
    return_usage: Literal[True],
    reasoning_effort: str | None = ...,
) -> tuple[str, ChatCompletionUsage]: ...


@overload
def post_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = ...,
    max_tokens: int = ...,
    api_key: str | None = ...,
    return_usage: bool = ...,
    reasoning_effort: str | None = ...,
) -> str | tuple[str, ChatCompletionUsage]: ...


def post_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 16,
    api_key: str | None = None,
    return_usage: bool = False,
    reasoning_effort: str | None = None,
) -> str | tuple[str, ChatCompletionUsage]:
    """Send a chat-completion request to OpenRouter and return the response text.

    Convenience function used by :mod:`benchmark.rubrics.judge_client` so that
    the judge can share the same OpenRouter HTTP logic as the adapter without
    importing the whole :class:`OpenRouterAdapter`.

    The request timeout is read from the ``GRADE_OPENROUTER_TIMEOUT``
    environment variable at call time (default: :data:`DEFAULT_OPENROUTER_TIMEOUT`
    seconds, i.e. 120 s).  When the timeout is exceeded, a :exc:`TimeoutError`
    is raised with a clear message so the runner's per-task resilience can
    record the failure and continue, rather than blocking the whole run.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        model: OpenRouter model slug.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        api_key: Optional API key; falls back to ``OPENROUTER_API_KEY`` env var.
        return_usage: When ``True``, return a ``(text, ChatCompletionUsage)``
            tuple instead of just the response text.  Existing callers that do
            not pass this flag are unaffected (backward-compatible).
        reasoning_effort: Optional reasoning effort level (e.g. ``"low"``,
            ``"medium"``, ``"high"``, ``"xhigh"``) forwarded to OpenRouter as
            ``{"reasoning": {"effort": ...}}``.  Only meaningful for
            reasoning-capable models; ``None`` (default) omits the field.
            NOTE: for reasoning models, *max_tokens* bounds reasoning +
            visible output combined, so callers must budget accordingly.

    Returns:
        When *return_usage* is ``False`` (default): the model's response text.
        When *return_usage* is ``True``: a ``(text, ChatCompletionUsage)`` tuple
        where :class:`ChatCompletionUsage` holds ``prompt_tokens``,
        ``completion_tokens``, and ``cost_usd`` (any may be ``None`` when the
        provider does not report them).

    Raises:
        EnvironmentError: If no API key is available.
        ImportError: If ``httpx`` is not installed.
        RuntimeError: If the API returns a non-2xx status.
        TimeoutError: If the request exceeds the configured timeout.
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
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}

    timeout = _get_openrouter_timeout()

    try:
        response = httpx.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": HTTP_REFERER,
                "X-Title": X_TITLE,
            },
            json=payload,
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise TimeoutError(
            f"OpenRouter request timed out after {timeout:.0f}s "
            f"(model={model!r}). "
            f"Increase GRADE_OPENROUTER_TIMEOUT to allow more time, "
            f"or investigate network/model availability."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API returned HTTP {response.status_code}: {response.text[:500]}"
        )

    body: dict[str, Any] = response.json()
    text = str(body["choices"][0]["message"]["content"] or "")

    if not return_usage:
        return text

    # Extract usage from the response body for callers that opt in.
    usage_raw: dict[str, Any] = body.get("usage", {})
    prompt_tokens: int | None = usage_raw.get("prompt_tokens")
    completion_tokens: int | None = usage_raw.get("completion_tokens")
    cost_usd: float | None = None
    if "cost" in usage_raw:
        try:
            cost_usd = float(usage_raw["cost"])
        except (TypeError, ValueError):
            cost_usd = None

    return text, ChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )


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
