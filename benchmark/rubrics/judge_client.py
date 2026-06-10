"""Deterministic judge-model client wrapper for GRADE rubric scoring (C2).

This module provides :class:`JudgeClient`, a thin wrapper around the
**OpenRouter** chat-completions API that enforces deterministic settings
(temperature 0, fixed seed) for rubric-dimension evaluation.

The client reuses :func:`runner.adapters.openrouter_adapter.post_chat_completion`
so that HTTP configuration (base URL, headers, auth) is defined in exactly one
place.  A single ``OPENROUTER_API_KEY`` is the only credential required.

**CI / test policy**: This module is *documented* but **not exercised in CI**.
Tests in ``tests/unit/test_rubric_scoring.py`` use a mock judge that follows
the same :class:`~benchmark.rubrics.rubric_scoring.JudgeClientProtocol`.  Only
call this class in manual integration runs or production pipelines.

Live-judge invocation example::

    import os
    from benchmark.rubrics.judge_client import JudgeClient
    from benchmark.rubrics.rubric_scoring import score_rubric

    client = JudgeClient(api_key=os.environ["OPENROUTER_API_KEY"])
    result = score_rubric(task, model_output, judge_client=client)
    print(result["composite"])

Judge prompt format
-------------------
For each dimension the client constructs a prompt that includes:

1. A system header identifying the judge's role.
2. The dimension name and any task-specific scorer guidance.
3. The task's ``user_prompt`` and key rubric material (gold facts, gold
   insights, required limitations, forbidden claims).
4. The full model output text (``raw_response_text`` if present, else the
   ``key_findings`` and ``limitations`` lists joined as prose).
5. An instruction to give a brief (2-3 sentence) justification, then a
   final ``SCORE: <float>`` line with the score in [0, 1]
   (rationale-then-score is measurably more reliable than bare-float
   judging).

The judge model is queried at **temperature 0** so that results are
reproducible across runs.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Upper bound of judge calls per score_rubric invocation (one per scored
#: dimension; the runner judges only the three C2-owned dimensions).
MAX_JUDGE_CALLS_PER_TASK: int = 6

#: Task fields that contain raw fixture data or internal dispatcher keys.
#: These are never relevant to the judge's evaluation and are explicitly
#: excluded from the judge prompt to prevent prompt-size blowout when a
#: 4000-row CSV is inlined by the dispatcher.
_TASK_FIELDS_EXCLUDED_FROM_JUDGE: frozenset[str] = frozenset(
    {
        "fixtures",  # {filename: full_csv_text} — can be megabytes
        "pack_id",  # internal dispatcher routing key, not a rubric concept
    }
)

#: Fixed random seed — kept for documentation / future use; OpenRouter does
#: not expose a ``seed`` parameter on all models, so reproducibility relies on
#: ``temperature=0`` instead.
JUDGE_SEED: int = 42

#: Default judge model identifier (OpenRouter slug).
#:
#: Judge-selection policy (decided 2026-06-10, see
#: ``docs/methodology_review_findings.md`` C-3): the strongest available
#: model judges everyone — Claude Opus 4.8 — EXCEPT when the candidate under
#: test is itself a Claude-family model, in which case GPT-5.5 at xhigh
#: reasoning effort judges instead.  No judge ever shares a model family with
#: its candidate, so house style cannot bias the judged dimensions.  Use
#: :func:`select_judge_model` to apply this policy; ``--judge-model`` on both
#: CLIs overrides it.
#:
#: NOTE: current OpenRouter slugs use dots in version numbers
#: (``claude-opus-4.8``), not dashes — the dash form is rejected by the API.
DEFAULT_JUDGE_MODEL: str = "anthropic/claude-opus-4.8"

#: Judge used when the candidate model is a Claude-family model (so a Claude
#: judge never grades a Claude candidate).  GPT-5.5 is a reasoning model, so
#: judge calls for it must send a reasoning-effort hint and a much larger
#: token budget.
CLAUDE_CANDIDATE_JUDGE_MODEL: str = "openai/gpt-5.5"

#: Reasoning effort used for the :data:`CLAUDE_CANDIDATE_JUDGE_MODEL` judge.
CLAUDE_CANDIDATE_JUDGE_REASONING_EFFORT: str = "xhigh"

#: max_tokens for reasoning-model judges.  Reasoning tokens and the visible
#: justification + SCORE line share this budget, so it must be large.
REASONING_JUDGE_MAX_TOKENS: int = 8192

#: Substrings identifying a Claude-family candidate (checked lowercase).
_CLAUDE_FAMILY_MARKERS: tuple[str, ...] = ("claude", "anthropic/", "opus", "sonnet", "haiku")


def select_judge_model(candidate_model_id: str | None) -> tuple[str, str | None]:
    """Pick the judge for *candidate_model_id* per the cross-family policy.

    The strongest model (Claude Opus 4.8) judges every candidate, except
    Claude-family candidates — those are judged by GPT-5.5 at xhigh reasoning
    effort, so no judge ever shares a family (and hence a house style) with
    the model it grades.

    Args:
        candidate_model_id: The model under test (shorthand or OpenRouter
            slug, e.g. ``"claude-opus-4-8"`` or ``"anthropic/claude-opus-4.8"``).
            ``None`` or empty selects the default judge.

    Returns:
        A ``(judge_model_slug, reasoning_effort)`` tuple ready to construct a
        :class:`JudgeClient`.  *reasoning_effort* is ``None`` for
        non-reasoning judges.
    """
    candidate = (candidate_model_id or "").lower()
    if any(marker in candidate for marker in _CLAUDE_FAMILY_MARKERS):
        return CLAUDE_CANDIDATE_JUDGE_MODEL, CLAUDE_CANDIDATE_JUDGE_REASONING_EFFORT
    return DEFAULT_JUDGE_MODEL, None


class JudgeScoreError(RuntimeError):
    """Raised when the judge model's response cannot be parsed as a score.

    Raised only after the configured retry has also failed.  Surfacing this
    as an exception (rather than silently scoring 0.0) ensures a judge-side
    failure is recorded as a task failure in ``failures.json`` instead of
    being charged against the candidate model's score.
    """


#: Regex that extracts a float (e.g. ``0.85``, ``1``, ``.5``) from a judge
#: reply, tolerating stray punctuation, markdown, or label prefixes.
_FLOAT_PATTERN = re.compile(r"-?(?:\d+\.?\d*|\.\d+)")

#: Regex for the canonical final ``SCORE: <float>`` line requested by the
#: judge prompt.  Matched case-insensitively and tolerant of markdown
#: emphasis around the label (e.g. ``**SCORE: 0.85**``).
_SCORE_LINE_PATTERN = re.compile(r"SCORE\s*[:=]\s*(-?(?:\d+\.?\d*|\.\d+))", re.IGNORECASE)

#: Default max_tokens for non-reasoning judges.  The rationale-then-score
#: protocol needs room for 2-3 sentences of justification plus the final
#: ``SCORE:`` line.
JUDGE_MAX_TOKENS: int = 384


def _parse_judge_score(raw: str) -> float | None:
    """Extract the judge's score from *raw* response text.

    Prefers the last ``SCORE: <float>`` line (the prompt asks for it as the
    final line; "last" guards against the justification quoting the format).
    Falls back to the last bare float in the text — under rationale-then-
    score, trailing numbers are far more likely to be the verdict than
    numbers quoted from the response being judged.

    Args:
        raw: Stripped judge response text.

    Returns:
        The parsed score, or ``None`` if no float is present at all.
    """
    score_matches = _SCORE_LINE_PATTERN.findall(raw)
    if score_matches:
        return float(score_matches[-1])
    float_matches = _FLOAT_PATTERN.findall(raw)
    if float_matches:
        return float(float_matches[-1])
    return None


def _build_judge_prompt(
    dimension: str,
    guidance: str,
    task: dict[str, Any],
    model_output: dict[str, Any],
) -> str:
    """Construct a judge evaluation prompt for a single rubric dimension.

    The prompt is built exclusively from the task fields that are relevant to
    rubric evaluation: ``task_id``, ``title``, ``user_prompt``, ``gold_facts``,
    ``gold_insights``, ``required_limitations``, ``forbidden_claims``, and
    ``reference_answer_outline``.

    Raw fixture data (``fixtures``, ``pack_id``) is **always excluded** —
    grounding accuracy is the job of the deterministic C1 scorer, not the
    judge.  Excluding bulk data keeps the prompt bounded regardless of fixture
    size (e.g. a 4 000-row attendance.csv would otherwise balloon the prompt
    to tens of thousands of tokens and cause the OpenRouter call to hang).

    Args:
        dimension: The rubric dimension name (e.g. ``"grounding_accuracy"``).
        guidance: Task-specific scorer guidance, or empty string.
        task: The full task definition dict.  Fields listed in
            :data:`_TASK_FIELDS_EXCLUDED_FROM_JUDGE` are ignored.
        model_output: The normalized model output dict.

    Returns:
        A fully formatted prompt string ready to send to the judge model.
    """
    # Strip bulk/internal fields before touching any task values so the prompt
    # size is bounded regardless of how large the inlined fixture data is.
    # This guard is intentionally placed here (not only at the call-site) so
    # the exclusion is enforced even if new callers pass a raw task dict.
    task = {k: v for k, v in task.items() if k not in _TASK_FIELDS_EXCLUDED_FROM_JUDGE}

    gold_facts_text = json.dumps(task.get("gold_facts", []), indent=2)
    gold_insights_text = "\n".join(f"- {ins}" for ins in task.get("gold_insights", []))
    required_limitations_text = "\n".join(
        f"- {lim}" for lim in task.get("required_limitations", [])
    )
    forbidden_claims_text = "\n".join(f"- {claim}" for claim in task.get("forbidden_claims", []))
    reference_outline = task.get("reference_answer_outline", "")

    # Build a readable representation of the model's response.
    if model_output.get("raw_response_text"):
        response_text = model_output["raw_response_text"]
    else:
        findings = model_output.get("key_findings", [])
        limitations = model_output.get("limitations", [])
        findings_text = "\n".join(findings)
        limits_text = ("\n\nLimitations:\n" + "\n".join(limitations)) if limitations else ""
        response_text = findings_text + limits_text

    guidance_section = f"\nScorer guidance: {guidance}" if guidance else ""

    task_id = task.get("task_id", "unknown")
    user_prompt = task.get("user_prompt", "")

    lines = [
        "You are a rigorous benchmark judge evaluating an AI model's response "
        "on the GRADE educational-data benchmark.",
        "",
        "## Task",
        f"Task ID: {task_id}",
        f"User prompt: {user_prompt}",
        "",
        f"## Rubric dimension: {dimension}{guidance_section}",
        "",
        "## Reference material",
        "Gold facts:",
        gold_facts_text,
        "",
        "Gold insights:",
        gold_insights_text,
        "",
        "Required limitations:",
        required_limitations_text,
        "",
        "Forbidden claims:",
        forbidden_claims_text,
        "",
    ]
    if reference_outline:
        lines += [
            "Reference answer outline:",
            reference_outline,
            "",
        ]
    lines += [
        "## Model response",
        response_text,
        "",
        "## Instruction",
        f"Score the model response on the '{dimension}' dimension only.",
        "First, briefly justify your assessment in 2-3 sentences, referring to",
        "the reference material above.",
        "Then, on the final line, output exactly:",
        "SCORE: <number>",
        "where <number> is a floating-point number between 0.0 and 1.0",
        "(inclusive), 0.0 = completely fails the dimension and 1.0 = fully",
        "satisfies it.",
    ]
    return "\n".join(lines)


class JudgeClient:
    """OpenRouter-backed deterministic judge client for GRADE rubric scoring.

    Uses temperature 0 so that repeated calls to the same
    (dimension, task, output) triple return the same score.
    The number of judge calls per task is capped at
    :data:`MAX_JUDGE_CALLS_PER_TASK` (one per dimension).

    **Not used in CI** — tests inject a mock judge.  See module docstring for
    live-judge usage instructions.

    Accumulated cost tracking
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Each :meth:`judge` call updates the following instance attributes so that
    the single :class:`JudgeClient` passed through a full benchmark run totals
    all judge-model API overhead:

    - :attr:`cumulative_cost_usd` (``float | None``): total USD across all
      :meth:`judge` calls; ``None`` if the provider has never reported cost.
    - :attr:`cumulative_prompt_tokens` (``int``): total prompt tokens (0 if
      never reported).
    - :attr:`cumulative_completion_tokens` (``int``): total completion tokens.
    - :attr:`judge_call_count` (``int``): number of :meth:`judge` calls made.

    Args:
        api_key: OpenRouter API key.  If *None*, falls back to the
            ``OPENROUTER_API_KEY`` environment variable.
        model: OpenRouter model slug to use as judge.  Defaults to
            :data:`DEFAULT_JUDGE_MODEL`.
        max_tokens: Maximum tokens to request in the judge response.  Defaults
            to :data:`JUDGE_MAX_TOKENS` (room for a short justification plus
            the final ``SCORE:`` line), or
            :data:`REASONING_JUDGE_MAX_TOKENS` when *reasoning_effort* is set
            (reasoning tokens share the budget with the visible answer).
        reasoning_effort: Optional reasoning effort (e.g. ``"xhigh"``) for
            reasoning-model judges; forwarded to OpenRouter as
            ``{"reasoning": {"effort": ...}}``.

    Example::

        client = JudgeClient(api_key="sk-or-...")
        score = client.judge("grounding_accuracy", "", task, model_output)
        print(client.cumulative_cost_usd)

        # Per the no-self-judging policy:
        judge_model, effort = select_judge_model("anthropic/claude-opus-4.8")
        client = JudgeClient(model=judge_model, reasoning_effort=effort)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_JUDGE_MODEL,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Initialize the JudgeClient with OpenRouter settings.

        Args:
            api_key: OpenRouter API key.  If *None*, reads from the
                ``OPENROUTER_API_KEY`` environment variable at call time.
            model: OpenRouter model slug for the judge.
            max_tokens: Maximum tokens for the judge response.  ``None``
                (default) resolves to :data:`JUDGE_MAX_TOKENS`, or
                :data:`REASONING_JUDGE_MAX_TOKENS` when *reasoning_effort*
                is set.
            reasoning_effort: Optional reasoning effort level for
                reasoning-model judges (``None`` omits the field).
        """
        self._api_key = api_key
        self._model = model
        self._reasoning_effort = reasoning_effort
        if max_tokens is not None:
            self._max_tokens = max_tokens
        elif reasoning_effort is not None:
            self._max_tokens = REASONING_JUDGE_MAX_TOKENS
        else:
            self._max_tokens = JUDGE_MAX_TOKENS

        # Accumulated judge-cost state — updated after every judge() call.
        #: Total cost in USD across all judge() calls; None until the provider
        #: reports at least one cost value.
        self.cumulative_cost_usd: float | None = None
        #: Total prompt tokens across all judge() calls (0 if never reported).
        self.cumulative_prompt_tokens: int = 0
        #: Total completion tokens across all judge() calls (0 if never reported).
        self.cumulative_completion_tokens: int = 0
        #: Number of judge() calls made since this instance was created.
        self.judge_call_count: int = 0

    def judge(
        self,
        dimension: str,
        guidance: str,
        task: dict[str, Any],
        model_output: dict[str, Any],
    ) -> float:
        """Call the judge model to score one rubric dimension.

        Uses :func:`runner.adapters.openrouter_adapter.post_chat_completion`
        so that the HTTP logic lives in one place.  Each call accumulates
        usage/cost into the instance-level counters
        (:attr:`cumulative_cost_usd`, :attr:`cumulative_prompt_tokens`,
        :attr:`cumulative_completion_tokens`, :attr:`judge_call_count`).

        Args:
            dimension: The rubric dimension name.
            guidance: Task-specific scorer guidance text, or empty string.
            task: Full task definition dict.
            model_output: Normalized model output dict.

        Returns:
            A float in ``[0.0, 1.0]`` parsed from the judge model's response.
            Parsing extracts the first float in the reply (tolerating stray
            punctuation or markdown).  If no float can be extracted, the API
            call is retried once; if the retry also fails,
            :exc:`JudgeScoreError` is raised so the failure surfaces as a
            task failure rather than silently scoring the candidate 0.0.

        Raises:
            JudgeScoreError: If the judge response contains no parseable
                float after one retry.
        """
        # Lazy import so this module is importable without httpx installed.
        from runner.adapters.openrouter_adapter import post_chat_completion

        prompt = _build_judge_prompt(dimension, guidance, task, model_output)

        last_raw: str = ""
        for _attempt in range(2):  # initial call + one retry on parse failure
            result = post_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                temperature=0.0,
                max_tokens=self._max_tokens,
                api_key=self._api_key,
                return_usage=True,
                reasoning_effort=self._reasoning_effort,
            )
            # post_chat_completion with return_usage=True always returns a tuple.
            raw, usage = result
            last_raw = raw.strip()

            # Accumulate usage into instance state.
            self.judge_call_count += 1
            if usage.cost_usd is not None:
                self.cumulative_cost_usd = (self.cumulative_cost_usd or 0.0) + usage.cost_usd
            if usage.prompt_tokens is not None:
                self.cumulative_prompt_tokens += usage.prompt_tokens
            if usage.completion_tokens is not None:
                self.cumulative_completion_tokens += usage.completion_tokens

            score = _parse_judge_score(last_raw)
            if score is not None:
                return max(0.0, min(1.0, score))

        raise JudgeScoreError(
            f"Judge model {self._model!r} returned no parseable score for "
            f"dimension {dimension!r} after retry.  Last response: {last_raw!r}"
        )
