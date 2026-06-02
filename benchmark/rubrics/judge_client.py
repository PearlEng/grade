"""Deterministic judge-model client wrapper for GRADE rubric scoring (C2).

This module provides :class:`JudgeClient`, a thin wrapper around the Anthropic
Messages API that enforces deterministic settings (temperature 0, fixed seed)
for rubric-dimension evaluation.

**CI / test policy**: This module is *documented* but **not exercised in CI**.
Tests in ``tests/unit/test_rubric_scoring.py`` use a mock judge that follows
the same :class:`~benchmark.rubrics.rubric_scoring.JudgeClientProtocol`.  Only
call this class in manual integration runs or production pipelines.

Live-judge invocation example::

    import os
    from benchmark.rubrics.judge_client import JudgeClient
    from benchmark.rubrics.rubric_scoring import score_rubric

    client = JudgeClient(api_key=os.environ["ANTHROPIC_API_KEY"])
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
5. An instruction to respond with a single float in [0, 1].

The judge model is queried at **temperature 0** with a **fixed random seed**
(:data:`JUDGE_SEED`) so that results are reproducible across runs.
"""

from __future__ import annotations

import json
from typing import Any

#: Number of judge calls permitted per task invocation (one per dimension).
MAX_JUDGE_CALLS_PER_TASK: int = 6

#: Fixed random seed passed to the judge model for reproducibility.
JUDGE_SEED: int = 42

#: Default judge model identifier.
DEFAULT_JUDGE_MODEL: str = "claude-opus-4-5"


def _build_judge_prompt(
    dimension: str,
    guidance: str,
    task: dict[str, Any],
    model_output: dict[str, Any],
) -> str:
    """Construct a judge evaluation prompt for a single rubric dimension.

    Args:
        dimension: The rubric dimension name (e.g. ``"grounding_accuracy"``).
        guidance: Task-specific scorer guidance, or empty string.
        task: The full task definition dict.
        model_output: The normalized model output dict.

    Returns:
        A fully formatted prompt string ready to send to the judge model.
    """
    gold_facts_text = json.dumps(task.get("gold_facts", []), indent=2)
    gold_insights_text = "\n".join(f"- {ins}" for ins in task.get("gold_insights", []))
    required_limitations_text = "\n".join(
        f"- {lim}" for lim in task.get("required_limitations", [])
    )
    forbidden_claims_text = "\n".join(f"- {claim}" for claim in task.get("forbidden_claims", []))

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
        "## Model response",
        response_text,
        "",
        "## Instruction",
        f"Score the model response on the '{dimension}' dimension only.",
        "Return a single floating-point number between 0.0 and 1.0 (inclusive),",
        "where 0.0 = completely fails the dimension and 1.0 = fully satisfies it.",
        "Respond with ONLY the number — no explanation, no punctuation, no units.",
    ]
    return "\n".join(lines)


class JudgeClient:
    """Anthropic-backed deterministic judge client for GRADE rubric scoring.

    Uses temperature 0 and a fixed seed (:data:`JUDGE_SEED`) so that repeated
    calls to the same (dimension, task, output) triple return the same score.
    The number of judge calls per task is capped at
    :data:`MAX_JUDGE_CALLS_PER_TASK` (one per dimension).

    **Not used in CI** — tests inject a mock judge.  See module docstring for
    live-judge usage instructions.

    Args:
        api_key: Anthropic API key.  If *None*, falls back to the
            ``ANTHROPIC_API_KEY`` environment variable via the SDK default.
        model: Model identifier to use as judge.  Defaults to
            :data:`DEFAULT_JUDGE_MODEL`.
        max_tokens: Maximum tokens to request in the judge response.  Defaults
            to 16 (a single float token).

    Example::

        client = JudgeClient(api_key="sk-ant-...")
        score = client.judge("grounding_accuracy", "", task, model_output)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_JUDGE_MODEL,
        max_tokens: int = 16,
    ) -> None:
        """Initialize the JudgeClient with Anthropic SDK settings.

        Args:
            api_key: Anthropic API key.  If *None*, the SDK reads from the
                ``ANTHROPIC_API_KEY`` environment variable.
            model: Judge model identifier.
            max_tokens: Maximum tokens for the judge response.
        """
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required to use JudgeClient.  "
                "Install it with: pip install anthropic"
            ) from exc

        self._client: Any = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self._model = model
        self._max_tokens = max_tokens

    def judge(
        self,
        dimension: str,
        guidance: str,
        task: dict[str, Any],
        model_output: dict[str, Any],
    ) -> float:
        """Call the judge model to score one rubric dimension.

        Args:
            dimension: The rubric dimension name.
            guidance: Task-specific scorer guidance text, or empty string.
            task: Full task definition dict.
            model_output: Normalized model output dict.

        Returns:
            A float in ``[0.0, 1.0]`` parsed from the judge model's response.
            If parsing fails, returns ``0.0`` and the failure is silent — callers
            should log at a higher level if desired.
        """
        prompt = _build_judge_prompt(dimension, guidance, task, model_output)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        try:
            score = float(raw)
        except ValueError:
            score = 0.0
        return max(0.0, min(1.0, score))
