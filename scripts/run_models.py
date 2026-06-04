r"""run_models — multi-model batch runner for the GRADE benchmark.

Runs one or more models over a task pack and produces a ranked leaderboard.

Usage::

    python -m scripts.run_models \\
        --models anthropic/claude-sonnet-4-5,anthropic/claude-haiku-3 \\
        --pack operations \\
        --adapter openrouter \\
        --runs 5 \\
        --out results/

Output layout::

    results/
      <safe_model_slug>/
        result.json          # per-model scorecard
        raw_outputs.jsonl    # per-run raw outputs
        failures.json        # present only when some tasks failed
      leaderboard.json       # aggregated N-way leaderboard
      leaderboard.md         # rendered Markdown table

Per-model resilience
--------------------
If a whole model errors (e.g. auth failure, adapter crash), the batch records
the failure in a ``"failed_models"`` list and continues to the next model.
The leaderboard is built from whichever models succeeded.  A non-zero exit
code is returned when at least one model failed.

Judge cost
----------
When ``--judge`` is supplied, a **fresh** :class:`~benchmark.rubrics.judge_client.JudgeClient`
is constructed for each model.  This ensures that ``judge_cost_usd`` in each
per-model result reflects only that model's judge overhead, not a cumulative
total across all models.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from benchmark.reports.leaderboard import build_leaderboard, render_leaderboard_markdown
from runner.run import run_pack

# ---------------------------------------------------------------------------
# Constants — mirrors cli._PACK_PATHS / _PACK_IDS
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[1]

_PACK_PATHS: dict[str, Path] = {
    "operations": _REPO_ROOT / "benchmark" / "tasks" / "operations_pack.jsonl",
    "outcomes": _REPO_ROOT / "benchmark" / "tasks" / "outcomes_pack.jsonl",
    "equity_research": _REPO_ROOT / "benchmark" / "tasks" / "equity_research_pack.jsonl",
}

_PACK_IDS: dict[str, str] = {
    "operations": "pack_operations",
    "outcomes": "pack_outcomes",
    "equity_research": "pack_equity_research",
}

_ADAPTER_REGISTRY: dict[str, type] = {}  # populated lazily below


def _get_adapter_registry() -> dict[str, type]:
    """Return the adapter registry (imported lazily to avoid circular deps)."""
    global _ADAPTER_REGISTRY
    if not _ADAPTER_REGISTRY:
        from runner.adapters.openrouter_adapter import OpenRouterAdapter
        from runner.adapters.stub_adapter import StubAdapter

        _ADAPTER_REGISTRY = {
            "openrouter": OpenRouterAdapter,
            "stub": StubAdapter,
        }
    return _ADAPTER_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_slug(model_id: str) -> str:
    """Convert a model ID to a filesystem-safe directory name.

    Replaces any character that is not alphanumeric, ``-``, ``_``, or ``.``
    with ``_`` and collapses consecutive underscores.

    Args:
        model_id: Raw model identifier, e.g. ``"anthropic/claude-haiku-3"``.

    Returns:
        Sanitised slug, e.g. ``"anthropic_claude-haiku-3"``.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", model_id)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown_model"


def _resolve_pack(
    pack_arg: str,
) -> tuple[Path, str | None]:
    """Resolve a pack name or path to ``(pack_path, pack_id)``.

    Args:
        pack_arg: Short pack name or absolute/relative file path.

    Returns:
        ``(resolved_path, pack_id)`` where *pack_id* may be ``None``.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    if pack_arg in _PACK_PATHS:
        return _PACK_PATHS[pack_arg], _PACK_IDS[pack_arg]

    resolved = Path(pack_arg)
    if not resolved.is_absolute():
        resolved = _REPO_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Pack file not found: {resolved}")
    return resolved.resolve(), None


def _build_adapter(
    adapter_name: str,
    model: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Instantiate an adapter for *model* from the registry.

    Args:
        adapter_name: Registry key, e.g. ``"openrouter"`` or ``"stub"``.
        model: Model slug forwarded to the adapter constructor.
        temperature: Sampling temperature forwarded to the adapter.

    Returns:
        Instantiated adapter object.

    Raises:
        KeyError: If *adapter_name* is not registered.
    """
    registry = _get_adapter_registry()
    if adapter_name not in registry:
        known = ", ".join(sorted(registry))
        raise KeyError(f"Unknown adapter {adapter_name!r}.  Known: {known}")

    factory = registry[adapter_name]
    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature
    if kwargs:
        try:
            return factory(**kwargs)
        except TypeError:
            pass
    return factory()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_models",
        description=(
            "GRADE multi-model batch runner.  "
            "Runs each listed model over a task pack and produces a ranked leaderboard."
        ),
    )
    parser.add_argument(
        "--models",
        required=True,
        metavar="M1,M2,...",
        help=(
            "Comma-separated list of model IDs to evaluate, "
            "e.g. 'anthropic/claude-haiku-3,anthropic/claude-sonnet-4-5'."
        ),
    )
    parser.add_argument(
        "--pack",
        required=True,
        metavar="NAME_OR_PATH",
        help=(
            "Task pack to evaluate.  Short name "
            "(operations | outcomes | equity_research) or path to a .jsonl file."
        ),
    )
    parser.add_argument(
        "--adapter",
        default="openrouter",
        metavar="ADAPTER_ID",
        help="Model adapter to use.  Default: 'openrouter'.  Also: 'stub'.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        metavar="N",
        help="Number of repetition runs per task per model (default: 5).",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Use a live OpenRouter-backed judge (C2/C3 rubric + claim validation). "
            "A fresh JudgeClient is constructed per model so judge costs are isolated. "
            "Requires OPENROUTER_API_KEY."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="TEMP",
        help="Sampling temperature (default: 1.0).  Ignored by stub adapter.",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help=(
            "Output root directory.  Per-model results land in "
            "<DIR>/<safe_model_slug>/result.json; "
            "leaderboard files are written to <DIR>/leaderboard.{md,json}."
        ),
    )
    parser.add_argument(
        "--grade-version",
        default="0.1.0",
        metavar="VERSION",
        help="GRADE benchmark version string (default: 0.1.0).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the multi-model batch runner.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).

    Returns:
        Exit code: 0 when all models succeeded, 1 when any model failed.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    models: list[str] = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("ERROR: --models must be a non-empty comma-separated list.", file=sys.stderr)
        return 1

    # Resolve pack path once (shared across models).
    try:
        pack_path, pack_id = _resolve_pack(args.pack)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(models)} model(s) over pack '{args.pack}' ({pack_path.name}).")
    print(f"Output root: {out_root}\n")

    # Collect (model_id, result) pairs for the leaderboard.
    leaderboard_results: list[tuple[str, dict[str, Any]]] = []
    failed_models: list[dict[str, str]] = []

    for model_id in models:
        print(f"=== Model: {model_id} ===")
        slug = _safe_slug(model_id)
        model_out_dir = out_root / slug

        # Build a fresh adapter for this model.
        try:
            adapter = _build_adapter(args.adapter, model=model_id, temperature=args.temperature)
        except (KeyError, Exception) as exc:  # noqa: BLE001
            print(f"  FAILED to build adapter: {exc}", flush=True)
            failed_models.append({"model_id": model_id, "error": str(exc)})
            continue

        # Build a fresh JudgeClient per model (isolates judge cost).
        judge_client: object | None = None
        if args.judge:
            try:
                from benchmark.rubrics.judge_client import JudgeClient

                judge_client = JudgeClient()
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: failed to build judge client: {exc}", flush=True)

        try:
            result = run_pack(
                pack_path=pack_path,
                pack_id=pack_id,
                adapter=adapter,
                runs=args.runs,
                judge_client=judge_client,
                grade_version=args.grade_version,
                model_id=model_id,
                out_dir=model_out_dir,
                verbose=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", flush=True)
            failed_models.append({"model_id": model_id, "error": str(exc)})
            continue

        leaderboard_results.append((model_id, result))
        composite = result.get("overall_composite")
        composite_str = f"{composite:.4f}" if composite is not None else "n/a"
        print(f"  => composite={composite_str}\n")

    # --- Build and write leaderboard ---
    if leaderboard_results:
        leaderboard = build_leaderboard(leaderboard_results)
        markdown = render_leaderboard_markdown(leaderboard)

        lb_json_path = out_root / "leaderboard.json"
        lb_md_path = out_root / "leaderboard.md"

        with lb_json_path.open("w", encoding="utf-8") as fh:
            json.dump(leaderboard, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

        lb_md_path.write_text(markdown, encoding="utf-8")

        print("\n" + markdown)
        print(f"Leaderboard JSON → {lb_json_path}")
        print(f"Leaderboard MD   → {lb_md_path}")
    else:
        print("\nNo models completed successfully — leaderboard not written.", file=sys.stderr)

    # --- Summary ---
    n_ok = len(leaderboard_results)
    n_fail = len(failed_models)
    print(f"\nBatch complete: {n_ok} succeeded, {n_fail} failed.")
    if failed_models:
        for fm in failed_models:
            print(f"  FAILED {fm['model_id']}: {fm['error']}", file=sys.stderr)

    return 1 if failed_models else 0


if __name__ == "__main__":
    sys.exit(main())
