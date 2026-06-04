"""GRADE generic benchmark runner CLI (C5 + C6).

Entry point::

    python -m runner.cli --pack <name> --adapter <id> --runs 5 --out <dir>

The CLI:

1. Resolves the task pack JSONL path from ``--pack`` (name or absolute path).
2. Instantiates the requested adapter via the adapter registry.
3. For each task in the pack, runs the adapter ``--runs`` times, validates
   outputs, and scores with C1/C2 scorers.  Tasks that raise an exception are
   **recorded and skipped** rather than aborting the whole run.
4. Aggregates results into the ``result_schema.json`` shape (over successful
   tasks only).
5. Writes ``raw_outputs.jsonl`` and ``result.json`` to ``--out``.  If any tasks
   failed, also writes ``failures.json`` (list of ``{task_id, error}`` dicts).

Partial runs
------------
If at least one task succeeds the runner exits with code 0 and writes a
partial scorecard.  If **all** tasks fail it prints an error summary and exits
with code 1 (no ``result.json`` / ``raw_outputs.jsonl`` are written).

``failures.json`` is written to ``--out`` whenever at least one task fails.
When all tasks succeed the file is not created (no empty file).

Adapter registry
----------------
Registered adapters:

- ``openrouter`` *(default)* — :class:`~runner.adapters.openrouter_adapter.OpenRouterAdapter`
  (requires ``OPENROUTER_API_KEY``).
- ``stub`` — :class:`~runner.adapters.stub_adapter.StubAdapter` (network-free,
  for tests and smoke runs).

Usage examples::

    # Smoke test — no network or API key required:
    python -m runner.cli --pack operations --adapter stub --runs 1 --out /tmp/grade_out

    # With the OpenRouter adapter:
    python -m runner.cli --pack outcomes --adapter openrouter \
        --model anthropic/claude-sonnet-4-5 --runs 5 --out /tmp/grade_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from runner.adapters.base import Adapter
from runner.adapters.openrouter_adapter import OpenRouterAdapter
from runner.adapters.stub_adapter import StubAdapter
from runner.dispatcher import load_pack
from runner.run import run_pack

# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

#: Map from CLI ``--adapter`` name to an adapter class (treated as a factory).
#: C6 adds OpenRouterAdapter as the default; the factory accepts an optional
#: model kwarg which is threaded through from ``--model``.
_ADAPTER_REGISTRY: dict[str, type] = {
    "openrouter": OpenRouterAdapter,
    "stub": StubAdapter,
}

#: Default adapter name when ``--adapter`` is not specified.
_DEFAULT_ADAPTER: str = "openrouter"

#: Known pack name → relative path within the repo.
_PACK_PATHS: dict[str, Path] = {
    "operations": Path("benchmark/tasks/operations_pack.jsonl"),
    "outcomes": Path("benchmark/tasks/outcomes_pack.jsonl"),
    "equity_research": Path("benchmark/tasks/equity_research_pack.jsonl"),
}

#: Map CLI pack name → canonical pack_id used in result_schema.
_PACK_IDS: dict[str, str] = {
    "operations": "pack_operations",
    "outcomes": "pack_outcomes",
    "equity_research": "pack_equity_research",
}


def _resolve_pack_path(pack_arg: str, repo_root: Path) -> tuple[Path, str | None]:
    """Resolve --pack argument to an absolute path and a canonical pack_id.

    Accepts either a known pack short name (``"operations"``) or an arbitrary
    file path.

    Args:
        pack_arg: Value of the ``--pack`` CLI argument.
        repo_root: Root directory of the GRADE repository.

    Returns:
        A ``(path, pack_id)`` tuple where *pack_id* may be ``None`` for
        unknown packs.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    if pack_arg in _PACK_PATHS:
        resolved = repo_root / _PACK_PATHS[pack_arg]
        pack_id: str | None = _PACK_IDS[pack_arg]
    else:
        resolved = Path(pack_arg)
        if not resolved.is_absolute():
            resolved = repo_root / resolved
        pack_id = None

    if not resolved.exists():
        raise FileNotFoundError(f"Pack file not found: {resolved}")

    return resolved.resolve(), pack_id


def _build_adapter(
    adapter_name: str,
    model: str | None = None,
    temperature: float | None = None,
) -> Adapter:
    """Instantiate an adapter from the registry by name.

    Args:
        adapter_name: Value of the ``--adapter`` CLI argument.
        model: Optional model slug/shorthand passed via ``--model``.  If
            provided and the adapter constructor accepts a ``model`` keyword
            argument, it is forwarded.  Ignored by adapters that don't accept
            it (e.g. ``StubAdapter``).
        temperature: Optional sampling temperature passed via ``--temperature``.
            If provided and the adapter constructor accepts a ``temperature``
            keyword argument, it is forwarded.  Ignored by adapters that don't
            accept it (e.g. ``StubAdapter``).

    Returns:
        An instantiated :class:`~runner.adapters.base.Adapter`.

    Raises:
        KeyError: If *adapter_name* is not in the adapter registry.
    """
    if adapter_name not in _ADAPTER_REGISTRY:
        known = ", ".join(sorted(_ADAPTER_REGISTRY))
        raise KeyError(f"Unknown adapter '{adapter_name}'.  Known adapters: {known}")
    factory = _ADAPTER_REGISTRY[adapter_name]
    kwargs: dict[str, object] = {}
    if model is not None:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature
    if kwargs:
        try:
            return factory(**kwargs)  # type: ignore[no-any-return]
        except TypeError:
            pass  # adapter doesn't accept these kwargs — fall through
    return factory()  # type: ignore[no-any-return]


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="python -m runner.cli",
        description=(
            "GRADE generic benchmark runner.  "
            "Loads a task pack, dispatches to a model adapter, scores outputs, "
            "and writes results to disk."
        ),
    )
    parser.add_argument(
        "--pack",
        required=True,
        metavar="NAME_OR_PATH",
        help=(
            "Task pack to evaluate.  Either a short name "
            "(operations | outcomes | equity_research) or a path to a .jsonl file."
        ),
    )
    parser.add_argument(
        "--adapter",
        default=_DEFAULT_ADAPTER,
        metavar="ADAPTER_ID",
        help=(
            f"Model adapter to use.  Default: {_DEFAULT_ADAPTER!r}.  "
            f"Available: {', '.join(sorted(_ADAPTER_REGISTRY))}."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help=(
            "Model slug or shorthand forwarded to the adapter "
            "(e.g. 'anthropic/claude-sonnet-4-5', 'claude-sonnet-4-6').  "
            "Required for the openrouter adapter; ignored by the stub."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="TEMP",
        help=(
            "Sampling temperature forwarded to the adapter (default: 1.0).  "
            "Use 0.0 for deterministic inference; higher values increase "
            "response variability, which is needed for the C4 consistency "
            "dimension.  Ignored by the stub adapter."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        metavar="N",
        help="Number of repetition runs per task (default: 5).",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="Output directory for raw_outputs.jsonl and result.json.",
    )
    parser.add_argument(
        "--grade-version",
        default="0.1.0",
        metavar="VERSION",
        help="GRADE benchmark version string embedded in result metadata (default: 0.1.0).",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        metavar="MODEL_ID",
        help=(
            "Override the model_id in result metadata.  "
            "Defaults to the adapter's reported model_id."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load tasks and validate configuration without running the adapter.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Use a live OpenRouter-backed judge for the rubric (C2) and claim-validation "
            "(C3) judge fallback, instead of the default null judge (0.5). Requires "
            "OPENROUTER_API_KEY."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).parent.parent

    # --- Resolve pack ---
    try:
        pack_path, pack_id = _resolve_pack_path(args.pack, repo_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # --- Load tasks ---
    tasks = load_pack(pack_path)
    print(f"Loaded {len(tasks)} tasks from {pack_path.name}.")

    if args.dry_run:
        print(
            f"Dry run — would run {len(tasks)} tasks × {args.runs} reps"
            f" with adapter '{args.adapter}'."
        )
        return 0

    # --- Build adapter ---
    try:
        adapter = _build_adapter(args.adapter, model=args.model, temperature=args.temperature)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)

    # --- Build judge (optional, live OpenRouter) ---
    judge_client: object | None = None
    if args.judge:
        from benchmark.rubrics.judge_client import JudgeClient

        judge_client = JudgeClient()

    # --- Run pack via shared helper ---
    try:
        run_pack(
            pack_path=pack_path,
            pack_id=pack_id,
            adapter=adapter,
            runs=args.runs,
            judge_client=judge_client,
            grade_version=args.grade_version,
            model_id=args.model_id,
            out_dir=out_dir,
            verbose=True,
        )
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
