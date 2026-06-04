"""Reusable single-model pack runner — extracted from cli.main for DRY reuse.

:func:`run_pack` encapsulates the complete *load → dispatch → aggregate* loop
for one (model, pack) pair.  Both the CLI (:mod:`runner.cli`) and the batch
runner (:mod:`scripts.run_models`) delegate to this function, ensuring the
two paths stay in sync.

Public API
----------
- :func:`run_pack` — run a task pack through an adapter and return the
  ``result_schema.json``-shaped dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runner.adapters.base import Adapter
from runner.aggregator import aggregate
from runner.dispatcher import TaskRunResult, load_pack, run_task
from runner.io import write_raw_outputs, write_result


def run_pack(
    pack_path: Path,
    pack_id: str | None,
    adapter: Adapter,
    runs: int = 5,
    judge_client: object | None = None,
    grade_version: str = "0.1.0",
    model_id: str | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all tasks in a pack through *adapter* and return the result dict.

    This is the canonical single-model execution loop, shared by the CLI and
    the multi-model batch runner.  Callers receive the fully-aggregated
    ``result_schema.json``-shaped dict; they are responsible for writing it
    to disk if required (the CLI does so via :func:`~runner.io.write_result`).

    Per-task resilience
    -------------------
    Tasks that raise an exception are **recorded and skipped** rather than
    aborting the whole run (identical to CLI behaviour).  If *all* tasks fail,
    :exc:`RuntimeError` is raised.

    Args:
        pack_path: Absolute path to the ``.jsonl`` task pack file.
        pack_id: Canonical pack identifier (e.g. ``"pack_operations"``), or
            ``None`` for unknown packs.
        adapter: Instantiated model adapter satisfying
            :class:`~runner.adapters.base.Adapter`.
        runs: Number of repetition runs per task.  Defaults to 5.
        judge_client: Optional live judge client.  When ``None``, the null
            judge (always 0.5) is used for C2/C3.
        grade_version: GRADE benchmark version string embedded in metadata.
        model_id: Override for the ``model_id`` field in the result.  When
            ``None``, the model ID is inferred from the first adapter output.
        out_dir: If provided, write ``raw_outputs.jsonl``, ``result.json``,
            and (if any tasks failed) ``failures.json`` to this directory.
            When ``None``, nothing is written to disk.
        verbose: If ``True``, print per-task progress lines to stdout.

    Returns:
        A ``result_schema.json``-shaped dict produced by
        :func:`~runner.aggregator.aggregate`.

    Raises:
        RuntimeError: If every task in the pack fails (no scorecard can be
            produced).
        FileNotFoundError: If *pack_path* does not exist.
    """
    tasks = load_pack(pack_path)

    task_results: list[TaskRunResult] = []
    all_outputs: list[dict[str, Any]] = []
    model_id_from_adapter: str | None = None
    failures: list[dict[str, str]] = []

    for i, task in enumerate(tasks, start=1):
        task_id: str = task.get("task_id", f"task_{i}")
        if verbose:
            print(f"  [{i}/{len(tasks)}] {task_id} ...", end=" ", flush=True)
        try:
            result = run_task(
                task=task,
                adapter=adapter,
                runs=runs,
                pack_id=pack_id,
                judge_client=judge_client,
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"FAILED ({exc})", flush=True)
            failures.append({"task_id": task_id, "error": str(exc)})
            continue

        task_results.append(result)
        all_outputs.extend(result.outputs)

        # Capture model_id from first output.
        if model_id_from_adapter is None and result.outputs:
            model_id_from_adapter = result.outputs[0].get("model_id")

        if verbose:
            composite_str = f"{result.composite:.3f}" if result.composite is not None else "n/a"
            print(f"composite={composite_str}", flush=True)

    if not task_results:
        raise RuntimeError(
            f"All {len(failures)} task(s) failed — no scorecard produced. "
            f"Failures: {[f['task_id'] for f in failures]}"
        )

    # Determine effective model_id.
    effective_model_id: str = model_id or model_id_from_adapter or adapter.name

    # Collect judge-client accumulated cost.
    judge_metrics_dict: dict[str, Any] | None = None
    if judge_client is not None:
        try:
            from benchmark.rubrics.judge_client import JudgeClient as _JC

            if isinstance(judge_client, _JC):
                judge_metrics_dict = {
                    "cumulative_cost_usd": judge_client.cumulative_cost_usd,
                    "cumulative_prompt_tokens": judge_client.cumulative_prompt_tokens,
                    "cumulative_completion_tokens": judge_client.cumulative_completion_tokens,
                    "judge_call_count": judge_client.judge_call_count,
                }
        except ImportError:
            pass

    scorecard = aggregate(
        task_results=task_results,
        model_id=effective_model_id,
        grade_version=grade_version,
        judge_metrics=judge_metrics_dict,
    )

    # Optionally write outputs to disk.
    if out_dir is not None:
        raw_path = write_raw_outputs(all_outputs, out_dir)
        result_path = write_result(scorecard, out_dir)
        if verbose:
            print(f"\nRaw outputs  → {raw_path}")
            print(f"Result JSON  → {result_path}")

        if failures:
            out_dir.mkdir(parents=True, exist_ok=True)
            failures_path = out_dir / "failures.json"
            with failures_path.open("w", encoding="utf-8") as fh:
                json.dump(failures, fh, indent=2)
                fh.write("\n")
            if verbose:
                print(f"Failures     → {failures_path}")

    if failures and verbose:
        n_ok = len(task_results)
        n_fail = len(failures)
        failed_ids = ", ".join(f["task_id"] for f in failures)
        print(f"\nSummary: {n_ok} task(s) succeeded, {n_fail} task(s) failed ({failed_ids}).")

    return scorecard
