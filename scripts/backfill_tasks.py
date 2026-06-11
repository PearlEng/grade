r"""backfill_tasks — re-run only the missing/failed tasks of an existing model row.

Reads the model's existing ``result.json`` + ``raw_outputs.jsonl`` from
``<out>/<safe_model_slug>/``, re-runs ONLY the requested task IDs against the
live pack, merges old and new per-task results, re-aggregates, and rewrites
both files in place (originals are backed up with a ``.bak`` suffix).

Usage::

    python -m scripts.backfill_tasks \
        --model anthropic/claude-sonnet-4.6 \
        --tasks T1-OPS-003,T1-OPS-004,T1-OPS-005,T1-OPS-006 \
        --pack operations --runs 5 --judge --out results/operations

If the model has no existing row (e.g. gpt-5.5@xhigh), all tasks in
``--tasks`` are run and the merged row is just the new tasks — equivalent to
a fresh partial run.

Judge cost in the merged ``cost_metrics`` is the sum of the old row's
``judge_cost_usd`` and the backfill's judge spend (token-level judge counts
are only available for the backfill portion and are reported as-is).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from runner.aggregator import aggregate
from runner.dispatcher import TaskRunResult, load_pack, run_task
from runner.io import write_raw_outputs, write_result
from scripts.run_models import _build_adapter, _resolve_pack, _safe_slug


def _load_existing(
    model_dir: Path,
    skip_task_ids: set[str],
) -> tuple[list[TaskRunResult], float]:
    """Reconstruct TaskRunResults for tasks NOT being backfilled.

    Args:
        model_dir: Existing per-model result directory.
        skip_task_ids: Task IDs being re-run (excluded from reconstruction).

    Returns:
        ``(kept_results, old_judge_cost_usd)``.  Empty list / 0.0 when no
        prior row exists.
    """
    result_path = model_dir / "result.json"
    raw_path = model_dir / "raw_outputs.jsonl"
    if not result_path.exists():
        return [], 0.0

    result = json.loads(result_path.read_text(encoding="utf-8"))

    outputs_by_task: dict[str, list[dict[str, Any]]] = {}
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out = json.loads(line)
                outputs_by_task.setdefault(out["task_id"], []).append(out)

    kept: list[TaskRunResult] = []
    for entry in result.get("per_task_scores", []):
        tid = entry["task_id"]
        if tid in skip_task_ids:
            continue
        kept.append(
            TaskRunResult(
                task_id=tid,
                track=entry["track"],
                pack_id=entry.get("pack_id"),
                run_count=entry["run_count"],
                outputs=outputs_by_task.get(tid, []),
                scores=dict(entry["scores"]),
                composite=entry.get("composite"),
                scorer_flags=list(entry.get("scorer_flags") or []),
            )
        )

    old_judge_cost = (result.get("cost_metrics") or {}).get("judge_cost_usd") or 0.0
    return kept, float(old_judge_cost)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: re-run the named tasks and merge into the existing row.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).

    Returns:
        Exit code: 0 on full success, 1 when any backfill task failed.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.backfill_tasks",
        description="Re-run only the named tasks for one model row and merge results.",
    )
    parser.add_argument("--model", required=True, help="Model ID (supports @<effort> suffix).")
    parser.add_argument(
        "--tasks", required=True, metavar="T1,T2,...", help="Comma-separated task IDs to re-run."
    )
    parser.add_argument("--pack", required=True, help="Pack short name or path.")
    parser.add_argument("--adapter", default="openrouter")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", required=True, help="Output root (same as the original run).")
    parser.add_argument("--grade-version", default="0.1.0")
    args = parser.parse_args(argv)

    wanted_ids = {t.strip() for t in args.tasks.split(",") if t.strip()}
    if not wanted_ids:
        print("ERROR: --tasks must be a non-empty comma-separated list.", file=sys.stderr)
        return 1

    pack_path, pack_id = _resolve_pack(args.pack)
    tasks = [t for t in load_pack(pack_path) if t.get("task_id") in wanted_ids]
    missing = wanted_ids - {t["task_id"] for t in tasks}
    if missing:
        print(f"ERROR: task IDs not found in pack: {sorted(missing)}", file=sys.stderr)
        return 1

    model_dir = Path(args.out) / _safe_slug(args.model)
    kept_results, old_judge_cost = _load_existing(model_dir, wanted_ids)
    print(
        f"Backfilling {len(tasks)} task(s) for {args.model}; "
        f"keeping {len(kept_results)} existing task result(s)."
    )

    adapter = _build_adapter(args.adapter, model=args.model, temperature=args.temperature)

    judge_client: Any | None = None
    if args.judge:
        from benchmark.rubrics.judge_client import JudgeClient, select_judge_model

        if args.judge_model:
            judge_model, judge_effort = args.judge_model, None
        else:
            judge_model, judge_effort = select_judge_model(args.model)
        print(f"  judge: {judge_model}" + (f" (effort: {judge_effort})" if judge_effort else ""))
        judge_client = JudgeClient(model=judge_model, reasoning_effort=judge_effort)

    new_results: list[TaskRunResult] = []
    failures: list[dict[str, str]] = []
    for i, task in enumerate(tasks, start=1):
        tid = task["task_id"]
        print(f"  [{i}/{len(tasks)}] {tid} ...", end=" ", flush=True)
        try:
            tr = run_task(
                task=task,
                adapter=adapter,
                runs=args.runs,
                pack_id=pack_id,
                judge_client=judge_client,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED ({exc})", flush=True)
            failures.append({"task_id": tid, "error": str(exc)})
            continue
        comp = f"{tr.composite:.4f}" if tr.composite is not None else "n/a"
        print(f"composite={comp}", flush=True)
        new_results.append(tr)

    if not new_results:
        print("No backfill task succeeded — leaving existing row untouched.", file=sys.stderr)
        return 1

    # Merge, preserving canonical pack order.
    pack_order = {t["task_id"]: i for i, t in enumerate(load_pack(pack_path))}
    merged = sorted(
        kept_results + new_results,
        key=lambda tr: pack_order.get(tr.task_id, 10_000),
    )

    judge_metrics: dict[str, Any] | None = None
    if judge_client is not None:
        judge_metrics = {
            "cumulative_cost_usd": (judge_client.cumulative_cost_usd or 0.0) + old_judge_cost,
            "cumulative_prompt_tokens": judge_client.cumulative_prompt_tokens,
            "cumulative_completion_tokens": judge_client.cumulative_completion_tokens,
            "judge_call_count": judge_client.judge_call_count,
        }
    elif old_judge_cost:
        judge_metrics = {"cumulative_cost_usd": old_judge_cost}

    scorecard = aggregate(
        task_results=merged,
        model_id=args.model,
        grade_version=args.grade_version,
        judge_metrics=judge_metrics,
    )

    # Back up originals, then write merged row.
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "raw_outputs.jsonl", "failures.json"):
        p = model_dir / name
        if p.exists():
            shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))

    all_outputs = [out for tr in merged for out in tr.outputs]
    raw_path = write_raw_outputs(all_outputs, model_dir)
    result_path = write_result(scorecard, model_dir)

    failures_path = model_dir / "failures.json"
    if failures:
        failures_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    elif failures_path.exists():
        failures_path.unlink()  # row is now complete — stale failure list removed

    overall = scorecard.get("overall_composite")
    comp_str = f"{overall:.4f}" if overall is not None else "n/a"
    print(f"\nMerged row: {len(merged)} task(s), composite={comp_str}")
    print(f"Raw outputs  → {raw_path}")
    print(f"Result JSON  → {result_path}")
    if failures:
        print(f"Backfill failures recorded → {failures_path}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
