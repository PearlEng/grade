"""Tests for the runner CLI ``--judge`` flag wiring (network-free)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import runner.cli as cli


def test_judge_flag_defaults_false() -> None:
    """Without --judge, the parsed flag is False."""
    args = cli.build_parser().parse_args(["--pack", "operations", "--out", "/tmp/x"])
    assert args.judge is False


def test_judge_flag_parses_true() -> None:
    """With --judge, the parsed flag is True."""
    args = cli.build_parser().parse_args(["--pack", "operations", "--out", "/tmp/x", "--judge"])
    assert args.judge is True


def test_main_passes_live_judge_to_run_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--judge builds a JudgeClient and threads it into run_pack as judge_client."""
    captured: dict[str, object | None] = {}

    def fake_run_pack(**kwargs: Any) -> dict[str, Any]:
        captured["judge_client"] = kwargs.get("judge_client")
        raise RuntimeError("stop-after-capture")  # cli.main catches -> returns 1

    sentinel = object()

    def fake_judge_client(**kwargs: Any) -> object:
        captured["judge_kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(cli, "run_pack", fake_run_pack)
    monkeypatch.setattr("benchmark.rubrics.judge_client.JudgeClient", fake_judge_client)

    rc = cli.main(
        [
            "--pack",
            "operations",
            "--adapter",
            "stub",
            "--runs",
            "1",
            "--judge",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 1
    assert captured["judge_client"] is sentinel
    # No --judge-model and a non-Opus (stub) candidate → policy default judge.
    from benchmark.rubrics.judge_client import DEFAULT_JUDGE_MODEL

    judge_kwargs = captured["judge_kwargs"]
    assert isinstance(judge_kwargs, dict)
    assert judge_kwargs["model"] == DEFAULT_JUDGE_MODEL
    assert judge_kwargs["reasoning_effort"] is None


def test_main_uses_null_judge_without_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without --judge, run_pack receives judge_client=None (dispatcher uses null judge)."""
    captured: dict[str, object | None] = {}

    def fake_run_pack(**kwargs: Any) -> dict[str, Any]:
        captured["judge_client"] = kwargs.get("judge_client")
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(cli, "run_pack", fake_run_pack)
    rc = cli.main(
        ["--pack", "operations", "--adapter", "stub", "--runs", "1", "--out", str(tmp_path)]
    )
    assert rc == 1
    assert captured["judge_client"] is None
