"""Smoke tests for the GRADE package scaffold."""


def test_benchmark_importable() -> None:
    """Verify the benchmark package can be imported.

    Expected: import succeeds without error.
    """
    import benchmark  # noqa: F401


def test_runner_importable() -> None:
    """Verify the runner package can be imported.

    Expected: import succeeds without error.
    """
    import runner  # noqa: F401
