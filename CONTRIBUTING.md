# Contributing to GRADE

Thank you for your interest in contributing to GRADE (Grounded Reasoning & Analysis for Data in Education), a public AI benchmark for education program analytics.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Running Tests](#running-tests)
- [Pull Request Process](#pull-request-process)

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/grade.git
   cd grade
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/PearlEng/grade.git
   ```

## Development Setup

GRADE requires Python 3.12 or later. Use a virtual environment to avoid conflicts with system packages.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

This installs the package in editable mode along with all development dependencies (Ruff, mypy, pytest, pre-commit).

## Code Style

**Formatter and linter:** [Ruff](https://docs.astral.sh/ruff/) is used for both formatting and linting.

```bash
.venv/bin/ruff format .            # auto-format
.venv/bin/ruff check .             # lint
.venv/bin/ruff check --fix .       # lint + auto-fix safe issues
```

**Type checking:** [mypy](https://mypy.readthedocs.io/) is enabled in strict mode for new code.

```bash
.venv/bin/mypy benchmark runner
```

**Docstrings:** Follow [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all public functions, classes, and modules.

**Pre-commit hooks:** The pre-commit configuration enforces trailing-whitespace removal, end-of-file newlines, and other file hygiene checks automatically on every commit. The hooks run automatically after `pre-commit install`, but you can also run them manually:

```bash
.venv/bin/pre-commit run --all-files
```

## Running Tests

```bash
.venv/bin/pytest
```

Tests live under `tests/`. Please add tests for any new functionality or bug fixes. A passing test suite is required before a PR can be merged.

## Pull Request Process

1. Create a branch from `main` with a short descriptive name, e.g. `fix/openrouter-timeout` or `feat/new-adapter`.
2. Make your changes, following the code-style guidelines above.
3. Ensure the full test suite and pre-commit hooks pass locally.
4. Open a pull request against `main` on GitHub. Fill in the PR template, including a reference to any related issue (e.g. `Closes #123`).
5. A maintainer listed in [CODEOWNERS](./CODEOWNERS) will review your PR. Please respond to review feedback promptly.
6. Once approved, the maintainer will merge the PR.

For questions or discussion, open a GitHub Issue rather than contacting maintainers privately (unless reporting a security vulnerability — see [SECURITY.md](./SECURITY.md)).
