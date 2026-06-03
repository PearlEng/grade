# GRADE Quick-Start — 10 Minutes from Clone to Scorecard

This guide walks you through cloning the repo, installing dependencies, running
the benchmark end-to-end with the built-in **stub adapter** (no API key needed),
and then switching to a real model via the OpenRouter adapter.

---

## Prerequisites

- Python 3.12 or later
- Git

---

## Step 1 — Clone and install

```bash
git clone https://github.com/PearlEng/grade.git
cd grade
pip install -e .
```

The base install pulls in only `jsonschema`.  No API key is required for
everything in this guide up through Step 3.

---

## Step 2 — Run the benchmark (stub adapter, network-free)

The **stub adapter** returns deterministic, schema-valid outputs without
calling any external API.  It is the fastest way to confirm your installation
works and to explore the benchmark structure.

```bash
python -m runner.cli \
    --pack operations \
    --adapter stub \
    --runs 1 \
    --out /tmp/grade_out
```

You should see output similar to:

```
Loaded 10 tasks from operations_pack.jsonl.
  [1/10] T1-OPS-001 ... composite=0.340
  [2/10] T1-OPS-002 ... composite=0.340
  ...
  [10/10] T3-OPS-003 ... composite=0.340

Raw outputs  → /tmp/grade_out/raw_outputs.jsonl
Result JSON  → /tmp/grade_out/result.json
Done.
```

Try the other packs too:

```bash
python -m runner.cli --pack outcomes        --adapter stub --runs 1 --out /tmp/grade_out_outcomes
python -m runner.cli --pack equity_research --adapter stub --runs 1 --out /tmp/grade_out_equity
```

### What was produced?

| File | Description |
|------|-------------|
| `raw_outputs.jsonl` | One JSON object per (task × run), conforming to `output_schema.json` |
| `result.json` | Aggregated scorecard conforming to `result_schema.json` |

---

## Step 3 — Render a Markdown scorecard (C9)

```bash
python -m benchmark.reports.scorecard \
    --results-dir /tmp/grade_out \
    --out /tmp/grade_reports
```

Open `/tmp/grade_reports/scorecard.md` to see the per-dimension and
per-track breakdown.

---

## Step 4 — Run with a real model (OpenRouter)

To evaluate an actual language model you need an
[OpenRouter](https://openrouter.ai) API key and the `openrouter` extra:

```bash
pip install -e ".[openrouter]"
export OPENROUTER_API_KEY="sk-or-..."

python -m runner.cli \
    --pack operations \
    --adapter openrouter \
    --model anthropic/claude-sonnet-4-5 \
    --runs 5 \
    --out /tmp/grade_real_out
```

The `--model` flag accepts any model slug available on OpenRouter, for
example `openai/gpt-4o` or `anthropic/claude-opus-4-7`.

---

## Step 5 — Explore the sample outputs

`examples/sample_outputs/` contains one pre-computed output JSON per track
that validate against `output_schema.json`.  Use them to understand the
expected output shape before plugging in a real model:

| File | Track | Pack |
|------|-------|------|
| `track1_operations.json` | 1 — Grounded Retrieval | operations |
| `track2_outcomes.json` | 2 — Snapshot & Trends | outcomes |
| `track3_coaching.json` | 3 — Coaching Recommendations | operations |
| `track4_equity.json` | 4 — Equity Interpretation | equity_research |
| `track5_effectiveness_research.json` | 5 — Effectiveness Research | equity_research |

Validate any sample output against the schema:

```python
import json
from benchmark.schemas import validate_output

with open("examples/sample_outputs/track1_operations.json") as f:
    output = json.load(f)

validate_output(output)   # raises jsonschema.ValidationError if invalid
print("Valid!")
```

---

## Step 6 — Write your own adapter

See `examples/adapter_examples/stub_adapter.py` for a minimal adapter
template with inline `# TODO` comments explaining each required section.

The adapter interface is defined in `runner/adapters/base.py`:

```python
class Adapter(Protocol):
    name: str
    def run(self, task: dict, run_index: int = 0) -> dict: ...
```

Once your adapter is ready, register it in `runner/cli.py`:

```python
_ADAPTER_REGISTRY: dict[str, type] = {
    "openrouter": OpenRouterAdapter,
    "stub": StubAdapter,
    "my_provider": MyProviderAdapter,   # add this line
}
```

Then run:

```bash
python -m runner.cli --pack operations --adapter my_provider --runs 1 --out /tmp/out
```

---

## CLI reference

```
python -m runner.cli --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pack` | *(required)* | `operations`, `outcomes`, `equity_research`, or path to a `.jsonl` |
| `--adapter` | `openrouter` | `stub` (network-free) or `openrouter` |
| `--model` | *(none)* | Model slug forwarded to the adapter (required for openrouter) |
| `--runs` | `5` | Repetitions per task |
| `--out` | *(required)* | Output directory |
| `--grade-version` | `0.1.0` | Version string embedded in result metadata |
| `--model-id` | *(adapter default)* | Override model_id in result metadata |
| `--dry-run` | `false` | Validate configuration without calling the adapter |

---

## Next steps

- Read the dataset cards in `docs/dataset_cards/` to understand each pack's
  tasks, fixtures, and scoring rubrics.
- See `CONTRIBUTING.md` for guidance on adding tasks, tracks, or adapters.
- Run the test suite: `pytest` (all tests are network-free; stub adapter only).
