# GRADE

**Grounded Reasoning & Analysis for Data in Education**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0.dev0-orange)](pyproject.toml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)

GRADE is an open benchmark that measures how accurately and usefully AI systems analyze education program data. It was built by [Pearl](https://tutorwithpearl.com) — an AI tutoring platform working directly with school districts — to answer a practical question: **can AI produce education analytics that a program manager or district analyst could trust and act on?**

---

## The Problem

AI tools are being adopted across education to help program managers and analysts make sense of attendance records, session logs, outcome summaries, and research evidence. The quality of that analysis directly affects the decisions practitioners make about students.

General-purpose benchmarks don't measure this. A model can excel at broad reasoning tasks while still hallucinating attendance rates, misattributing causation to correlated signals, or failing to flag that a subgroup is too small to report on. GRADE was built to fill that gap.

---

## Leaderboard

Official run, June 10–12 2026. Three packs: **operations** (11 tasks × 5 runs),
**outcomes** (5 tasks × 1 run), **equity research** (10 tasks × 1 run). All rows
ran with a 65,536-token completion budget so reasoning never crowds out the
answer; judging is cross-family (no judge scores its own model family). Full
caveats in [docs/methodology.md](docs/methodology.md); run notes in
[RUN_FINDINGS.md](RUN_FINDINGS.md).

### Operations (program-data analysis, 5 repetitions)

| Rank | Model | Composite | Grounding | Insight | Evidence | Calibration | Consistency | Coverage |
|------|-------|:---------:|:---------:|:-------:|:--------:|:-----------:|:-----------:|:--------:|
| 1 | `openai/gpt-5.5@xhigh` | **66.9%** | 91.1% | 45.0% | 77.5% | 22.4% | 63.0% | full |
| 2 | `openai/gpt-5.5@high` | **65.4%** | 87.7% | 46.3% | 76.9% | 18.5% | 62.9% | full |
| 3 | `openai/gpt-5.5@medium` | **64.9%** | 88.6% | 44.2% | 73.4% | 17.8% | 66.2% | full |
| 4 | `google/gemini-3.1-pro-preview` | **63.5%** | 79.2% | 45.0% | 76.9% | 31.2% | 61.3% | 8/11 [1] |
| 5 | `google/gemini-3.5-flash` | **61.0%** | 79.5% | 44.7% | 69.4% | 23.6% | 61.2% | 10/11 [1] |
| 6 | `openai/gpt-5.5@low` | **52.4%** | 62.4% | 35.7% | 59.3% | 23.2% | 64.2% | full |
| 7 | `anthropic/claude-opus-4.8` | **41.9%** | 45.5% | 30.1% | 31.2% | 27.3% | 75.5% | full |
| 8 | `anthropic/claude-sonnet-4.6` | **38.3%** | 42.6% | 29.2% | 27.7% | 18.3% | 76.7% | full |
| 9 | `anthropic/claude-haiku-4.5` | **37.9%** | 47.9% | 22.0% | 20.6% | 20.5% | 70.7% | full |
| 10 | `openai/gpt-oss-120b` | **25.2%** | 22.2% | 7.7% | 18.7% | 25.6% | 74.2% | 3/11 DNF [2] |
| 11 | `nvidia/nemotron-3-ultra-550b-a55b` | **25.1%** | 30.2% | 7.3% | 26.9% | 8.3% | 67.6% | full |

> [1] Remaining tasks in flight at publication; composite covers listed tasks only.
> [2] `gpt-oss-120b`'s 131k context cannot fit the ~190k-token data payloads on 8 of 11 tasks — reported as a finding, not ranked competitively.

### Outcomes (trend analysis, single run)

| Rank | Model | Composite | Grounding | Calibration | Model Cost |
|------|-------|:---------:|:---------:|:-----------:|:----------:|
| 1 | `openai/gpt-5.5@low` | **90.1%** | 100.0% | 59.3% | $0.17 |
| 2 | `openai/gpt-5.5@medium` | **90.0%** | 100.0% | 55.7% | $0.21 |
| 3 | `nvidia/nemotron-3-ultra-550b-a55b` | **88.3%** | 100.0% | 45.8% | $0.04 |
| 4 | `openai/gpt-5.5@xhigh` | **86.9%** | 100.0% | 45.0% | $0.67 |
| 5 | `openai/gpt-5.5@high` | **86.6%** | 100.0% | 39.2% | $0.51 |
| 6 | `anthropic/claude-opus-4.8` | **86.2%** | 100.0% | 46.0% | $0.30 |
| 7 | `openai/gpt-oss-120b` | **84.0%** | 100.0% | 39.9% | $0.00 |
| 8 | `anthropic/claude-sonnet-4.6` | **83.0%** | 100.0% | 43.6% | $0.14 |
| 9 | `google/gemini-3.5-flash` | **81.2%** | 100.0% | 29.9% | $0.17 |
| 10 | `google/gemini-3.1-pro-preview` | **80.5%** | 100.0% | 46.1% | $0.17 |
| 11 | `anthropic/claude-haiku-4.5` | **76.7%** | 90.0% | 43.8% | $0.05 |

### Equity research (subgroup & research reasoning, single run)

| Rank | Model | Composite | Grounding | Calibration | Model Cost |
|------|-------|:---------:|:---------:|:-----------:|:----------:|
| 1 | `openai/gpt-5.5@medium` | **83.0%** | 79.8% | 58.9% | $1.10 |
| 2 | `openai/gpt-5.5@xhigh` | **82.8%** | 81.5% | 53.1% | $2.45 |
| 3 | `nvidia/nemotron-3-ultra-550b-a55b` | **81.8%** | 78.5% | 52.8% | $0.10 |
| 4 | `openai/gpt-5.5@high` | **81.1%** | 77.7% | 52.8% | $1.97 |
| 5 | `openai/gpt-5.5@low` | **80.6%** | 76.0% | 58.3% | $0.64 |
| 6 | `anthropic/claude-opus-4.8` | **75.4%** | 69.8% | 49.7% | $0.95 |
| 7 | `google/gemini-3.1-pro-preview` | **73.8%** | 67.8% | 48.3% | $0.59 |
| 8 | `google/gemini-3.5-flash` | **73.0%** | 67.8% | 39.2% | $0.53 |
| 9 | `openai/gpt-oss-120b` | **70.0%** | 62.3% | 32.8% | $0.01 |
| 10 | `anthropic/claude-sonnet-4.6` | **68.7%** | 55.8% | 42.8% | $0.47 |
| 11 | `anthropic/claude-haiku-4.5` | **66.9%** | 66.0% | 35.8% | $0.17 |

> Composite = 35% Grounding + 20% Insight + 15% Evidence + 15% Calibration + 10% Consistency + 5% Structure (consistency excluded and weights renormalized on single-run packs).

**Key findings.** (1) *Reasoning effort pays only where tasks are hard:* GPT-5.5's
effort curve rises steeply from low→medium on heavy data analysis
(52%→65%) then saturates, and **inverts** on light trend summaries (low beats
xhigh). (2) *Scores are envelope-relative:* under a 4k token budget the
default-thinking models (Gemini, Nemotron) score near the floor because
reasoning consumes the budget before any answer appears — the same models are
mid-pack or better at 64k. Nemotron still exhausts even 64k on the largest
tasks (it reasons by exhaustive enumeration), so its operations score reflects
budget economics, not analytical ability — it places top-three on both smaller
packs. (3) *Accuracy and stability trade off:* GPT-5.5 leads grounding accuracy
while Claude models lead run-to-run consistency; Claude Haiku 4.5 delivers
~90% of Opus's operations composite at ~18% of its cost. (4) *Calibration is
universally weak* (18–31%): no model reliably acknowledges data limitations
without prompting — the clearest open problem this benchmark surfaces.

---

---

## What GRADE Measures

Five tracks across three synthetic fixture packs — 26 tasks total.

| Track | Tasks | Core challenge |
|-------|------:|----------------|
| **1 · Grounded Retrieval & Computation** | 6 | Exact counts, rates, and date-filtered aggregates from CSVs — no inference required; the only failure mode is hallucination |
| **2 · Program Snapshot & Trend Interpretation** | 5 | Month-over-month summaries and anomaly detection with epistemic discipline: the data contains correlations that don't support causal claims |
| **3 · Operational Coaching & Recommendations** | 5 | Evidence-backed recommendations with entity IDs, numeric rates, and source-file citations — generic advice scores zero |
| **4 · Equity & Subgroup Interpretation** | 5 | Suppressed low-N groups, attendance and outcome disparities, and plain-language translation for practitioners |
| **5 · Program Effectiveness & Research Reasoning** | 5 | Synthesis of local fixture data with external research references, including contradictory findings |

### Six Scoring Dimensions

Every response is scored on six dimensions combined into a weighted composite:

| Dimension | Weight | How it's scored |
|-----------|:------:|-----------------|
| Grounding Accuracy | **35%** | Deterministic: structured numeric outputs vs. gold facts with tolerance |
| Insight Quality | **20%** | LLM judge: coverage and depth of expected analytical findings |
| Evidence Linkage | **15%** | LLM judge: are claims traced to specific files and columns? |
| Calibration & Limitation Handling | **15%** | Deterministic + judge fallback: required caveats present, forbidden claims absent |
| Consistency | **10%** | Automated: finding/ranking/metric stability across 5 repeated runs |
| Structure & Usability | **5%** | LLM judge: format clarity and scannability |

---

## Quick Start

No API key required to validate your installation.

**Requirements:** Python 3.12+

```bash
git clone https://github.com/PearlEng/grade.git
cd grade
pip install -e .

# Smoke test with the built-in stub adapter (network-free, deterministic)
python -m runner.cli \
    --pack operations \
    --adapter stub \
    --runs 1 \
    --out /tmp/grade_out
```

Produces `raw_outputs.jsonl` (per-run model outputs) and `result.json` (aggregated scorecard) under `/tmp/grade_out`.

### Run a real model

```bash
pip install -e ".[openrouter]"
export OPENROUTER_API_KEY="sk-or-..."

python -m runner.cli \
    --pack operations \
    --adapter openrouter \
    --model openai/gpt-4o \
    --runs 5 \
    --out /tmp/grade_real
```

Any model available on [OpenRouter](https://openrouter.ai/models) works. A full 26-task × 5-run evaluation is 130 model calls and typically completes in under two hours.

### Render a scorecard

```bash
python -m benchmark.reports.scorecard \
    --results-dir /tmp/grade_real \
    --out /tmp/grade_reports
# → /tmp/grade_reports/scorecard.md  (Markdown, per-track breakdown)
# → /tmp/grade_reports/result.json   (machine-readable, result_schema)
```

For a full walkthrough — running all three packs, writing a custom adapter, and interpreting track profiles — see [examples/quickstart.md](examples/quickstart.md).

---

## How It Works

### Synthetic Fixtures

All data is generated from a deterministic seed (42) and represents a fictional high-dosage tutoring program. Three packs are layered:

| Pack | ID | Tracks | What's included |
|------|----|--------|-----------------|
| Operations | `pack_operations` | 1, 3 | Students, sessions, tutors, attendance, survey responses |
| Outcomes | `pack_outcomes` | 2 | All of Operations + month-over-month summary tables |
| Equity & Research | `pack_equity_research` | 4, 5 | All of Outcomes + subgroup summaries + 10 research references |

No real students, schools, or programs are represented. Dataset cards: [`docs/dataset_cards/`](docs/dataset_cards/).

### Scoring Pipeline

For each (model × task) pair:

1. **Adapter** calls the model N times (default: 5) with the task prompt and fixture context
2. **C1 — Grounding** compares `structured_metrics` in the response against `gold_facts` with numeric tolerance
3. **C2 — Rubric** passes the response to a judge model for quality dimensions (insight, evidence, structure)
4. **C3 — Claim validation** checks deterministically that required limitations are present and forbidden claims are absent
5. **C4 — Consistency** measures finding/ranking/metric stability across the N runs
6. **C9 — Aggregation** rolls up to task, track, and overall composites and writes `result.json`

### Cross-Family Judging

No judge ever evaluates a model from the same family. Claude-family models are judged by GPT-5.5 (xhigh reasoning effort); all others are judged by Claude Opus 4.8. This eliminates house-style bias in the 40% of the composite scored by an LLM judge.

### Reproducibility

Gold fact values are derived from `ground_truth.json` files computed from the generated data — not from the generator's input constants. Because the generator applies stochastic noise, realized values differ from spec targets. GRADE mitigates benchmark contamination by regenerating fixtures (new seed, recomputed `ground_truth.json`) for each numbered version; scores are only comparable within a version.

---

## Pluggable Adapters

Adapters implement a minimal Protocol:

```python
class Adapter(Protocol):
    name: str
    def run(self, task: dict, run_index: int = 0) -> dict: ...
```

| Adapter | Flag | Purpose |
|---------|------|---------|
| Stub | `--adapter stub` | Deterministic, network-free — validates installation |
| OpenRouter | `--adapter openrouter` | Live inference for any OpenRouter model |

To evaluate a model not on OpenRouter, implement the Protocol, register it in `runner/cli.py`, and pass `--adapter your_name`. See [examples/quickstart.md](examples/quickstart.md).

---

## Repository Layout

```
grade/
├── benchmark/
│   ├── datagen/        # Synthetic fixture generator (seeded, deterministic)
│   ├── reports/        # Scorecard and leaderboard renderers
│   ├── rubrics/        # Scorer implementations (C1–C4, C9)
│   ├── schemas/        # JSON Schema contracts (task, output, result)
│   └── tasks/          # 26 task definitions across 3 JSONL packs
├── runner/
│   ├── adapters/       # Adapter Protocol + built-in adapters
│   ├── cli.py          # Entry point: python -m runner.cli
│   ├── dispatcher.py   # Task loader and run coordinator
│   └── aggregator.py   # Result aggregation
├── fixtures/           # Pre-generated synthetic data (3 packs)
├── results/            # Pre-computed results from model runs
├── examples/           # Quickstart guide and sample outputs
└── docs/               # Methodology, scoring reference, dataset cards
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/methodology.md](docs/methodology.md) | Mission, task taxonomy, scoring design, limitations, judging policy |
| [docs/scoring.md](docs/scoring.md) | Full dimension definitions, aggregation rules, per-track rubric defaults |
| [docs/dataset_cards/](docs/dataset_cards/) | File inventories, column contracts, intentional signal descriptions per pack |
| [examples/quickstart.md](examples/quickstart.md) | End-to-end walkthrough: install → smoke test → real model → scorecard → custom adapter |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to add tasks, fixture packs, and adapter implementations |

---

## Contributing

We welcome contributions — new tasks, fixture packs, adapter implementations, and methodology proposals. See [CONTRIBUTING.md](CONTRIBUTING.md) and [GOVERNANCE.md](GOVERNANCE.md).

---

## About Pearl

GRADE is built and maintained by [Pearl](https://tutorwithpearl.com), an AI tutoring platform that partners with school districts to deliver high-dosage tutoring programs. Pearl works with program managers and district analysts daily, and GRADE reflects the analytical questions and failure modes we encounter in practice.

If you're a researcher, AI developer, or education technologist and want to discuss the benchmark or collaborate, reach out via GitHub Issues or at [hello@tutorwithpearl.com](mailto:hello@tutorwithpearl.com).

---

## Citing GRADE

If you use GRADE in your research, please cite the repository:

```bibtex
@software{pearl2026grade,
  author       = {Pearl},
  title        = {{GRADE}: Grounded Reasoning \& Analysis for Data in Education},
  year         = {2026},
  publisher    = {GitHub},
  version      = {0.1.0},
  url          = {https://github.com/PearlEng/grade},
  note         = {Open benchmark for AI evaluation in education analytics}
}
```

A Zenodo DOI for stable versioned citation will be added at the v0.1.0 release. Once available, prefer the DOI-based entry:

```bibtex
@software{pearl2026grade,
  author       = {Pearl},
  title        = {{GRADE}: Grounded Reasoning \& Analysis for Data in Education},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

GitHub also provides a formatted citation via the **Cite this repository** button in the sidebar (powered by [`CITATION.cff`](CITATION.cff)).

---

## License

Apache 2.0. See [LICENSE](LICENSE).
