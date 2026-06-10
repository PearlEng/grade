# GRADE Methodology Review — Findings & Fix Plan

**Date:** 2026-06-10
**Scope:** Full review of the evaluation harness (`runner/`, `benchmark/rubrics/`, `scripts/run_models.py`) ahead of the first real leaderboard run (issue PearlEng/platform-ai#724).
**Status legend:** `FIXED` (this session) · `OPEN` (assigned to follow-up session) · `DISCLOSE` (methodology-page disclosure, not a code change)

---

## Critical (launch-blocking)

### C-1. Model registry mislabels which models actually run — `FIXED`
`SEED_MODELS` in `runner/adapters/openrouter_adapter.py` mapped shorthands to
stale placeholder slugs: `"gpt-5"` → `openai/gpt-4o`, `"gemini-2.5-pro"` →
`google/gemini-pro-1.5`, `"claude-opus-4-7"` → `anthropic/claude-opus-4.5`.
A leaderboard published from these would attribute GPT-4o scores to GPT-5.
The adapter's default model and the judge default also used dash-form slugs
(`anthropic/claude-sonnet-4-5`) that do not exist on OpenRouter (the live
slugs use dots, e.g. `anthropic/claude-sonnet-4.6`).

**Fix applied:** `SEED_MODELS` replaced with the verified launch lineup (see
"Launch model lineup" below; every slug checked against the live
`/api/v1/models` endpoint on 2026-06-10). Adapter default model and judge
default corrected to live slugs.
**Follow-up (separate repo):** `grade-app/arena/pairs.py`
`SEED_MODEL_DISPLAY_NAMES` must be updated to match the new lineup when real
results are imported (part of #724).

### C-2. Forgetting `--judge` silently produces a fake leaderboard — `FIXED`
Without a judge client, `insight_quality`, `evidence_linkage`, and
`structure_usability` (40% of the composite) were scored as a flat `0.5` for
every model by `_NullJudge`, and nothing in the result recorded this. A
misconfigured run would yield a plausible-looking leaderboard ranked only by
fact-matching.

**Fix applied:**
- `runner/dispatcher.py` now stamps a `null_judge` flag into every task's
  `scorer_flags` when no live judge is configured.
- `runner/cli.py` and `scripts/run_models.py` now **refuse to run** the
  `openrouter` adapter without `--judge` unless `--allow-null-judge` is
  passed explicitly (stub adapter unaffected; smoke tests keep working).

### C-3. Judge reliability: invalid slug, silent zeros, no retry — `FIXED` (partially `DISCLOSE`)
`benchmark/rubrics/judge_client.py`:
- `DEFAULT_JUDGE_MODEL` was `anthropic/claude-opus-4-5` — an invalid slug
  (live slug is `anthropic/claude-opus-4.5`), so every live judge call would
  have failed.
- A judge reply that didn't parse as a float silently became `0.0` — a judge
  hiccup charged against the candidate model, with no log, retry, or flag.

**Fix applied:**
- Default judge slug corrected to a live dot-form slug (now
  `anthropic/claude-opus-4.8` per the selection policy below).
- Response parsing now extracts the first float via regex (tolerates stray
  punctuation/markdown), retries the API call once on parse failure, and
  raises `JudgeScoreError` if both attempts fail — so the failure surfaces in
  `failures.json` instead of silently zeroing a dimension.
- `--judge-model` flag added to both CLIs so the judge is configurable per
  run.

**Judge selection policy (decided 2026-06-10, revised same day, implemented):**
the strongest available model judges everyone — the principle being that a
weaker model should not grade a stronger one — and **no judge ever shares a
model family with its candidate**, so house style cannot bias the judged
dimensions. Default judge is **Claude Opus 4.8**; every **Claude-family
candidate** (Opus, Sonnet, Haiku) is instead judged by **GPT-5.5 at xhigh
reasoning effort**. Implemented in
`benchmark.rubrics.judge_client.select_judge_model()` and applied
automatically by both CLIs (`--judge-model` overrides). Reasoning-effort
passthrough was added to `post_chat_completion` for this (the GPT-5.5 judge
sends `reasoning: {effort: "xhigh"}` with an 8192-token budget, since
reasoning tokens share the budget with the one-float answer).

**`DISCLOSE` — two residual caveats for the methodology page:**
1. *Contestant judges*: both judges are themselves contestants (Opus 4.8 is
   judged by GPT-5.5; GPT-5.5 rows are judged by Opus 4.8). Neither ever
   judges its own family, but neither is a disinterested third party either.
2. *Calibration mismatch*: Claude rows are scored by a different judge than
   all other rows. If the two judges differ in harshness, the Claude rows
   sit on a slightly different scale for the three judged dimensions (40% of
   composite). Mitigation if it proves material: run both judges over a
   small overlap sample and report (or correct for) the judge-to-judge
   offset.

---

## High (affects score validity — fix before the real run)

### H-1. Grounding accuracy is credit-only and key-agnostic → rewards number spam — `FIXED`
`benchmark/rubrics/fact_scoring.py`: a gold fact was credited if **any**
number anywhere in the output (up to 50 prose sentences, plus a `/100`
percent-normalized variant of every candidate) fell within tolerance.
Leniency floors widened the window further (gold count 30 with authored
tolerance 0 gets ±0.6, so an unrelated "29.6%" anywhere credited it).

**Fix applied:** free-text numeric matches (`key_findings` / `limitations`)
now pass a **claim-context gate** — the containing sentence must share at
least one content token with the gold claim (stopwords/numbers excluded,
plural-`s` normalized; see `_shares_claim_context`). The "29.6% of survey
responses arrived late" example no longer credits a gold count of 30.
`structured_metrics` matching deliberately stays key-agnostic: structured
values are deliberate model assertions, and key-name matching was previously
found too brittle against real model outputs. Covered by the new
context-gate tests in `test_fact_scoring.py`.

### H-2. Non-numeric gold facts effectively require verbatim echo — `FIXED`
`fact_scoring.py:score_fact` found a candidate finding by substring
containment but then scored it with **exact** string equality, so a finding
containing the claim plus any other words scored 0 — non-numeric facts were
near-universal misses.

**Fix applied:** the match is now scored directly. Stage 1 is substring
containment (either direction); stage 2 credits paraphrases when >= 50% of
the claim's content tokens appear in the entry
(`TEXT_FACT_OVERLAP_THRESHOLD`, mirroring C3). Match provenance is recorded
in the `method` label (`text_match[key_findings+token_overlap]` etc.). The
doc/code mismatch (docstring promised token overlap) is resolved.

### H-3. `runs=1` gives every model a free 10% (consistency) — `OPEN`
All three consistency sub-metrics default to 1.0 with a single run. The
leaderboard run must use `--runs ≥ 3` (default 5 is good). Suggested fix:
stamp a `consistency_trivial` scorer flag when `runs < 2`, or exclude the
dimension from the composite in that case.

### H-4. `max_tokens=1024` truncates analyses; reasoning models break — `OPEN`
`OpenRouterAdapter.DEFAULT_MAX_TOKENS = 1024` is tight for a multi-section
prose analysis; limitations sections come last and get cut first (deflating
`calibration_limitation_handling`). `finish_reason` is not checked, so
truncation is invisible. Reasoning models (GPT-5.5 at high effort) can burn
the entire budget on reasoning tokens and return an empty visible response.

**Suggested fix:** raise the default to ≥ 4096, check
`choices[0].finish_reason == "length"` and stamp a `truncated` flag, and add
a `reasoning` parameter passthrough (see H-5).

### H-5. No reasoning-effort support — required for the GPT-5.5 sweep — `OPEN` (partial)
The launch plan includes GPT-5.5 at xhigh/high/medium/low effort.
**Done (2026-06-10):** `post_chat_completion` now accepts a
`reasoning_effort` parameter (used by the GPT-5.5 judge).
**Still open:** `OpenRouterAdapter` (the candidate-side path) does not yet
accept/forward a reasoning effort, and results need distinct `model_id`
labels per effort level (e.g. `openai/gpt-5.5@xhigh`) so leaderboard rows
don't collide. Wire the adapter through the same parameter and add a
`--reasoning-effort` CLI flag (or per-model syntax in `--models`).

---

## Medium (quality / cost — assigned to follow-up session)

### M-1. C1 never sees the `limitations` field — `FIXED`
`runner/dispatcher.py:_score_c1_grounding` now passes the output's
`limitations` list to `score_facts` as the documented fallback search target.
Covered by `test_c1_grounding_searches_limitations`.

### M-2. Half the judge spend is wasted — `FIXED`
`score_rubric` now accepts a `dimensions` subset (full-rubric weight
validation unchanged), and the dispatcher judges only the three C2-owned
dimensions (`insight_quality`, `evidence_linkage`, `structure_usability`).
This halves judge calls: 3 per run instead of 6. Covered by
`test_dimensions_subset_judges_only_those` and
`test_c2_judge_not_called_for_non_c2_dimensions`.

### M-3. Small task count → noisy rankings — `DISCLOSED`
26 tasks total (11 operations, 5 outcomes, 10 equity_research); per-track N
is as low as 5. Disclosure added to `docs/methodology.md` ("Statistical
precision") and to the grade-app public methodology page (treat close calls
as ties). Bootstrap CIs over per-task composites remain a nice-to-have.

### M-4. Public fixtures start the contamination clock — `DISCLOSED`
Disclosure + policy added to `docs/methodology.md` ("Benchmark
contamination"): fixtures are regenerated (new seed + recomputed
`ground_truth.json`) per numbered benchmark version, and scores are only
comparable within a version. Plain-language version added to the grade-app
methodology page.

### M-5. Judge cannot verify evidence against data — `DISCLOSED`
Disclosure added to `docs/methodology.md` ("Judge scope") and, in plain
language, to the grade-app methodology page: the judge scores against gold
material; numbers are verified separately by the deterministic C1 scorer.

### M-6. Bare-float judging (no rationale) — `FIXED`
The judge prompt now asks for a brief (2-3 sentence) justification followed
by a final `SCORE: <float>` line; parsing prefers the last `SCORE:` line and
falls back to the last float, so numbers quoted in the justification are
never mistaken for the verdict. Non-reasoning judge `max_tokens` raised
16 → 384 to fit the rationale. Locked in before the first official run, so
no published scores change.

---

## Launch model lineup (verified against OpenRouter 2026-06-10)

| Leaderboard entry | OpenRouter slug | Pricing ($/M in / out) |
|---|---|---|
| Claude Opus 4.8 | `anthropic/claude-opus-4.8` | (see openrouter.ai) |
| Claude Sonnet 4.6 | `anthropic/claude-sonnet-4.6` | |
| Claude Haiku 4.5 | `anthropic/claude-haiku-4.5` | |
| GPT-5.5 (xhigh) | `openai/gpt-5.5` + `reasoning.effort=xhigh` | 5.00 / 30.00 |
| GPT-5.5 (high) | `openai/gpt-5.5` + `reasoning.effort=high` | 5.00 / 30.00 |
| GPT-5.5 (medium) | `openai/gpt-5.5` + `reasoning.effort=medium` | 5.00 / 30.00 |
| GPT-5.5 (low) | `openai/gpt-5.5` + `reasoning.effort=low` | 5.00 / 30.00 |
| Gemini 3.5 Flash | `google/gemini-3.5-flash` | 1.50 / 9.00 |
| Gemini 3.1 Pro | `google/gemini-3.1-pro-preview` | 2.00 / 12.00 |
| Nemotron 3 Ultra | `nvidia/nemotron-3-ultra-550b-a55b` | 0.50 / 2.50 |
| GPT-OSS 120B | `openai/gpt-oss-120b` | ~0.10 / 0.45 |

Notes:
- Gemini 3.1 Pro only exists as a `-preview` slug; re-verify before the run.
- The four GPT-5.5 effort rows depend on H-5 (reasoning passthrough).
- Open-weight rows (Nemotron, GPT-OSS) chosen for brand recognition with US
  educators; Llama and Mistral dropped per product decision 2026-06-10.

## Run protocol for the official leaderboard (after fixes)

```bash
python -m scripts.run_models \
    --models <comma-separated slugs above> \
    --pack operations --runs 5 --judge --out results/operations
# repeat for outcomes and equity_research, then merge per-pack results
```

- `--runs 5` (never 1 — see H-3), temperature 1.0 (default), judge required.
- Judge is auto-selected per candidate (Opus 4.8 default; GPT-5.5 xhigh for
  all Claude-family candidates — no judge shares a family with its
  candidate). `--judge-model` overrides.
- Record judge model + version in the methodology page alongside results,
  including the per-candidate judge assignment and the calibration caveat
  noted under C-3.
