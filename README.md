# GRADE

> **Private pre-release** — this repo is private/internal until launch; APIs and schemas are unstable.

repository for pearl AI benchmark

---

## Quick start

Get from clone to a scored result in under 10 minutes — no API key required.

```bash
git clone https://github.com/PearlEng/grade.git
cd grade
pip install -e .

python -m runner.cli \
    --pack operations \
    --adapter stub \
    --runs 1 \
    --out /tmp/grade_out
```

The **stub adapter** is network-free and deterministic.  It validates your
installation end-to-end without any model credentials.

For a full walkthrough — including switching to a real model via OpenRouter,
rendering a Markdown scorecard, and writing your own adapter — see
[examples/quickstart.md](examples/quickstart.md).

### Run all three task packs

```bash
python -m runner.cli --pack operations        --adapter stub --runs 1 --out /tmp/grade_ops
python -m runner.cli --pack outcomes          --adapter stub --runs 1 --out /tmp/grade_out
python -m runner.cli --pack equity_research   --adapter stub --runs 1 --out /tmp/grade_eq
```

### Use a real model (OpenRouter)

```bash
pip install -e ".[openrouter]"
export OPENROUTER_API_KEY="sk-or-..."

python -m runner.cli \
    --pack operations \
    --adapter openrouter \
    --model anthropic/claude-sonnet-4.6 \
    --runs 5 \
    --out /tmp/grade_real
```

---
