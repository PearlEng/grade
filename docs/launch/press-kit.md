# GRADE Press Kit

**G**rounded **R**easoning & **A**nalysis for **D**ata in **E**ducation

*Open benchmark · Education program analytics · Synthetic fixtures · Apache 2.0*

<!-- NOTE: <NAME> is a placeholder for the final repository name, pending the G3 naming decision. -->

---

## One-line description

GRADE is an open benchmark that measures how accurately and usefully AI systems analyze
realistic education program data — attendance records, outcome summaries, and research
references — against ground-truth validated fixtures and a six-dimension scoring rubric.

---

## Short description (two sentences)

GRADE is an open benchmark for AI in education program analytics. It evaluates whether
AI systems can produce grounded, evidence-linked, appropriately calibrated answers to
the kinds of questions program managers, district analysts, and researchers actually ask
about tutoring and intervention program data.

---

## Extended description (one paragraph)

GRADE — Grounded Reasoning and Analysis for Data in Education — is a public, open-source
benchmark that measures AI performance on education program analytics tasks. The V1
benchmark contains 26 tasks organized across five tracks: grounded retrieval and
computation, program snapshot and trend interpretation, operational coaching and
recommendations, equity and subgroup interpretation, and program effectiveness and
research reasoning. All tasks run against three fully synthetic fixture packs modeled on
a realistic tutoring program. Each (model × task) combination is evaluated across 5
repeated runs and scored on six dimensions: grounding accuracy (35%), insight quality
(20%), evidence linkage (15%), calibration and limitation handling (15%), consistency
(10%), and structure and usability (5%). The benchmark runner, schemas, fixtures, and
scoring pipeline are open-source under the Apache 2.0 license.

---

## The problem

AI tools are being adopted across education to help program managers, coordinators, and
analysts make sense of program data. Whether analyzing attendance trends, comparing
subgroup outcomes, or synthesizing research to guide coaching decisions, the quality of
AI-generated analysis directly affects the decisions practitioners make.

Existing general-purpose benchmarks do not measure this. A model can excel at broad
reasoning tasks while still hallucinating attendance rates, misattributing causation to
correlated signals, or failing to flag that a subgroup is too small to support a
reportable conclusion. GRADE addresses this gap with a benchmark purpose-built for
education program analytics.

---

## What we built

**Five analytical tracks** covering the range of questions educators and program analysts
ask about tutoring and intervention programs:

1. Grounded retrieval and computation — accurate fact lookup and arithmetic from structured data
2. Program snapshot and trend interpretation — calibrated summaries of month-over-month signals
3. Operational coaching and recommendations — evidence-backed, traceable recommendations
4. Equity and subgroup interpretation — responsible reasoning about attendance and outcome gaps
5. Program effectiveness and research reasoning — synthesis of local data and published findings

**Three synthetic fixture packs** representing a realistic tutoring program across an
operational quarter, including attendance records, session logs, survey responses,
month-over-month summaries, subgroup outcome data, and structured research references.
All data is fully synthetic, seeded for reproducibility, and designed to include the
kinds of structural messiness found in real program records.

**A six-dimension scoring rubric** with locked weights, explicit per-task guidance, and
automated scoring via ground-truth validated gold facts, deterministic claim validation,
and a judge-model rubric scorer.

**A repeated-run methodology** — each model is run 5 times per task — that distinguishes
consistent analytical behavior from single-run performance.

---

## Why it matters

- **Practitioners need trustworthy AI analysis.** Education program decisions affect
  students. Benchmarks that measure analytical accuracy, evidence citation, and epistemic
  calibration are more relevant to those decisions than general reasoning scores.

- **Equity tasks are first-class.** Track 4 specifically evaluates whether AI systems
  handle subgroup data responsibly — including correct suppression of low-N groups and
  appropriate caveating of disparities. This reflects real obligations that program
  analysts face when reporting on student outcomes.

- **Reproducibility is built in.** All fixtures are synthetically generated with a
  deterministic seed. Any researcher can run the same evaluation and get comparable
  results. The runner, schemas, and scoring code are all open-source.

- **The methodology is transparent.** Rubric weights, gold facts, forbidden claims, and
  required limitations are all documented in the task files. There are no hidden scoring
  criteria.

---

## How to participate

**Run the benchmark.** Clone the repository and run any OpenRouter-accessible model
against the full task set in under an hour. Results include a Markdown scorecard and a
machine-readable `results.json`.

**Contribute tasks or fixtures.** New tasks and fixture packs can be contributed via pull
request. Contribution requirements — task schema, gold-fact validation, forbidden-claim
documentation — are in `CONTRIBUTING.md`.

**Discuss methodology.** Open a GitHub Issue to propose changes to the scoring rubric,
track definitions, or fixture design. See `GOVERNANCE.md` for how decisions are made.

---

## Fact sheet

| Property | Value |
|---|---|
| Benchmark name | GRADE |
| Full name | Grounded Reasoning & Analysis for Data in Education |
| Version | 0.1.0.dev0 |
| License | Apache 2.0 |
| Repository | `github.com/PearlEng/<NAME>` |
| Task count (V1) | 26 tasks |
| Tracks | 5 |
| Fixture packs | 3 (synthetic, seeded) |
| Scoring dimensions | 6 |
| Runs per model × task | 5 |
| Runner | Generic CLI + OpenRouter adapter |
| Primary audience | Education researchers, AI developers, district analysts |
| Domain | Tutoring and intervention program analytics |

---

## Frequently asked questions

**Is this a real-world dataset?**

No. All fixture data is fully synthetic and seeded. No real students, tutors, schools, or
programs are represented. The synthetic data is designed to be structurally realistic —
including intentional messiness like uneven survey coverage, partial attendance records,
and embedded operational anomalies — but does not correspond to any actual program.

**Can I run this benchmark on a closed-source or proprietary model?**

Yes. The benchmark runner uses the OpenRouter API, which provides access to a wide range
of models. You can also implement a custom adapter that conforms to the runner's output
schema to evaluate any model accessible to you.

**How long does a full evaluation run take?**

This depends on the model and API latency. A full run of 26 tasks × 5 repetitions = 130
model calls. With typical API response times, a full evaluation completes in well under
two hours for most models.

**What does a "good" score look like?**

No external model results have been published yet. The benchmark is designed so that a
model producing factually accurate, well-cited, appropriately calibrated responses on
all 26 tasks would score near 1.0. Track profiles matter more than the overall
composite — a model profile shows which analytical capabilities are strong and which
need improvement.

**Does GRADE measure general AI capability?**

No. GRADE is scoped to education program analytics tasks using realistic synthetic data.
A high GRADE score is evidence of strong analytical behavior within this domain; it does
not generalize to other domains or task types.

**How is the benchmark licensed?**

The benchmark, runner, fixtures, and scoring code are released under the Apache 2.0
license. Task schema, rubric format, and output format are open.

**Can I extend the benchmark for a different program type?**

Yes. The fixture schema supports different program structures, grade levels, and
assessment types. The task schema accommodates new task types and track definitions.
See `CONTRIBUTING.md` for guidance on adding fixture packs and tasks.

**What if I disagree with a rubric or gold fact?**

Open a GitHub Issue. The `GOVERNANCE.md` file describes the process for proposing
changes to scoring criteria. All changes to locked weights or gold facts require a
documented rationale and go through the standard pull-request review process.

---

## Contact and links

| Resource | Location |
|---|---|
| Repository | `github.com/PearlEng/<NAME>` |
| Methodology | `docs/methodology.md` |
| Scoring reference | `docs/scoring.md` |
| Contributing guide | `CONTRIBUTING.md` |
| Governance | `GOVERNANCE.md` |
| Issues and discussion | GitHub Issues |

<!-- NOTE: <NAME> placeholders above must be replaced with the final repository name before publication. See G3 (naming decision) and G4 (launch runbook). -->
