# ml-agent

Agentic ML experimentation assistant — an LLM (Gemini) orchestrates dataset
inspection, model proposal, training, and evaluation over scikit-learn via
native function-calling, iterating toward a target metric in a supervised
loop.

**Status: in progress.** Project 5 of a 10-step self-directed learning
roadmap toward Data Science / ML Engineering / Agentic AI.

For the full session-by-session development history — real dead ends,
debugging stories, and design discussions — see [`DEVELOPMENT_LOG.md`](./DEVELOPMENT_LOG.md).
For implementation depth, see [`TECHNICAL_NOTES.md`](./TECHNICAL_NOTES.md);
for data-science methodology and findings, see
[`DATA_SCIENCE_ANALYSIS.md`](./DATA_SCIENCE_ANALYSIS.md).

---

## How It Works

```
User: "Find the best classifier for this dataset, optimizing for recall"
        ↓
Agent inspects the dataset (via Pandas) — shape, types, target distribution
        ↓
Agent (via Gemini, native function-calling) proposes candidate models + reasoning
        ↓
Agent trains each via scikit-learn, computes metrics
        ↓
Agent (via Gemini) interprets results, decides: good enough, or iterate further?
        ↓
Loop until convergence or max iterations
        ↓
Agent reports: best model, why, plain-English summary
```

This extends the agentic loop pattern from a prior project, `sql-agent`
(question → LLM reasoning → execution → interpretation → loop), into a
second domain, using **native Gemini function-calling** rather than the
simpler prompt-schema approach used previously.

---

## Architecture & Design Choices

**Core principle: deterministic code handles fact-gathering; the LLM
handles reasoning under uncertainty.** Dataset inspection computes shape,
class balance, and feature statistics — but makes no judgment calls.
Deciding what those facts *imply* is left to Gemini, reasoning over
structured facts it's handed. Which class counts as "positive" per
dataset, and whether a proposed hyperparameter is even valid, are likewise
resolved deterministically — never left for Gemini to guess or supply.

**Tool architecture — two genuinely different tool categories, kept
strictly separate:**

- **Category A — execution tools** (`list_available_models`, `train_model`,
  `evaluate_model`). Ordinary deterministic functions with real side
  effects. Same inputs → same outputs, every time. No awareness an LLM
  exists.
- **Category B — structured-decision tools** (`record_model_proposal`,
  `record_convergence_decision`). No execution — these force Gemini's
  free-form reasoning into parseable JSON.

```
     CATEGORY A — EXECUTION                     CATEGORY B — DECISION
     (real scikit-learn side effects,           (schema-enforced reasoning,
      deterministic, LLM-unaware)                nothing executes)

     list_available_models  ─┐                  record_model_proposal
     train_model              │◄──── verbatim ───── (model_type, hparams,
     evaluate_model           │      handoff         reasoning)
                              ▼
                    record_convergence_decision
                    (continue: bool, reasoning)
```

All of the agent's "agency" lives inside Category B calls — Category A is
plumbing that executes whatever B decided. If Gemini were swapped for a
different model, Category A would not need to change at all. Full
diagrams and the human-in-the-loop extension point (designed, not yet
built) are in `DEVELOPMENT_LOG.md`.

---

## Quickstart

```bash
git clone git@github.com:MelanieM2/ml-agent.git
cd ml-agent
uv sync
cp .env.example .env   # then fill in your real GEMINI_API_KEY
```

```bash
uv run python main.py
```

Prompts interactively for a dataset and optimization target if not given
as flags; everything else defaults sensibly.

```bash
uv run python main.py --dataset climate --target recall
uv run python main.py --dataset breast_cancer --target f1 --no-log-iterations
```

Other subcommands:

```bash
uv run python main.py compare --dataset climate    # cross-run comparison
uv run python main.py export --dataset climate     # CSV export
uv run python main.py report results/result_log_*.json   # Markdown report
```

---

## Example Session

Reconstructed from a real logged run (`DATA_SCIENCE_ANALYSIS.md` §9.1–9.2,
Climate Crashes, `optimization_target="recall"`) — every model, metric, and
quoted reasoning fragment below is real, from that actual run:

```
$ uv run python main.py --dataset climate --target recall

Running: dataset=climate, target=recall, model=gemini-3.1-flash-lite,
max_iterations=15, random_state=42, log_iterations=True

── Iteration 1–4 ──────────────────────────────────────────
Agent proposes: Logistic Regression (class_weight="balanced")
→ recall 0.78, precision 0.25
Agent's reasoning: "a strong baseline... significant room for
improvement, especially in precision"

── Iteration 5–8 ──────────────────────────────────────────
Agent proposes: Random Forest (n_estimators=200, max_depth=10,
class_weight="balanced")
→ recall drops to 0.22, precision rises to 0.4
Agent REJECTS this model: "dropped in recall significantly, making
it unsuitable for the stated optimization target"

── Iteration 9–12 ─────────────────────────────────────────
Agent proposes: SVM (kernel="rbf", C=1, class_weight="balanced")
→ recall 1.0, precision 0.18
Agent ACCEPTS — converged.

Final report: SVM (rbf kernel), recall=1.0, precision=0.18
```

The Random Forest rejection step is the interesting part — direct
evidence the agent isn't just landing on a good final answer, it's
actively tracking the stated metric through its intermediate reasoning
too, dropping a model specifically because it regressed on recall.

---

## Datasets

- **Primary: Climate Model Simulation Crashes** (OpenML id 1467). Binary
  classification — predict whether a combination of 18 physical
  parameters causes a numerical simulation crash. 540 rows, imbalanced
  (~91.5% success / 8.5% failure). `pos_label=1` (failure, the rare
  class).
- **Fallback: Breast Cancer Wisconsin** (`sklearn.datasets`). Used for
  fast, network-free debugging. Its positive-class convention (`1 =
  benign`, the majority class) is the *opposite* sense from Climate
  Crashes — `pos_label` is resolved per-dataset, never assumed
  consistent.

Datasets are loaded through a small registry in `dataset.py`; adding a new
one means writing one loader function — nothing else in the project needs
to change.

**Available models** (via `list_available_models`): Logistic Regression,
Random Forest, SVM — deliberately kept to three rather than an exhaustive
sklearn zoo, since this project's explicit growth area is agentic
tool-calling engineering, not exhaustive model search.

---

## Selected Findings

A few real, verified results from this project's development — full
methodology and numbers in `DATA_SCIENCE_ANALYSIS.md`:

- **Finding a real behavioral gap, fixing it, then verifying the fix.**
  Early runs showed the agent consistently trading recall away for
  accuracy or F1 — even on a dataset where missing a rare failure case is
  the costlier mistake — because nothing in its prompt ever stated which
  metric actually mattered. Adding an explicit `optimization_target`
  fixed this: three follow-up runs all converged on recall-maximizing
  models, each explicitly citing the stated target in its own reasoning.
  The fix also surfaced a real trade-off worth being honest about —
  optimizing for recall alone, with no precision floor, tends to produce
  models that catch every case at the cost of many false alarms — noted
  as an open refinement, not a solved problem.
- **Isolating a causal driver, not just observing a correlation.** Two
  runs that changed `max_iter` and `C` together looked like `max_iter`
  drove a recall improvement. A controlled follow-up changing `C` alone
  confirmed `C` — not `max_iter` — was the actual driver; `max_iter` only
  affects whether sklearn's solver reports convergence, not what the
  model predicts.
- **A real reproducibility bug, found via a contradiction, not a code
  review.** Three `RandomForestClassifier` runs with different
  hyperparameters produced identical results, suggesting a fixed seed —
  until a fourth run, same hyperparameters as the first, produced a
  *different* result. Root cause: `random_state` was never threaded into
  the estimator at all; the matching results had been coincidence.
- **A silent bug in "final model" reporting, found and fixed.** A
  heuristic assuming "last evaluated model = final model" was confirmed
  wrong on a real run, where the actual chosen model (per the agent's own
  reasoning) wasn't the last one evaluated. Fixed by resolving the final
  model structurally, by best score on the run's own optimization target.

---

## Project Structure

```
ml-agent/
├── DATA_SCIENCE_ANALYSIS.md    # methodology, findings, statistical reasoning
├── DEMO_RUN.md                 # verified command sequence: setup, tests, two live runs, compare, export, report
├── DEVELOPMENT_LOG.md          # full session-by-session development history
├── README.md
├── SECURITY.md
├── TECHNICAL_NOTES.md
├── main.py                     # CLI entry point (run/compare/export/report)
├── ml_agent/
│   ├── agent.py                 # orchestration loop
│   ├── compare_runs.py          # cross-run comparison
│   ├── dataset.py                # dataset registry + inspection
│   ├── gemini_client.py         # Gemini client + function-call dispatch loop
│   ├── reporting.py              # CSV export + Markdown viewer
│   ├── tools.py                   # tool schemas + dispatch functions
│   └── trainer.py                 # model storage, validation, sklearn fit/evaluate
├── results/                     # gitignored — per-run and comparison outputs
├── tests/                       # 52 tests passing
└── uv.lock
```

`ml_agent/` is a real installed package. See `DEVELOPMENT_LOG.md` for the
fully annotated tree and per-file history.

**Results & Reports.** Every run persists to
`results/result_log_<timestamp>_<dataset>.json` — the full
proposal/evaluation/rejection trail plus the final model and metrics.
`compare` folds several runs into one `comparison_*.json`; `export`/
`report` turn either into a flat CSV or a human-readable Markdown
summary, all built on one shared `summarize_run()` path rather than
three separate implementations. Full format details in
[`TECHNICAL_NOTES.md`](./TECHNICAL_NOTES.md).

---

## Security

Summary — full policy in [`SECURITY.md`](./SECURITY.md):

- **Dependency pinning:** exact versions only (`add-bounds = "exact"`),
  with a 7-day quarantine on newly published releases
  (`exclude-newer = "7 days ago"`) to reduce exposure to freshly injected
  malicious packages.
- **Vulnerability scanning:** `uv audit` against the OSV database before
  running application code.
- **Cryptographic lockfile:** `uv.lock` is committed, with checksums
  verified on every `uv sync`.
- **API key handling:** loaded from a gitignored `.env` via
  `python-dotenv`; never hardcoded.
- **LLM-generated tool-call guard** (the project-specific risk this
  project's domain introduces): dispatch is restricted to an explicit
  whitelist of 5 known tool names, and every Gemini-proposed
  `model_type`/hyperparameter is validated against a known schema
  *before* touching scikit-learn — a hallucinated or out-of-range value
  raises a clear, named error rather than failing deep inside a
  third-party constructor.

---

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Gemini API key — [get one here](https://aistudio.google.com/)

---

## Development Notes

This project was developed as part of a structured learning portfolio
toward Data Science, ML Engineering, and Agentic AI.

**Development tool:** Claude Sonnet 5 (Anthropic), used as a
pair-programming and design-review collaborator — for architecture
decisions, code review, concept explanations, and test design. All
architectural decisions were reviewed and made by the project author; AI
assistance focused on explaining concepts, generating boilerplate, and
surfacing issues for human review, not unsupervised code generation.

**Runtime AI:** `gemini-3.1-flash-lite` (Google) is the model actually
orchestrating the agent's tool-calling loop at runtime — this is the AI
system the project *is*, as distinct from the AI assistance used to
*build* it.

---

## Roadmap / Future Improvements

- [ ] **Human-in-the-loop hook** — design substantially explored (live
      interrupt on stall/crash + post-hoc debug assist on a flagged run);
      implementation deferred to its own dedicated session.
- [ ] CLI-layer test coverage for `main.py`'s `export`/`report`
      subcommands (the underlying logic is tested; the argument-parsing
      glue isn't yet).
- [ ] A "register/list datasets from the terminal" feature
      (`dataset list` / `dataset upload`).
- [ ] An optional second `--model-random-state` flag, to isolate
      split-variance from model-internal-variance as an opt-in advanced
      knob.
- [ ] Housekeeping: delete pre-2026-07-29 legacy result files no longer
      needed for reference.

Full, currently-open TODO list with status and context in
`DEVELOPMENT_LOG.md`.

---

## Project Context

This project is **Project 5** in a broader personal systems engineering
and AI learning stack, including:

* **Project 0:** Local development infrastructure & backup system (SSH, rsync, Git mirroring)
* **Project 1:** AI-assisted research digester (arXiv + Gemini pipeline)
* **Project 2:** Linux system inspection and analysis tooling
* **Project 3:** Log Analyzer — Pandas-based log parsing and AI summarization
* **Project 4:** [`sql-agent`](https://github.com/MelanieM2/sql-agent) — agentic natural language querying over SQLite
* **Project 5:** `ml-agent` (this project) — agentic ML experimentation over scikit-learn

This roadmap is a flexible, evolving experimental playground for learning
rather than a fixed plan, bridging mathematical and machine learning
foundations with practical systems engineering and production-style
automation workflows.

**Connections to other projects:**
- Extends the agentic-loop pattern established in `sql-agent` (question →
  LLM reasoning → execution → interpretation → loop) into a second
  domain, stepping up to native Gemini function-calling.
- Inherits the dependency-pinning and vulnerability-scanning security
  framework from `sql-agent`, adapted to this project's actual risk (LLM
  tool-call dispatch to scikit-learn, vs. SQL injection there).
- Planned: Project 6 will extend this project's agentic patterns into
  broader AI-assisted workflows.

---

## Screenshots

A full, real command sequence — environment setup, tests, two live runs,
compare, export, and report — is documented in [`DEMO_RUN.md`](./DEMO_RUN.md).
Terminal screenshots are still planned for this section.