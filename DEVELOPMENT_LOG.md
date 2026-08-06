# ml-agent

This file is the original, unedited development log — every session, in the order it happened, including dead ends and corrections made in place rather than edited away. `README.md`, `TECHNICAL_NOTES.md`, and `DATA_SCIENCE_ANALYSIS.md` each give a cleaner, shorter cut through parts of this same story; this file is where the reader can dig in much deeper if they want to follow the full development journey behind this project.

**How to read this file.** Sessions here have same-day counterparts in `TECHNICAL_NOTES.md` (and, from 2026-07-17 onward, `DATA_SCIENCE_ANALYSIS.md`) — see those files' own session maps for how the three line up. A short "Problem, briefly" box sits at the start of each major thread here, pointing forward to where that thread gets its cleaner treatment. Some reference sections (setup, datasets, security, project structure) now just link to `README.md`'s shorter version rather than repeat it — nothing was cut, only de-duplicated.

---

## Status index

| Section | Date | Covers | See also |
|---|---|---|---|
| [Concept](#concept) | — | Core architectural principle | [README §Architecture](../README.md#architecture--design-choices) |
| [Tool architecture: Category A vs Category B](#tool-architecture-category-a-execution-vs-category-b-decision) | — | Execution vs. decision tool split | [README §Architecture](../README.md#architecture--design-choices) |
| [`test_tools.py`](#test_toolspy--the-schemafunction-drift-check) | 2026-07-15 → 07-27 | Schema/function drift check | — |
| [`gemini_client.py` — the agent loop](#gemini_clientpy--the-agent-loop) | — | Native function-calling loop design | — |
| [Per-iteration logging](#per-iteration-logging-2026-07-19) | 2026-07-19 | Opt-in per-iteration logging | [TN Part 3](./TECHNICAL_NOTES.md#part-3-per-iteration-logging-in-run_agent_loop-2026-07-19)/[DS §9](./DATA_SCIENCE_ANALYSIS.md#9-update-2026-07-19-per-iteration-logging--first-look-inside-a-runs-actual-search-path) |
| [Cross-run comparison](#cross-run-comparison-2026-07-24-renamed-and-dataset-scoped-2026-07-29) | 2026-07-24 → 07-29 | `compare_runs.py`, `compare` subcommand | [TN Part 4](./TECHNICAL_NOTES.md#part-4-fit-time-warning-capture-corrected-run-persistence-and-cross-run-comparison-2026-07-24)/[DS §10](./DATA_SCIENCE_ANALYSIS.md#10-update-2026-07-24-cross-run-comparison-goes-live--the-8493-question-finally-has-real-data-behind-it) |
| [Results reporting](#results-reporting-csv-export-and-the-markdown-viewer-2026-08-01) | 2026-08-01 | `reporting.py`, final-model bug fix | [TN Part 7](./TECHNICAL_NOTES.md#part-7-results-reporting-reportingpy-a-real-final-model-bug-fix-and-a-pytest-collection-bug-2026-08-01)/[DS §13](./DATA_SCIENCE_ANALYSIS.md#13-update-2026-08-01-summarize_runs-final-model-resolution-corrected--a-real-breast_cancer-runs-reported-outcome-was-wrong) |
| [Fit-time warning capture](#fit-time-warning-capture-2026-07-24) | 2026-07-24 | Warning capture, all categories | [TN Part 4](./TECHNICAL_NOTES.md#part-4-fit-time-warning-capture-corrected-run-persistence-and-cross-run-comparison-2026-07-24) |
| [Closing the `ConvergenceWarning` shortlist](#closing-the-convergencewarning-shortlist-2026-07-27) | 2026-07-27 | `max_iter` exposed, shortlist decided | [TN Part 5](./TECHNICAL_NOTES.md#part-5-max_iter-exposed-as-a-hyperparameter-confirming-the-agent-reasoning-pathway-was-already-wired-2026-07-27)/[DS §11](./DATA_SCIENCE_ANALYSIS.md#11-update-2026-07-27-max_iter-exposed-as-a-tunable-hyperparameter--the-convergencewarning-shortlist-decision-tested-against-3-live-runs) |
| [Reproducibility: seeding every estimator](#reproducibility-seeding-every-estimator-2026-07-29) | 2026-07-29 | `random_state` bug found & fixed | [TN Part 6](./TECHNICAL_NOTES.md#part-6-results-file-renaming-compare-subcommand-and-the-reproducibility-fix-2026-07-29)/[DS §12](./DATA_SCIENCE_ANALYSIS.md#12-update-2026-07-29-the-max_iter-vs-c-confound-resolved-117s-anomaly-explained-and-new-breast-cancer-results) |
| [`Trainer` — model storage and encapsulation](#trainer--model-storage-and-encapsulation) | — | Storage/validation design | — |
| [Validating Gemini's arguments](#validating-geminis-arguments-before-training) | — | Hyperparameter validation design | — |
| [`agent.py` — wiring the dispatch table](#agentpy--wiring-the-dispatch-table) | 2026-07-17 | `DispatchResult`, `run_session`, optimization-target fix | [TN Part 2](./TECHNICAL_NOTES.md#part-2-orchestration-loop-wiring--implementation-details-2026-07-17)/[DS §8](./DATA_SCIENCE_ANALYSIS.md#8-update-2026-07-17-the-optimization-target-fix-tested-for-the-first-time-against-a-live-agent) |
| [`main.py` — the CLI entry point](#mainpy--the-cli-entry-point-2026-07-27-extended-2026-07-29-2026-08-01) | 2026-07-27 → 08-01 | CLI design, dataset menu | — |
| [Project structure](#project-structure) | — | Annotated file tree | [README §Project Structure](../README.md#project-structure) (current, non-annotated version) |
| [Datasets](#datasets) | — | Dataset registry, `pos_label` | [README §Datasets](../README.md#datasets) (quick version) |
| [Available models](#available-models-via-list_available_models) | — | Hyperparameter ranges | [README §Datasets](../README.md#datasets) (quick version) |
| [Setup](#setup) | — | Install steps | [README §Quickstart](../README.md#quickstart) |
| [Running](#running) | — | Run commands, subcommands | [README §Quickstart](../README.md#quickstart) |
| [Testing](#testing) | — | Test suite, `testpaths` bug | — |
| [Security](#security) | — | (was just a pointer) | [README §Security](../README.md#security), [SECURITY.md](./SECURITY.md) |
| [Roadmap context](#roadmap-context) | — | Progression + human-in-the-loop design | [README §Roadmap](../README.md#roadmap--future-improvements) (brief version) |
| [Development Notes](#development-notes) | — | Per-session recap, all sessions | — |

---

## Table of contents

<details>
<summary><a href="#concept">Concept</a></summary>

- [Core architectural principle](#core-architectural-principle)

</details>

<details>
<summary><a href="#tool-architecture-category-a-execution-vs-category-b-decision">Tool architecture: Category A (execution) vs Category B (decision)</a></summary>

</details>

<details>
<summary><a href="#test_toolspy--the-schemafunction-drift-check">`test_tools.py` — the schema/function drift check</a></summary>

- [Why this exists](#why-this-exists)
- [The mechanism](#the-mechanism)
- [Extension (2026-07-19): guarding the dispatch-table override keys](#extension-2026-07-19-guarding-the-dispatch-table-override-keys)
- [Reviewed and resolved (2026-07-27)](#reviewed-and-resolved-2026-07-27)

</details>

<details>
<summary><a href="#gemini_clientpy--the-agent-loop">`gemini_client.py` — the agent loop</a></summary>

- [Design decisions, stated explicitly](#design-decisions-stated-explicitly)
- [`client.chats.create()` — internal dynamics](#clientchatscreate--internal-dynamics)

</details>

<details>
<summary><a href="#per-iteration-logging-2026-07-19">Per-iteration logging (2026-07-19)</a></summary>

- [Why this exists](#why-this-exists-1)
- [How it works](#how-it-works)
- [Reading a log](#reading-a-log)
- [Persistence (smoke-test usage only, not committed package behavior)](#persistence-smoke-test-usage-only-not-committed-package-behavior)

</details>

<details>
<summary><a href="#cross-run-comparison-2026-07-24-renamed-and-dataset-scoped-2026-07-29">Cross-run comparison (2026-07-24; renamed and dataset-scoped 2026-07-29)</a></summary>

- [Why this exists](#why-this-exists-2)
- [How it works](#how-it-works-1)
- [Renamed and dataset-scoped (2026-07-29)](#renamed-and-dataset-scoped-2026-07-29)
- [Discoverability: the `compare` subcommand (2026-07-29)](#discoverability-the-compare-subcommand-2026-07-29)
- [Verified against real data](#verified-against-real-data)
- [Known limitation, reproduced and documented, still not fixed](#known-limitation-reproduced-and-documented-still-not-fixed)

</details>

<details>
<summary><a href="#results-reporting-csv-export-and-the-markdown-viewer-2026-08-01">Results reporting: CSV export and the Markdown viewer (2026-08-01)</a></summary>

- [Why this exists](#why-this-exists-3)
- [How it works](#how-it-works-2)
- [A real bug found while building this: "last evaluated = final model" was wrong](#a-real-bug-found-while-building-this-last-evaluated--final-model-was-wrong)
- [The `export` and `report` subcommands](#the-export-and-report-subcommands)
- [Verified against real data](#verified-against-real-data-1)

</details>

<details>
<summary><a href="#fit-time-warning-capture-2026-07-24">Fit-time warning capture (2026-07-24)</a></summary>

- [Why this exists](#why-this-exists-4)
- [How it works](#how-it-works-3)
- [Verified against real data](#verified-against-real-data-2)
- [Resolved (2026-07-27)](#resolved-2026-07-27)

</details>

<details>
<summary><a href="#closing-the-convergencewarning-shortlist-2026-07-27">Closing the `ConvergenceWarning` shortlist (2026-07-27)</a></summary>

- [The shortlist, and the decision made](#the-shortlist-and-the-decision-made)
- [The plumbing was already there — nothing needed building for option B](#the-plumbing-was-already-there--nothing-needed-building-for-option-b)
- [Schema change: `max_iter` added to `list_available_models`](#schema-change-max_iter-added-to-list_available_models)
- [Verified against 3 live runs](#verified-against-3-live-runs)
- [The `max_iter`-vs-`C` confound, isolated (2026-07-29)](#the-max_iter-vs-c-confound-isolated-2026-07-29)

</details>

<details>
<summary><a href="#reproducibility-seeding-every-estimator-2026-07-29">Reproducibility: seeding every estimator (2026-07-29)</a></summary>

- [The bug, found via a real anomaly, not a code review](#the-bug-found-via-a-real-anomaly-not-a-code-review)
- [The fix](#the-fix)
- [A deliberate trade-off, decided rather than defaulted into](#a-deliberate-trade-off-decided-rather-than-defaulted-into)
- [Verified against real data](#verified-against-real-data-3)
- [Test coverage (`test_trainer.py`, 2026-08-03)](#test-coverage-test_trainerpy-2026-08-03)

</details>

<details>
<summary><a href="#trainer--model-storage-and-encapsulation">`Trainer` — model storage and encapsulation</a></summary>

- [The problem](#the-problem)
- [`Trainer` class (final form)](#trainer-class-final-form)

</details>

<details>
<summary><a href="#validating-geminis-arguments-before-training">Validating Gemini's arguments before training</a></summary>

- [The problem this solves](#the-problem-this-solves)
- [Design: a standalone function, not a `Trainer` method](#design-a-standalone-function-not-a-trainer-method)
- [A schema ambiguity, caught and fixed](#a-schema-ambiguity-caught-and-fixed)

</details>

<details>
<summary><a href="#agentpy--wiring-the-dispatch-table">`agent.py` — wiring the dispatch table</a></summary>

- [The problem](#the-problem-1)
- [`DispatchResult`: why `build_dispatch_table`'s return type changed (2026-07-17)](#dispatchresult-why-build_dispatch_tables-return-type-changed-2026-07-17)
- [`run_session`: the orchestration entry point (2026-07-17; extended 2026-07-19)](#run_session-the-orchestration-entry-point-2026-07-17-extended-2026-07-19)
- [What's genuinely still missing](#whats-genuinely-still-missing)

</details>

<details>
<summary><a href="#mainpy--the-cli-entry-point-2026-07-27-extended-2026-07-29-2026-08-01">`main.py` — the CLI entry point (2026-07-27; extended 2026-07-29, 2026-08-01)</a></summary>

- [Why this needed building specifically](#why-this-needed-building-specifically)
- [How the dataset menu stays in sync with the registry, automatically](#how-the-dataset-menu-stays-in-sync-with-the-registry-automatically)
- [What's a genuine judgment call vs. a secondary knob](#whats-a-genuine-judgment-call-vs-a-secondary-knob)
- [The `compare` subcommand (2026-07-29)](#the-compare-subcommand-2026-07-29)
- [The `export` and `report` subcommands (2026-08-01)](#the-export-and-report-subcommands-2026-08-01)
- [Free-tier rate limit note (2026-07-29)](#free-tier-rate-limit-note-2026-07-29)
- [Renamed results-file convention (2026-07-29)](#renamed-results-file-convention-2026-07-29)
- [Verified against real live runs](#verified-against-real-live-runs)

</details>

<details>
<summary><a href="#project-structure">Project structure</a></summary>

- [Why `agent.py` and `gemini_client.py` are separate](#why-agentpy-and-gemini_clientpy-are-separate)

</details>

<details>
<summary><a href="#datasets">Datasets</a></summary>

- [What `pos_label` actually is](#what-pos_label-actually-is)

</details>

<details>
<summary><a href="#available-models-via-list_available_models">Available models (via `list_available_models`)</a></summary>

</details>

<details>
<summary><a href="#setup">Setup</a></summary>

</details>

<details>
<summary><a href="#running">Running</a></summary>

</details>

<details>
<summary><a href="#testing">Testing</a></summary>

</details>

<details>
<summary><a href="#security">Security</a></summary>

</details>

<details>
<summary><a href="#roadmap-context">Roadmap context</a></summary>

- [Progression of "next up"](#progression-of-next-up)
- [Human-in-the-loop design (explored 2026-08-03, implementation deferred)](#human-in-the-loop-design-explored-2026-08-03-implementation-deferred)

</details>

<details>
<summary><a href="#development-notes">Development Notes</a></summary>

- [Real-world debugging notes (2026-07-13 session)](#real-world-debugging-notes-2026-07-13-session)
- [Testing/tooling notes (2026-07-15 session)](#testingtooling-notes-2026-07-15-session)
- [Orchestration wiring notes (2026-07-17 session)](#orchestration-wiring-notes-2026-07-17-session)
- [CLI entry point and threshold-revisiting notes (2026-07-27 session)](#cli-entry-point-and-threshold-revisiting-notes-2026-07-27-session)
- [Reproducibility, naming, and discoverability notes (2026-07-29 session)](#reproducibility-naming-and-discoverability-notes-2026-07-29-session)
- [Results reporting and two real bugs found (2026-08-01 session)](#results-reporting-and-two-real-bugs-found-2026-08-01-session)
- [Test design and empirical verification notes (2026-08-03 session)](#test-design-and-empirical-verification-notes-2026-08-03-session)

</details>

---

_Last updated: 2026-08-03_

Agentic ML experimentation assistant — an LLM (Gemini) orchestrates dataset inspection, model proposal, training, and evaluation over scikit-learn via native function-calling, iterating toward a target metric in a supervised loop.

**Status: in progress.** This is Project 5 in a 10-step self-directed learning roadmap toward Data Science / ML Engineering / Agentic AI.

- `dataset.py` — ✅ done and tested.
- `tools.py` — ✅ done (5 tool schemas + inclusive-bounds convention documented; `TOOL_FUNCTIONS` name→callable mapping added 2026-07-15, now actively reused by `agent.py`'s dispatch-table wiring — see "Project structure" below). **`max_iter` exposed as a tunable Logistic Regression hyperparameter (2026-07-27)** — gives the agent a concrete lever to act on the `ConvergenceWarning` `trainer.py` already surfaces — see "Closing the `ConvergenceWarning` shortlist" below.
- `trainer.py` — ✅ done, **and now fully test-covered (2026-08-03).** Real sklearn fit/predict logic implemented: `train_model` fits the correct estimator per `model_type` via a registry lookup against `tools.py`'s schema; `evaluate_model` computes accuracy, precision, recall, f1, and a correctly-oriented confusion matrix using `pos_label`. **Fit-time warning capture added (2026-07-24)** — `train_model` now returns any warning scikit-learn raised during fitting (e.g. a `ConvergenceWarning`), never just silently letting it print to the terminal — see "Fit-time warning capture" below. **`random_state` threaded into every estimator (2026-07-29)** — closes a real reproducibility bug: `RandomForestClassifier` was previously instantiated with no seed at all, so identical hyperparameters could (and did) produce different results across separate runs — see "Reproducibility: seeding every estimator" below. **17 new tests (`test_trainer.py`, 2026-08-03)** close out the item named on 2026-08-01 as "still never started" — see "Test coverage" under "Reproducibility" below for what they check and a real, three-attempt debugging story worth reading if you're curious how a test like this actually gets built.
- `agent.py` — ✅ **orchestration loop wired end-to-end (2026-07-17), verified via 3 real live runs.** `build_dispatch_table` now returns a `DispatchResult` (dispatch table + the run's `X`/`y`), and `run_session(dataset_name, optimization_target)` connects it, `inspect_dataset`, and `gemini_client.run_agent_loop` into one real, callable entry point — see "`agent.py` — wiring the dispatch table" below. `run_session` also now forwards an optional `log_iterations` flag (2026-07-19) straight through to `run_agent_loop` — see "Per-iteration logging" below. **`build_dispatch_table`'s `random_state` is now also bound into `Trainer.train_model` (2026-07-29)**, not just the train/test split — see "Reproducibility: seeding every estimator" below.
- `gemini_client.py` — ✅ **done and verified.** Implements the full agent loop: sends dataset context + tool schemas to Gemini, dispatches whichever tool call comes back, feeds results back, repeats until convergence or a max-iterations guard. Verified via multiple real end-to-end runs across both datasets (see "Data Science Notes" below). **Now also supports optional per-iteration logging (2026-07-19)** — see "Per-iteration logging" below. **`MAX_ITERATIONS` raised from 10 to 15 (2026-07-27)** — the 07-12 debugging-safety-net rationale for keeping it small was explicitly conditioned on "once the loop is trusted end-to-end"; 9 live runs across two sessions, plus a `main.py` run that hit the old ceiling mid-retry on legitimate in-progress work, met that condition — see "`gemini_client.py` — the agent loop" below.
- `test_tools.py` — ✅ **done, 12 tests passing (2026-07-15, extended 2026-07-19).** The `TOOL_SCHEMAS` ↔ real-function signature drift check, using `inspect.signature()`, plus a guard on the two dispatch-table override keys. See "`test_tools.py` — the schema/function drift check" below. **Review resolved (2026-07-27):** `train_model`'s new `warnings` return key needed no update here — every assertion in this file inspects a function's *parameters* via `inspect.signature()`, never its return value, and `train_model`'s own parameter list is unchanged.
- **`compare_runs.py` (2026-07-24) — ✅ done, verified against 6 real live runs, plus several more since.** Turns several persisted run files into one `comparison_<timestamp>_<dataset>.json`, summarizing each run's model sequence, final metrics, any warnings encountered, and elapsed time. **Renamed and dataset-scoped (2026-07-29):** the underlying files are now `result_log_<timestamp>_<dataset_name>.json` (was `smoke_test_log_<timestamp>.json`), and `build_comparison` now accepts an optional `dataset_name` filter — comparing runs across two different datasets was judged never acceptable, since their metrics aren't comparable against different data. **Final-model resolution fixed, and the standalone entry point dataset-restricted (2026-08-01):** `summarize_run`'s "last evaluated = final model" rule was confirmed wrong on a real file; final model is now resolved by best score on the run's own optimization target, with a `final_model_ambiguous` flag surfacing any disagreement. `python -m ml_agent.compare_runs` now also requires a dataset, matching `main.py`. See "Cross-run comparison" and "Results reporting" below.
- **`main.py` (2026-07-27; extended 2026-07-29, 2026-08-01) — ✅ done.** The real, committed public entry point — `run_smoke_test.py` (gitignored, hardcoded) was never reachable by anyone cloning the repo. Interactive/CLI hybrid: prompts for `--dataset`/`--target` only when omitted, everything else silently defaults. **`compare` subcommand (2026-07-29):** `python main.py compare --dataset NAME` surfaces `compare_runs.py` under the same entry point as `run`. **`export` and `report` subcommands (2026-08-01):** CSV spreadsheet export and a human-friendly Markdown viewer, both mirroring `compare`'s pattern — see "`main.py` — the CLI entry point" and "Results reporting" below.
- **`ml_agent/reporting.py` (2026-08-01) — ✅ done.** CSV export (the project's named gate before going public) and a Markdown viewer (single-run or cross-run comparison, auto-detected by filename) — both built on `compare_runs.py`'s `summarize_run()`. See "Results reporting: CSV export and the Markdown viewer" below.
- **Human-in-the-loop hook: scope agreed 2026-07-19, design substantially refined 2026-08-03, implementation still deferred to its own dedicated session.** Two genuinely different modes, not one: a **live** interrupt, triggered by deterministic code (not Gemini itself) watching for known signals — a convergence stall, a hard error/crash — synchronously pausing to ask the human before continuing; and a **post-hoc** debug assist, triggered only when a completed run is flagged as needing a closer look, offering a plain-language explanation of the run plus a genuine data-science causal analysis (not just a summary) — by default reasoning over what `reporting.py` already captures (no new logging needed), with an optional deeper escalation to the raw `result_log_*.json` on request. What still needs a dedicated design session: the precise definition of "flagged" beyond the existing `final_model_ambiguous` (candidates: a suspiciously low final metric, a warning that never got resolved — neither yet defined precisely). Subsumes and supersedes the earlier separate `Agent-decisions.md` idea (2026-07-27) and the "Gemini adjudicates `final_model_ambiguous`" idea (2026-08-01) — both folded into this one design. See "Roadmap context" below for the full write-up and diagram.
- Also not yet started: `AGENTS.md`, and CLI-layer tests for `main.py`'s `export`/`report` subcommands (flagged 2026-08-01 — the underlying `reporting.py`/`compare_runs.py` logic is tested; the argument-parsing/file-writing glue around it isn't yet). **A general repo-wide readability pass** (trimming this README down from its current session-log shape into something more genuinely browsable, cleaning up stray commented-out lines across the codebase) is explicitly planned but deferred to its own dedicated session — see "Roadmap context."

---

## Concept

Instead of manually writing and tuning scikit-learn models, an LLM orchestrates the pipeline:

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

This extends the agentic loop pattern from a prior project, `sql-agent` (question → LLM reasoning → execution → interpretation → loop), into a second domain, using **native Gemini function-calling** rather than the simpler prompt-schema approach used previously — a deliberate step up in tool-calling sophistication.

### Core architectural principle

Deterministic code handles fact-gathering; the LLM handles reasoning under uncertainty. `dataset.py`'s `inspect_dataset()` computes shape, class balance, missing values, and feature statistics — but makes no judgment calls. Deciding what those facts *imply* (e.g. "this class imbalance means recall-aware modeling is appropriate") is left to Gemini, reasoning over the structured facts it's handed. The project does not automate data-science judgment — it automates the mechanical prep work so that judgment has good inputs to work from.

This same principle extends deeper into the codebase over time:
- Into `tools.py`: **which class counts as "positive"** for a given dataset is also a fact, not a judgment call, and is never something Gemini supplies or guesses (see "Datasets" → "What `pos_label` actually is" below).
- Into `trainer.py`: **whether a proposed model/hyperparameter combination is even valid** is likewise a fact checked deterministically — against `list_available_models()`'s own schema — before any training happens, rather than left for Gemini's arguments to be trusted blindly (see "Validating Gemini's arguments before training" below).
- Into `tools.py`'s parameter-visibility convention (confirmed 2026-07-15, project-wide, not a one-off): **keyword-only parameters are always internally injected, never something Gemini supplies or sees.** `evaluate_model`'s `pos_label` is the canonical example — it sits after `*` in the signature and is deliberately absent from its `TOOL_SCHEMAS` entry, because which class is "positive" is a documented dataset fact, not something Gemini should ever be asked to guess.
- Into `agent.py`'s `run_session` (2026-07-17): **which class is "positive" is a fact, resolved once per dataset — but what metric to optimize for (recall, precision, accuracy) is a judgment call**, made once per *run*, not baked into the dataset registry alongside `pos_label`. This is why `optimization_target` is a plain parameter on `run_session`, not a dataset fact — see "`agent.py` — wiring the dispatch table" below for the full reasoning, and for what "optimizing for recall" concretely means on this project's primary dataset.
- Into `run_agent_loop`'s new `log_iterations` flag (2026-07-19): **whether to log is a caller's choice, not something the loop decides for itself.** `run_session` makes no logging decisions of its own — it just forwards whatever the caller asked for, keeping the same separation of concerns as everywhere else in this list.
- Into `main.py` (2026-07-27): **`optimization_target` is now constrained to the exact four metrics `evaluate_model` actually computes** (`recall`/`precision`/`accuracy`/`f1`), closing a gap that existed as long as it was unvalidated free text: a typo'd or invented target would previously reach Gemini's prompt with no connection to what the tool results could actually support — not malicious, just silently disconnected from reality. See "`main.py` — the CLI entry point" below.
- Into `trainer.py`/`agent.py` (2026-07-29): **reproducibility is a single, shared knob, not two.** `--random-state` already existed for the train/test split; it now also seeds every model's own internal randomness (`RandomForestClassifier`'s bagging, in particular). A second, separate seed for "the models" specifically was considered and deliberately rejected for now — see "Reproducibility: seeding every estimator" below for the real trade-off this gives up (the ability to isolate split-variance from model-variance), and why it's an acceptable one given this project's actual goal.

---

## Tool architecture: Category A (execution) vs Category B (decision)

A key design decision, made explicit early rather than left implicit in code: native function-calling is used here for two genuinely different purposes, kept strictly separate.

- **Category A — execution tools** (`list_available_models`, `train_model`, `evaluate_model`). Ordinary deterministic Python/scikit-learn functions with real side effects (train a model, compute metrics). They have no awareness that an LLM exists; they just receive arguments and run. Same inputs → same outputs, every time.
- **Category B — structured-decision tools** (`record_model_proposal`, `record_convergence_decision`). No execution happens here at all. Calling one of these captures Gemini's chosen arguments as a dict — the tool exists purely to force free-form reasoning into parseable JSON instead of prose.

**The consequence that matters:** all of the agent's "agency" lives entirely inside Category B tool calls. Category A is just plumbing that executes whatever B decided — Category B's output doesn't just inform Category A, it *becomes* Category A's function arguments, verbatim, via a direct handoff. If Gemini were swapped for a different model, or even a human typing choices into a CLI, Category A would not need to change at all.

```
                              ┌─────────────────────────────┐
                              │   Front-loaded once at start │
                              │   inspect_dataset(X, y)      │
                              │   → dict (shape, balance,    │
                              │     missing values, feature  │
                              │     stats) — part of the     │
                              │     initial prompt context,  │
                              │     NOT a callable tool       │
                              └───────────────┬───────────────┘
                                              │
                                              ▼
                    ┌──────────────────────────────────────────────┐
                    │              agent.py  (the loop)             │
                    │   owns: state, iteration count, results-so-far│
                    └───────────────┬────────────────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────────────┐
                    │  gemini_client.py sends context + tool specs   │
                    │  to Gemini, dispatches whichever tool call     │
                    │  comes back                                    │
                    └───────────────┬────────────────────────────────┘
                                     │
                 ┌───────────────────┴────────────────────┐
                 ▼                                         ▼
     CATEGORY A — EXECUTION                     CATEGORY B — DECISION
     (real scikit-learn side effects,           (schema-enforced reasoning,
      deterministic, LLM-unaware)                nothing executes)

     ┌─────────────────────────┐                ┌──────────────────────────┐
     │ list_available_models   │                │ record_model_proposal    │
     │ → supported model types │                │ (model_type, hparams,    │
     │   + hyperparam ranges   │                │  reasoning)              │
     └────────────┬────────────┘                └────────────┬─────────────┘
                  │                                            │
                  ▼                                            │  ← HUMAN-IN-THE-LOOP
     ┌─────────────────────────┐                               │    extension point:
     │ train_model              │◄─────────────────────────────┘    could slot in
     │ (model_type, hparams)    │   agent.py copies proposal        exactly here
     │ → model reference/id     │   fields verbatim into            (scope agreed
     └────────────┬────────────┘   train_model's arguments           2026-07-19, build
                  ▼                                                  still deferred)
     ┌─────────────────────────┐
     │ evaluate_model            │
     │ (model_ref, *, pos_label)  │  ← pos_label supplied internally
     │ → accuracy, precision,     │     by agent.py, NEVER by Gemini
     │   recall, f1,               │     (see keyword-only convention,
     │   confusion_matrix          │     "Core architectural principle" above)
     └────────────┬────────────────┘
                  │
                  ▼
     ┌──────────────────────────────┐
     │ record_convergence_decision   │◄── Category B again
     │ (continue: bool, reasoning,   │
     │  next_step?)                  │
     └────────────┬───────────────────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
   continue = True      continue = False
        │                    │
        │                    ▼
        │         ┌───────────────────────┐
        │         │ Final report:          │
        │         │ best model, why,       │
        │         │ plain-English summary  │
        │         └───────────────────────┘
        │
        └──── loop back to record_model_proposal
              (max-iterations guard checked each pass)
```

**Designed extension point:** a human-in-the-loop confirmation step (e.g. "approve this proposal before training?") slots in cleanly at the Category B → Category A handoff — between `record_model_proposal`'s return and `train_model`'s call — without requiring changes to either category's internals. **Not yet implemented; a `# TODO` marker sits at this exact point in `gemini_client.py`.** As of 2026-07-19, the intended *scope* of this hook is now decided (see the status block at the top of this README and "Roadmap context" below) — but the build itself remains deliberately deferred until the project is closer to a finished, working state, so its design can draw on real agent behavior across many runs rather than being built off a single early example.

The two Category B tools are worth a brief word each, since their names alone don't say much. `record_model_proposal` is how Gemini puts a candidate on the table — a model type, its hyperparameters, and the reasoning behind choosing them — without anything being trained yet. `record_convergence_decision` is the checkpoint after a model has actually been evaluated: Gemini looks at the metrics it just got back and states, explicitly, whether to stop here or try again, along with why.

All five tools above exist as ordinary Python functions in `tools.py`, but Gemini never sees Python — it needs each tool declared as a JSON schema (name, description, parameter types) before it can call any of them. That declaration lives in `TOOL_SCHEMAS`, a list at the bottom of `tools.py` that `gemini_client.py` hands to the Gemini API. It's kept as its own structure rather than auto-generated from the functions' docstrings above it, on purpose: **the schema text is what *Gemini* reads to decide when and how to call a tool, while the docstring is what a *human* reads to understand the code — the two audiences don't always need the same amount of detail, so letting the wording diverge slightly where it helps is a small deliberate cost, not an oversight**. The trade-off is that a signature change to a function requires a matching by-hand update to its schema entry — easy to forget. **This mitigation is now implemented** — see "`test_tools.py` — the schema/function drift check" below.

---

## `test_tools.py` — the schema/function drift check

_Added 2026-07-15; extended 2026-07-19; review resolved 2026-07-27._

### Why this exists

`TOOL_SCHEMAS` (what Gemini reads) and the real tool functions in `tools.py` describe the same thing twice, by hand, in two different places. If someone changes a function's parameters without updating the schema to match, Gemini either can't call the tool correctly, or silently sends an argument the function doesn't expect — a failure mode that doesn't surface until runtime, deep inside a live agent loop, usually as a confusing `TypeError`. `test_tools.py` catches that drift automatically, at test time, using `inspect.signature()` — Python's own reflection tool for reading a function's real parameter list without calling it.

### The mechanism

```
        ┌────────────────────────────┐        ┌─────────────────────────────┐
        │  Real function (tools.py)   │        │  TOOL_SCHEMAS entry          │
        │  def train_model(            │        │  {"name": "train_model",     │
        │      model_type: str,        │        │   "parameters": {            │
        │      hyperparameters: dict)  │        │     "model_type": {...},     │
        │                              │        │     "hyperparameters":{...}} │
        └──────────────┬───────────────┘        └───────────────┬───────────────┘
                       ▼                                        ▼
              inspect.signature(fn)                     schema["parameters"]
              → {"model_type", "hyperparameters"}         .keys()
                       │                                        │
                       └───────────────────┬────────────────────┘
                                           ▼
                              set comparison: do these
                              two parameter-name sets
                              match exactly?
```

Keyword-only parameters (the internally-injected ones, e.g. `evaluate_model`'s `pos_label`) are deliberately excluded from this comparison — they're never part of what Gemini is meant to see or supply, so comparing them against the schema would be comparing the wrong thing entirely. Three checks run per tool: parameter names match in both directions, `required` in the schema matches which parameters actually lack a default, and every schema entry has a real registered function (and vice versa).

**Result:** 12/12 passing, alongside the existing 9 in `test_dataset.py` (21/21 total).

### Extension (2026-07-19): guarding the dispatch-table override keys

A fourth test, `test_dispatch_table_override_keys_exist()`, asserts that `"train_model"` and `"evaluate_model"` still exist as keys in `TOOL_FUNCTIONS` — added as explicit, by-name documentation of a concern originally raised during the 2026-07-17 orchestration wiring. What this test actually guards, precisely: the original stated rationale for it turned out to be inaccurate on closer inspection — `agent.py`'s dispatch-table override uses **literal string keys**, not a lookup into `TOOL_FUNCTIONS`, so a rename inside `TOOL_FUNCTIONS` cannot silently break that override the way it was originally described as doing. The real failure mode a rename *can* cause — `TOOL_SCHEMAS` declaring a tool name no longer present in `TOOL_FUNCTIONS` — was already caught by the pre-existing "every schema has a registered function" check above. This new test is retained as cheap, explicit documentation of the concern, not because it closes a gap that genuinely existed. Full detail in `TECHNICAL_NOTES.md` §2.2's correction note and Part 3, §3.1.

### Reviewed and resolved (2026-07-27)

`trainer.py`'s `train_model` gained a new `"warnings"` key on its return value in the 2026-07-24 session (see "Fit-time warning capture" below). Reviewed this session: no update needed. Every test in this file operates on `_gemini_visible_params`, which wraps `inspect.signature()` — a function's declared *parameters*, never its return value. `train_model(model_type, hyperparameters)`'s own parameter list is byte-for-byte unchanged; only what it returns grew a key. This distinction — parameters vs. return shape — is the reason none of this file's four tests were, or needed to be, affected by either the 2026-07-24 warning-capture change or the 2026-07-27 `max_iter` schema addition (see "Closing the `ConvergenceWarning` shortlist" below).

---

## `gemini_client.py` — the agent loop

`run_agent_loop(dispatch_table, initial_context, *, model=DEFAULT_MODEL, max_iterations=MAX_ITERATIONS, log_iterations=False)` is the thin, deliberately "dumb" messenger between Gemini and the real scikit-learn machinery. It never trains a model or judges a metric itself — it relays Gemini's decisions to `dispatch_table`, and relays the real results back to Gemini, until Gemini calls `record_convergence_decision` with `continue_iterating=False`, or `max_iterations` is reached.

```
                    ┌─────────────────────────────────────┐
                    │  Initial prompt: inspect_dataset()   │
                    │  output front-loaded as context      │
                    │  (locked decision — not a tool call)  │
                    └──────────────────┬────────────────────┘
                                       ▼
                          ┌─────────────────────┐
                     ┌───▶│   Send to Gemini     │
                     │    │ (message + tool defs)│
                     │    └──────────┬────────────┘
                     │               ▼
                     │    ┌─────────────────────────┐
                     │    │ Gemini responds with:    │
                     │    │  (a) plain text, or      │
                     │    │  (b) a function call     │
                     │    └──────────┬────────────────┘
                     │               ▼ (b)
                     │    ┌─────────────────────────────┐
                     │    │ dispatch_table[tool_name]    │
                     │    │        (**args)              │
                     │    └──────────┬────────────────────┘
                     │               ▼
                     │    ┌─────────────────────────────┐
                     │    │ [HUMAN-IN-THE-LOOP HOOK —    │
                     │    │  only between record_model_  │
                     │    │  proposal's return and       │
                     │    │  train_model's call — scope  │
                     │    │  agreed 2026-07-19, build     │
                     │    │  still deferred]              │
                     │    └──────────┬────────────────────┘
                     │               ▼
                     │    ┌─────────────────────────────┐
                     └────┤ Feed tool result back to     │
                          │ Gemini as next message         │
                          └──────────┬────────────────────┘
                                     ▼
                     Loop continues until record_convergence_
                     decision is called (or max-iterations guard
                     trips) — then stop and report the outcome.
```

### Design decisions, stated explicitly

- **`automatic_function_calling` is disabled.** If Gemini's own SDK executed functions directly, it would bypass `dispatch_table` entirely — meaning it would miss the real, per-run partial-bound `Trainer`/`X_train`/`y_train`/`pos_label` that `build_dispatch_table` sets up. Every tool call must go through `dispatch_table`, no exceptions.
- **`TOOL_SCHEMAS` entries are wrapped explicitly into `FunctionDeclaration` objects** rather than passed as raw dicts, even though the SDK's Pydantic-based constructor accepts raw dicts fine at runtime via internal coercion — the explicit wrap avoids relying on undocumented behavior the SDK's own type stubs don't advertise.
- **Conversation history uses `client.chats.create()`** (the SDK's own automatic history tracking) rather than manually assembled `types.Content` lists. Simplest available option; the trade-off (no manual control over pruning) is discussed in `TECHNICAL_NOTES.md`.
- **Known limitation, not yet handled:** the loop assumes exactly one function call per Gemini turn. Gemini's function-calling mode can in principle return several parallel calls in a single response; this case isn't currently handled. **Still open as of 2026-07-24** — untouched again this session, deliberately, consistent with the standing 2026-07-17 agreement to leave it deferred.
- **Known limitation, not yet handled:** `record_convergence_decision`'s result is not echoed back to Gemini when the loop stops (`continue_iterating=False`) — the loop just returns. **Also still open as of 2026-07-24**, same status as above.
- **Implemented (2026-07-19): optional per-iteration logging.** See "Per-iteration logging" below.
- **Resolved (2026-07-27): `MAX_ITERATIONS` raised from 10 to 15.** The original 07-12 value was deliberately small as a cheap debugging safety net, on the explicit condition that it be raised "once the loop is trusted end-to-end." That condition is now judged met — 9 live runs across two sessions have shown coherent tool sequencing with no sign of the failure mode the cap was guarding against, and a real `main.py` run this session hit the old ceiling mid-retry, clipping legitimate in-progress work rather than catching a bug. Full rationale in `TECHNICAL_NOTES.md` Part 5, §5.9.

### `client.chats.create()` — internal dynamics

```
┌──────────────────────────────────────────────────────────────┐
│  client.chats.create(model=..., config=...)                   │
│  → returns a Chat object                                      │
│  → internal history: []   (empty at creation)                 │
└──────────────────────────┬──────────────────────────────────┘
                            ▼
        ┌──────────────────────────────────────┐
        │  chat.send_message(initial_context)    │
        └──────────────────┬─────────────────────┘
                            ▼
   ┌───────────────────────────────────────────────────────┐
   │ INSIDE chat.send_message (SDK-managed, not our code):   │
   │  1. Appends UserContent(initial_context) to history      │
   │  2. Calls models.generate_content(                       │
   │        contents=<full history so far>,                   │
   │        config=<our tools + disabled auto-FC>             │
   │     )                                                    │
   │  3. Appends the model's ModelContent response to history │
   │  4. Returns that response to our code                    │
   └──────────────────────────┬────────────────────────────┘
                              ▼
              (loop repeats: steps 1–4 above, every
               turn, history strictly append-only —
               nothing is ever pruned or summarized)
```

Every `chat.send_message()` call — the first `initial_context` or the tenth `function_response` — replays the *entire* accumulated history to `generate_content` each time. This is why `gemini_client.py` never has to manually remember prior iterations, but it's also why request size grows with iteration count. See `TECHNICAL_NOTES.md` for the full cost-scaling discussion — not an issue at the current `MAX_ITERATIONS = 15` (raised from 10 on 2026-07-27), but worth understanding before that ceiling is ever raised significantly (past ~20-30).

---

## Per-iteration logging (2026-07-19)

### Why this exists

Comparing multiple live runs against each other (see "Data Science Notes" below) surfaced a real limitation: `run_agent_loop` returned only the final decision — no record of which models were proposed, evaluated, or rejected along the way, and no way to answer questions like "why did this run converge faster than that one." `log_iterations` closes that gap for a single run.

### How it works

`log_iterations: bool = False` — opt-in, matching this project's existing default-off convention elsewhere (`class_weight=None`, `next_step_hint=""`). When enabled, every iteration of the loop appends one entry to a list, with a consistent key set regardless of what happened that iteration:

```python
{
    "iteration": int,
    "tool_name": str | None,       # None only when Gemini replied with plain text, no tool call
    "tool_args": dict | None,      # same
    "result": Any | None,          # same
    "response_text": str | None,   # populated only on that same no-tool-call branch
    "timestamp": str,              # UTC, ISO 8601
}
```

The full log is returned under `result["log"]` — but only when `log_iterations=True`; callers that don't ask for it see no change at all in the function's return shape. `run_session` forwards this flag unchanged, making no logging decisions of its own.

```
                    ┌────────────────────────────────────────┐
                    │  run_agent_loop(dispatch_table,          │
                    │    initial_context, *, model,            │
                    │    max_iterations, log_iterations=False) │
                    └──────────────────┬─────────────────────┘
                                       │
                          log: list = []   ◄── empty unless log_iterations=True;
                                              caller sees no change if unused
                                       │
                    ┌──────────────────▼─────────────────────┐
                    │  chat = client.chats.create(...)         │
                    │  response = chat.send_message(           │
                    │      initial_context)                    │
                    └──────────────────┬─────────────────────┘
                                       │
                     ┌─────────────────▼──────────────────┐
                ┌───▶│  for iteration in range(max_iters):  │
                │    └─────────────────┬─────────────────────┘
                │                      ▼
                │        function_calls = response.function_calls
                │                      │
                │         ┌────────────┴─────────────┐
                │         ▼ empty                     ▼ has call(s)
                │  ┌──────────────────────┐  call = function_calls[0]
                │  │ if log_iterations:      │  (only first call used —
                │  │  log.append({           │   known limitation, unchanged)
                │  │   iteration,            │           │
                │  │   tool_name: None,      │           ▼
                │  │   tool_args: None,      │  tool_name, tool_args = ...
                │  │   result: None,         │           │
                │  │   response_text,        │           ▼
                │  │   timestamp })          │  result = dispatch_table[tool_name](**tool_args)
                │  └──────────┬──────────────┘           │
                │             ▼                ┌─────────▼──────────────┐
                │   return {status:             │ if log_iterations:      │
                │    "stopped_without_          │  log.append({            │
                │    convergence_call", ...     │   iteration, tool_name,  │
                │    (+ "log": log if           │   tool_args, result,     │
                │     log_iterations)}          │   response_text: None,   │
                │                               │   timestamp })           │
                │                               └─────────┬──────────────┘
                │                                          ▼
                │                     record_convergence_decision AND
                │                     continue_iterating == False ?
                │                              │              │
                │                          yes │              │ no
                │                              ▼              ▼
                │              return {status: "converged",   record_model_proposal?
                │               decision: result, ...          → human-in-the-loop TODO
                │               (+ "log": log if                 (see above)
                │                log_iterations)}                    │
                │                                                     ▼
                │                              response = chat.send_message(
                │                                  function_response_part)
                └───────────────────────────────────────────────────┘
                          (loop back — next iteration)

                    ── loop exhausts max_iterations without stopping ──
                                       │
                                       ▼
                    return {status: "max_iterations_reached", ...
                     (+ "log": log if log_iterations)}
```

**Why a consistent key set on every entry, rather than only including the applicable fields:** so `pd.DataFrame(result["log"])` produces a fully rectangular table immediately, and hand-written iteration (`entry["tool_name"]`) never hits a missing-key error. The trade-off is a few `None` values sitting in fields that don't apply to a given entry — judged a reasonable cost for that convenience.

### Reading a log

`format_log(log)`, also in `gemini_client.py`, renders a log list as a series of `=== Iteration N (timestamp) ===` blocks for terminal/human reading — kept as a shared, standalone utility (not private to any one caller), since anything else that ends up with a log list in hand later can reuse it.

### Persistence (smoke-test usage only, not committed package behavior)

`run_smoke_test.py` (gitignored) writes the full `result` dict as raw JSON to `results/` on every run with `log_iterations=True`, with a wall-clock `elapsed_seconds` value added at write time. **Corrected (2026-07-24):** each run now gets its own timestamped filename (`results/smoke_test_log_<YYYY_MM_DD_HHMMSS>.json`) rather than a single file overwritten every time. The original overwrite-by-design choice — documented here as of 2026-07-19, reasoned as "individual agent runs aren't reproducible, so there's no meaningful continuous history to preserve" — was superseded once comparing multiple runs against each other became the actual goal: a single overwritten file made that structurally impossible by construction. `results/` remains a pre-existing, gitignored directory; no `.gitignore` change was needed for either the new filename pattern or `compare_runs.py`'s own output files (see below).

**What this does and doesn't solve:** a single run's full proposal/evaluation/rejection trail is inspectable after the fact (see "Data Science Notes" below for a concrete example), and — as of this session — so is a side-by-side comparison across several runs (see "Cross-run comparison" immediately below). What's still open: a genuinely human-friendly rendering of either a single run's or a comparison's raw JSON (design agreed 2026-07-24 — auto-detect by filename, render as Markdown — not yet built; see "Roadmap context").

---

## Cross-run comparison (2026-07-24; renamed and dataset-scoped 2026-07-29)

### Why this exists

With runs now persisted individually instead of overwritten, the original motivating question — why does one run converge faster than another, with or without a `ConvergenceWarning` — finally has real, comparable data behind it. `ml_agent/compare_runs.py` turns several raw run files into one summary.

### How it works

```
results/*.json (several completed runs, ONE dataset)
        │
        ▼
summarize_run()  — pure function, one run in, one flat row out
        │
        ▼
list of per-run summary rows
        │
        ▼
results/comparison_<timestamp>_<dataset>.json  — one row per run
```

`summarize_run(run_data, source_file=None)` reads one already-loaded run dict — no file I/O, no live API call — and extracts: `status`, `iterations`, `elapsed_seconds`, the ordered `model_sequence` actually tried, the `final_model_type` and its metrics, every `warnings_encountered` entry (flattened, with the iteration and model type each fired on), and the run's final `convergence_reasoning`. Kept as a pure function of its input dict, for the same reason `validate_split` and `validate_hyperparameters` are kept standalone elsewhere in this project: testable directly against a hand-built dict, with no dataset, `Trainer`, or live API call required.

`build_comparison(results_dir=Path("results"), dataset_name=None)` scans for matching result files and calls `summarize_run` on each — not capped at any particular count, whether that's 2 files or 20.

**One of these two judgment calls has since been resolved with a real fix; the other remains a stated limitation.** "Final model" was originally read as the *last* `evaluate_model` call in a run's log — **confirmed wrong on a real file (2026-08-01):** a run can evaluate one extra model purely for comparison after already deciding, and the convergence reasoning can name an earlier evaluation as the actual choice. Final model is now resolved as whichever evaluated model scores highest on the run's own optimization target instead, with a new `final_model_ambiguous` field flagging any disagreement between the two approaches for a human reader's attention — see "Results reporting" below for the full writeup and the real example that surfaced it. `convergence_reasoning` is still taken from the *last* `record_convergence_decision` entry seen, even on a run that hit `max_iterations` without ever setting `continue_iterating=False` — unchanged, not independently verified as universal.

### Renamed and dataset-scoped (2026-07-29)

Once a second dataset (Breast Cancer) came into regular use, two real problems surfaced with the original `smoke_test_log_<timestamp>.json` / `comparison_<timestamp>.json` convention:

1. Nothing in either filename indicated which dataset it belonged to.
2. `build_comparison`'s original unfiltered scan would silently mix both datasets' runs into one comparison file — combining metrics (accuracy/precision/recall/f1) computed against genuinely different data, which was judged never acceptable, not even as an opt-in "compare everything" mode.

**Resolved:** run files are now named `result_log_<YYYY_MM_DD_HHMMSS>_<dataset_name>.json` — dataset name deliberately placed *after* the timestamp, not before, so a whole-filename string sort still sorts chronologically across every dataset (putting the dataset name first would have grouped, say, all `breast_cancer` runs before all `climate` runs regardless of when they actually ran — `build_comparison`'s own sort relies on this). `build_comparison` now accepts an optional `dataset_name` filter, restricting its scan to `result_log_*_<dataset_name>.json`; `main.py`'s `compare` subcommand (see "`main.py`" below) makes this filter effectively mandatory for its users, never silently defaulting to "everything." Pre-existing files were migrated with a one-time script (`rename_results.py`, gitignored — not project code, a throwaway migration helper).

### Discoverability: the `compare` subcommand (2026-07-29)

`compare_runs.py`'s existence as a separate script meant it was easy for a first-time reader to miss entirely. `main.py` now exposes the same capability as a sibling subcommand:

```
python main.py                          # (implicit "run" — unchanged from before)
python main.py run --dataset climate    # explicit form, same result
python main.py compare --dataset climate
```

`--dataset` is required for `compare` — if omitted or misspelled, it prints a warning explaining why (mixing datasets isn't allowed) and falls back to the same interactive picker `run` already uses, rather than a bare argparse usage error. **Resolved (2026-08-01):** the standalone `python -m ml_agent.compare_runs` invocation now requires a dataset too (positional argument, or the same interactive prompt on omission) — it previously scanned everything unfiltered by default, an inconsistency flagged as an open question back on 2026-07-29 and left undecided until now.

Two further subcommands, `export` and `report`, build directly on this same comparison data — see "Results reporting: CSV export and the Markdown viewer" below.

### Verified against real data

Across six real Climate Crashes runs from a single evening: correctly distinguished the one run that never proposed Logistic Regression — and so never hit the `ConvergenceWarning` — from the five that did. Since then, verified again post-rename against 12 real Climate Crashes runs and 4 Breast Cancer runs, correctly kept separate by dataset. Full run-by-run findings, including a discussion of run-to-run path variation, are in [`DATA_SCIENCE_ANALYSIS.md`](./DATA_SCIENCE_ANALYSIS.md).

### Known limitation, reproduced and documented, still not fixed

A run that crashes before completing (e.g. hitting the Gemini API's own free-tier rate limit — a real 429 `RESOURCE_EXHAUSTED` error, reproduced 2026-07-29 by running two sessions back-to-back within the same minute) writes no file at all — a failed attempt is currently invisible to this comparison, not just excluded from it. `main.py` now prints a note at startup warning that back-to-back runs can hit this limit. The underlying gap remains an accepted limitation, not a planned fix; see "Roadmap context."

---

> **Problem, briefly:** a heuristic assuming "last evaluated model = final model" was quietly wrong on a real run — the model the agent actually chose wasn't the last one it had evaluated.
> **Thread:** this section fixes it; a cleaner, condensed account is in [TN Part 7](./TECHNICAL_NOTES.md#part-7-results-reporting-reportingpy-a-real-final-model-bug-fix-and-a-pytest-collection-bug-2026-08-01)/[DS §13](./DATA_SCIENCE_ANALYSIS.md#13-update-2026-08-01-summarize_runs-final-model-resolution-corrected--a-real-breast_cancer-runs-reported-outcome-was-wrong).

## Results reporting: CSV export and the Markdown viewer (2026-08-01)

### Why this exists

Two long-standing items on the project's own named roadmap — a human-friendly viewer for result/comparison files, and a CSV/spreadsheet export (explicitly named as the gate before making this repo public) — turned out to share one underlying need: turning a `result_log_*.json` or `comparison_*.json` file into flat, comparable rows. `ml_agent/reporting.py` builds that shared path once, then exposes two thin, format-specific outputs.

### How it works

```
result_log_*.json ──► summarize_run() ──► one row
                         │  (dataset/target/random_state from config,
                         │   final_hyperparameters + final_model_ambiguous
                         │   resolved by model_ref — see below)
                         ▼
comparison_*.json  = [ rows ]   (build_comparison(), unchanged shape)

reporting.py:
  result_log_*.json ──► load_rows() ──► summarize_run() ──► [1 row]
  comparison_*.json ──► load_rows() ──► json.loads()     ──► [N rows]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                to_csv()           to_markdown()
              (main.py export)    (main.py report)
```

`load_rows(path)` auto-detects which kind of file it's looking at by filename prefix only (`result_log_` vs `comparison_`) — no content inspection, since these filenames aren't expected to be renamed by hand. An unrecognized filename raises a clean, caught error rather than a raw traceback.

`to_csv()` writes one row per run (not one row per model evaluated within a run) against a fixed column order, so every export has the same shape regardless of which fields a given row happens to have populated; nested fields (hyperparameters, confusion matrix, warnings) are serialized as compact JSON strings within their cell. `to_markdown()` renders a single run as a narrative report (config, final model, full model-sequence trail, convergence reasoning verbatim, warnings) or several runs as a GitHub-native Markdown table — auto-selected by the same `load_rows()` detection.

### A real bug found while building this: "last evaluated = final model" was wrong

Extending `summarize_run()` (see "Cross-run comparison" above) to also surface hyperparameters surfaced a genuine bug in its existing final-model heuristic. On a real file, three models were evaluated — `logistic_regression` (recall 0.9583), `logistic_regression` with `max_iter=500` (recall **0.9722**), then `random_forest` evaluated *last* (recall 0.9583) purely as a comparison point. The run's own `record_convergence_decision` reasoning explicitly named the 0.9722 model as chosen — but the old "last `evaluate_model` call in the log wins" rule picked `random_forest` instead, silently misreporting the run.

**Fixed:** the final model is now resolved as whichever evaluated model scores highest on the run's own optimization target (`config["target"]`) — a structural, metric-based resolution, not text-parsing of the free-text reasoning field (considered and rejected as too fragile). Older files with no `config` block fall back to the original last-evaluated behavior unchanged. Because this heuristic can, rarely, still disagree with the literal reasoning text (the case above is proof it happens), `summarize_run()` also returns a new `final_model_ambiguous` field — `True` when the best-by-target pick differs from the old last-evaluated pick, `False` when they agree, `None` when there's no target to compare against. This surfaces as a `⚠️` note in the Markdown viewer and a plain column in the CSV export, rather than being silently resolved either way.

**Explicitly considered and deferred:** using a second, live Gemini call to adjudicate a flagged mismatch. This would add a network/API dependency to what's otherwise an offline, deterministic module, and overlaps directly with the human-in-the-loop design (see "Roadmap context") — folded into that future design conversation rather than built here.

The full data-science-layer writeup of this finding — the real run involved, and an explicit check of whether any of this document's own earlier figures could be affected — is in [`DATA_SCIENCE_ANALYSIS.md`](./DATA_SCIENCE_ANALYSIS.md) §13 (added 2026-08-03). This section here covers the implementation side; that one covers what it means for trusting a run's reported outcome.

### The `export` and `report` subcommands

Both live in `main.py`, mirroring `compare`'s existing pattern exactly:

```bash
uv run python main.py export --dataset climate    # → results/export_<timestamp>_<dataset>.csv
uv run python main.py report results/result_log_2026_07_27_223458_climate.json
uv run python main.py report results/comparison_2026_08_01_125444_climate.json
```

`export` requires `--dataset`, same as `compare` — comparing/exporting across datasets is never allowed. `report` accepts either a single-run or comparison file, auto-detected as above, and writes a `.md` file next to the input by default (overridable with `--out`).

### Verified against real data

Both subcommands run against real Climate Crashes and Breast Cancer result files, including the confirmed final-model fix (re-verified on the file that originally surfaced the bug, plus a second, independent file). Two automated test files (`tests/test_compare_runs.py`, `tests/test_reporting.py`) now lock in the resolution logic and rendering behavior — see "Testing" below.

---

## Fit-time warning capture (2026-07-24)

### Why this exists

An earlier smoke test had already surfaced a real `ConvergenceWarning` (Logistic Regression's `lbfgs` solver hitting the default `max_iter=100`) — but the only record of it was watching raw terminal output during a live run. Nothing in `train_model`'s return value captured it, so it couldn't be persisted, compared across runs, or relied on for anything beyond that one observation.

### How it works

`train_model` wraps only the `estimator.fit(...)` call — not the whole method — in `warnings.catch_warnings`:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    estimator.fit(X_train, y_train)

fit_warnings = [
    {"category": w.category.__name__, "message": str(w.message)}
    for w in caught
]
```

`train_model` now always returns a `"warnings"` key alongside `model_ref` — an empty list when nothing fired, matching this project's existing "consistent key-set regardless of branch" convention (see "Per-iteration logging" above).

```
Before the with block  →  Enter (start recording)  →  Run estimator.fit()
    →  Exit (recording stops, even on error)  →  After (caught holds every
       warning seen)
```

`catch_warnings(record=True)` temporarily redirects Python's warning system into a list instead of the terminal, for exactly the duration of the indented block — and, like any context manager, the "stop recording" step happens automatically even if `fit()` raises an exception partway through, which a hand-rolled start/stop pair would only guarantee if every call site remembered to handle the error path too.

**Deliberately captures every warning category, not just `ConvergenceWarning`.** Narrowing the filter to one already-known warning type would hardcode today's one observed case and miss anything else scikit-learn might raise later (e.g. a `DataConversionWarning`) — a data scientist reading this repo would want visibility into any fit-time warning, not only the one that happened to be found first.

`warnings.simplefilter("always")`, set inside the block, deliberately overrides Python's own default warning behavior (showing only the *first* occurrence of an identical warning per process) — necessary here because a second run hitting the exact same `ConvergenceWarning` from a fresh fit would otherwise go silently unrecorded the second time. The full reasoning for why this matters specifically for this project's data science is written up in [`DATA_SCIENCE_ANALYSIS.md`](./DATA_SCIENCE_ANALYSIS.md) / [`TECHNICAL_NOTES.md`](./TECHNICAL_NOTES.md).

### Verified against real data

Across six real live runs: correctly returned an empty list for models that didn't warn (Random Forest, SVM), and correctly captured the real `ConvergenceWarning` every time Logistic Regression was proposed and trained.

### Resolved (2026-07-27)

Capturing the warning is not the same as fixing the underlying cause — this was true as of 2026-07-24, when the shortlist of options wasn't even drafted yet. It now is: see "Closing the `ConvergenceWarning` shortlist" immediately below for the decision made and what it looks like verified against real runs.

---

> **Problem, briefly:** `LogisticRegression` kept emitting a `ConvergenceWarning` across runs — an open question of whether it was a real correctness issue or just a `max_iter` cap set too low.
> **Thread:** this section exposes `max_iter` and decides the shortlist; a cleaner, condensed account is in [TN Part 5](./TECHNICAL_NOTES.md#part-5-max_iter-exposed-as-a-hyperparameter-confirming-the-agent-reasoning-pathway-was-already-wired-2026-07-27)/[DS §11](./DATA_SCIENCE_ANALYSIS.md#11-update-2026-07-27-max_iter-exposed-as-a-tunable-hyperparameter--the-convergencewarning-shortlist-decision-tested-against-3-live-runs).

## Closing the `ConvergenceWarning` shortlist (2026-07-27)

### The shortlist, and the decision made

Three options were on the table for what to actually do about `lbfgs` hitting its default `max_iter=100`:

```
(A) Raise the default             (B) Let the agent reason        (C) Expose max_iter as
    (e.g. to 500) so the              about the warning using         a hyperparameter, giving
    warning stops firing               train_model's already-          the agent something
    in most/all runs                   captured "warnings" list        concrete to act on

    REJECTED — would silently     DECIDED — primary approach      DECIDED — the mechanism
    remove the exact signal                                        B needs to act on
    §10's six-run comparison
    depended on
```

**Decided: B, using C as the acting mechanism.** Raising the default (A) would have made every future run converge silently, quietly breaking the exact warning-presence/absence signal the 2026-07-24 session's six-run comparison methodology relied on to distinguish "Logistic Regression was tried and struggled" from "Logistic Regression was never tried."

### The plumbing was already there — nothing needed building for option B

Before treating "the agent can reason about the warning" as a real capability, it was traced directly through `gemini_client.py`'s `run_agent_loop`, not assumed from `agent.py`'s wiring code alone:

```python
result = dispatch_table[tool_name](**tool_args)
...
function_response_part = types.Part.from_function_response(
    name=tool_name,
    response={"result": result},   # the FULL result dict, unfiltered
)
response = chat.send_message(function_response_part)
```

This runs unconditionally, on every single tool call, independent of `log_iterations`. `train_model`'s complete return value — including `"warnings"` — was already being fed back to Gemini verbatim, every turn, from the moment the 2026-07-24 session added warning capture. Option B required zero new code; only option C (the schema addition below) needed building.

### Schema change: `max_iter` added to `list_available_models`

A single-site addition to `tools.py`'s `logistic_regression` hyperparameters — `type: "int"`, `range: [50, 1000]`, `default: 100` (preserving today's exact behavior unless the agent deliberately raises it). Neither `TOOL_SCHEMAS` nor `validate_hyperparameters` needed changes: both already treat `hyperparameters` generically, keyed by whatever `list_available_models` declares.

### Verified against 3 live runs

All three runs (Climate Crashes, `optimization_target="recall"`) show the intended chain working end-to-end: a `ConvergenceWarning` appears in `train_model`'s result → the agent's own `record_model_proposal`/`record_convergence_decision` reasoning references it explicitly → the agent re-proposes Logistic Regression with `max_iter` raised → the warning clears. One run isolated `max_iter` as the *only* changed value on retry — recall/precision/the full confusion matrix came back bit-identical to the pre-retry attempt, meaning the warning was a reporting problem, not an accuracy problem, in that instance. Two other runs changed `max_iter` and `C` together, reaching a better precision (0.25 vs. the prior typical 0.18) than either the original six-run table's usual outcome — but since both changed together, which one actually drove the improvement is a plausible inference from comparing across runs, not proven directly. Full run-by-run findings, including this confound and a proposed follow-up to isolate it, are in [`DATA_SCIENCE_ANALYSIS.md`](./DATA_SCIENCE_ANALYSIS.md) §11.

**Precisely, not overclaimed:** this confirms the data *reaches* Gemini and that the mechanism functions end-to-end on 3 real runs — it isn't a guarantee that every future run will reason about a warning this well. "Wired" and "reasons about it well" are different claims.

### The `max_iter`-vs-`C` confound, isolated (2026-07-29)

**Correction to how this was first described:** the two confounded runs above were originally written up as reaching "a better precision" — this was imprecise. Re-checking the actual numbers: precision was **0.25 in every relevant run** (the original baseline, the confounded runs, and the isolation run below) — it never moved. The metric that actually changed was **recall**: 0.778 in the run that changed `max_iter` alone, vs. 1.0 in every run that also changed `C`.

A controlled follow-up run isolated `C` alone (`C=0.1`, `max_iter` left at its untouched default of 100) — deliberately *not* fixing the convergence issue, to separate the two effects cleanly:

```
                                   warning fires?   recall   precision
max_iter changed alone (isolated) →     no      →  0.778  →  0.25
max_iter + C changed together     →     no      →  1.0    →  0.25
C changed alone (isolation run)   →    YES      →  1.0    →  0.25
```

The isolation run's `ConvergenceWarning` still fired (expected — `max_iter` was untouched), yet recall still jumped to 1.0, matching the runs where `C` and `max_iter` moved together. **This confirms `C` — not `max_iter` — drove the recall improvement, independent of convergence status.** `max_iter` only resolves whether sklearn's solver believes it converged; it does not by itself change *what the model predicts*. `C` (inverse regularization strength) is what actually reshapes the decision boundary. The warning and prediction quality are decoupled: a model can still be "unconverged" by sklearn's own internal criterion while making the same predictions as one that isn't.

This is a genuine causal-isolation finding, not just an observed correlation — a controlled, single-variable follow-up, in the same spirit this project treats as most valuable for its data-science content (see "Data Science Notes" in `DATA_SCIENCE_ANALYSIS.md`). A natural further extension — sweeping `C` across several values (e.g. `1.0, 0.5, 0.1, 0.01`) with `max_iter` held constant, to characterize the shape of the relationship rather than just its direction — is noted as a low-priority future addition; see "Roadmap context."

---

> **Problem, briefly:** three `RandomForest` runs with different hyperparameters produced suspiciously identical results — until a fourth run, same hyperparameters as the first, produced a *different* result.
> **Thread:** this section traces and fixes the real bug (`random_state` never threaded into the estimator); a cleaner, condensed account is in [TN Part 6](./TECHNICAL_NOTES.md#part-6-results-file-renaming-compare-subcommand-and-the-reproducibility-fix-2026-07-29)/[DS §12](./DATA_SCIENCE_ANALYSIS.md#12-update-2026-07-29-the-max_iter-vs-c-confound-resolved-117s-anomaly-explained-and-new-breast-cancer-results).

## Reproducibility: seeding every estimator (2026-07-29)

### The bug, found via a real anomaly, not a code review

Running Breast Cancer sessions back-to-back surfaced something that looked like reproducible behavior at first, then contradicted itself: three separate `random_forest` calls, each with *different* hyperparameters (library defaults; `n_estimators=200`; `n_estimators=200, max_depth=10`), all produced the exact same recall and confusion matrix — suggestive of a fixed seed making RF insensitive to those particular hyperparameter changes. Then a fourth run, using the *exact same* hyperparameters as the very first of those three, produced a **different** result.

Checked directly against `trainer.py`'s real code, not inferred: `estimator = estimator_class(**hyperparameters)` — every estimator was instantiated with only whatever `tools.py`'s schema allows Gemini to tune (`n_estimators`, `max_depth`, `class_weight` for Random Forest; none of them `random_state`). `RandomForestClassifier` therefore always ran with sklearn's own default, `random_state=None`, pulling from numpy's global RNG — freshly seeded differently in every separate process. The three matching results were coincidence (a small, strongly-separable dataset where these particular hyperparameter changes happened not to matter), not evidence of determinism; the fourth run was the real behavior showing through.

`LogisticRegression`'s `lbfgs` solver looked reproducible the entire time for a different, unrelated reason: it's a deterministic optimizer solving a fixed convex objective against a fixed data split — no bootstrap sampling, no per-split feature randomness — so it never needed a seed to behave consistently. It was never actually seeded either; it simply didn't need to be.

### The fix

`Trainer.train_model` now accepts a keyword-only `random_state` parameter (default `42`), passed into *every* estimator's constructor — `LogisticRegression`, `RandomForestClassifier`, and `SVC` all accept this kwarg, so no per-model-type branching is needed:

```python
estimator = estimator_class(**hyperparameters, random_state=random_state)
```

Deliberately **not** exposed as a Gemini-tunable hyperparameter in `tools.py`'s schema — it's a reproducibility knob, not a modeling choice the agent should be making run to run (same fact-vs-judgment separation as `pos_label` and `optimization_target` elsewhere — see "Core architectural principle" above). `build_dispatch_table` threads the *same* `random_state` value already used for the train/test split into this new parameter via `functools.partial`, rather than introducing a second, separate seed:

```python
bound_train = partial(
    trainer.train_model,
    X_train=X_train, y_train=y_train,
    random_state=random_state,
)
```

### A deliberate trade-off, decided rather than defaulted into

A genuinely separate design — a second, independent `--model-random-state` flag (defaulting to match `--random-state`, so nothing changes for a typical user) — was considered instead. The single-shared-seed design that was actually built is simpler (one flag, one mental model, full end-to-end reproducibility from one number) but gives up the ability to isolate "how much does the split affect results" from "how much does a model's own internal randomness affect results" — changing `--random-state` moves both at once. Given this project's actual goal (a working data-science agent, not a formal variance study), the single-seed design was judged the right call for now. The two-seed alternative is kept as a low-priority future item; see "Roadmap context."

### Verified against real data

Confirmed via two Breast Cancer runs with `--random-state 42` (the default): despite using different hyperparameter combinations, `random_forest` now produces the identical result across both — recall 0.9444, confusion matrix `[[40,2],[4,68]]` in both. This is the specific behavior the fix was meant to produce, and the confusion matrix differs from any pre-fix run (expected — `random_state=42` is a genuinely different seed than sklearn's old unseeded default), while now agreeing with itself run to run.

### Test coverage (`test_trainer.py`, 2026-08-03)

17 tests, real sklearn fits throughout (not mocked estimators — a mock would confirm `Trainer` *calls* the estimator correctly, but the original bug was about the estimator's actual numeric behavior, which only a real fit can expose). Parametrized across all three registered estimator types, not just `RandomForestClassifier`: the registry pattern that made adding a new model type easy is exactly what would let a future `random_state`-threading regression on any *other* estimator go just as unnoticed as this one did.

The most direct regression guard: fitting the *same* model type, hyperparameters, and `random_state` twice and asserting identical predictions — precisely the test that would have caught the original bug.

**Worth telling honestly, since it's a good example of how a test like this actually gets built:** the one test needing a genuinely *hard-to-converge* case (to confirm `ConvergenceWarning` really does get captured) took three attempts, not one, each empirically checked rather than assumed correct:

1. **Uniform feature scaling** (multiplying every feature by the same large number) — failed. Scaling every feature identically barely changes `lbfgs`'s internal behavior at all; it doesn't touch the *relative* conditioning between features, which is what a quasi-Newton solver actually struggles with.
2. **Near-perfect linear separability** (classes deliberately hard to tell apart... too easy, actually) — also failed, for a more interesting reason: `lbfgs`'s stopping rule checks *gradient norm*, and gradient norm can shrink below tolerance quickly even while the true optimum (infinite coefficients, for perfectly separable data) is never actually reached. Theoretically correct, practically irrelevant to a 50-iteration cap.
3. **Mismatched per-feature scales** (some features ~1e-5, others ~1e6, on the *same* dataset) — this is what actually works, verified empirically across 5 data seeds × 2 training seeds before being written into the final test. It's not scale itself that matters, it's the *mismatch* between features' scales, which genuinely distorts `lbfgs`'s internal curvature approximation. This is also, not coincidentally, the real-world reason linear models are usually fit on standardized features — this test doubles as a working demonstration of why that convention exists.

Full technical account, including the exact commands and numbers from each attempt, is in `TECHNICAL_NOTES.md` Part 8.

---

## `Trainer` — model storage and encapsulation

### The problem

`train_model` needs to hand Gemini back an opaque `model_ref` — a string id — never the fitted scikit-learn object itself. But *something* has to hold onto the actual fitted model between the `train_model` call and a later `evaluate_model` call that looks it up by that same id.

Two shapes were considered:

```
OPTION A — module-level dict                         OPTION B — class-based Trainer (chosen)
(plain functions, global state)                       (instance holds state, no globals)

  trainer.py                                            trainer.py
  ┌─────────────────────────────┐                       ┌─────────────────────────────┐
  │ _MODEL_STORE: dict = {}     │                       │ class Trainer:              │
  │  (reachable by ANY import)  │                       │     def __init__(self):     │
  │                             │                       │         self._models = {}   │ ← local,
  │ def train_model(...):       │                       │             ↑               │   not global,
  │     _MODEL_STORE[ref]=model │                       │      only reachable          │   only reachable
  │                             │                       │      via self                │   through self
  │ def evaluate_model(...):    │                       │                              │
  │     m = _MODEL_STORE[ref]   │                       │     def train_model(self,..) │
  └─────────────────────────────┘                       │         self._models[ref]=m  │
                                                          │     def evaluate_model(self,│
                                                          │         model_ref,*,pos_lbl)│
                                                          │         m=self._models[ref] │
                                                          └─────────────────────────────┘
```

**Chosen: Option B**, for two reasons. First, encapsulation: a module-level dict is reachable and mutable by *any* code that imports the module. A class instance's `self._models` is only reachable through that specific instance's own methods. Second, testing: each test can construct a fresh `Trainer()` with a guaranteed-empty store and no manual reset step.

### `Trainer` class (final form)

`train_model(self, model_type, hyperparameters, *, X_train, y_train)` validates hyperparameters against `list_available_models()`'s schema via `validate_hyperparameters` before instantiating anything, fits the correct sklearn estimator per `model_type`, stores it under a generated `model_ref`, and returns that ref plus any warning captured during fitting (see "Fit-time warning capture" above). `evaluate_model(self, model_ref, *, pos_label, X_test, y_test)` looks up the stored model, predicts against the test set, and computes accuracy, precision, recall, f1, and a confusion matrix computed with explicit label ordering (`labels=[1 - pos_label, pos_label]`) rather than sklearn's default ascending sort — critical given the two datasets' opposite `pos_label` semantics (see "Datasets" below).

---

## Validating Gemini's arguments before training

### The problem this solves

Without a check, a hallucinated `model_type`, or a `hyperparameters` value outside its documented range/choices, would fail deep inside whatever code instantiates the sklearn estimator — a confusing, hard-to-trace failure far from its actual cause. `list_available_models()` already *is* the schema — the same dict shown to Gemini describing what's available is also the ground truth to check its choices against.

### Design: a standalone function, not a `Trainer` method

`validate_hyperparameters(model_type, hyperparameters, schema)` is a pure function of its three arguments — no instance state needed, so it lives standalone in `trainer.py` rather than as a `Trainer` method. Checks `model_type` validity, then each hyperparameter's presence and range/choice validity, raising a specific, named `ValueError` on any mismatch.

### A schema ambiguity, caught and fixed

The schema's numeric `"range": [min, max]` field is project-specific, not a real JSON Schema keyword — nothing stated whether boundary values themselves were meant to be valid. Resolved by adding an explicit inclusive-bounds convention statement to `list_available_models()`'s docstring; `validate_hyperparameters` uses inclusive bounds (`low <= value <= high`) matching it.

---

> **Problem, briefly:** early runs showed the agent trading recall away for other metrics, because nothing in its prompt ever stated which metric actually mattered.
> **Thread:** this section wires the fix; a cleaner, condensed account is in [TN Part 2](./TECHNICAL_NOTES.md#part-2-orchestration-loop-wiring--implementation-details-2026-07-17)/[DS §8](./DATA_SCIENCE_ANALYSIS.md#8-update-2026-07-17-the-optimization-target-fix-tested-for-the-first-time-against-a-live-agent).

## `agent.py` — wiring the dispatch table

### The problem

Everything dataset-specific (which `pos_label` applies) and run-specific (the train/test split, which fitted models exist so far) needs to be resolved exactly once, in one place, so nothing downstream has to rediscover or re-pass it. `build_dispatch_table` is that one place.

```python
def build_dispatch_table(
    dataset_name: str, random_state: int = 42
) -> DispatchResult:
    """Builds one run's complete tool dispatch table for gemini_client.py,
    plus the (X, y) that table was built from."""
    X, y = load_dataset(dataset_name)
    pos_label = get_pos_label(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    validate_split(y_train, y_test, pos_label, min_count=5)

    trainer = Trainer()
    bound_train = partial(
        trainer.train_model,
        X_train=X_train, y_train=y_train,
        random_state=random_state,  # also seeds every estimator — see
    )                                # "Reproducibility" above (2026-07-29)
    bound_evaluate = partial(
        trainer.evaluate_model, X_test=X_test, y_test=y_test, pos_label=pos_label
    )

    dispatch_table = {
        **TOOL_FUNCTIONS,
        "train_model": bound_train,
        "evaluate_model": bound_evaluate,
    }

    return DispatchResult(dispatch_table=dispatch_table, X=X, y=y)
```

Note the asymmetry visible directly in the dict spread: 3 of 5 entries come straight from `TOOL_FUNCTIONS` (`tools.py`'s own name→callable mapping) unchanged; 2 are overridden with `functools.partial`-bound versions built here, since only `build_dispatch_table` has the real per-run training data those two need. That asymmetry *is* the Category A/B split, made concrete in code rather than only described in prose. (The override uses literal string keys, not a `TOOL_FUNCTIONS` lookup — see `test_tools.py`'s section above for why that matters for the drift-check test added 2026-07-19.)

### `DispatchResult`: why `build_dispatch_table`'s return type changed (2026-07-17)

Before this session, `build_dispatch_table` returned only the dispatch dict — meaning any caller needing the same `(X, y)` it had already loaded (to build `initial_context` via `inspect_dataset`) had to call `load_dataset` a second time. For Climate Crashes specifically, this meant a second OpenML network fetch per run, not just wasted CPU.

```python
@dataclass(frozen=True)
class DispatchResult:
    dispatch_table: dict[str, Callable[..., Any]]
    X: pd.DataFrame
    y: pd.Series
```

Deliberately narrow — it carries `(X, y)` and the dispatch table only, **not** a formed prompt string. Prompt assembly stays outside this function (see `run_session` below), so `build_dispatch_table`'s own job stays exactly what its docstring already committed it to: resolving what's dataset- and run-specific, not writing prompts. Shaped as a frozen dataclass, matching this project's existing convention for named, self-documenting return values (`DatasetSpec` in `dataset.py` is the precedent).

### `run_session`: the orchestration entry point (2026-07-17; extended 2026-07-19)

The piece that was missing until the 2026-07-17 session — everything it calls already existed and worked in isolation; nothing previously called them together with real, non-hand-built inputs in the committed codebase.

```python
def run_session(
    dataset_name: str,
    optimization_target: str,
    *,
    random_state: int = 42,
    model: str = DEFAULT_MODEL,
    max_iterations: int = MAX_ITERATIONS,
    log_iterations: bool = False,
) -> dict[str, Any]:
    result = build_dispatch_table(dataset_name, random_state=random_state)
    facts = inspect_dataset(result.X, result.y)
    initial_context = _format_initial_context(facts, optimization_target)
    return run_agent_loop(
        result.dispatch_table, initial_context,
        model=model, max_iterations=max_iterations,
        log_iterations=log_iterations,
    )
```

**`optimization_target` is a plain parameter, deliberately not hardcoded per dataset.** This preserves the same fact-vs-judgment separation already applied to `pos_label` elsewhere in this project (see "Core architectural principle" above): which class is "positive" is a documented fact, resolved once per dataset; what to optimize for is a judgment call, made once per *run*, by whoever calls `run_session`.

**What "optimizing for recall" concretely means, on this project's primary dataset:** on Climate Crashes, `pos_label=1` marks a *simulation failure* — the rare (~8.5%), undesirable outcome. Optimizing for recall there means minimizing how many real crashes slip through undetected, even if that means more false alarms (flagging a run as risky when it would've been fine). The alternative, precision, asks the opposite question: of everything the model flagged as a crash, how many actually were one. Which one matters more is a judgment call that depends on which mistake is more costly in context — missing a real crash, vs. chasing a false alarm. This is exactly the judgment `optimization_target` exists to make explicit, rather than leaving it for Gemini to guess at (see "Data Science Notes" below for what happened when it wasn't stated).

**`log_iterations` (2026-07-19)** is forwarded unchanged to `run_agent_loop` — `run_session` makes no logging decisions of its own; see "Per-iteration logging" above.

`run_session` never calls `input()` — it stays directly callable with a hardcoded string, no interactive I/O to mock in a future test. Whichever CLI entry point is eventually built (`main.py`, not started) is responsible for actually asking the person what to optimize for.

### What's genuinely still missing

The orchestration loop itself is wired and verified (see "Data Science Notes" below for live runs). What remains open: the human-in-the-loop hook (scope agreed 2026-07-19, build still deferred), the two `gemini_client.py` limitations noted above (single-call-per-turn, convergence result not echoed on stop), and a human-friendly viewer for the result/comparison JSON files (design agreed 2026-07-24, not yet built). The cross-run comparison itself — previously the biggest open gap — is now built; see "Cross-run comparison" above. **`main.py`'s real interactive CLI loop is now also built (2026-07-27)** — see immediately below.

---

## `main.py` — the CLI entry point (2026-07-27; extended 2026-07-29, 2026-08-01)

### Why this needed building specifically

`run_smoke_test.py` is gitignored — meaning it was, and remains, invisible to anyone cloning this repo. Before the 2026-07-27 session, there was no committed way for a stranger to actually *run* a session at all, regardless of how complete `run_session` itself already was. `main.py` is that missing public entry point; it adds no new orchestration logic of its own — it asks a real person the two genuine judgment calls `run_session` needs, then calls it exactly the way `run_smoke_test.py` already does.

### How the dataset menu stays in sync with the registry, automatically

```
dataset.py
┌───────────────────────────────────────────────┐
│ DATASET_LOADERS: dict[str, DatasetSpec]        │
│   "climate"        → DatasetSpec(...)          │
│   "breast_cancer"  → DatasetSpec(...)           │
│   [new entry]      → DatasetSpec(...)  ← the ONLY
│                                            place a
│                                            new dataset
│                                            gets added
└───────────────────────────────────────────────┘
                    │  imported by reference —
                    │  main.py never hardcodes names
                    ▼
main.py (every run, at startup)
┌───────────────────────────────────────────────┐
│ names = list(DATASET_LOADERS.keys())           │
│                                                 │
│  --dataset given?                              │
│    yes → argparse's choices=names rejects an    │
│          unknown name immediately, before        │
│          anything runs                           │
│    no  → interactive menu lists names +          │
│          descriptions, generated live from       │
│          the registry, never a fixed list        │
└───────────────────────────────────────────────┘
                    │  dataset_name (validated)
                    ▼
run_session(dataset_name, ...) → build_dispatch_table
                    │
                    ▼
   load_dataset(name)   → DATASET_LOADERS[name].loader()
   get_pos_label(name)  → DATASET_LOADERS[name].pos_label
```

Adding a new dataset means writing one loader function and one `DatasetSpec` entry in `dataset.py` — nothing in `main.py` needs to change; the CLI's `--dataset` choices and the interactive menu both pick up a new registry entry automatically. **Constraint, not yet lifted:** this assumes binary classification throughout (`evaluate_model`'s `precision_score`/`recall_score`/`f1_score` all use `average="binary"` with a single `pos_label`); a multi-class dataset would need real changes beyond the registry. See `TECHNICAL_NOTES.md` Part 5, §5.8 for the full step-by-step (referenced directly from `main.py`'s own terminal output, so it's discoverable without reading source).

### What's a genuine judgment call vs. a secondary knob

`--dataset`/`--target` prompt interactively when omitted — these are the two facts/judgment-calls `run_session` genuinely needs a person to supply (see "Core architectural principle" above). `--target` is constrained to the exact four metrics `evaluate_model` computes (`recall`/`precision`/`accuracy`/`f1`), not free text. Everything else (`--model`, `--max-iterations`, `--random-state`, `--log-iterations`) silently uses `run_session`'s own defaults unless explicitly overridden — no prompts, no interruption to a normal run. The effective configuration actually used is printed once, at the start of every run, and persisted into the saved result file's `config` key — discoverable without ever pausing execution to ask:

```
Running: dataset=climate, target=recall, model=gemini-3.1-flash-lite,
max_iterations=15, random_state=42, log_iterations=True
(override any of these next time with --dataset, --target, --model,
--max-iterations, --random-state, --no-log-iterations)
```

`--log-iterations` defaults to `True` here — deliberately different from `run_session`'s own library default of `False` — since a first-time user benefits from seeing the agent's full reasoning trail, and it can be turned off with `--no-log-iterations` for a quieter run. Unlike `run_smoke_test.py`, `main.py` persists the result to `results/` unconditionally, even on a quiet run, rather than gating the save behind `log_iterations` — losing a run's outcome entirely just because someone wanted less terminal output was judged an avoidable loss.

### The `compare` subcommand (2026-07-29)

`main.py` now handles two subcommands under one entry point, without breaking any existing invocation:

```
python main.py                          # implicit "run" — unchanged from before this session
python main.py run --dataset climate    # explicit "run", identical result
python main.py compare --dataset climate
```

The first CLI token is sniffed before argparse ever sees it: `"compare"` hands off entirely to a small dedicated parser and returns, never touching `run_session`; `"run"` has that one token stripped and everything proceeds exactly as it did before; anything else (or nothing) proceeds exactly as it always has. This was a deliberate choice over requiring `run` as an explicit subcommand everywhere (the more conventional argparse pattern) — with the project not yet public and no existing users besides its author, retraining muscle memory for zero benefit was judged not worth it; this can be revisited if/when the project goes public.

`compare` requires `--dataset` — comparing across two different datasets would mix incomparable metrics into one file, which is never allowed (see "Cross-run comparison" above). If omitted or misspelled, it prints a warning and falls back to the same interactive picker `run` already uses, rather than a bare argparse error. Output is always named `comparison_<timestamp>_<dataset>.json`.

### The `export` and `report` subcommands (2026-08-01)

Two more subcommands, sniffed the same way as `compare`:

```
python main.py export --dataset climate
python main.py report results/result_log_2026_07_27_223458_climate.json
```

`export` (CSV) requires `--dataset`, identical rule to `compare`. `report` (Markdown) takes a file path instead, auto-detecting a single-run vs. comparison file by name — see "Results reporting: CSV export and the Markdown viewer" above for the full design and the real bug this feature surfaced and fixed.

### Free-tier rate limit note (2026-07-29)

Running sessions back-to-back can exceed the Gemini API's free-tier per-minute quota, surfacing as an uncaught `429 RESOURCE_EXHAUSTED` error partway through a run — and since this happens before the result is persisted, the run leaves no file behind at all (see "Cross-run comparison" → "Known limitation" above). `main.py` now prints a note about this at startup, alongside the existing "want to add a new dataset?" pointer.

### Renamed results-file convention (2026-07-29)

`results/smoke_test_log_<timestamp>.json` → `results/result_log_<timestamp>_<dataset_name>.json`. See "Cross-run comparison" above for the full reasoning (why the dataset name comes after the timestamp, not before, and how pre-existing files were migrated).

### Verified against real live runs

Confirmed working end-to-end across both datasets: interactive dataset/target prompts, the printed config summary, the full run through `run_session`, and persistence under the new naming convention. The `compare` subcommand verified against real accumulated data: 12 Climate Crashes runs and 4 Breast Cancer runs, each correctly kept separate.

---

## Project structure

```
.
├── .env
├── .env.example
├── .gitignore
├── .python-version
├── .vscode/
│   └── settings.json
├── AGENTS.md               # agent rules, guardrails, tool list, convergence criteria — not yet written
├── DATA_SCIENCE_ANALYSIS.md
├── README.md
├── SECURITY.md
├── TECHNICAL_NOTES.md
├── main.py                 # ✅ CLI entry point (2026-07-27) — real public replacement for run_smoke_test.py's role; 'compare' subcommand + rate-limit note (2026-07-29); 'export'/'report' subcommands (2026-08-01)
├── ml_agent/
│   ├── __init__.py
│   ├── agent.py             # ✅ orchestration loop wired end-to-end (2026-07-17); log_iterations threaded through (2026-07-19); random_state now also bound into Trainer.train_model (2026-07-29); stale TOOL_FUNCTIONS docstring note corrected (2026-08-01)
│   ├── compare_runs.py      # ✅ turns several results/result_log_*.json runs (one dataset at a time) into one comparison_<timestamp>_<dataset>.json; final-model resolution fixed + final_model_ambiguous flag added, standalone entry point now dataset-required (2026-08-01)
│   ├── dataset.py           # ✅ done — dataset-agnostic loading + inspection
│   ├── gemini_client.py     # ✅ done — Gemini client + function-call dispatch loop, verified; per-iteration logging + format_log (2026-07-19); MAX_ITERATIONS raised 10→15 (2026-07-27)
│   ├── reporting.py          # ✅ NEW (2026-08-01) — CSV export (to_csv) + Markdown viewer (to_markdown), auto-detecting result_log_*/comparison_* by filename
│   ├── tools.py              # ✅ done — Gemini function-calling tool schemas + TOOL_FUNCTIONS; max_iter hyperparameter added (2026-07-27)
│   └── trainer.py            # ✅ done — storage, validation, real sklearn fit/predict/evaluate; fit-time warning capture (2026-07-24); random_state threaded into every estimator for reproducibility (2026-07-29)
├── pyproject.toml            # testpaths = ["tests"] added (2026-08-01) — see "Testing" below for why
├── rename_results.py         # one-time migration script (2026-07-29), gitignored — not project code; renamed pre-existing results/ files to the result_log_*_<dataset>.json convention
├── results/                   # gitignored; one result_log_<timestamp>_<dataset_name>.json per run, plus comparison_<timestamp>_<dataset>.json (compare_runs.py) and export_<timestamp>_<dataset>.csv / *.md (reporting.py, 2026-08-01)
├── run_smoke_test.py         # manual live-API smoke test of run_session(); gitignored; superseded by main.py as the real entry point, kept for quick manual debugging
├── smoke_test.py              # isolated schema-construction check, no API key/network; gitignored
├── tests/
│   ├── __init__.py
│   ├── test_compare_runs.py  # ✅ NEW (2026-08-01), 5 passing tests — locks in the final-model resolution fix
│   ├── test_dataset.py       # ✅ done, 9 passing tests
│   ├── test_reporting.py     # ✅ NEW (2026-08-01), 9 passing tests — auto-detect, CSV/Markdown rendering
│   ├── test_tools.py         # ✅ done, 12 passing tests (2026-07-15; extended 2026-07-19) — reviewed, resolved (2026-07-27), no changes needed
│   └── test_trainer.py       # not yet started
└── uv.lock
```

`ml_agent/` is a real installed package — imports elsewhere use `from ml_agent.tools import ...`, `from ml_agent.dataset import ...`, etc., never bare top-level module names.

_For the current, non-annotated version of this tree, see [README §Project Structure](../README.md#project-structure)._

### Why `agent.py` and `gemini_client.py` are separate
`gemini_client.py` only knows how to talk to Gemini and dispatch function calls. `agent.py` owns the actual multi-step state — what's been tried, results so far, whether to keep iterating. This separation allows convergence logic to be unit-tested without a live API call.

---

## Datasets

The two datasets used (Climate Model Simulation Crashes, Breast Cancer Wisconsin) and why each is here are described in [README §Datasets](../README.md#datasets). The registry mechanics behind them:

Datasets are loaded through a small registry (`DATASET_LOADERS`) in `dataset.py`, keyed by short name (`"climate"`, `"breast_cancer"`). Each registry entry is a `DatasetSpec` — a frozen dataclass pairing the loader function with `pos_label` and a human-readable `description`.

### What `pos_label` actually is

In a binary classification task, the target column only ever holds two values — but *which one counts as the "positive" outcome* isn't a mathematical given, it's a convention someone has to fix per dataset. It matters because precision, recall, and F1 aren't symmetric in the two classes. That's why it's stored as a registry fact rather than inferred at evaluation time.

A dedicated accessor, `get_pos_label(name)`, mirrors `load_dataset(name)`'s pattern — one accessor function per registry fact, never reach into `DATASET_LOADERS` directly outside `dataset.py`.

---

## Available models (via `list_available_models`)

Why only three models is explained in [README §Datasets](../README.md#datasets). All three expose `class_weight: ["balanced", null]`, the single most relevant lever for the recall-optimization framing on Climate Crashes.

| Model | Key hyperparameters |
|---|---|
| Logistic Regression | `C` (0.001–100, inclusive), `class_weight` |
| Random Forest | `n_estimators` (10–500, inclusive), `max_depth` (int or null), `class_weight` |
| SVM | `C`, `kernel` (linear/rbf/poly), `class_weight` |

---

## Setup

Install steps are in [README §Quickstart](../README.md#quickstart). Dependency-pinning policy is in [README §Security](../README.md#security) and `SECURITY.md`.

## Running

Basic run/compare/export/report commands are in [README §Quickstart](../README.md#quickstart). Below: a couple of invocations and notes not covered there.

For a manual, real-API smoke test with hardcoded defaults (the original way this loop was exercised before `main.py` existed):

```bash
uv run python run_smoke_test.py
```

The standalone module invocation now requires a dataset too (resolved 2026-08-01 — see "Cross-run comparison" above):

```bash
uv run python -m ml_agent.compare_runs climate
```

## Testing

```bash
uv run pytest -v
```

Network-dependent behavior (the Climate Crashes OpenML fetch) is not exercised directly in tests — it's mocked. Currently **52 tests passing**: 9 in `test_dataset.py`, 12 in `test_tools.py` (added 2026-07-15, extended 2026-07-19; reviewed and resolved 2026-07-27 against `trainer.py`'s `warnings` key and `tools.py`'s `max_iter` addition — no changes needed), 5 in `test_compare_runs.py` and 9 in `test_reporting.py` (both 2026-08-01 — see "Results reporting" above), and **17 in `test_trainer.py` (new 2026-08-03)** — see "Test coverage" under "Reproducibility: seeding every estimator" above.

**`numpy` added as a dev dependency (2026-08-03):** needed only by `test_trainer.py`'s ill-conditioned-data fixture (the per-feature scale-mismatch array used to reliably trigger `ConvergenceWarning`) — never imported inside `ml_agent/` itself, so it lives in the `dev` dependency group, not the package's main dependencies.

**`pyproject.toml`'s `testpaths` setting (2026-08-01):** without it, pytest's default file-discovery glob (`test_*.py` **or** `*_test.py`) also matches `run_smoke_test.py`, importing and — since its body is unguarded top-level code — actually *executing* it during test collection: a real Gemini API call and a real `results/smoke_test_log_*.json` write, on every `pytest` run, silently. Reproduced three times before being traced to this cause. `testpaths = ["tests"]` restricts collection to the `tests/` directory only. See `TECHNICAL_NOTES.md` Part 7, §7.6 for the full root-cause writeup, including its likely (newly discovered, not previously suspected) contribution to the free-tier rate-limit errors documented elsewhere in this README.

---

## Security

Summary in [README §Security](../README.md#security); full policy in [`SECURITY.md`](./SECURITY.md).

---

## Roadmap context

Project 5 of a 10-step learning roadmap: Linux → Git → Professional Python → Pandas → SQL → **Scikit-learn (this project)** → AI-assisted workflows → PyTorch → Transformers → Geometric Deep Learning. Builds directly on the agentic-loop pattern established in `sql-agent` (Project 3).

### Progression of "next up"

1. **After `dataset.py`:** implement `trainer.py`'s real bodies, resolving `pos_label` per-dataset via `get_pos_label()`.
2. **After `Trainer` storage/validation scaffolding:** fill in real fit/evaluate logic. Then `test_trainer.py`, then `gemini_client.py` and the real orchestration loop in `agent.py`.
3. **As of 2026-07-10:** sklearn fit/evaluate logic and the `agent.py` train/test split + `validate_split` gate in place. `gemini_client.py` the explicit next milestone.
4. **As of 2026-07-13:** `gemini_client.py` is complete and verified via multiple real end-to-end runs on both datasets. Remaining before the project's next phase: wire `agent.py`'s actual orchestration loop (currently only exercised via a hand-built standalone script, not real `agent.py` code), then `test_tools.py` (the `inspect.signature()` drift check), then `test_trainer.py`, then `AGENTS.md`.
5. **As of 2026-07-15:** `test_tools.py` complete — 11/11 passing, 20/20 across the full suite. Next milestone: `agent.py`'s orchestration loop wiring — deciding how `build_dispatch_table` should expose `(X, y)`/`initial_context` to the real loop, injecting an explicit optimization target into `initial_context`, and deciding whether the human-in-the-loop hook gets built now or stays deferred. Recommended as its own dedicated session given its scope.
6. **As of 2026-07-17:** `agent.py`'s orchestration loop is wired end-to-end and verified via 3 live runs (`build_dispatch_table` → `DispatchResult` → `run_session` → `run_agent_loop`). The optimization-target injection is built and confirmed working against a live agent. **Not resolved this session:** the human-in-the-loop hook's timing (still deferred, undecided either way — not the same as "rejected"). **New, explicitly requested:** per-iteration logging inside `run_agent_loop`, to support deeper run-to-run analysis — planned for a future session.
7. **As of 2026-07-19:** per-iteration logging is built, verified via a live run, and made human-readable (`format_log`) and persistable (`results/smoke_test_log.json`, at that point still overwritten per run — see item 8). The `TOOL_FUNCTIONS`-rename test safeguard from item 6 is also built, though its original rationale was found to be inaccurate on closer inspection (see "Development Notes" below) — kept as documentation, not as a fix for a real gap. **The human-in-the-loop hook's scope is now decided** (hyperparameter edge-case flagging, reasoning/action contradiction detection, stalled/repeated-proposal detection — deliberately not per-proposal approval, deliberately not convergence-only review), but its **build remains deferred** until the project is close to a finished, working state.
8. **As of 2026-07-24:** run persistence corrected to one timestamped file per run instead of an overwritten single file, wall-clock timing added, fit-time warning capture built and verified in `trainer.py` (captures every warning category, not just the one already known about), and `ml_agent/compare_runs.py` built and verified against 6 real runs — the cross-run comparison goal from item 7 is now met. **Not resolved this session:** whether/how to actually address the underlying `ConvergenceWarning` itself — not yet even narrowed to a shortlist. **Also carried forward:** a `test_tools.py` review against `train_model`'s new `warnings` key, a minor, low-priority timestamp-format inconsistency between `run_smoke_test.py` (local time) and `compare_runs.py` (UTC), and the still-undecided choice between building `main.py` next or the human-friendly results viewer.
9. **As of 2026-07-27:** `test_tools.py` reviewed and resolved (no changes needed — see its section above); the `ConvergenceWarning` shortlist narrowed to a decision and built (`max_iter` exposed, verified against 3 live runs — see "Closing the `ConvergenceWarning` shortlist" above); `main.py` built as the real public CLI entry point, chosen over the results viewer as this session's priority specifically because `run_smoke_test.py` being gitignored meant no committed entry point existed at all; and `MAX_ITERATIONS` raised from 10 to 15, closing a 07-12 open decision once real live-run evidence met its stated condition.
10. **As of 2026-07-29 (current):** the `max_iter`-vs-`C` confound (item 9's carried-forward follow-up) resolved via a real controlled run — `C`, not `max_iter`, confirmed as the driver of the recall improvement, and a metric-terminology correction made in the same write-up (see "Closing the `ConvergenceWarning` shortlist" above). Results-file naming renamed and dataset-scoped (`result_log_<timestamp>_<dataset_name>.json`, `compare_runs.py`'s `build_comparison` now filters by dataset), closing a 2026-07-27 open item, with a one-time migration script for pre-existing files. `main.py` gained a `compare` subcommand (dataset-required, sharing one entry point with `run`) and a free-tier rate-limit note, after reproducing a real `429 RESOURCE_EXHAUSTED` crash from running sessions back-to-back. A genuine reproducibility bug was found and fixed: `RandomForestClassifier` had no `random_state` at all, so identical hyperparameters could produce different results across separate runs — `random_state` is now threaded through every estimator, reusing the same seed already used for the train/test split (see "Reproducibility: seeding every estimator" above).

    **Carried forward from 2026-07-29 into this session, both now resolved:** the human-friendly results viewer and the CSV/spreadsheet export — see item 11 below.

11. **As of 2026-08-01 (current):** both remaining named blockers to making the repo public are now built. `ml_agent/reporting.py` (CSV export + Markdown viewer, sharing one data-prep path built on `compare_runs.py`'s `summarize_run()`) and two new `main.py` subcommands, `export` and `report`, mirroring `compare`'s pattern. Building this surfaced and fixed a real bug: `summarize_run`'s "last evaluated model = final model" heuristic was confirmed wrong on a real file — final model is now resolved by best score on the run's own optimization target, with a new `final_model_ambiguous` flag surfacing any disagreement rather than silently picking one. Also resolved this session: the 2026-07-29 open question on whether the standalone `python -m ml_agent.compare_runs` entry point should require a dataset (it now does, matching `main.py`), and a stale docstring/TODO note about an already-existing `test_tools.py` safeguard (added 2026-07-17, not "not yet built" as both had claimed). A separate, unrelated bug was also found via manual testing and fixed: `pyproject.toml` had no `testpaths` setting, so pytest was silently importing and executing `run_smoke_test.py` (a real API call) on every test run — see "Testing" above. Two new test files (`test_compare_runs.py`, `test_reporting.py`) bring the suite to 35 passing tests.

    **Carried forward from 2026-08-01 into this session, now resolved:** `test_trainer.py` (item below), and the `DATA_SCIENCE_ANALYSIS.md` writeup of the `final_model_ambiguous` finding — see item 12.

12. **As of 2026-08-03 (current):** `tests/test_trainer.py` built — 17 tests, real sklearn fits, parametrized across all three registered estimators, closing the item named "still never started" on 2026-08-01 (see "Test coverage" under "Reproducibility" above for the full writeup, including a three-attempt debugging story worth reading). `numpy` added as a dev dependency to support it. `DATA_SCIENCE_ANALYSIS.md` gained §13, the data-science-layer writeup of the `final_model_ambiguous` finding from 2026-08-01 (see "Results reporting" above for the pointer). **The human-in-the-loop design was substantially explored and refined** — not built, but no longer just a named placeholder either; see "Human-in-the-loop design" below for the full write-up and diagram.

    **Carried forward, full running list, for the next session:**
    - **The human-in-the-loop hook's actual implementation** — design refined this session (see "Human-in-the-loop design" below), still not built. What's specifically still open before it can be built: a precise definition of "untrustworthy"/"flagged" beyond the existing `final_model_ambiguous` (candidate criteria: a suspiciously low final metric, a warning that never got resolved — both need real design time, not a quick call).
    - **A general repo-wide readability pass (named this session, not yet started):** this README has grown as a genuine session-by-session log, which was the right shape while the project was actively taking form, but isn't yet the shape a first-time public reader wants. Planned: restructure into something more browsable, and clean up unnecessary commented-out lines across the codebase and docs. Explicitly deferred to its own dedicated session, given its scope.
    - Timestamp-format inconsistency, `run_smoke_test.py` (local) vs. `compare_runs.py` (UTC) — low priority, unfixed.
    - **A "register/list datasets from the terminal" feature (expanded 2026-07-29, merging in a 2026-07-24 idea raised in the same spirit):** `dataset list` / `dataset upload` subcommands, mirroring `compare`'s pattern — `list` shows the current `DATASET_LOADERS` registry, `upload` lets a user register a new dataset without hand-editing `dataset.py`. Not scoped in detail; only worth building once a third dataset is realistically on the horizon.
    - **A second, optional `--model-random-state` flag (2026-07-29), low priority:** would default to match `--random-state`, restoring the ability to isolate split-variance from model-internal-variance as an opt-in advanced knob — considered and deliberately deferred in favor of the simpler single-seed design actually built (see "Reproducibility" above for the full trade-off).
    - A further extension to the `C`-vs-`max_iter` isolation finding (2026-07-29), low priority: sweeping `C` across several values with `max_iter` held constant, to characterize the shape of the relationship rather than just confirm its direction — nice-to-have for the portfolio angle, not blocking anything.
    - `main.py`'s `export`/`report` CLI wiring (argument parsing, interactive-prompt fallback, file-write side effects) has no automated test coverage yet (flagged 2026-08-01) — only the `reporting.py`/`compare_runs.py` logic it calls does. Would need `tmp_path`, `pytest-mock`'s `mocker`, and `capsys`.
    - Housekeeping (flagged 2026-08-01): the pre-2026-07-29 `result_log_*.json` files (no `config` block, fall back to the old last-evaluated heuristic, show `?` for dataset/target in reports) are planned for eventual deletion once no longer needed for reference — not done yet.
    - The two original 07-13 `run_agent_loop` judgment calls (single-call-per-turn; convergence result not echoed on stop) — still untouched, still deliberately deferred.

> **Why this section exists at this length:** the human-in-the-loop hook was scoped back in 2026-07-19, then merged with two other separate ideas (an `Agent-decisions.md` generator, and using Gemini to adjudicate a `final_model_ambiguous` flag) into one coherent design — explored in real depth, but never built. `README.md`'s Roadmap only has a single line pointing here; this is the only place the actual design lives. Two modes: **live** (pauses training synchronously on a known signal, resumes after human input) and **post-hoc** (a Gemini-assisted explanation for a run flagged after it finishes, not during). One piece is still genuinely open: a precise definition of "flagged," beyond the one concrete criterion (`final_model_ambiguous`) that already exists.

### Human-in-the-loop design (explored 2026-08-03, implementation deferred)

Three previously-separate ideas — the human-in-the-loop hook (scoped 2026-07-19), the `Agent-decisions.md` generator (2026-07-27), and using Gemini to adjudicate a `final_model_ambiguous` flag (2026-08-01) — turned out, on closer discussion, to fit into **one coherent design with two genuinely different modes**, not three separate features:

```
                    ┌────────────────────────┐
                    │  Training loop running  │
                    └────────────┬─────────────┘
                 if stall/crash  │  otherwise
              ┌───────────────────┴───────────────┐
              ▼                                    ▼
   ┌───────────────────────┐            ┌──────────────────┐
   │  Live sync interrupt   │            │   Loop finishes    │
   │  Pauses, asks human    │            └─────────┬──────────┘
   └───────────┬─────────────┘           clean, unambiguous │  flagged/ambiguous
        ↻ resumes training               ┌──────────┴──────────┐
        loop (back to top)               ▼                     ▼
                              ┌────────────────────┐  ┌───────────────────────┐
                              │    reporting.py      │  │  Gemini debug assist   │
                              │  Standard run report  │  │  Explains what & why   │
                              └────────────────────┘  └───────────┬───────────────┘
                                                                    │ optional, on request
                                                                    ▼
                                                        ┌──────────────────────────┐
                                                        │ Deeper DS analysis over    │
                                                        │ the raw result_log_*.json  │
                                                        └──────────────────────────┘
```

**Live mode:** triggered by deterministic code inside `run_agent_loop` watching for known signals (e.g. a `ConvergenceWarning` already present in `train_model`'s returned `warnings`), not by Gemini deciding for itself whether to ask — the same orchestration-code-owns-the-judgment-call pattern used everywhere else in this project (see "Core architectural principle" above). Pauses synchronously, explains what's happening, gets human input, then resumes.

**Post-hoc mode, two-tiered:** only for a run flagged as needing a closer look after it finishes.
- **Default:** a single Gemini call reasoning over what `reporting.py`'s existing report already captures — final model, hyperparameters, metrics, full model-sequence trail, convergence reasoning verbatim, warnings. No new logging infrastructure needed; this is a new *consumer* of data the project already collects, not a new capture mechanism. Produces both a plain-language explanation (what happened, in words a less technical reader can follow) and a genuine data-science causal analysis (why it likely happened) — not just a restatement of the report.
- **Optional escalation, on request:** if the default summary-level analysis isn't enough, a deeper investigation directly over the raw `result_log_*.json` (full per-iteration detail, not just the final narrative).

**What decides a run is "flagged," precisely, is the one piece still genuinely open** — `final_model_ambiguous` (see "Results reporting" above) is the only concrete criterion that exists today. A suspiciously low final metric and a warning that never got resolved are logged as candidate additions, not yet precisely defined. This — along with the live-trigger signal list, and where exactly the interrupt slots into `run_agent_loop` — needs a dedicated design session before any of this gets built.

---

## Development Notes

This project was developed with AI assistance:
- **Development tool:** Claude Sonnet 5 (Anthropic), used as a pair-programming and design-review collaborator.
- **Runtime AI component:** `gemini-3.1-flash-lite` (Google), the model actually orchestrating the agent's tool-calling loop at runtime — this is the AI system the project *is*, as distinct from the AI assistance used to *build* it.

All architectural decisions were reviewed and made by the project author. AI assistance focused on explaining concepts, generating boilerplate, and surfacing issues for human review — not on unsupervised code generation.

### Real-world debugging notes (2026-07-13 session)

Building and verifying `gemini_client.py` surfaced two genuinely instructive technical patterns worth documenting plainly, since they're the kind of thing likely to recur on any project using a fast-moving SDK:

**Static type stubs and runtime behavior can disagree, and both can be "correct."** The `google-genai` SDK's own type hints declare `Tool(function_declarations=...)` as expecting a list of `FunctionDeclaration` objects — but its underlying Pydantic model actually accepts plain dicts too, coercing them internally. A type checker (Pylance) flagging this isn't wrong, and the code isn't broken either — they're answering different questions ("does this match the declared contract?" vs. "does this work?"). The practical lesson: when a type checker and an actual test run disagree, that's worth investigating explicitly rather than trusting either alone — and when in doubt, matching the declared contract explicitly (wrapping dicts into the real object type) is cheap insurance against undocumented behavior changing in a future SDK version.

**SDK fields typed `Optional` should be guarded, not assumed.** `FunctionCall.name` and `.args` are both typed as possibly-`None` in the SDK, even though in ordinary use they're essentially always populated. Code that reads them without checking works fine until the one time it doesn't — and when it doesn't, the failure surfaces confusingly far from the actual cause. Adding an explicit `if call.name is None: raise ValueError(...)` guard cost two lines and turned a hypothetical confusing failure into an immediate, clearly-named one.

A third, unrelated but equally real finding: `uv run` does not automatically load a project's `.env` file — a dependency (`python-dotenv`) can be correctly installed and completely unused if nothing in the code actually calls `load_dotenv()`. Worth checking explicitly, not assuming, when an API key "should" be available but isn't.

Full technical write-up of a related, deliberately-deferred design question (conversation-history growth and cost scaling as `MAX_ITERATIONS` increases) lives in [`TECHNICAL_NOTES.md`](./TECHNICAL_NOTES.md).

### Testing/tooling notes (2026-07-15 session)

Building `test_tools.py` surfaced one genuinely instructive Python-mechanics detail worth keeping documented: **`inspect.signature()` on a `functools.partial` object automatically excludes the already-bound arguments** from the signature it reports. This matters here because `evaluate_model`/`train_model` get several arguments pre-bound by `build_dispatch_table` before Gemini ever sees them — the drift check therefore compares `TOOL_SCHEMAS` against the *bare*, unbound functions in `tools.py`, not the partial-bound versions in the dispatch table, since Gemini only ever fills in the parameters that are still open.

### Orchestration wiring notes (2026-07-17 session)

Wiring `agent.py`'s real orchestration loop for the first time surfaced one design question worth documenting: `build_dispatch_table`'s return type needed to change (to also expose `(X, y)`, avoiding a second `load_dataset` call — see "`agent.py`" above), which raised the question of *where* the newly-required data should live. The choice made — a narrow `DispatchResult` carrying just `(X, y)` and the dispatch table, with prompt assembly kept as a separate caller-side step — was deliberately the smaller of two possible signature changes, favoring keeping `build_dispatch_table`'s job narrow over folding prompt construction into it as well.

Reusing `tools.py`'s `TOOL_FUNCTIONS` mapping for 3 of the dispatch table's 5 entries (rather than hand-building all 5, as before) closes an explicitly-flagged open question from the 2026-07-15 session. It was originally documented here as introducing a risk — that a rename of `train_model`/`evaluate_model` in `tools.py` would cause the 2 override entries to "silently stop overriding." **Correction (2026-07-19):** this turned out to be inaccurate. The override in `agent.py` uses literal string keys, not a `TOOL_FUNCTIONS` lookup, so it cannot fail this way — see `TECHNICAL_NOTES.md` §2.2's correction note for the full explanation, and Part 3 for the test that was built anyway (as documentation of the concern, not as a fix for a real gap).

### CLI entry point and threshold-revisiting notes (2026-07-27 session)

Building `main.py` surfaced a distinction easy to conflate: an unvalidated `optimization_target` string reaching Gemini's prompt unchecked is an **input-validation gap** (a typo or invented metric silently has no connection to what the tool results can support) — not **prompt injection** (which specifically means malicious instructions smuggled into content the model processes, hijacking it against its own instructions). The fix for the former (constrain to a known set) is a normal input-validation measure, not a security control against the latter — know which problem a given safeguard actually addresses.

Separately, this session closed an open decision (`MAX_ITERATIONS`, 10 → 15) using a criterion worth naming explicitly for future similar cases: a debugging-safety-net value should be revisited once its own stated condition is met by real evidence, not left in place indefinitely just because it was never causing visible errors. Here, the cap wasn't failing loudly — it was silently truncating legitimate in-progress agent reasoning, a quieter failure mode that only became visible by actually reading what a real run's tool-call sequence was doing when it got cut off. Full rationale in `TECHNICAL_NOTES.md` Part 5, §5.9.

### Reproducibility, naming, and discoverability notes (2026-07-29 session)

Two Breast Cancer runs looking identical, followed by a third with matching hyperparameters producing a *different* result, is the kind of anomaly that's easy to wave away as noise if it isn't actually chased down. It wasn't: checked directly against `trainer.py`'s real code rather than assumed, this turned out to be a genuine bug (`RandomForestClassifier` never received a `random_state`, see "Reproducibility: seeding every estimator" above) rather than a fluke — a reminder that an *apparent* pattern across a small number of runs (three matching results) can still be coincidence, and the anomaly that breaks the pattern is sometimes more informative than the pattern itself.

A related terminology correction, made explicitly rather than fixed quietly: an earlier write-up of the `max_iter`-vs-`C` confound (2026-07-27) described the confounded runs as reaching "a better precision." Re-checking the actual numbers this session showed precision never moved at all across any of the relevant runs — the metric that changed was recall. The distinction matters beyond wording: `max_iter` and `C` do fundamentally different things (solver iteration cap vs. regularization strength), and getting the metric right is part of getting the causal claim right, not a cosmetic detail.

The results-file naming change (`smoke_test_log_<timestamp>.json` → `result_log_<timestamp>_<dataset_name>.json`) surfaced a design fork worth documenting: placing `dataset_name` *before* the timestamp (`result_log_<dataset_name>_<timestamp>.json`) was the initially-proposed convention, but would have broken `compare_runs.py`'s existing assumption that a whole-filename string sort produces chronological order — all of one dataset's runs would sort before all of another's, regardless of when they actually ran. Checking a downstream consumer's assumptions before finalizing an upstream naming change caught this before it shipped, rather than after.

Also worth recording: a subcommand design (`main.py compare`, alongside the existing default `run` behavior) was chosen over requiring an explicit `run` subcommand for the existing default path, specifically because there were no existing users of the CLI besides this project's own author at the time of the change — retraining muscle memory for a convention with zero present benefit was judged the wrong trade-off for a not-yet-public project, though worth revisiting if the project goes public later and a stranger's first interaction is `--help`.

### Results reporting and two real bugs found (2026-08-01 session)

Building `reporting.py` surfaced a genuine bug rather than just a missing feature: a heuristic ("last evaluated model in a run's log = the final model") that had sat undisturbed since 2026-07-24, explicitly flagged in its own docstring as "not observed in any real run yet," turned out to be wrong the first time a real file actually exercised the case it warned about. The lesson generalizes beyond this specific bug: a documented-but-unverified assumption is a liability that accumulates silently until something forces it to be checked — in this case, extending `summarize_run()` for an unrelated reason (surfacing hyperparameters) was what happened to surface it. Full technical writeup, including the exact real numbers involved, is in `TECHNICAL_NOTES.md` Part 7, §7.1–7.2.

A second, unrelated bug was found the same way — not by design review, but by a person noticing something that shouldn't have been happening (an unexplained file appearing after running `pytest`) and pushing to actually trace it rather than dismiss it as unrelated. The root cause — pytest's default collection glob silently importing and executing an unguarded top-level script — was several layers removed from the symptom, and easy to have written off as coincidental timing. See `TECHNICAL_NOTES.md` Part 7, §7.6 for the full trace.

A smaller, purely process-level finding worth naming directly: a mitigation described in `agent.py`'s own docstring as "not built this session" (2026-07-19) had, in fact, already been built — two sessions later, in a commit that never updated the docstring or the running TODO list to say so. Neither the code nor the fix was wrong; the *documentation about the code* had simply drifted from reality without anything catching it until a `git log` review this session did. Worth deliberately checking a project's actual current state (via `git log`, `grep`, or the test suite itself) before trusting a docstring or TODO's claim that something "isn't built yet" — the claim itself can be the stale part, not the thing it's describing.

### Test design and empirical verification notes (2026-08-03 session)

Building `test_trainer.py`'s `ConvergenceWarning` test surfaced a lesson easy to repeat if unstated: a *plausible-sounding* mechanism for a numerical result isn't the same as a *verified* one, even when the reasoning sounds like it should be right. Two consecutive hypotheses — that uniform feature scaling would slow `lbfgs` convergence, and separately, that near-perfect linear separability would prevent it from converging at all — both sounded like reasonable applications of real optimization theory, and both failed the moment they were actually run. The theory wasn't exactly *wrong* in either case; each simply missed a detail that only running the code surfaced (uniform scaling doesn't touch relative feature conditioning; `lbfgs`'s gradient-norm stopping rule can be satisfied well before a theoretically-unreachable optimum matters in practice). The fix, in both the debugging process and the practice this encourages generally: when a numerical claim can actually be checked by running code, check it — don't ship a second theoretical guess without running it first, however confident it sounds. See "Reproducibility: seeding every estimator" → "Test coverage" above for the full three-attempt account, and `TECHNICAL_NOTES.md` Part 8 for the exact commands and numbers from each attempt.