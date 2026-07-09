# ml-agent

_Last updated: 2026-07-07_

Agentic ML experimentation assistant — an LLM (Gemini) orchestrates dataset inspection, model proposal, training, and evaluation over scikit-learn via native function-calling, iterating toward a target metric in a supervised loop.

**Status: in progress.** This is Project 5 in a 10-step self-directed learning roadmap toward Data Science / ML Engineering / Agentic AI. `dataset.py` and `tools.py` are complete (the latter with two contract stubs awaiting `trainer.py`); `gemini_client.py`, `trainer.py`, and `agent.py` are in progress.

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

This same principle extends one level deeper into `tools.py`: **which class counts as "positive"** for a given dataset is also a fact, not a judgment call, and is never something Gemini supplies or guesses (see "Tool architecture" below).

---

## Tool architecture: Category A (execution) vs Category B (decision)

A key design decision, made explicit early rather than left implicit in code: native function-calling is used here for two genuinely different purposes, kept strictly separate.

- **Category A — execution tools.** Ordinary deterministic Python/scikit-learn functions with real side effects (train a model, compute metrics). They have no awareness that an LLM exists; they just receive arguments and run. Same inputs → same outputs, every time.
- **Category B — structured-decision tools.** No execution happens here at all. Calling one of these captures Gemini's chosen arguments as a dict — the tool exists purely to force free-form reasoning into parseable JSON instead of prose.

**The consequence that matters:** all of the agent's "agency" lives entirely inside Category B tool calls. Category A is just plumbing that executes whatever B decided — Category B's output doesn't just inform Category A, it *becomes* Category A's function arguments, verbatim, via a direct handoff in `agent.py`. If Gemini were swapped for a different model, or even a human typing choices into a CLI, Category A would not need to change at all.

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
     │ → model reference/id     │   fields verbatim into
     └────────────┬────────────┘   train_model's arguments
                  ▼
     ┌─────────────────────────┐
     │ evaluate_model            │
     │ (model_ref, *, pos_label)  │  ← pos_label supplied internally
     │ → accuracy, precision,     │     by agent.py, NEVER by Gemini
     │   recall, f1,               │
     │   confusion_matrix          │
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
              (agent.py increments iteration count,
               checks max-iterations guard too)
```

**Designed extension point:** a human-in-the-loop confirmation step (e.g. "approve this proposal before training?") slots in cleanly at the Category B → Category A handoff — between `record_model_proposal`'s return and `train_model`'s call — without requiring changes to either category's internals. Not yet implemented; noted here as a deliberate future hook.

The two Category B tools are worth a brief word each, since their names alone don't say much. `record_model_proposal` is how Gemini puts a candidate on the table — a model type, its hyperparameters, and the reasoning behind choosing them — without anything being trained yet; it's a proposal in the literal sense, waiting to be acted on. `record_convergence_decision` is the checkpoint after a model has actually been evaluated: Gemini looks at the metrics it just got back and states, explicitly, whether to stop here or try again, along with why. Neither tool does anything on its own — they exist so that a step which would otherwise be free-form prose ("I think we should try a random forest next...") becomes a structured, reviewable decision instead. (Both descriptions here are intentionally brief — they'll be filled out further once `trainer.py` exists and these tools have real evaluation results to react to, rather than a stub.)

All five tools above exist as ordinary Python functions in `tools.py`, but Gemini never sees Python — it needs each tool declared as a JSON schema (name, description, parameter types) before it can call any of them. That declaration lives in `TOOL_SCHEMAS`, a list at the bottom of `tools.py` that `gemini_client.py` hands to the Gemini API. It's kept as its own structure rather than auto-generated from the functions' docstrings above it, on purpose: **the schema text is what *Gemini* reads to decide when and how to call a tool, while the docstring is what a *human* reads to understand the code — the two audiences don't always need the same amount of detail, so letting the wording diverge slightly where it helps is a small deliberate cost, not an oversight**. The trade-off is that a signature change to a function requires a matching by-hand update to its schema entry — easy to forget, worth double-checking when either changes. Concretely: nothing enforces that the two stay in sync, so a forgotten update either leaves a new parameter invisible to Gemini (it simply never learns the option exists — no error, just a silently unreachable feature) or, worse, causes a `TypeError` at dispatch time if a parameter gets renamed — and that error only surfaces whenever Gemini next happens to call that specific tool, which may be well after the change was made. A planned mitigation (not yet implemented) is a `test_tools.py` check using `inspect.signature()` to compare each function's real parameters against its `TOOL_SCHEMAS` entry automatically, rather than relying on remembering to check by hand.

---

## Project structure

```
ml-agent/
├── pyproject.toml
├── .env.example
├── .gitignore
├── AGENTS.md              # not yet written — see Roadmap / TODOs
├── main.py
├── ml_agent/
│   ├── dataset.py         # dataset-agnostic loading + inspection ✅ done
│   ├── tools.py            # Gemini function-calling tool schemas ✅ done
│   │                        #   (train_model/evaluate_model are contract
│   │                        #   stubs pending trainer.py)
│   ├── gemini_client.py    # Gemini client wrapper + function-call dispatch — not started
│   ├── trainer.py          # scikit-learn training + metric computation — not started
│   └── agent.py             # orchestration: state, convergence logic, main loop — not started
└── tests/
    ├── test_dataset.py    # ✅ done, 9 passing tests
    ├── test_tools.py       # not started (Category B only, until trainer.py exists)
    └── test_trainer.py     # not started
```

---

## Datasets

- **Primary: Climate Model Simulation Crashes** (OpenML id 1467). Binary classification — predict whether a given combination of 18 physical parameters (from the POP2 ocean model component of CCSM4) causes a numerical simulation crash. 540 rows, genuinely imbalanced (~91.5% success / 8.5% failure), which gives a "optimize for recall" objective real substance. Loader explicitly drops two non-predictive identifier columns present in the raw source (documented by the dataset's own maintainers as unfit for prediction) and remaps OpenML's string target labels to standard `0`/`1` semantics, with `1` established as the failure/positive class via cross-checking against the dataset's documented failure count.
- **Fallback: Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`). Used for fast, network-free debugging of the pipeline in isolation. Note: its positive-class convention (`1 = benign`) is the *opposite* sense from Climate Crashes (`1 = failure`) — `pos_label` is established per-dataset, never assumed consistent.

Datasets are loaded through a small registry (`DATASET_LOADERS`) in `dataset.py`. Each registry entry is a `DatasetSpec` — a frozen dataclass pairing the loader function with two dataset-level facts that must never be guessed or re-derived at evaluation time: `pos_label` (the positive class for precision/recall) and a human-readable `description`. Adding a new dataset means writing one loader function and one `DatasetSpec` registry line.

### What `pos_label` actually is

In a binary classification task, the target column only ever holds two values (here, `0` and `1`) — but *which one counts as the "positive" outcome* isn't a mathematical given, it's a convention someone has to fix per dataset. `pos_label` is that fixed value.

It matters for one concrete reason: precision, recall, and F1 aren't symmetric in the two classes — each is defined relative to whichever class is "positive." Recall answers *"of the actual positives, how many did we catch"*; that's a different question depending on whether "positive" means *failure* (Climate Crashes) or *benign* (Breast Cancer). Compute the same metric with the wrong `pos_label` and you silently get a correct-looking number answering the wrong question — not an error, just quietly wrong.

That's why it's stored as a registry fact rather than inferred at evaluation time: it removes any ambiguity about which label the evaluation pipeline is tracking, for every dataset, every run — nothing is left for the code (or the LLM) to guess.

```
DATASET_LOADERS
  "climate"        ──► DatasetSpec(loader=..., pos_label=1, description="1 = simulation failure (rare class, ~8.5%)")
  "breast_cancer"  ──► DatasetSpec(loader=..., pos_label=1, description="1 = benign (majority class) — opposite sense from climate")
```

A dedicated accessor, `get_pos_label(name)`, mirrors `load_dataset(name)`'s pattern (including its clear `ValueError` for an unknown name) — the house rule going forward is: **one accessor function per registry fact, never reach into `DATASET_LOADERS` directly outside `dataset.py`.**

---

## Available models (via `list_available_models`)

Deliberately kept to three model types rather than an exhaustive sklearn zoo — this project's explicit growth area is agentic/tool-calling engineering, not exhaustive model search. All three expose `class_weight: ["balanced", null]`, the single most relevant lever for the recall-optimization framing on Climate Crashes.

| Model | Key hyperparameters |
|---|---|
| Logistic Regression | `C` (0.001–100), `class_weight` |
| Random Forest | `n_estimators` (10–500), `max_depth` (int or null), `class_weight` |
| SVM | `C`, `kernel` (linear/rbf/poly), `class_weight` |

---

## Setup

```bash
git clone git@github.com:MelanieM2/ml-agent.git
cd ml-agent
uv sync
cp .env.example .env   # then fill in your real GEMINI_API_KEY
```

Dependencies are pinned exactly (`add-bounds = "exact"` in `[tool.uv]`) and restricted to packages no newer than 7 days old at install time (`exclude-newer = "7 days ago"`), as a routine security precaution applied to every project in this series.

## Running

```bash
uv run python main.py
```

(Interactive loop entry point — details to be filled in as `agent.py` is completed.)

## Testing

```bash
uv run pytest -v
```

Network-dependent behavior (the Climate Crashes OpenML fetch) is not exercised directly in tests — it's mocked, keeping the suite fast and independent of external service availability, consistent with how Gemini API calls are mocked in this project's `sql-agent` predecessor. Currently 9 tests passing (`test_dataset.py`).

---

## Security

- `.env` holds the real `GEMINI_API_KEY`; it is gitignored from project creation, before any commit, so the key never enters git history. (Verified this session via `git log --all --full-history -- .env` returning empty — confirmed `.env` has never been committed at any point.)
- `.env.example` is committed as a placeholder template.
- Dependency versions are exact-pinned and freshness-bounded (see `pyproject.toml`).

---

## Development Notes

This project was developed with AI assistance:
- **Development tool:** Claude Sonnet 5 (Anthropic), used as a pair-programming and design-review collaborator — explaining rationale, reviewing architecture decisions, and catching data-quality issues (e.g. identifying and correcting two non-predictive identifier columns and a target-label encoding ambiguity in the Climate Crashes dataset, and the per-dataset `pos_label` convention risk) before they reached the codebase.
- **Runtime AI component:** `gemini-3.1-flash-lite` (Google), the model actually orchestrating the agent's tool-calling loop at runtime — this is the AI system the project *is*, as distinct from the AI assistance used to *build* it.

All architectural decisions (including the Category A/B tool split, the `DatasetSpec` registry design, and the human-in-the-loop extension point), dataset choices, and correctness checks (e.g. cross-referencing dataset documentation against observed data, verifying `.env` was never committed) were reviewed and made by the project author. AI assistance focused on explaining concepts, generating boilerplate, and surfacing issues for human review — not on unsupervised code generation.

---

## Roadmap context

Project 5 of a 10-step learning roadmap: Linux → Git → Professional Python → Pandas → SQL → **Scikit-learn (this project)** → AI-assisted workflows → PyTorch → Transformers → Geometric Deep Learning. Builds directly on the agentic-loop pattern established in `sql-agent` (Project 3). Grokking and Geometric Deep Learning are deliberately deferred to later projects; this project's explicit focus is agentic/software engineering skill, not ML theory.

### Next up: `trainer.py`

Implements the real bodies of `train_model` and `evaluate_model`, currently contract stubs in `tools.py`. Must resolve `pos_label` per-dataset via `get_pos_label()` — never a blanket assumption across datasets.
