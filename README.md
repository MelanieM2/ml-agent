# ml-agent

_Last updated: 2026-07-13_

Agentic ML experimentation assistant — an LLM (Gemini) orchestrates dataset inspection, model proposal, training, and evaluation over scikit-learn via native function-calling, iterating toward a target metric in a supervised loop.

**Status: in progress.** This is Project 5 in a 10-step self-directed learning roadmap toward Data Science / ML Engineering / Agentic AI.

- `dataset.py` — ✅ done and tested.
- `tools.py` — ✅ done (5 tool schemas + inclusive-bounds convention documented).
- `trainer.py` — ✅ done. Real sklearn fit/predict logic implemented: `train_model` fits the correct estimator per `model_type` via a registry lookup against `tools.py`'s schema; `evaluate_model` computes accuracy, precision, recall, f1, and a correctly-oriented confusion matrix using `pos_label`.
- `agent.py` — owns the train/test split for a run: loads the dataset, performs a stratified `train_test_split`, runs a statistical validation gate (`validate_split`) before any model touches the data, and binds the resulting train/test arrays into `Trainer` via `functools.partial`. **The actual multi-step orchestration loop (calling Gemini, dispatching, feeding results back) is still not wired into `agent.py`** — that logic exists and works in `gemini_client.py`, but `agent.py` doesn't yet call it directly; see "Development Notes" below.
- `gemini_client.py` — ✅ **done and verified.** Implements the full agent loop: sends dataset context + tool schemas to Gemini, dispatches whichever tool call comes back, feeds results back, repeats until convergence or a max-iterations guard. Verified via multiple real end-to-end runs across both datasets (see "Data Science Notes" below).
- Also not yet started: `test_trainer.py`, `test_tools.py`, `AGENTS.md`.

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
     │ → model reference/id     │   fields verbatim into            (deferred — see
     └────────────┬────────────┘   train_model's arguments           Development Notes)
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
              (max-iterations guard checked each pass)
```

**Designed extension point:** a human-in-the-loop confirmation step (e.g. "approve this proposal before training?") slots in cleanly at the Category B → Category A handoff — between `record_model_proposal`'s return and `train_model`'s call — without requiring changes to either category's internals. **Not yet implemented; a `# TODO` marker sits at this exact point in `gemini_client.py`, deliberately deferred rather than built prematurely.**

The two Category B tools are worth a brief word each, since their names alone don't say much. `record_model_proposal` is how Gemini puts a candidate on the table — a model type, its hyperparameters, and the reasoning behind choosing them — without anything being trained yet. `record_convergence_decision` is the checkpoint after a model has actually been evaluated: Gemini looks at the metrics it just got back and states, explicitly, whether to stop here or try again, along with why.

All five tools above exist as ordinary Python functions in `tools.py`, but Gemini never sees Python — it needs each tool declared as a JSON schema (name, description, parameter types) before it can call any of them. That declaration lives in `TOOL_SCHEMAS`, a list at the bottom of `tools.py` that `gemini_client.py` hands to the Gemini API. It's kept as its own structure rather than auto-generated from the functions' docstrings above it, on purpose: **the schema text is what *Gemini* reads to decide when and how to call a tool, while the docstring is what a *human* reads to understand the code — the two audiences don't always need the same amount of detail, so letting the wording diverge slightly where it helps is a small deliberate cost, not an oversight**. The trade-off is that a signature change to a function requires a matching by-hand update to its schema entry — easy to forget. A planned mitigation (not yet implemented) is a `test_tools.py` check using `inspect.signature()` to compare each function's real parameters against its `TOOL_SCHEMAS` entry automatically.

---

## `gemini_client.py` — the agent loop

`run_agent_loop(dispatch_table, initial_context, *, model=DEFAULT_MODEL, max_iterations=MAX_ITERATIONS)` is the thin, deliberately "dumb" messenger between Gemini and the real scikit-learn machinery. It never trains a model or judges a metric itself — it relays Gemini's decisions to `dispatch_table`, and relays the real results back to Gemini, until Gemini calls `record_convergence_decision` with `continue_iterating=False`, or `max_iterations` is reached.

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
                     │    │  train_model's call — not    │
                     │    │  implemented yet, deferred]  │
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
- **Known limitation, not yet handled:** the loop assumes exactly one function call per Gemini turn. Gemini's function-calling mode can in principle return several parallel calls in a single response; this case isn't currently handled.

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

Every `chat.send_message()` call — the first `initial_context` or the tenth `function_response` — replays the *entire* accumulated history to `generate_content` each time. This is why `gemini_client.py` never has to manually remember prior iterations, but it's also why request size grows with iteration count. See `TECHNICAL_NOTES.md` for the full cost-scaling discussion — not an issue at the current `MAX_ITERATIONS = 10`, but worth understanding before that ceiling is ever raised significantly.

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

`train_model(self, model_type, hyperparameters, *, X_train, y_train)` validates hyperparameters against `list_available_models()`'s schema via `validate_hyperparameters` before instantiating anything, fits the correct sklearn estimator per `model_type`, stores it under a generated `model_ref`, and returns that ref. `evaluate_model(self, model_ref, *, pos_label, X_test, y_test)` looks up the stored model, predicts against the test set, and computes accuracy, precision, recall, f1, and a confusion matrix computed with explicit label ordering (`labels=[1 - pos_label, pos_label]`) rather than sklearn's default ascending sort — critical given the two datasets' opposite `pos_label` semantics (see "Datasets" below).

---

## Validating Gemini's arguments before training

### The problem this solves

Without a check, a hallucinated `model_type`, or a `hyperparameters` value outside its documented range/choices, would fail deep inside whatever code instantiates the sklearn estimator — a confusing, hard-to-trace failure far from its actual cause. `list_available_models()` already *is* the schema — the same dict shown to Gemini describing what's available is also the ground truth to check its choices against.

### Design: a standalone function, not a `Trainer` method

`validate_hyperparameters(model_type, hyperparameters, schema)` is a pure function of its three arguments — no instance state needed, so it lives standalone in `trainer.py` rather than as a `Trainer` method. Checks `model_type` validity, then each hyperparameter's presence and range/choice validity, raising a specific, named `ValueError` on any mismatch.

### A schema ambiguity, caught and fixed

The schema's numeric `"range": [min, max]` field is project-specific, not a real JSON Schema keyword — nothing stated whether boundary values themselves were meant to be valid. Resolved by adding an explicit inclusive-bounds convention statement to `list_available_models()`'s docstring; `validate_hyperparameters` uses inclusive bounds (`low <= value <= high`) matching it.

---

## `agent.py` — wiring the dispatch table

### The problem

Everything dataset-specific (which `pos_label` applies) and run-specific (the train/test split, which fitted models exist so far) needs to be resolved exactly once, in one place, so nothing downstream has to rediscover or re-pass it. `build_dispatch_table` is that one place.

```python
def build_dispatch_table(
    dataset_name: str, random_state: int = 42
) -> dict[str, Callable[..., Any]]:
    """Builds one run's complete tool dispatch table for gemini_client.py."""
    X, y = load_dataset(dataset_name)
    pos_label = get_pos_label(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    validate_split(y_train, y_test, pos_label, min_count=5)

    trainer = Trainer()
    bound_train = partial(trainer.train_model, X_train=X_train, y_train=y_train)
    bound_evaluate = partial(
        trainer.evaluate_model, X_test=X_test, y_test=y_test, pos_label=pos_label
    )

    return {
        "list_available_models": list_available_models,
        "train_model": bound_train,
        "evaluate_model": bound_evaluate,
        "record_model_proposal": record_model_proposal,
        "record_convergence_decision": record_convergence_decision,
    }
```

Note the asymmetry visible directly in the dict: some entries are bound/wrapped (`bound_train`, `bound_evaluate`), some are bare top-level functions (`list_available_models`, both Category B tools). That asymmetry *is* the Category A/B split, made concrete in code rather than only described in prose.

### The `pos_label` / split-array binding mechanism: `functools.partial`

Binding `pos_label`, `X_train`, `y_train`, etc. into `Trainer`'s methods is an instance of **partial function application** — pre-loading a function with specific arguments so it can be saved and called later without those arguments needing to be supplied again.

```
CANDIDATE 1 — functools.partial (chosen)            CANDIDATE 2 — closure (not used, but a
                                                      real alternative worth knowing)
 bound_evaluate = partial(                           def make_evaluate_dispatch(trainer, pos_label):
     trainer.evaluate_model,                             def _dispatch(model_ref):
     pos_label=pos_label,                                     return trainer.evaluate_model(
     X_test=X_test, y_test=y_test                                 model_ref, pos_label=pos_label)
 )                                                        return _dispatch
```

**Why `partial` here specifically:** the freezing needed is only ever a handful of arguments, with no additional behavior wrapped around the call — no logging, no transformation, no conditionals. That is precisely the case `functools.partial` is built for. (If a future requirement ever needs extra logic around the call — e.g. catching a specific exception type before it reaches Gemini's dispatch — that would be the signal to switch to the closure form instead; not a current need.)

### What's genuinely still missing: the orchestration loop itself

`build_dispatch_table` builds the dispatch table correctly and does **not** return `(X, y)` or an `initial_context` — meaning nothing in the committed codebase yet calls `run_agent_loop` with real, non-hand-built inputs. This is `agent.py`'s own documented next step, not an oversight discovered this session — see "Roadmap context" below.

---

## Project structure

```
ml-agent/
├── pyproject.toml
├── .env.example
├── .gitignore
├── AGENTS.md              # agent rules, guardrails, tool list, convergence criteria — not yet written
├── main.py                # CLI entry point / interactive loop
├── ml_agent/
│   ├── dataset.py         # dataset-agnostic loading + inspection ✅ done
│   ├── tools.py            # Gemini function-calling tool schemas ✅ done
│   ├── gemini_client.py    # ✅ done — Gemini client + function-call dispatch loop, verified
│   ├── trainer.py          # ✅ done — storage, validation, and real sklearn fit/predict/evaluate
│   └── agent.py             # ✅ dispatch-table + split wiring done; orchestration loop still pending
└── tests/
    ├── test_dataset.py    # ✅ done, 9 passing tests
    ├── test_tools.py       # not started
    └── test_trainer.py     # not started
```

### Why `agent.py` and `gemini_client.py` are separate
`gemini_client.py` only knows how to talk to Gemini and dispatch function calls. `agent.py` owns the actual multi-step state — what's been tried, results so far, whether to keep iterating. This separation allows convergence logic to be unit-tested without a live API call.

---

## Datasets

- **Primary: Climate Model Simulation Crashes** (OpenML id 1467). Binary classification — predict whether a given combination of 18 physical parameters (from the POP2 ocean model component of CCSM4) causes a numerical simulation crash. 540 rows, genuinely imbalanced (~91.5% success / 8.5% failure). `pos_label=1` (failure, the rare class).
- **Fallback: Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`). Used for fast, network-free debugging. Its positive-class convention (`1 = benign`, the majority class) is the *opposite* sense from Climate Crashes — `pos_label` is established per-dataset, never assumed consistent.

Datasets are loaded through a small registry (`DATASET_LOADERS`) in `dataset.py`, keyed by short name (`"climate"`, `"breast_cancer"`). Each registry entry is a `DatasetSpec` — a frozen dataclass pairing the loader function with `pos_label` and a human-readable `description`.

### What `pos_label` actually is

In a binary classification task, the target column only ever holds two values — but *which one counts as the "positive" outcome* isn't a mathematical given, it's a convention someone has to fix per dataset. It matters because precision, recall, and F1 aren't symmetric in the two classes. That's why it's stored as a registry fact rather than inferred at evaluation time.

A dedicated accessor, `get_pos_label(name)`, mirrors `load_dataset(name)`'s pattern — one accessor function per registry fact, never reach into `DATASET_LOADERS` directly outside `dataset.py`.

---

## Available models (via `list_available_models`)

Deliberately kept to three model types rather than an exhaustive sklearn zoo — this project's explicit growth area is agentic/tool-calling engineering, not exhaustive model search. All three expose `class_weight: ["balanced", null]`, the single most relevant lever for the recall-optimization framing on Climate Crashes.

| Model | Key hyperparameters |
|---|---|
| Logistic Regression | `C` (0.001–100, inclusive), `class_weight` |
| Random Forest | `n_estimators` (10–500, inclusive), `max_depth` (int or null), `class_weight` |
| SVM | `C`, `kernel` (linear/rbf/poly), `class_weight` |

---

## Setup

```bash
git clone git@github.com:MelanieM2/ml-agent.git
cd ml-agent
uv sync
cp .env.example .env   # then fill in your real GEMINI_API_KEY
```

Dependencies are pinned exactly (`add-bounds = "exact"` in `[tool.uv]`) and restricted to packages no newer than 7 days old at install time (`exclude-newer = "7 days ago"`), as a routine security precaution applied to every project in this series. See `SECURITY.md` for the full policy.

## Running

```bash
uv run python main.py
```

(Interactive loop entry point — details to be filled in as `agent.py`'s orchestration wiring is completed.)

## Testing

```bash
uv run pytest -v
```

Network-dependent behavior (the Climate Crashes OpenML fetch) is not exercised directly in tests — it's mocked. Currently 9 tests passing (`test_dataset.py`).

---

## Security

See [`SECURITY.md`](./SECURITY.md) for the full policy (dependency pinning, vulnerability scanning, lockfile verification, API key handling, and the LLM-generated-tool-call guard specific to this project).

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

### Data Science Notes — cross-dataset smoke test (2026-07-10)

Two datasets are used deliberately for their **opposite class semantics**: Climate Crashes' `pos_label=1` marks a rare (8.5%), *undesirable* outcome (simulation failure), while Breast Cancer's `pos_label=1` marks a majority (62.7%), *desirable* outcome (benign). This asymmetry is a real risk for any code that computes a confusion matrix or class-conditional metrics: an implementation that silently assumes "positive = rare" or "positive = bad" will produce a subtly wrong result on one dataset while looking correct on the other.

| | Climate Crashes | Breast Cancer |
|---|---|---|
| `pos_label=1` means | simulation *failure* — rare (8.5%), undesirable | *benign* — majority (62.7%), desirable |
| Test set size | 108 | 114 |
| Accuracy | 0.917 | 0.947 |
| Precision (`pos_label`) | 0.500 | 0.958 |
| Recall (`pos_label`) | 0.333 | 0.958 |
| F1 (`pos_label`) | 0.400 | 0.958 |
| Confusion matrix `[[TN,FP],[FN,TP]]` | `[[96, 3], [6, 3]]` | `[[39, 3], [3, 69]]` |

**What this confirms about the implementation specifically:** the confusion matrix is computed with an explicit label ordering (`labels=[1 - pos_label, pos_label]`) rather than relying on `sklearn`'s default ascending sort, which would silently assume class `0` is always "negative" — true for Climate Crashes but false for Breast Cancer. Running both datasets through the same code path confirmed this ordering logic holds in both directions.

A much more extensive live-agent version of this cross-dataset comparison — six real end-to-end Gemini-orchestrated runs, convergence-rate differences, and a concrete finding about how an unstated optimization target affects model selection differently per dataset — is written up in full in [`DATA_SCIENCE_ANALYSIS.md`](./DATA_SCIENCE_ANALYSIS.md).

### Statistical rationale — `validate_split`'s threshold

`build_dispatch_table` runs a validation gate (`validate_split`) between splitting the data and constructing any model: both the train and test splits must contain at least 5 examples of the dataset's `pos_label` class, or the pipeline halts with a clear error before any training begins.

The number 5 was chosen after directly computing the actual outcome of a stratified 80/20 split on Climate Crashes' documented class distribution (540 rows, 46 positive): the test split produces exactly **9** rare-class examples. The threshold is grounded in how much a single misclassification can swing a metric computed on so few examples: at 9 examples, one misclassification moves recall by about 11 percentage points (1/9); a threshold set at or below 4 would allow swings of 25% or more (1/4). 5 provides real headroom above that volatility floor while still catching a genuinely broken split.

---

## Roadmap context

Project 5 of a 10-step learning roadmap: Linux → Git → Professional Python → Pandas → SQL → **Scikit-learn (this project)** → AI-assisted workflows → PyTorch → Transformers → Geometric Deep Learning. Builds directly on the agentic-loop pattern established in `sql-agent` (Project 3).

### Progression of "next up"

1. **After `dataset.py`:** implement `trainer.py`'s real bodies, resolving `pos_label` per-dataset via `get_pos_label()`.
2. **After `Trainer` storage/validation scaffolding:** fill in real fit/evaluate logic. Then `test_trainer.py`, then `gemini_client.py` and the real orchestration loop in `agent.py`.
3. **As of 2026-07-10:** sklearn fit/evaluate logic and the `agent.py` train/test split + `validate_split` gate in place. `gemini_client.py` the explicit next milestone.
4. **As of 2026-07-13 (current):** `gemini_client.py` is complete and verified via multiple real end-to-end runs on both datasets. Remaining before the project's next phase: wire `agent.py`'s actual orchestration loop (currently only exercised via a hand-built standalone script, not real `agent.py` code), then `test_tools.py` (the `inspect.signature()` drift check), then `test_trainer.py`, then `AGENTS.md`.