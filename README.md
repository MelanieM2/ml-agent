# ml-agent

_Last updated: 2026-07-10_

Agentic ML experimentation assistant — an LLM (Gemini) orchestrates dataset inspection, model proposal, training, and evaluation over scikit-learn via native function-calling, iterating toward a target metric in a supervised loop.

**Status: in progress.** This is Project 5 in a 10-step self-directed learning roadmap toward Data Science / ML Engineering / Agentic AI.

- `dataset.py` — ✅ done and tested.
- `tools.py` — ✅ done (5 tool schemas + inclusive-bounds convention documented). Initially shipped with `train_model`/`evaluate_model` as contract stubs pending `trainer.py`.
- `trainer.py` — real sklearn fit/predict logic is now implemented: `train_model` fits the correct estimator per `model_type` via a registry lookup against `tools.py`'s schema; `evaluate_model` computes accuracy, precision, recall, f1, and a correctly-oriented confusion matrix using `pos_label`. (Storage + validation scaffolding was completed first, with the real fit/evaluate logic filled in during the following session.)
- `agent.py` — now owns the train/test split for a run: loads the dataset, performs a stratified `train_test_split` (with a caller-adjustable `random_state`, default `42`), runs a statistical validation gate (`validate_split`) before any model touches the data, and binds the resulting train/test arrays into `Trainer` via `functools.partial` — the same binding mechanism already used for `pos_label`. (Before this, `agent.py` only wired the complete 5-tool dispatch table together, without owning the split.)
- `gemini_client.py` — not yet started; this is now the **explicit next milestone**.
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

The two Category B tools are worth a brief word each, since their names alone don't say much. `record_model_proposal` is how Gemini puts a candidate on the table — a model type, its hyperparameters, and the reasoning behind choosing them — without anything being trained yet; it's a proposal in the literal sense, waiting to be acted on. `record_convergence_decision` is the checkpoint after a model has actually been evaluated: Gemini looks at the metrics it just got back and states, explicitly, whether to stop here or try again, along with why. Neither tool does anything on its own — they exist so that a step which would otherwise be free-form prose ("I think we should try a random forest next...") becomes a structured, reviewable decision instead. (Both descriptions were initially left intentionally brief, pending `trainer.py`'s completion so these tools would have real evaluation results to react to rather than a stub.)

All five tools above exist as ordinary Python functions in `tools.py`, but Gemini never sees Python — it needs each tool declared as a JSON schema (name, description, parameter types) before it can call any of them. That declaration lives in `TOOL_SCHEMAS`, a list at the bottom of `tools.py` that `gemini_client.py` hands to the Gemini API. It's kept as its own structure rather than auto-generated from the functions' docstrings above it, on purpose: **the schema text is what *Gemini* reads to decide when and how to call a tool, while the docstring is what a *human* reads to understand the code — the two audiences don't always need the same amount of detail, so letting the wording diverge slightly where it helps is a small deliberate cost, not an oversight**. The trade-off is that a signature change to a function requires a matching by-hand update to its schema entry — easy to forget, worth double-checking when either changes. Concretely: nothing enforces that the two stay in sync, so a forgotten update either leaves a new parameter invisible to Gemini (it simply never learns the option exists — no error, just a silently unreachable feature) or, worse, causes a `TypeError` at dispatch time if a parameter gets renamed — and that error only surfaces whenever Gemini next happens to call that specific tool, which may be well after the change was made. A planned mitigation (not yet implemented) is a `test_tools.py` check using `inspect.signature()` to compare each function's real parameters against its `TOOL_SCHEMAS` entry automatically, rather than relying on remembering to check by hand.

---

## `Trainer` — model storage and encapsulation

### The problem

`train_model` needs to hand Gemini back an opaque `model_ref` — a string id — never the fitted scikit-learn object itself (Gemini only ever sees ids and numbers). But *something* has to hold onto the actual fitted model between the `train_model` call and a later `evaluate_model` call that looks it up by that same id.

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

**Chosen: Option B**, for two reasons. First, encapsulation: a module-level dict is reachable and mutable by *any* code that imports the module — nothing structurally prevents a stray line elsewhere in the codebase from reading or corrupting it. A class instance's `self._models` is only reachable through that specific instance's own methods — there's exactly one sanctioned path to the state, enforced by Python's object model rather than by convention alone. Second, testing: each test can construct a fresh `Trainer()` with a guaranteed-empty store and no manual reset step — Option A would need explicit teardown logic to avoid one test's fitted models leaking into the next.

### `Trainer` class

```python
class Trainer:
    """Owns the model_ref -> fitted model mapping for one agent run.

    This is the encapsulated 'filing cabinet': one Trainer instance holds
    its own private dict of trained models. Nothing outside this class can
    read or mutate that dict directly — the only way in or out is through
    train_model() and evaluate_model() on this specific instance.
    """

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] = {}

    def train_model(
        self, model_type: str, hyperparameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Fits model_type with hyperparameters, stores it, returns a ref."""
        schema = list_available_models()["models"]
        validate_hyperparameters(model_type, hyperparameters, schema)
        # ^ raises ValueError immediately on any mismatch — nothing below
        #   this line runs on invalid input.

        model_ref = uuid.uuid4().hex
        label = f"{model_type}_{model_ref[:6]}"  # human-readable, debug-only

        self._models[model_ref] = {
            "model": None,  # placeholder for the real fitted estimator
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "label": label,
        }
        return {"model_ref": model_ref}

    def evaluate_model(
        self, model_ref: str, *, pos_label: int
    ) -> dict[str, Any]:
        """Looks up model_ref, computes metrics against pos_label."""
        if model_ref not in self._models:
            raise ValueError(f"Unknown model_ref: {model_ref!r}")
        entry = self._models[model_ref]
        return {"model_ref": model_ref, "model_type": entry["model_type"],
                 "pos_label": pos_label}
```

_(Shown above in its original "storage + validation scaffolding" form — `model: None` and the absence of real prediction logic. As of the 2026-07-10 session this scaffolding was completed: `train_model` now instantiates and fits the correct sklearn estimator per `model_type` via a registry lookup against `tools.py`'s schema, and `evaluate_model` now actually predicts and computes accuracy, precision, recall, f1, and a correctly-oriented confusion matrix using `pos_label`, rather than returning the stub fields shown above.)_

---

## Validating Gemini's arguments before training

### The problem this solves

Without a check, a hallucinated `model_type`, or a `hyperparameters` value outside its documented range/choices, would fail deep inside whatever code instantiates the sklearn estimator — a confusing, hard-to-trace failure far from its actual cause (Gemini choosing something it wasn't offered). `list_available_models()` already *is* the schema — the same dict shown to Gemini describing what's available is also the ground truth to check its choices against.

**What a failure should look like:** a clear, specifically-named `ValueError` — naming exactly which key, which value, or which `model_type` was invalid — never a bare crash surfacing from deep inside scikit-learn's own constructor. Errors that are actionable at the point they're raised are worth the small extra code; an error five layers deep in someone else's library, with no context about which of Gemini's arguments caused it, is not something to leave to chance.

### Design: a standalone function, not a `Trainer` method

`validate_hyperparameters(model_type, hyperparameters, schema)` is a pure function of its three arguments — it needs no instance state, so it lives as a standalone function in `trainer.py` rather than a method on `Trainer`. This keeps it testable in complete isolation (no `Trainer()` construction needed at all) and keeps `Trainer`'s own responsibility narrowly scoped to storage, not validation.

```python
def validate_hyperparameters(
    model_type: str,
    hyperparameters: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Checks Gemini's proposed model_type/hyperparameters against the
    known schema returned by list_available_models()["models"], before
    anything gets instantiated. Raises ValueError with a specific, named
    reason on any mismatch; returns None silently if everything checks out.
    """
    if model_type not in schema:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Valid options: {list(schema.keys())}"
        )

    valid_params = schema[model_type]["hyperparameters"]

    for key, value in hyperparameters.items():
        if key not in valid_params:
            raise ValueError(
                f"Unknown hyperparameter {key!r} for model_type "
                f"{model_type!r}. Valid params: {list(valid_params.keys())}"
            )

        spec = valid_params[key]
        param_type = spec["type"]

        if param_type in ("float", "int"):
            low, high = spec["range"]
            if not (low <= value <= high):
                raise ValueError(
                    f"{key}={value!r} out of range for {model_type!r}. "
                    f"Valid range: [{low}, {high}]"
                )
        elif param_type == "int_or_null":
            if value is not None:
                low, high = spec["range"]
                if not (low <= value <= high):
                    raise ValueError(
                        f"{key}={value!r} out of range for {model_type!r}. "
                        f"Valid range: [{low}, {high}] or null"
                    )
        elif param_type == "choice":
            if value not in spec["options"]:
                raise ValueError(
                    f"{key}={value!r} not a valid choice for {model_type!r}. "
                    f"Valid options: {spec['options']}"
                )
```

### A schema ambiguity, caught and fixed

The schema's numeric `"range": [min, max]` field is project-specific, not a real JSON Schema keyword — so nothing in it stated whether the boundary values themselves (e.g. `C=100.0` exactly) were meant to be valid, or just descriptive endpoints. This kind of silently-undefined convention is exactly the sort of thing that validation code ends up guessing about unless it's made explicit. Resolved by adding a one-line convention statement directly to `list_available_models()`'s docstring:

> *"Convention: all numeric "range" bounds are inclusive on both ends (e.g. C=100.0 is itself a valid value, not just a descriptive ceiling). This mirrors standard JSON Schema's minimum/maximum default behavior, even though "range" itself is a project-specific field, not a real JSON Schema keyword."*

`validate_hyperparameters` uses inclusive bounds (`low <= value <= high`) on both ends, matching this now-explicit convention rather than an unstated assumption.

---

## `agent.py` — wiring the dispatch table

### The problem

Everything dataset-specific (which `pos_label` applies) and run-specific (which fitted models exist so far) needs to be resolved exactly once, in one place, so nothing downstream has to rediscover or re-pass it. `build_dispatch_table` is that one place.

```
  build_dispatch_table(dataset_name="climate")
  │
  ├─► get_pos_label("climate")  ──────────────►  pos_label = 1
  │
  ├─► Trainer()  ──────────────────────────────►  trainer
  │        (fresh, empty filing cabinet)
  │
  ├─► partial(trainer.evaluate_model,
  │           pos_label=pos_label)  ───────────► bound_evaluate
  │        (freezes pos_label=1 into the call,     (looks like it only
  │         permanently, for this run)              takes model_ref)
  │
  ▼
  returns one dict — the dispatch table:

  ┌─────────────────────────────────────────────────────────┐
  │ dispatch_table = {                                        │
  │   "list_available_models" ─────► list_available_models     │  (stateless)
  │   "train_model"           ─────► trainer.train_model       │  (bound method)
  │   "evaluate_model"        ─────► bound_evaluate             │  (pos_label baked
  │                                                              │   in, hidden from
  │                                                              │   Gemini entirely)
  │   "record_model_proposal" ─────► record_model_proposal      │  (stateless,
  │   "record_convergence_decision" ► record_convergence_decision│   Category B)
  │ }                                                            │
  └─────────────────────────────────────────────────────────┘
                          │
                          ▼
        handed to gemini_client.py (not yet written) —
        it just does dispatch_table[tool_name](**args)
        and never needs to know dataset_name, pos_label,
        or the Trainer instance exist at all.
```

```python
def build_dispatch_table(dataset_name: str) -> dict[str, Callable[..., Any]]:
    """Builds one run's complete tool dispatch table for gemini_client.py.

    Resolves everything dataset- or run-specific exactly once, here:
      - one Trainer instance (private model_ref -> fitted model store)
      - this dataset's pos_label, pre-filled into evaluate_model via
        functools.partial, so Gemini's dispatch never sees it at all

    Category A (list_available_models, train_model, evaluate_model) are
    real executions; Category B (record_model_proposal,
    record_convergence_decision) are stateless structured-decision
    capture, wired in as-is.
    """
    trainer = Trainer()
    pos_label = get_pos_label(dataset_name)
    bound_evaluate = partial(trainer.evaluate_model, pos_label=pos_label)

    return {
        "list_available_models": list_available_models,
        "train_model": trainer.train_model,
        "evaluate_model": bound_evaluate,
        "record_model_proposal": record_model_proposal,
        "record_convergence_decision": record_convergence_decision,
    }
```

Note the asymmetry visible directly in the dict: some entries are bound/wrapped (`trainer.train_model`, `bound_evaluate`), some are bare top-level functions (`list_available_models`, both Category B tools). That asymmetry *is* the Category A/B split, made concrete in code rather than only described in prose.

### Update (2026-07-10 session): `agent.py` now also owns the train/test split

As of this session, `agent.py` does more than wire the dispatch table — it now also:
- Loads the dataset.
- Performs a stratified `train_test_split`, with a caller-adjustable `random_state` (default `42`).
- Runs a statistical validation gate, `validate_split`, before any model touches the data (see "Statistical rationale — `validate_split`'s threshold" below).
- Binds the resulting train/test arrays into `Trainer` via `functools.partial` — the same binding mechanism already used for `pos_label`.

`build_dispatch_table` now runs this `validate_split` gate between splitting the data and constructing any model: both the train and test splits must contain at least 5 examples of the dataset's `pos_label` class, or the pipeline halts with a clear error before any training begins.

### The `pos_label` binding mechanism: `functools.partial`

Binding `pos_label` into `evaluate_model` is an instance of the general concept of **partial function application** — pre-loading a function with specific arguments so it can be saved and called later without those arguments needing to be supplied again. Two candidate mechanisms exist in Python for this:

```
CANDIDATE 1 — functools.partial (chosen)            CANDIDATE 2 — closure (not used, but a
                                                      real alternative worth knowing)
 pos_label = get_pos_label(dataset)                  def make_evaluate_dispatch(trainer, pos_label):
 trainer = Trainer()                                     def _dispatch(model_ref):
 bound_eval = partial(                                       return trainer.evaluate_model(
     trainer.evaluate_model,                                     model_ref, pos_label=pos_label)
     pos_label=pos_label                                    return _dispatch
 )
                                                       bound_eval = make_evaluate_dispatch(
                                                           trainer, pos_label)
```

`functools.partial` takes an existing callable and produces a new one with one argument permanently pre-filled — calling `bound_eval(model_ref="x")` is exactly equivalent to `trainer.evaluate_model(model_ref="x", pos_label=<baked-in value>)`. A closure achieves the identical effect by hand, via a nested function that "remembers" its enclosing scope's variables after the outer function returns.

**Why `partial` here specifically:** the freezing needed is only ever one argument, with no additional behavior wrapped around the call — no logging, no transformation, no conditionals. That is precisely the case `functools.partial` is built for. (If a future requirement ever needs extra logic around the call — e.g. catching a specific exception type before it reaches Gemini's dispatch — that would be the signal to switch to the closure form instead; not a current need.)

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
│   │                        #   (5 tool schemas + inclusive-bounds convention documented)
│   ├── gemini_client.py    # Gemini client wrapper + function-call dispatch — not started
│   │                        #   (explicit next milestone as of 2026-07-10)
│   ├── trainer.py          # ✅ storage + validation scaffolding done, PLUS real
│   │                        #   sklearn fit/predict/evaluate logic now implemented
│   └── agent.py             # ✅ dispatch-table wiring done; now also owns the
│                              #   train/test split, validate_split gate, and
│                              #   functools.partial binding of train/test arrays
│                              #   into Trainer; orchestration loop still pending
└── tests/
    ├── test_dataset.py    # ✅ done, 9 passing tests
    ├── test_tools.py       # not started (Category B only, until trainer.py existed)
    └── test_trainer.py     # not started
```

### Why `agent.py` and `gemini_client.py` are separate
`gemini_client.py` only knows how to talk to Gemini and dispatch function calls. `agent.py` owns the actual multi-step state — what's been tried, results so far, whether to keep iterating. This separation allows convergence logic to be unit-tested without a live API call.

---

## Datasets

- **Primary: Climate Model Simulation Crashes** (OpenML id 1467). Binary classification — predict whether a given combination of 18 physical parameters (from the POP2 ocean model component of CCSM4) causes a numerical simulation crash. 540 rows, genuinely imbalanced (~91.5% success / 8.5% failure), which gives a "optimize for recall" objective real substance. Loader explicitly drops two non-predictive identifier columns present in the raw source (documented by the dataset's own maintainers as unfit for prediction) and remaps OpenML's string target labels to standard `0`/`1` semantics, with `1` established as the failure/positive class via cross-checking against the dataset's documented failure count. `pos_label=1` (failure, the rare class).
- **Fallback: Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`). Used for fast, network-free debugging of the pipeline in isolation. Note: its positive-class convention (`1 = benign`, the majority class) is the *opposite* sense from Climate Crashes (`1 = failure`, the rare class) — `pos_label` is established per-dataset, never assumed consistent.

Datasets are loaded through a small registry (`DATASET_LOADERS`) in `dataset.py`, keyed by short name (`"climate"`, `"breast_cancer"`). Each registry entry is a `DatasetSpec` — a frozen dataclass pairing the loader function with two dataset-level facts that must never be guessed or re-derived at evaluation time: `pos_label` (the positive class for precision/recall) and a human-readable `description`. Adding a new dataset later means writing only one loader function and one `DatasetSpec` registry line.

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

- `.env` holds the real `GEMINI_API_KEY`; it is gitignored from project creation, before any commit, so the key never enters git history. (Verified via `git log --all --full-history -- .env` returning empty — confirmed `.env` has never been committed at any point.)
- `.env.example` is committed as a placeholder template.
- Dependency versions are exact-pinned and freshness-bounded (see `pyproject.toml`).

---

## Development Notes

This project was developed with AI assistance:
- **Development tool:** Claude Sonnet 5 (Anthropic), used as a pair-programming and design-review collaborator — explaining rationale, reviewing architecture decisions, and catching data-quality issues (e.g. identifying and correcting two non-predictive identifier columns and a target-label encoding ambiguity in the Climate Crashes dataset, and the per-dataset `pos_label` convention risk) and schema-ambiguity issues before they reached the codebase.
- **Runtime AI component:** `gemini-3.1-flash-lite` (Google), the model actually orchestrating the agent's tool-calling loop at runtime — this is the AI system the project *is*, as distinct from the AI assistance used to *build* it. These are two distinct roles: one is a tool used while building the project, the other is a component *of* the project itself.

All architectural decisions — including the Category A/B tool split, the `DatasetSpec` registry design, the `Trainer` encapsulation design, the `functools.partial` pos_label binding, the inclusive-bounds schema convention, and the human-in-the-loop extension point — along with dataset choices and correctness checks (e.g. cross-referencing dataset documentation against observed data, verifying `.env` was never committed) were reviewed and made by the project author. AI assistance focused on explaining concepts, generating boilerplate, and surfacing issues for human review — not on unsupervised code generation.

### Data Science Notes — cross-dataset smoke test (2026-07-10)

Two datasets are used deliberately for their **opposite class semantics**: Climate Crashes' `pos_label=1` marks a rare (8.5%), *undesirable* outcome (simulation failure), while Breast Cancer's `pos_label=1` marks a majority (62.7%), *desirable* outcome (benign). This asymmetry is a real risk for any code that computes a confusion matrix or class-conditional metrics: an implementation that silently assumes "positive = rare" or "positive = bad" will produce a subtly wrong result on one dataset while looking correct on the other. A manual end-to-end run on both datasets was used specifically to catch this.

| | Climate Crashes | Breast Cancer |
|---|---|---|
| `pos_label=1` means | simulation *failure* — rare (8.5%), undesirable | *benign* — majority (62.7%), desirable |
| Test set size | 108 | 114 |
| Accuracy | 0.917 | 0.947 |
| Precision (`pos_label`) | 0.500 | 0.958 |
| Recall (`pos_label`) | 0.333 | 0.958 |
| F1 (`pos_label`) | 0.400 | 0.958 |
| Confusion matrix `[[TN,FP],[FN,TP]]` | `[[96, 3], [6, 3]]` | `[[39, 3], [3, 69]]` |

**Why the gap between the two runs is expected, not a bug.** Climate Crashes' weak recall (0.333 — the model caught only 3 of the 9 actual failures present in the test set) is a direct consequence of the severe class imbalance the dataset is built on: with so few positive examples to learn from, a Random Forest trained with default-ish hyperparameters struggles to generalize the failure pattern, and each of the 9 rare cases in the test set carries real statistical weight — a single flipped prediction moves recall by roughly 11 percentage points (1/9). Breast Cancer's much stronger, tied precision/recall (0.958, because false positives and false negatives happened to tie at 3 each) reflects both a more balanced class ratio and a comparatively easier decision boundary for this classic benchmark dataset.

**What this confirms about the implementation specifically:** the confusion matrix is computed with an explicit label ordering (`labels=[1 - pos_label, pos_label]`) rather than relying on `sklearn`'s default ascending sort, which would silently assume class `0` is always "negative" — true for Climate Crashes but false for Breast Cancer. Running both datasets through the same code path and manually verifying the matrix's internal consistency (cell sums matching known split sizes, and all four derived metrics recomputing correctly from the raw matrix values) confirmed this ordering logic holds in both directions, which was the specific risk this test was designed to catch.

**Why this matters as a methodology note, not just a smoke test result:** the two runs are a small but genuine illustration of a standard imbalanced-classification lesson — accuracy alone (0.917 vs. 0.947) looks deceptively similar for both runs, and would mask how differently the two models are actually performing on the class that matters. Only examining recall/precision on `pos_label` specifically reveals that the Climate Crashes model, as currently configured, is not yet reliable for the task it's meant to perform — exactly the kind of signal `record_convergence_decision` is meant to act on once the orchestration loop exists.

### Statistical rationale — `validate_split`'s threshold

`build_dispatch_table` runs a validation gate (`validate_split`) between splitting the data and constructing any model: both the train and test splits must contain at least 5 examples of the dataset's `pos_label` class, or the pipeline halts with a clear error before any training begins.

The number 5 was chosen after directly computing the actual outcome of a stratified 80/20 split on Climate Crashes' documented class distribution (540 rows, 46 positive): the test split produces exactly **9** rare-class examples. The threshold is grounded in how much a single misclassification can swing a metric computed on so few examples: at 9 examples, one misclassification moves recall by about 11 percentage points (1/9); a threshold set at or below 4 would allow swings of 25% or more (1/4), making the resulting recall/precision numbers too volatile to trust. 5 provides real headroom above that volatility floor while still catching a genuinely broken split (e.g. stratification failing silently, or a dataset shrinking well below its documented size). Only the rare (`pos_label`) class is checked, since both datasets' class ratios make the majority class dropping anywhere near this floor a non-issue in practice under stratified sampling.

---

## Roadmap context

Project 5 of a 10-step learning roadmap: Linux → Git → Professional Python → Pandas → SQL → **Scikit-learn (this project)** → AI-assisted workflows → PyTorch → Transformers → Geometric Deep Learning. Builds directly on the agentic-loop pattern established in `sql-agent` (Project 3). Grokking and Geometric Deep Learning are deliberately deferred to later projects; this project's explicit focus is agentic/software engineering skill, not ML theory.

### Progression of "next up"

1. **After `dataset.py`:** implement `trainer.py`'s real bodies for `train_model` and `evaluate_model` (then contract stubs in `tools.py`), resolving `pos_label` per-dataset via `get_pos_label()` — never a blanket assumption across datasets.
2. **After `Trainer` storage/validation scaffolding:** fill in the `TODO` blocks inside `Trainer.train_model` (instantiate the correct sklearn estimator, call `.fit`) and `Trainer.evaluate_model` (predict, compute accuracy/precision/recall/f1/confusion_matrix using `pos_label`). Then `test_trainer.py`, then `gemini_client.py` and the real orchestration loop in `agent.py`.
3. **Current (as of 2026-07-10):** the real sklearn fit/evaluate logic and the `agent.py` train/test split + `validate_split` gate are now in place. `gemini_client.py` is the explicit next milestone, followed by `test_trainer.py`, `test_tools.py`, and `AGENTS.md`.