# ml-agent

_Last updated: 2026-07-09_

Agentic ML experimentation assistant — an LLM (Gemini) orchestrates dataset inspection, model proposal, training, and evaluation over scikit-learn via native function-calling, iterating toward a target metric in a supervised loop.

**Status: in progress.** This is Project 5 in a 10-step self-directed learning roadmap toward Data Science / ML Engineering / Agentic AI. `dataset.py` and `tools.py` are complete. `trainer.py` now has its full storage and validation scaffolding in place (real sklearn fit/evaluate logic still to come). `agent.py` now wires the complete 5-tool dispatch table together. `gemini_client.py` and the real orchestration loop are not yet started.

---

## Concept

_(unchanged from prior version — see the pipeline diagram and core architectural principle sections below, both still accurate)_

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

### Core architectural principle

Deterministic code handles fact-gathering; the LLM handles reasoning under uncertainty. `dataset.py`'s `inspect_dataset()` computes shape, class balance, missing values, and feature statistics — but makes no judgment calls. Deciding what those facts *imply* is left to Gemini, reasoning over the structured facts it's handed.

This same principle extends into `tools.py` and now `trainer.py`: **which class counts as "positive"** for a given dataset is a fact, never something Gemini supplies or guesses. And, as of this session, **whether a proposed model/hyperparameter combination is even valid** is likewise a fact checked deterministically — against `list_available_models()`'s own schema — before any training happens, rather than left for Gemini's arguments to be trusted blindly.

---

## Tool architecture: Category A (execution) vs Category B (decision)

_(unchanged — see prior version for the full pipeline diagram)_

Native function-calling is used for two genuinely different purposes, kept strictly separate:

- **Category A — execution tools** (`list_available_models`, `train_model`, `evaluate_model`). Ordinary deterministic Python/scikit-learn functions with real side effects. No awareness that an LLM exists.
- **Category B — structured-decision tools** (`record_model_proposal`, `record_convergence_decision`). No execution — capture Gemini's reasoning as parseable JSON instead of prose.

All of the agent's "agency" lives inside Category B; Category A is LLM-unaware plumbing that executes whatever B decided.

---

## New this session: `Trainer` — model storage and encapsulation

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

### `Trainer` class (storage + validation scaffolding — real sklearn fit/predict logic still pending)

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

        # TODO (real logic, next session): instantiate matching sklearn
        # estimator, call .fit(X_train, y_train)

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
        # TODO (real logic, next session): predict + compute
        # accuracy/precision/recall/f1/confusion_matrix
        return {"model_ref": model_ref, "model_type": entry["model_type"],
                 "pos_label": pos_label}
```

---

## New this session: validating Gemini's arguments before training

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

## New this session: `agent.py` — wiring the dispatch table

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
├── AGENTS.md              # not yet written — see Roadmap / TODOs
├── main.py
├── ml_agent/
│   ├── dataset.py         # ✅ done
│   ├── tools.py            # ✅ done (5 tool schemas + inclusive-bounds
│   │                        #   convention documented)
│   ├── gemini_client.py     # not started
│   ├── trainer.py           # ✅ storage + validation scaffolding done;
│   │                        #   real sklearn fit/evaluate logic pending
│   └── agent.py              # ✅ dispatch-table wiring done;
│                              #   orchestration loop pending
└── tests/
    ├── test_dataset.py    # ✅ 9 passing tests
    ├── test_tools.py       # not started
    └── test_trainer.py     # not started
```

---

## Datasets

_(unchanged — see prior version for full detail)_

- **Primary: Climate Model Simulation Crashes** (OpenML id 1467) — 540 rows, 18 features after dropping two non-predictive identifier columns, imbalanced ~91.5%/8.5%, `pos_label=1` (failure, the rare class).
- **Fallback: Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`) — fast, network-free debugging. `pos_label=1` here means *benign* (majority class) — the opposite sense from Climate Crashes. `pos_label` is established per-dataset, never assumed consistent.

## Available models

_(unchanged — see prior version's table)_

| Model | Key hyperparameters |
|---|---|
| Logistic Regression | `C` (0.001–100, inclusive), `class_weight` |
| Random Forest | `n_estimators` (10–500, inclusive), `max_depth` (int or null), `class_weight` |
| SVM | `C`, `kernel` (linear/rbf/poly), `class_weight` |

---

## Setup / Running / Testing

_(unchanged from prior version)_

```bash
git clone git@github.com:MelanieM2/ml-agent.git
cd ml-agent
uv sync
cp .env.example .env   # then fill in your real GEMINI_API_KEY
uv run python main.py
uv run pytest -v
```

---

## Security

_(unchanged from prior version)_

---

## Development Notes

This project was developed with AI assistance:
- **Development tool:** Claude Sonnet 5 (Anthropic), used as a pair-programming and design-review collaborator — explaining rationale, reviewing architecture decisions, catching data-quality and schema-ambiguity issues before they reached the codebase.
- **Runtime AI component:** `gemini-3.1-flash-lite` (Google), the model actually orchestrating the agent's tool-calling loop at runtime.

All architectural decisions (including the Category A/B tool split, the `Trainer` encapsulation design, the `functools.partial` pos_label binding, and the inclusive-bounds schema convention), dataset choices, and correctness checks were reviewed and made by the project author. AI assistance focused on explaining concepts, generating boilerplate, and surfacing issues for human review — not on unsupervised code generation.

---

## Roadmap context

Project 5 of a 10-step learning roadmap: Linux → Git → Professional Python → Pandas → SQL → **Scikit-learn (this project)** → AI-assisted workflows → PyTorch → Transformers → Geometric Deep Learning.

### Next up: real `trainer.py` logic

Fill in the `TODO` blocks inside `Trainer.train_model` (instantiate the correct sklearn estimator, call `.fit`) and `Trainer.evaluate_model` (predict, compute accuracy/precision/recall/f1/confusion_matrix using `pos_label`). Then `test_trainer.py`, then `gemini_client.py` and the real orchestration loop in `agent.py`.
