# ml-agent

_Last updated: 2026-07-27_

Agentic ML experimentation assistant — an LLM (Gemini) orchestrates dataset inspection, model proposal, training, and evaluation over scikit-learn via native function-calling, iterating toward a target metric in a supervised loop.

**Status: in progress.** This is Project 5 in a 10-step self-directed learning roadmap toward Data Science / ML Engineering / Agentic AI.

- `dataset.py` — ✅ done and tested.
- `tools.py` — ✅ done (5 tool schemas + inclusive-bounds convention documented; `TOOL_FUNCTIONS` name→callable mapping added 2026-07-15, now actively reused by `agent.py`'s dispatch-table wiring — see "Project structure" below). **`max_iter` exposed as a tunable Logistic Regression hyperparameter (2026-07-27)** — gives the agent a concrete lever to act on the `ConvergenceWarning` `trainer.py` already surfaces — see "Closing the `ConvergenceWarning` shortlist" below.
- `trainer.py` — ✅ done. Real sklearn fit/predict logic implemented: `train_model` fits the correct estimator per `model_type` via a registry lookup against `tools.py`'s schema; `evaluate_model` computes accuracy, precision, recall, f1, and a correctly-oriented confusion matrix using `pos_label`. **Fit-time warning capture added (2026-07-24)** — `train_model` now returns any warning scikit-learn raised during fitting (e.g. a `ConvergenceWarning`), never just silently letting it print to the terminal — see "Fit-time warning capture" below.
- `agent.py` — ✅ **orchestration loop wired end-to-end (2026-07-17), verified via 3 real live runs.** `build_dispatch_table` now returns a `DispatchResult` (dispatch table + the run's `X`/`y`), and `run_session(dataset_name, optimization_target)` connects it, `inspect_dataset`, and `gemini_client.run_agent_loop` into one real, callable entry point — see "`agent.py` — wiring the dispatch table" below. `run_session` also now forwards an optional `log_iterations` flag (2026-07-19) straight through to `run_agent_loop` — see "Per-iteration logging" below.
- `gemini_client.py` — ✅ **done and verified.** Implements the full agent loop: sends dataset context + tool schemas to Gemini, dispatches whichever tool call comes back, feeds results back, repeats until convergence or a max-iterations guard. Verified via multiple real end-to-end runs across both datasets (see "Data Science Notes" below). **Now also supports optional per-iteration logging (2026-07-19)** — see "Per-iteration logging" below. **`MAX_ITERATIONS` raised from 10 to 15 (2026-07-27)** — the 07-12 debugging-safety-net rationale for keeping it small was explicitly conditioned on "once the loop is trusted end-to-end"; 9 live runs across two sessions, plus a `main.py` run that hit the old ceiling mid-retry on legitimate in-progress work, met that condition — see "`gemini_client.py` — the agent loop" below.
- `test_tools.py` — ✅ **done, 12 tests passing (2026-07-15, extended 2026-07-19).** The `TOOL_SCHEMAS` ↔ real-function signature drift check, using `inspect.signature()`, plus a guard on the two dispatch-table override keys. See "`test_tools.py` — the schema/function drift check" below. **Review resolved (2026-07-27):** `train_model`'s new `warnings` return key needed no update here — every assertion in this file inspects a function's *parameters* via `inspect.signature()`, never its return value, and `train_model`'s own parameter list is unchanged.
- **`compare_runs.py` (2026-07-24) — ✅ new, done, verified against 6 real live runs, plus 3 more this session.** Turns several persisted `results/smoke_test_log_*.json` files into one `comparison_<timestamp>.json`, summarizing each run's model sequence, final metrics, any warnings encountered, and elapsed time — see "Cross-run comparison" below.
- **`main.py` (2026-07-27) — ✅ new, done.** The real, committed public entry point — `run_smoke_test.py` (gitignored, hardcoded) was never reachable by anyone cloning the repo. Interactive/CLI hybrid: prompts for `--dataset`/`--target` only when omitted, everything else silently defaults. See "`main.py` — the CLI entry point" below.
- **Human-in-the-loop hook (2026-07-19): scope agreed, implementation still deferred.** The confirmation step described below (Category B → Category A handoff) will combine hyperparameter edge-case flagging, reasoning/action contradiction detection, and stalled/repeated-proposal detection — deliberately not a per-proposal approval gate, and deliberately not limited to reviewing only the final result. Build is intentionally deferred until the project is close to a finished, working state, not before — see "Roadmap context" below. **A related, distinct idea raised 2026-07-27:** a post-hoc `Agent-decisions.md` generator (human-readable decision/reasoning report per run, written incrementally) — architecture agreed, not built; see "Roadmap context."
- Also not yet started: `test_trainer.py`, `AGENTS.md`, and a human-friendly viewer for the JSON result files (design agreed 2026-07-24 — auto-detects a single-run vs. a comparison file by name, renders as Markdown — not yet built).

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

A fourth test, `test_dispatch_table_override_keys_exist()`, asserts that `"train_model"` and `"evaluate_model"` still exist as keys in `TOOL_FUNCTIONS` — added as explicit, by-name documentation of a concern originally raised during the 2026-07-17 orchestration wiring. Worth being precise about what this test actually guards, since the original stated rationale for it turned out to be inaccurate on closer inspection: `agent.py`'s dispatch-table override uses **literal string keys**, not a lookup into `TOOL_FUNCTIONS`, so a rename inside `TOOL_FUNCTIONS` cannot silently break that override the way it was originally described as doing. The real failure mode a rename *can* cause — `TOOL_SCHEMAS` declaring a tool name no longer present in `TOOL_FUNCTIONS` — was already caught by the pre-existing "every schema has a registered function" check above. This new test is retained as cheap, explicit documentation of the concern, not because it closes a gap that genuinely existed. Full detail in `TECHNICAL_NOTES.md` §2.2's correction note and Part 3, §3.1.

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

## Cross-run comparison (2026-07-24)

### Why this exists

With runs now persisted individually instead of overwritten, the original motivating question — why does one run converge faster than another, with or without a `ConvergenceWarning` — finally has real, comparable data behind it. `ml_agent/compare_runs.py` turns several raw run files into one summary.

### How it works

```
results/*.json (several completed runs)
        │
        ▼
summarize_run()  — pure function, one run in, one flat row out
        │
        ▼
list of per-run summary rows
        │
        ▼
results/comparison_<timestamp>.json  — one row per run
```

`summarize_run(run_data, source_file=None)` reads one already-loaded run dict — no file I/O, no live API call — and extracts: `status`, `iterations`, `elapsed_seconds`, the ordered `model_sequence` actually tried, the `final_model_type` and its metrics, every `warnings_encountered` entry (flattened, with the iteration and model type each fired on), and the run's final `convergence_reasoning`. Kept as a pure function of its input dict, for the same reason `validate_split` and `validate_hyperparameters` are kept standalone elsewhere in this project: testable directly against a hand-built dict, with no dataset, `Trainer`, or live API call required.

`build_comparison(results_dir=Path("results"))` scans for every `smoke_test_log_*.json` file present and calls `summarize_run` on each — not capped at any particular count, whether that's 2 files or 20.

**Two judgment calls worth stating plainly** — true of every run observed so far, but not independently verified as universal: "final model" is read as the *last* `evaluate_model` call in a run's log, not cross-checked against the wording of its convergence decision; and `convergence_reasoning` is taken from the *last* `record_convergence_decision` entry seen, even on a run that hit `max_iterations` without ever setting `continue_iterating=False`.

### Verified against real data

Across six real Climate Crashes runs from a single evening: correctly distinguished the one run that never proposed Logistic Regression — and so never hit the `ConvergenceWarning` — from the five that did. Full run-by-run findings, including a discussion of run-to-run path variation, are in [`DATA_SCIENCE_ANALYSIS.md`](./DATA_SCIENCE_ANALYSIS.md).

### Known limitation, not fixed

A run that crashes before completing (e.g. hitting the Gemini API's own rate limit) writes no file at all — a failed attempt is currently invisible to this comparison, not just excluded from it. Accepted for now; see "Roadmap context."

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

**Worth stating plainly, not overclaiming:** this confirms the data *reaches* Gemini and that the mechanism functions end-to-end on 3 real runs — it isn't a guarantee that every future run will reason about a warning this well. "Wired" and "reasons about it well" are different claims.

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
    bound_train = partial(trainer.train_model, X_train=X_train, y_train=y_train)
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

## `main.py` — the CLI entry point (2026-07-27)

### Why this needed building specifically

`run_smoke_test.py` is gitignored — meaning it was, and remains, invisible to anyone cloning this repo. Before this session, there was no committed way for a stranger to actually *run* a session at all, regardless of how complete `run_session` itself already was. `main.py` is that missing public entry point; it adds no new orchestration logic of its own — it asks a real person the two genuine judgment calls `run_session` needs, then calls it exactly the way `run_smoke_test.py` already does.

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

`--dataset`/`--target` prompt interactively when omitted — these are the two facts/judgment-calls `run_session` genuinely needs a person to supply (see "Core architectural principle" above). `--target` is constrained to the exact four metrics `evaluate_model` computes (`recall`/`precision`/`accuracy`/`f1`), not free text — see the new bullet under "Core architectural principle" above for why. Everything else (`--model`, `--max-iterations`, `--random-state`, `--log-iterations`) silently uses `run_session`'s own defaults unless explicitly overridden — no prompts, no interruption to a normal run. The effective configuration actually used is printed once, at the start of every run, and persisted into the saved result file's `config` key — discoverable without ever pausing execution to ask:

```
Running: dataset=climate, target=recall, model=gemini-3.1-flash-lite,
max_iterations=15, random_state=42, log_iterations=True
(override any of these next time with --dataset, --target, --model,
--max-iterations, --random-state, --no-log-iterations)
```

`--log-iterations` defaults to `True` here — deliberately different from `run_session`'s own library default of `False` — since a first-time user benefits from seeing the agent's full reasoning trail, and it can be turned off with `--no-log-iterations` for a quieter run. Unlike `run_smoke_test.py`, `main.py` persists the result to `results/` unconditionally, even on a quiet run, rather than gating the save behind `log_iterations` — losing a run's outcome entirely just because someone wanted less terminal output was judged an avoidable loss.

### Verified against a real live run

Confirmed working end-to-end: interactive dataset/target prompts, the printed config summary, the full run through `run_session`, and persistence to `results/smoke_test_log_<timestamp>.json` — same naming convention as `run_smoke_test.py`, deliberately, so both entry points feed the same directory `compare_runs.py` already scans.

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
├── main.py                 # ✅ CLI entry point (2026-07-27) — real public replacement for run_smoke_test.py's role
├── ml_agent/
│   ├── __init__.py
│   ├── agent.py             # ✅ orchestration loop wired end-to-end (2026-07-17); log_iterations threaded through (2026-07-19)
│   ├── compare_runs.py      # ✅ turns several results/smoke_test_log_*.json runs into one comparison_<timestamp>.json
│   ├── dataset.py           # ✅ done — dataset-agnostic loading + inspection
│   ├── gemini_client.py     # ✅ done — Gemini client + function-call dispatch loop, verified; per-iteration logging + format_log (2026-07-19); MAX_ITERATIONS raised 10→15 (2026-07-27)
│   ├── tools.py              # ✅ done — Gemini function-calling tool schemas + TOOL_FUNCTIONS; max_iter hyperparameter added (2026-07-27)
│   └── trainer.py            # ✅ done — storage, validation, real sklearn fit/predict/evaluate; fit-time warning capture (2026-07-24)
├── pyproject.toml
├── results/                   # gitignored; one smoke_test_log_<timestamp>.json per run (2026-07-24: no longer overwritten; written by both run_smoke_test.py and main.py), plus comparison_<timestamp>.json output from compare_runs.py
├── run_smoke_test.py         # manual live-API smoke test of run_session(); gitignored; superseded by main.py as the real entry point, kept for quick manual debugging
├── smoke_test.py              # isolated schema-construction check, no API key/network; gitignored
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py       # ✅ done, 9 passing tests
│   ├── test_tools.py         # ✅ done, 12 passing tests (2026-07-15; extended 2026-07-19) — reviewed, resolved (2026-07-27), no changes needed
│   └── test_trainer.py       # not yet started
└── uv.lock
```

`ml_agent/` is a real installed package — imports elsewhere use `from ml_agent.tools import ...`, `from ml_agent.dataset import ...`, etc., never bare top-level module names.

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

The real, public entry point. Prompts interactively for a dataset and optimization target if not given as flags; everything else defaults sensibly. See `python main.py --help` for the full flag list, or "`main.py` — the CLI entry point" above for the design rationale.

```bash
uv run python main.py --dataset climate --target recall
uv run python main.py --dataset breast_cancer --target f1 --no-log-iterations
```

For a manual, real-API smoke test with hardcoded defaults (the original way this loop was exercised before `main.py` existed):

```bash
uv run python run_smoke_test.py
```

To compare several already-completed runs sitting in `results/`:

```bash
uv run python -m ml_agent.compare_runs
```

## Testing

```bash
uv run pytest -v
```

Network-dependent behavior (the Climate Crashes OpenML fetch) is not exercised directly in tests — it's mocked. Currently **21 tests passing**: 9 in `test_dataset.py`, 12 in `test_tools.py` (added 2026-07-15, extended 2026-07-19; reviewed and resolved 2026-07-27 against `trainer.py`'s `warnings` key and `tools.py`'s `max_iter` addition — no changes needed).

---

## Security

See [`SECURITY.md`](./SECURITY.md) for the full policy (dependency pinning, vulnerability scanning, lockfile verification, API key handling, and the LLM-generated-tool-call guard specific to this project).

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
9. **As of 2026-07-27 (current):** `test_tools.py` reviewed and resolved (no changes needed — see its section above); the `ConvergenceWarning` shortlist narrowed to a decision and built (`max_iter` exposed, verified against 3 live runs — see "Closing the `ConvergenceWarning` shortlist" above); `main.py` built as the real public CLI entry point, chosen over the results viewer as this session's priority specifically because `run_smoke_test.py` being gitignored meant no committed entry point existed at all; and `MAX_ITERATIONS` raised from 10 to 15, closing a 07-12 open decision once real live-run evidence met its stated condition.

    **Carried forward, full running list, for the next session:**
    - The human-friendly results viewer (item 4 from 2026-07-24; design agreed, not built).
    - Timestamp-format inconsistency, `run_smoke_test.py` (local) vs. `compare_runs.py` (UTC) — low priority.
    - CSV/spreadsheet export — the explicitly named gate before making the repo public.
    - Rename the `smoke_test_log_<timestamp>.json` convention to `result_log_<dataset_name>_<timestamp>.json` — needs `compare_runs.py`'s real scan logic checked first, in case it globs a fixed prefix rather than everything in `results/`.
    - `Agent-decisions.md` generator — a post-hoc, human-readable report of `record_model_proposal`/`record_convergence_decision` reasoning per run, written incrementally (open-at-start, append-per-iteration) as partial mitigation for the invisible-failed-run limitation (item 8's known limitation, `compare_runs.py` section above). Architecture agreed: an injected callback into `run_agent_loop` (mirroring the still-deferred human-in-the-loop hook's shape), not yet built — to be designed together with that hook, since both need the same injection point in the same loop.
    - A possible future "register a dataset from the terminal, no code edit" feature — raised, not scoped or committed to.
    - A follow-up run isolating whether `C` or `max_iter` actually drove two of this session's three runs' improved precision (`DATA_SCIENCE_ANALYSIS.md` §11.5's confound) — a manual `Trainer.train_model`/`evaluate_model` call with `C=0.1` alone, `max_iter` left at default, documented as a ready-to-run snippet in the session's context handoff file.
    - The human-in-the-loop hook itself, and the two original 07-13 `run_agent_loop` judgment calls (single-call-per-turn; convergence result not echoed on stop) — all still untouched, still deliberately deferred.

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

Building `main.py` surfaced a distinction worth stating plainly, since it's easy to conflate: an unvalidated `optimization_target` string reaching Gemini's prompt unchecked is an **input-validation gap** (a typo or invented metric silently has no connection to what the tool results can support) — not **prompt injection** (which specifically means malicious instructions smuggled into content the model processes, hijacking it against its own instructions). The fix for the former (constrain to a known set) is a normal input-validation measure, not a security control against the latter; worth being precise about which problem a given safeguard actually addresses.

Separately, this session closed an open decision (`MAX_ITERATIONS`, 10 → 15) using a criterion worth naming explicitly for future similar cases: a debugging-safety-net value should be revisited once its own stated condition is met by real evidence, not left in place indefinitely just because it was never causing visible errors. Here, the cap wasn't failing loudly — it was silently truncating legitimate in-progress agent reasoning, a quieter failure mode that only became visible by actually reading what a real run's tool-call sequence was doing when it got cut off. Full rationale in `TECHNICAL_NOTES.md` Part 5, §5.9.