# Technical Notes

## Part 1: `gemini_client.py` Conversation-History Cost Scaling

_Status: speculative, deferred, NOT IMPLEMENTED. Written 2026-07-13 as a
deliberate evaluate-and-defer analysis, not a plan of record. Revisit only
if `MAX_ITERATIONS` is raised significantly above its current default of
10._

---

### Why this matters (and why it doesn't, yet)

`run_agent_loop` uses `client.chats.create()` for conversation-history
management — the SDK's own automatic tracking, rather than a manually
assembled message list. This is simple and correct, but it has one
structural property worth understanding precisely before ever scaling the
loop's iteration count up.

### `client.chats.create()`'s internal dynamics

```
┌──────────────────────────────────────────────────────────────┐
│  client.chats.create(model=..., config=...)                   │
│  → returns a Chat object                                      │
│  → internal history: []   (empty at creation, unless history= │
│    is passed explicitly — it isn't, here)                     │
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
              ┌───────────────────────────────┐
              │ Our loop reads response.text    │
              │ or response.function_calls      │
              └──────────────┬───────────────────┘
                              ▼
        ┌────────────────────────────────────────────┐
        │ chat.send_message(function_response_part)    │
        │ — SAME chat object, SAME growing history      │
        └──────────────────┬─────────────────────────┘
                            ▼
              (loop repeats: steps 1–4 above, every
               turn, history strictly append-only —
               nothing is ever pruned or summarized)
```

Every single `send_message()` call — the first `initial_context`, or the
tenth `function_response` deep into the loop — goes through the same four
steps: append the new message, replay the *entire* accumulated history to
`generate_content`, append the reply, return it. There's no special
handling for tool-result messages versus plain text ones.

### The actual cost shape — precisely, not loosely

Worth correcting an earlier imprecise framing (from mid-session
discussion): it's important to separate two different things that grow at
different rates.

- **Per-request size** (what a single `send_message` call sends) grows
  **linearly** with iteration count — turn *N* resends roughly *N* times
  the size of a single turn's content.
- **Cumulative cost across the whole run** (summing every request's input
  tokens over all turns) grows **quadratically** — turn *N*'s cost is
  itself proportional to *N*, and summing that from 1 to *N* gives an
  O(N²) total.

At `MAX_ITERATIONS = 10`, this is a non-issue — the absolute token counts
involved are small. The concern is purely about what happens if that
ceiling is raised significantly (e.g. past 20–30), where the quadratic
cumulative-cost curve starts to matter for budgeting, and where the linear
per-request growth starts to matter for approaching context-window limits.

### Mitigation options considered

#### 1. Explicit context caching (`client.caches.create()`) — NOT a fit, and why

Real, confirmed-existing feature in `google-genai`. Designed for a large,
*static* prefix reused across many separate, otherwise-unrelated requests
(e.g. one big reference document queried many times). This loop's
expensive part — the accumulated tool-call history — changes on *every*
turn, so a cache would need re-creating almost as often as it would save
anything. There's also a real minimum-size threshold and the cache's own
storage cost/TTL to manage. Evaluated and ruled out, not just left
unconsidered.

#### 2. Implicit caching — exists, unconfirmed relevance here

The SDK exposes `response.usage_metadata.total_cached_tokens`, suggesting
Gemini may automatically cache repeated prefixes across turns in the same
session, with no code changes needed. Whether this actually engages for a
same-`Chat`-session loop like this one is **not confirmed** — this is
inference from a metadata field's existence, not a tested claim. Worth
checking that field's value on a real run out of curiosity; free
information either way, no development cost.

#### 3. History windowing/truncation — the real lever, real trade-off

Instead of `client.chats.create()`'s never-pruned automatic history,
manually build the message list each turn: keep the original
`initial_context` plus only the last *k* tool exchanges, dropping older
raw `function_response` payloads. This directly caps both the linear
per-request size and the quadratic cumulative cost.

**The real cost of this option:** Gemini genuinely loses the ability to
"look back" past *k* iterations at what it already tried — which matters
specifically for a model-search loop, where avoiding repeated dead-end
hyperparameter choices is part of the point. Not a free win.

#### 4. Condensed running scratchpad — the recommended approach, if ever needed

Rather than truncating, replace old raw tool-call turns with a single
compact summary line per iteration (`model_type`, key hyperparameters,
headline metric) built directly in Python from the already-structured
dispatch results — no summarization LLM call required. This keeps
Gemini's memory of *what's been tried* intact while dropping the bulky
raw metadata (full confusion matrices, etc.) that isn't needed turn after
turn.

### Recommendation

At `MAX_ITERATIONS = 10`, none of the above is worth building — real
added complexity for a cost curve that's genuinely small at this scale.
If `MAX_ITERATIONS` is ever raised significantly, option 4 (condensed
scratchpad) is the one actually recommended: option 3 costs real search
quality, and options 1/2 don't target the part of the problem that's
actually growing (the changing tool-call history, not a static prefix).

This entire analysis is deferred, evaluated-not-implemented status — see
TODO #4 in `context_ml-agent_2026-07-13.md` and the Detailed Session
Summary for the same date.

---

## Part 2: Orchestration-loop wiring — implementation details (2026-07-17)

_Status: IMPLEMENTED and verified via 3 live runs (see
`DATA_SCIENCE_ANALYSIS.md` §8). This section documents how, and why,
rather than the evaluate-and-defer style of Part 1 above._

This session wired `agent.py`'s previously-missing orchestration caller.
Four sub-decisions were resolved; each is documented below with the
reasoning behind the choice, not just the final shape.

### 2.1 `build_dispatch_table`'s return type: `DispatchResult`

**The problem.** `build_dispatch_table` already loaded `(X, y)` internally
(to build the split), but discarded them at return — meaning any caller
needing `(X, y)` for `inspect_dataset` (to build `initial_context`) had to
call `load_dataset(dataset_name)` a second time. For Climate Crashes
specifically, this means a second OpenML network fetch per run, not just
a wasted CPU cycle.

**The fix.** `build_dispatch_table`'s return type changed from
`dict[str, Callable[..., Any]]` to a new frozen dataclass:

```python
@dataclass(frozen=True)
class DispatchResult:
    dispatch_table: dict[str, Callable[..., Any]]
    X: pd.DataFrame
    y: pd.Series
```

Shaped as a dataclass, not a bare tuple, to match this project's existing
convention for named, self-documenting return values (`DatasetSpec` in
`dataset.py` is the precedent). Lives in `agent.py`, not `dataset.py`,
since it describes `build_dispatch_table`'s own return shape specifically
— nothing outside `agent.py` has reason to construct one.

**Deliberately narrow.** `DispatchResult` carries `(X, y)` only — *not* a
formed `initial_context` string. An earlier version of this decision
considered having `build_dispatch_table` also assemble the full prompt
text, but that was rejected: it would widen the function's job from
"resolve dataset/run-specific facts" to "also write prompts," a real scope
creep for a function whose docstring already commits it to the former.
Prompt assembly stays the caller's job (see §2.3), using this same `X`/`y`
— so `load_dataset` still only ever runs once per run, without
`build_dispatch_table` taking on a second responsibility to get there.

### 2.2 `TOOL_FUNCTIONS` reuse in `build_dispatch_table`

**The problem.** `tools.py`'s `TOOL_FUNCTIONS` dict maps all five tool
names to their bare, module-level functions — including `train_model` and
`evaluate_model`, which are placeholder stubs that `raise
NotImplementedError`. `build_dispatch_table` needs the real,
`Trainer`-backed, `functools.partial`-bound versions of exactly those two
functions. The two dicts can therefore never be identical.

**The fix (chosen over hand-building all 5 entries from scratch).**
`build_dispatch_table` now starts from `TOOL_FUNCTIONS` and overrides only
the 2 entries that need per-run binding:

```python
dispatch_table = {
    **TOOL_FUNCTIONS,
    "train_model": bound_train,
    "evaluate_model": bound_evaluate,
}
```

The 3 entries that need no binding (`list_available_models`, and both
Category B tools) come from `TOOL_FUNCTIONS` as-is; the 2 that do are
supplied directly. This directly fulfils `TOOL_FUNCTIONS`'s own stated
purpose ("so any future dispatch-table wiring in `agent.py` can reuse the
same mapping") and removes a small drift risk: if a 6th, unbound tool were
ever added to `tools.py`, it would be picked up here automatically via the
dict spread, with no second place in `agent.py` to remember to update.

**Risk, flagged rather than silently accepted:** if `train_model` or
`evaluate_model` were ever renamed in `tools.py`, these two override lines
would silently stop overriding anything — the stale, `NotImplementedError`
-raising version from `TOOL_FUNCTIONS` would quietly take their place,
undetected until a real run hit it (no import error, no exception until
the tool is actually called by Gemini).

**Mitigation, planned but not built this session:** extend
`test_tools.py`'s existing drift-check philosophy with an assertion that
`build_dispatch_table`'s two override keys still exist in `TOOL_FUNCTIONS`
— would catch a rename at test time instead of runtime. Tracked as a
follow-up item, not implemented 2026-07-17.

### 2.3 Optimization target: explicit parameter, not hardcoded

**The problem.** `DATA_SCIENCE_ANALYSIS.md` §4 found, from real run
evidence, that `initial_context` never stated what metric to optimize for
— and that this measurably affected model selection across 3 separate
Climate runs.

**The fix, and why this specific shape.** `optimization_target` is a
plain string parameter on the new `run_session` function (see §2.4) —
*not* a value hardcoded per dataset inside `dataset.py`'s registry. This
was a deliberate choice to preserve an existing project principle: *which
class is "positive" is a documented fact* (`pos_label`, resolved once per
dataset in `dataset.py`'s registry); *what to optimize for* is a
judgment call, made once per run, by whoever calls `run_session` — not
something the dataset registry or `build_dispatch_table` should decide on
the caller's behalf. Baking it into the dataset registry would have
quietly turned a judgment call into a fact, contradicting the same
fact-vs-judgment separation this project already applies to `pos_label`
itself.

`optimization_target` is substituted into a fixed prompt template inside a
small, separately-testable helper:

```python
def _format_initial_context(facts: dict[str, Any], optimization_target: str) -> str:
    ...
    return (
        "...Optimization target: {optimization_target}. Propose and "
        "evaluate models with this target explicitly in mind, using ..."
    )
```

Only the substituted value changes between calls; the surrounding template
text is fixed. Nothing currently validates that the supplied string is one
of a known set (e.g. `"recall"`, `"precision"`, `"accuracy"`) — Gemini
reads whatever text is supplied as-is. This is a plausible future gap (a
typo'd or nonsensical target would silently reach Gemini unchanged), noted
here but not addressed this session — out of scope for today's wiring
work.

**Interactive prompting deliberately excluded from `run_session` itself.**
`run_session` never calls `input()` — it stays callable directly from a
test with a hardcoded string, with no interactive I/O to mock. Whichever
CLI entry point is eventually built (`main.py`, not started as of this
session) is responsible for actually asking the person what to optimize
for and passing the answer in as this parameter.

### 2.4 `run_session`: the new orchestration entry point

The piece that was "genuinely still missing" — everything it calls
(`build_dispatch_table`, `inspect_dataset`, `run_agent_loop`) already
existed and worked in isolation before this session; nothing previously
called them together with real, non-hand-built inputs in the committed
codebase.

```python
def run_session(
    dataset_name: str,
    optimization_target: str,
    *,
    random_state: int = 42,
    model: str = DEFAULT_MODEL,
    max_iterations: int = MAX_ITERATIONS,
) -> dict[str, Any]:
    result = build_dispatch_table(dataset_name, random_state=random_state)
    facts = inspect_dataset(result.X, result.y)
    initial_context = _format_initial_context(facts, optimization_target)
    return run_agent_loop(
        result.dispatch_table, initial_context,
        model=model, max_iterations=max_iterations,
    )
```

Does **not** touch either of the two judgment calls left open since
2026-07-13 (single-function-call-per-turn assumption;
`record_convergence_decision`'s result not echoed back on stop) — both
live entirely inside `run_agent_loop`, which `run_session` calls
unmodified. This was an explicit, agreed checkpoint this session (not a
silent omission) — see `context_ml-agent_2026-07-17.md`.

### 2.5 `run_smoke_test.py`: replaced, not just superseded

`DispatchResult` (§2.1) is a real, breaking change to
`build_dispatch_table`'s return type. The pre-2026-07-17 version of
`run_smoke_test.py` called `build_dispatch_table(DATASET_NAME)` and passed
its result directly to `run_agent_loop`, which expects a plain
`dict[str, Callable]` — so that file would have raised a type mismatch if
left unedited, not merely become redundant. It was rewritten to call the
real `run_session` directly instead of hand-building the dispatch table,
a second `load_dataset` call, and a placeholder prompt string, as it
previously did. Confirmed working via 3 live runs against the real Gemini
API (see `DATA_SCIENCE_ANALYSIS.md` §8).

### 2.6 Planned, not built: per-iteration logging in `run_agent_loop`

Comparing the 3 live runs in §8 surfaced a real limitation: `run_agent_loop`
currently returns only the final decision, with no record of which models
were proposed, evaluated, or rejected along the way. This makes questions
like "why did run 3 converge two iterations faster with no
`ConvergenceWarning`" unanswerable from the printed result alone.
Melanie has requested this as a future capability — optional logging of
each iteration's tool call (name, arguments, result) inside
`run_agent_loop` — flagged here as planned future work, not implemented
2026-07-17. See `context_ml-agent_2026-07-17.md` for next-session framing.