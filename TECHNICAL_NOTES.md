# Technical Notes: `gemini_client.py` Conversation-History Cost Scaling

_Status: speculative, deferred, NOT IMPLEMENTED. Written 2026-07-13 as a
deliberate evaluate-and-defer analysis, not a plan of record. Revisit only
if `MAX_ITERATIONS` is raised significantly above its current default of
10._

---

## Why this matters (and why it doesn't, yet)

`run_agent_loop` uses `client.chats.create()` for conversation-history
management — the SDK's own automatic tracking, rather than a manually
assembled message list. This is simple and correct, but it has one
structural property worth understanding precisely before ever scaling the
loop's iteration count up.

## `client.chats.create()`'s internal dynamics

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

## The actual cost shape — precisely, not loosely

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

## Mitigation options considered

### 1. Explicit context caching (`client.caches.create()`) — NOT a fit, and why

Real, confirmed-existing feature in `google-genai`. Designed for a large,
*static* prefix reused across many separate, otherwise-unrelated requests
(e.g. one big reference document queried many times). This loop's
expensive part — the accumulated tool-call history — changes on *every*
turn, so a cache would need re-creating almost as often as it would save
anything. There's also a real minimum-size threshold and the cache's own
storage cost/TTL to manage. Evaluated and ruled out, not just left
unconsidered.

### 2. Implicit caching — exists, unconfirmed relevance here

The SDK exposes `response.usage_metadata.total_cached_tokens`, suggesting
Gemini may automatically cache repeated prefixes across turns in the same
session, with no code changes needed. Whether this actually engages for a
same-`Chat`-session loop like this one is **not confirmed** — this is
inference from a metadata field's existence, not a tested claim. Worth
checking that field's value on a real run out of curiosity; free
information either way, no development cost.

### 3. History windowing/truncation — the real lever, real trade-off

Instead of `client.chats.create()`'s never-pruned automatic history,
manually build the message list each turn: keep the original
`initial_context` plus only the last *k* tool exchanges, dropping older
raw `function_response` payloads. This directly caps both the linear
per-request size and the quadratic cumulative cost.

**The real cost of this option:** Gemini genuinely loses the ability to
"look back" past *k* iterations at what it already tried — which matters
specifically for a model-search loop, where avoiding repeated dead-end
hyperparameter choices is part of the point. Not a free win.

### 4. Condensed running scratchpad — the recommended approach, if ever needed

Rather than truncating, replace old raw tool-call turns with a single
compact summary line per iteration (`model_type`, key hyperparameters,
headline metric) built directly in Python from the already-structured
dispatch results — no summarization LLM call required. This keeps
Gemini's memory of *what's been tried* intact while dropping the bulky
raw metadata (full confusion matrices, etc.) that isn't needed turn after
turn.

## Recommendation

At `MAX_ITERATIONS = 10`, none of the above is worth building — real
added complexity for a cost curve that's genuinely small at this scale.
If `MAX_ITERATIONS` is ever raised significantly, option 4 (condensed
scratchpad) is the one actually recommended: option 3 costs real search
quality, and options 1/2 don't target the part of the problem that's
actually growing (the changing tool-call history, not a static prefix).

This entire analysis is deferred, evaluated-not-implemented status — see
TODO #4 in `context_ml-agent_2026-07-13.md` and the Detailed Session
Summary for the same date.
