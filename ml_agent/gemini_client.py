"""
gemini_client.py — orchestrates the agent loop: sends context + tool
definitions to Gemini, dispatches whichever tool call comes back via
dispatch_table, feeds the tool's result back to Gemini as the next
message, and repeats until record_convergence_decision is called (with
continue_iterating=False) or MAX_ITERATIONS is reached.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from google import genai
from google.genai import types

from ml_agent.tools import TOOL_SCHEMAS

# Open decision #2 (07-12 session): 10 chosen deliberately small for
# early debugging — a broken convergence check surfaces fast and cheaply.
# Raise this once the loop is trusted end-to-end.
MAX_ITERATIONS = 10
DEFAULT_MODEL = "gemini-3.1-flash-lite"


def run_agent_loop(
    dispatch_table: dict[str, Callable[..., Any]],
    initial_context: str,
    *,
    model: str = DEFAULT_MODEL,
    max_iterations:  int = MAX_ITERATIONS,
    log_iterations: bool = False,
) -> dict[str, Any]:
    """
    Runs the Gemini-orchestrated model-search loop against an
    already-built dispatch table (see agent.py's build_dispatch_table).

    initial_context is the fully-formed first prompt — including
    inspect_dataset()'s output — built by agent.py before this function
    is ever called (open decision #4, resolved 07-12: gemini_client.py
    deliberately knows nothing about datasets, splits, or pos_label; it
    only knows how to talk to Gemini and dispatch whatever tool name
    comes back).

    log_iterations: off by default (False). When True, every iteration
    is recorded — tool name, tool arguments, result, and a UTC
    timestamp — including the case where Gemini replies with plain text
    instead of a tool call (tool_name/tool_args/result are None on that
    kind of entry; response_text is populated instead). The full list is
    returned under this function's result dict, keyed "log", only when
    log_iterations=True — callers that don't ask for it see no change
    in the returned dict's shape. Nothing is written to disk here; if
    you want the log to persist across runs, the caller is responsible
    for doing something with result["log"] itself.
    """
    client = genai.Client()
    log: list[dict[str, Any]] = []

    # automatic_function_calling is disabled on purpose: if Gemini's own
    # SDK executed functions directly, it would bypass dispatch_table
    # entirely — meaning it would miss the real, per-run partial-bound
    # Trainer/X_train/y_train/pos_label that build_dispatch_table sets up.
    # Every tool call must go through dispatch_table, no exceptions.
    #
    # TOOL_SCHEMAS entries are wrapped explicitly into FunctionDeclaration
    # objects rather than passed as raw dicts. Confirmed (07-13 smoke
    # test) that google-genai's Tool(function_declarations=...) accepts
    # raw dicts fine at runtime via Pydantic coercion — but its own type
    # stubs declare list[FunctionDeclaration] | None, not dicts, so the
    # explicit wrap keeps static type-checking clean and doesn't rely on
    # coercion behavior that isn't part of the SDK's declared contract.
    tool_declarations = [types.FunctionDeclaration(**schema) for schema in TOOL_SCHEMAS]
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=tool_declarations)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Judgment call (open decision #5, 07-12): using the SDK's own
    # chats.create() history tracking rather than manually assembling a
    # list of types.Content. Simplest option available; revisit only if
    # a real need for manual control over history surfaces (matching the
    # existing partial-over-closure trigger condition elsewhere in this
    # project — flag, don't silently switch, if that need appears).
    chat = client.chats.create(model=model, config=config)

    response = chat.send_message(initial_context)

    for iteration in range(max_iterations):
        function_calls = response.function_calls

        if not function_calls:
            # Gemini replied with plain text, not a tool call — treat this
            # as the loop ending without a formal convergence decision
            # (e.g. an early stray text reply). Not necessarily an error,
            # but worth agent.py's caller knowing it happened this way.
            if log_iterations:
                log.append({
                    "iteration": iteration,
                    "tool_name": None,
                    "tool_args": None,
                    "result": None,
                    "response_text": response.text,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            outcome = {
                "status": "stopped_without_convergence_call",
                "final_text": response.text,
                "iterations": iteration,
            }
            if log_iterations:
                outcome["log"] = log
            return outcome

        # Judgment call, flagging explicitly: this assumes exactly one
        # function_call per Gemini turn. TOOL_SCHEMAS/dispatch_table are
        # both built around single calls, but Gemini's function-calling
        # mode can in principle return several parallel calls in one
        # response — that case isn't handled here. Flag if you've seen
        # this happen in practice; I haven't built handling for it.
        call = function_calls[0]

        # FunctionCall.name and .args are typed Optional in the SDK — a
        # malformed or edge-case response could omit either. Guard
        # explicitly rather than assume both are always present.
        if call.name is None:
            raise ValueError("Gemini returned a function call with no name — cannot dispatch.")
        tool_name = call.name
        tool_args = dict(call.args) if call.args is not None else {}

        result = dispatch_table[tool_name](**tool_args)

        if log_iterations:
            log.append({
                "iteration": iteration,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": result,
                "response_text": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        if tool_name == "record_convergence_decision" and not tool_args.get("continue_iterating", False):
            # Gemini decided to stop — report the outcome and end the loop.
            outcome = {"status": "converged", "decision": result, "iterations": iteration}
            if log_iterations:
                outcome["log"] = log
            return outcome

        if tool_name == "record_model_proposal":
            # TODO: human-in-the-loop hook goes here — between
            # record_model_proposal's return (result, above) and
            # train_model's call (which happens on Gemini's *next* turn,
            # driven by whatever it does with this fed-back result).
            # Deferred per open decision #3 (07-12 session): currently
            # auto-approves everything. Real hook (e.g. an
            # on_proposal: Callable | None parameter) is a later,
            # separate piece of work — not implemented here.
            pass

        # Feed the tool's result back to Gemini as the next message. The
        # SDK's function_response Part tells Gemini which of its calls
        # this result answers, so the loop can continue coherently.
        function_response_part = types.Part.from_function_response(
            name=tool_name,
            response={"result": result},
        )
        response = chat.send_message(function_response_part)

    # Exhausted max_iterations without a convergence decision either way.
    outcome = {"status": "max_iterations_reached", "iterations": max_iterations}
    if log_iterations:
        outcome["log"] = log
    return outcome


def format_log(log: list[dict[str, Any]]) -> str:
    """Human-readable rendering of run_agent_loop's log entries (the
    list returned under result["log"] when log_iterations=True) — one
    block per iteration, separated for easy scanning in a terminal or
    a pasted-in doc. Purely a display helper; does not touch or
    reinterpret the log's actual data, so it stays safe to call from
    anywhere that has a log list in hand — run_smoke_test.py today,
    potentially main.py or other callers later.
    """
    blocks = []
    for entry in log:
        lines = [f"=== Iteration {entry['iteration']} ({entry['timestamp']}) ==="]
        if entry["tool_name"] is not None:
            lines.append(f"tool: {entry['tool_name']}")
            lines.append(f"args: {entry['tool_args']}")
            lines.append(f"result: {entry['result']}")
        else:
            lines.append("(no tool call — plain text response)")
            lines.append(f"response_text: {entry['response_text']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
