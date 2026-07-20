# agent.py — wiring skeleton; orchestration loop still pending (2a resolved
# this session: build_dispatch_table now also returns (X, y) via
# DispatchResult, so a caller never needs a second load_dataset() call)
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_agent.trainer import Trainer, validate_hyperparameters
from ml_agent.tools import TOOL_FUNCTIONS
from ml_agent.dataset import load_dataset, get_pos_label, inspect_dataset
from ml_agent.gemini_client import run_agent_loop, DEFAULT_MODEL, MAX_ITERATIONS


def validate_split(
    y_train: pd.Series, y_test: pd.Series, pos_label: int, min_count: int = 5
) -> None:
    """Raises ValueError if either split has too few pos_label examples.

    Pure function of its four arguments -- no Trainer/dataset construction
    needed to test it, same testing rationale as validate_hyperparameters
    in trainer.py: kept standalone so it's independently testable with
    zero setup.

    Guards against unreliable recall/precision on a near-empty rare class.
    A threshold of 5 keeps a single misclassification's swing on the
    metric below ~20% (1/5), while still catching a genuinely broken or
    collapsed split (e.g. stratify silently failing). See README Data
    Science Notes for the full statistical reasoning behind this choice.

    Only the rare class (pos_label) is checked. Given these datasets'
    class ratios (Climate Crashes: 91.48%/8.52%; Breast Cancer: 66%/34%),
    the majority class is never close to this floor under a stratified
    split, so checking it would add code without adding real safety.
    """
    # Count how many rows in each split actually belong to the rare class.
    train_count = int((y_train == pos_label).sum())
    test_count = int((y_test == pos_label).sum())

    if train_count < min_count:
        raise ValueError(
            f"Train split has only {train_count} pos_label={pos_label} "
            f"examples (minimum {min_count}) -- split is unreliable."
        )
    if test_count < min_count:
        raise ValueError(
            f"Test split has only {test_count} pos_label={pos_label} "
            f"examples (minimum {min_count}) -- split is unreliable."
        )


@dataclass(frozen=True)
class DispatchResult:
    """One run's dispatch table, plus the raw data it was built from.

    NEW (2026-07-17 session). Added because callers (e.g. the future
    run_session()) need the same (X, y) that build_dispatch_table already
    loads internally, in order to build initial_context via
    inspect_dataset(X, y) -- without this, a caller would have to call
    load_dataset(dataset_name) a SECOND time itself, redoing the same
    load (and, for Climate Crashes specifically, the same OpenML network
    fetch) for no reason. This was flagged as a known, open gap as far
    back as the 2026-07-15 context handoff.

    Deliberately narrow: carries the dispatch table and (X, y) only --
    NOT a formed prompt string. Building initial_context (via
    inspect_dataset + formatting) stays the caller's job, done once,
    using this same X/y. Keeping that logic outside this function/class
    keeps build_dispatch_table's own job unchanged: resolving what's
    dataset- and run-specific, not writing prompts.

    Shaped as a frozen dataclass (not a bare tuple) to match this
    project's existing convention for named, self-documenting return
    values -- see DatasetSpec in dataset.py for the precedent. Unlike
    DatasetSpec, this lives here in agent.py rather than dataset.py,
    since it describes build_dispatch_table's own return shape
    specifically, not a general dataset-registry fact used elsewhere.
    """

    dispatch_table: dict[str, Callable[..., Any]]
    # X/y types confirmed directly against dataset.py's real
    # load_dataset() signature this session -- not inferred/guessed.
    X: pd.DataFrame
    y: pd.Series


def build_dispatch_table(
    dataset_name: str, random_state: int = 42
) -> DispatchResult:
    """Builds one run's complete tool dispatch table for gemini_client.py,
    plus the (X, y) that table was built from.

    CHANGED (2026-07-17): return type is now DispatchResult, not a bare
    dict -- see DispatchResult's own docstring above for why. Everything
    below this point is UNCHANGED from the original version; only the
    final return statement and the return-type annotation differ.

    Given the active dataset's name, this resolves everything that's
    dataset- or run-specific exactly once, here, so nothing downstream
    ever needs to rediscover it:
      - the full dataset (X, y) and its pos_label, loaded via
        load_dataset() / get_pos_label()
      - a stratified train/test split of that data -- a run-construction
        concern, deliberately kept out of dataset.py, which only knows
        how to load/describe a *whole* dataset, never how to partition
        one for a specific run
      - a hard validation gate (validate_split) on that split, run before
        Trainer is even constructed
      - one Trainer instance (the private model_ref -> fitted model
        store for this run; see Trainer's own docstring)
      - X_train/y_train and X_test/y_test/pos_label, pre-filled into
        train_model/evaluate_model via functools.partial, so Gemini's
        dispatch never sees any of this data as an argument at all

    The returned DispatchResult.dispatch_table maps each of the five tool
    names exactly as Gemini will return them to a ready-to-call callable:
      - Category A (list_available_models, train_model, evaluate_model)
        are real executions - train_model validates its own arguments
        against list_available_models()'s schema before touching sklearn.
      - Category B (record_model_proposal, record_convergence_decision)
        are pure structured-decision capture, stateless, wired in as-is.

    gemini_client.py's dispatch logic can then do
    dispatch_table[tool_name](**args) without knowing dataset_name,
    pos_label, the split, or the Trainer instance exist.
    """
    X, y = load_dataset(dataset_name)
    pos_label = get_pos_label(dataset_name)

    # stratify=y preserves the class ratio in both splits -- critical
    # given Climate Crashes' 91.48%/8.52% imbalance; a plain random split
    # risks an unlucky test set with almost no rare-class examples.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Halts here, before any Trainer/Gemini involvement, if the split is
    # statistically unreliable for this dataset's class balance.
    validate_split(y_train, y_test, pos_label, min_count=5)

    trainer = Trainer()
    bound_train = partial(trainer.train_model, X_train=X_train, y_train=y_train)
    bound_evaluate = partial(
        trainer.evaluate_model, X_test=X_test, y_test=y_test, pos_label=pos_label
    )

    # 2d, RESOLVED (2026-07-17, Option B): start from TOOL_FUNCTIONS
    # (tools.py) -- its own docstring says it exists so dispatch-table
    # wiring here can reuse it -- then override just the 2 of 5 entries
    # that TOOL_FUNCTIONS can't supply working versions of on its own:
    # train_model/evaluate_model there are bare, unbound, and still
    # raise NotImplementedError, since the real per-run Trainer/X_train/
    # y_train/pos_label binding can only happen here, inside a live run.
    # list_available_models and both Category B tools need no binding at
    # all, so TOOL_FUNCTIONS' versions of those three are used as-is.
    #
    # RISK, flagged for the session summary/context/README: if
    # train_model or evaluate_model are ever renamed in tools.py, these
    # two override lines would silently stop overriding anything -- the
    # stale, NotImplementedError-raising version from TOOL_FUNCTIONS
    # would quietly take their place instead, undetected until a real
    # run hit it. MITIGATION (not built this session): extend
    # test_tools.py's drift-check philosophy with an assertion that
    # these two override keys still exist in TOOL_FUNCTIONS.
    dispatch_table = {
        **TOOL_FUNCTIONS,
        "train_model": bound_train,
        "evaluate_model": bound_evaluate,
    }

    return DispatchResult(dispatch_table=dispatch_table, X=X, y=y)


def _format_initial_context(facts: dict[str, Any], optimization_target: str) -> str:
    """Turns inspect_dataset()'s facts + the chosen optimization target
    into the single prompt string run_agent_loop sends to Gemini first.

    NEW (2026-07-17 session, Step 3). Kept as its own small function
    (rather than inlined into run_session) so its output can be checked
    directly in a test without needing a real dataset/Trainer/API call --
    same "isolate the pure, checkable part" instinct as validate_split
    and validate_hyperparameters elsewhere in this project.

    2b, RESOLVED (2026-07-17, Option 2): optimization_target is an
    explicit argument here, never hardcoded per dataset. This was
    DATA_SCIENCE_ANALYSIS.md's most actionable finding -- the agent's
    model choice was previously undetermined because nothing ever told
    Gemini what to optimize for. Kept as a plain string parameter (not
    baked into dataset facts) to preserve this project's existing
    fact-vs-judgment split: which class is "positive" is a fact
    (pos_label, resolved in dataset.py); what to optimize for is a
    judgment call, made once per run, by whoever calls run_session --
    not something dataset.py or build_dispatch_table should silently
    decide on the caller's behalf.
    """
    import json

    facts_json = json.dumps(facts, indent=2)
    return (
        "You are helping select and tune a scikit-learn classifier for "
        "the following dataset. Here are the dataset's facts -- computed "
        "deterministically, not your judgment to make:\n\n"
        f"{facts_json}\n\n"
        f"Optimization target: {optimization_target}. Propose and evaluate "
        "models with this target explicitly in mind, using "
        "list_available_models, record_model_proposal, train_model, "
        "evaluate_model, and record_convergence_decision as needed."
    )


def run_session(
    dataset_name: str,
    optimization_target: str,
    *,
    random_state: int = 42,
    model: str = DEFAULT_MODEL,
    max_iterations: int = MAX_ITERATIONS,
    log_iterations: bool = False,
) -> dict[str, Any]:
    """Runs one full agent session end-to-end: builds the dispatch table
    and dataset facts, assembles the first prompt, then hands off to
    gemini_client.run_agent_loop and returns its result.

    NEW (2026-07-17 session, Step 3) -- this is the piece that was
    "genuinely still missing" per the corrected TODO this session
    replaced: everything it calls (build_dispatch_table, inspect_dataset,
    run_agent_loop) already existed and worked in isolation; nothing
    previously called them together with real, non-hand-built inputs.
    run_smoke_test.py did this by hand, with its own duplicate
    load_dataset() call -- this function is the real, committed
    replacement for that pattern (see DispatchResult's docstring above
    for why the duplicate load is now avoidable).

    optimization_target is deliberately a plain parameter here, not
    something this function prompts for itself (no input() call) -- so
    it stays callable directly from a test with a hardcoded string.
    Whichever CLI entry point (main.py, not built this session) is
    responsible for actually asking the person what to optimize for and
    passing the answer in here.

    log_iterations: off by default (False), passed straight through to
    run_agent_loop unchanged -- see that function's docstring for what
    gets logged. run_session makes no logging decisions of its own; it
    just forwards the caller's choice.

    Does NOT touch either open 07-13 judgment call (single-function-
    call-per-turn assumption; record_convergence_decision's result not
    echoed back on stop) -- both live entirely inside run_agent_loop,
    untouched by this function, per this session's explicit Step 2.5
    agreement to leave them deferred.
    """
    result = build_dispatch_table(dataset_name, random_state=random_state)
    facts = inspect_dataset(result.X, result.y)
    initial_context = _format_initial_context(facts, optimization_target)

    return run_agent_loop(
        result.dispatch_table,
        initial_context,
        model=model,
        max_iterations=max_iterations,
        log_iterations=log_iterations,
    )

# Still open, unchanged by this session (deferred deliberately -- see
# Step 2.5 discussion): the human-in-the-loop extension point, inside
# gemini_client.py's run_agent_loop, between record_model_proposal's
# return and train_model's call. Not touched by run_session above --
# run_session only wires existing pieces together, it doesn't change
# what happens inside the loop itself.