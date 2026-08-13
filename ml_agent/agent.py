# agent.py — orchestration layer: builds each run's tool dispatch table
# (build_dispatch_table) and runs a full agent session end-to-end
# (run_session).
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

    Kept standalone (pure function of its four arguments, no Trainer/
    dataset setup needed) so it's independently testable -- same
    rationale as validate_hyperparameters in trainer.py.

    Guards against unreliable recall/precision on a near-empty rare
    class: with too few examples of the class actually being predicted,
    a single misclassification swings the metric by a large, misleading
    amount. A threshold of 5 keeps that swing below ~20% (1/5) per
    example, while still catching a genuinely broken or collapsed split
    (e.g. stratify silently failing to preserve class balance).

    Only the rare class is checked -- given these datasets' ratios
    (Climate Crashes 91.48%/8.52%, Breast Cancer 66%/34%), the majority
    class is never close to this floor under a stratified split, so
    checking it would add code without adding safety.
    """
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

    Carries dispatch_table and (X, y) only -- not a formed prompt string.
    Callers need this same (X, y) to build initial_context via
    inspect_dataset(), without loading the dataset a second time (costly
    for Climate Crashes, which requires an OpenML network fetch).
    Building the prompt itself stays the caller's job.

    A frozen dataclass rather than a bare tuple, matching this project's
    convention for named return values (see DatasetSpec in dataset.py).
    Lives here rather than in dataset.py because it describes
    build_dispatch_table's return shape specifically, not a general
    dataset-registry fact.
    """

    dispatch_table: dict[str, Callable[..., Any]]
    X: pd.DataFrame
    y: pd.Series


def build_dispatch_table(
    dataset_name: str, random_state: int = 42
) -> DispatchResult:
    """Builds one run's complete tool dispatch table for gemini_client.py,
    plus the (X, y) it was built from.

    Given the active dataset's name, resolves everything dataset- or
    run-specific exactly once, so nothing downstream has to rediscover
    it:
      - the full dataset (X, y) and its pos_label
      - a stratified train/test split
      - a hard validation gate (validate_split) on that split, before
        Trainer is even constructed
      - one Trainer instance for this run
      - train_model/evaluate_model pre-bound to X_train/y_train/X_test/
        y_test/pos_label via functools.partial, so Gemini's dispatch
        never sees any of this data as an argument

    random_state is reused for both the train/test split and every
    model's own internal randomness (e.g. RandomForestClassifier's
    bagging), rather than using two separate seeds -- otherwise two runs
    with identical hyperparameters could still produce different
    confusion matrices.

    Starts from TOOL_FUNCTIONS (tools.py) and overrides only train_model
    and evaluate_model, the two tools that need live per-run binding
    (Trainer, X_train, pos_label, etc.) and are otherwise unbound stubs.
    The other three tools (list_available_models, and the two Category B
    decision-capture tools) need no binding and are used as-is.

    The override below replaces those two entries by literal string key,
    not by looking them up in TOOL_FUNCTIONS first -- so renaming
    train_model or evaluate_model inside tools.py can't silently make
    this override stop applying. What a rename *can* still break --
    TOOL_SCHEMAS naming a tool that no longer exists in TOOL_FUNCTIONS --
    is caught separately, by test_tools.py's schema/function-presence
    check.
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
    bound_train = partial(
        trainer.train_model,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
    )
    bound_evaluate = partial(
        trainer.evaluate_model, X_test=X_test, y_test=y_test, pos_label=pos_label
    )

    # Overridden by literal key, not a TOOL_FUNCTIONS lookup -- see
    # docstring above for why that matters.
    dispatch_table = {
        **TOOL_FUNCTIONS,
        "train_model": bound_train,
        "evaluate_model": bound_evaluate,
    }

    return DispatchResult(dispatch_table=dispatch_table, X=X, y=y)


def _format_initial_context(facts: dict[str, Any], optimization_target: str) -> str:
    """Turns inspect_dataset()'s facts + the chosen optimization target
    into the single prompt string run_agent_loop sends to Gemini first.

    Kept as its own small function (rather than inlined into run_session)
    so its output can be checked directly in a test, without a real
    dataset/Trainer/API call -- same isolate-the-pure-part rationale as
    validate_split and validate_hyperparameters elsewhere in this
    project.

    optimization_target is an explicit argument here, never hardcoded
    per dataset. This keeps a deliberate distinction: which class counts
    as "positive" is a fixed fact about a dataset (pos_label, resolved in
    dataset.py), but what to optimize for -- recall, precision, F1 -- is
    a judgment call about what mistake matters more in a given run, made
    by whoever calls run_session, not decided on their behalf here.
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

    optimization_target is a plain parameter, not something this
    function prompts for itself -- so it stays callable directly from a
    test with a hardcoded string. A CLI entry point is responsible for
    asking the person what to optimize for and passing the answer in
    here.

    random_state is passed straight through to build_dispatch_table; see
    that function's docstring for what it's used for beyond the
    train/test split.

    log_iterations is off by default and forwarded to run_agent_loop
    unchanged -- run_session makes no logging decisions of its own.

    Two known limitations live inside run_agent_loop itself, untouched by
    run_session: it assumes Gemini returns at most one function call per
    turn (parallel function calls aren't handled), and it doesn't echo
    record_convergence_decision's result back to Gemini when the loop
    stops. Both are open, deliberately deferred design questions in
    gemini_client.py, not something run_session's own wiring affects.
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

# Still open, deferred deliberately: the human-in-the-loop extension
# point inside gemini_client.py's run_agent_loop, between
# record_model_proposal's return and train_model's call. run_session
# only wires existing pieces together -- it doesn't touch what happens
# inside the loop itself.
