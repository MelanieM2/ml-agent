# agent.py — wiring skeleton only; no orchestration loop yet
from functools import partial
from typing import Any, Callable

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_agent.trainer import Trainer, validate_hyperparameters
from ml_agent.tools import (
    list_available_models,
    record_model_proposal,
    record_convergence_decision,
)
from ml_agent.dataset import load_dataset, get_pos_label


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


def build_dispatch_table(
    dataset_name: str, random_state: int = 42
) -> dict[str, Callable[..., Any]]:
    """Builds one run's complete tool dispatch table for gemini_client.py.

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

    The returned dict maps each of the five tool names exactly as
    Gemini will return them to a ready-to-call callable:
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

    return {
        "list_available_models": list_available_models,
        "train_model": bound_train,
        "evaluate_model": bound_evaluate,
        "record_model_proposal": record_model_proposal,
        "record_convergence_decision": record_convergence_decision,
    }


# TODO (later, once gemini_client.py exists):
#   - the actual agent loop: call Gemini, read back which tool it picked,
#     look it up in dispatch_table, call it, feed the result back to Gemini
#   - iteration count + max-iterations guard
#   - the human-in-the-loop extension point, precisely at the handoff
#     between record_model_proposal's return and train_model's call
#     (unchanged from before -- the split itself does NOT get a human
#     checkpoint; deferred deliberately, see this session's discussion)