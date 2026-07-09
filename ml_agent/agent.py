# agent.py — wiring skeleton only; no orchestration loop yet

from functools import partial
from typing import Any, Callable

from ml_agent.trainer import Trainer, validate_hyperparameters
from ml_agent.tools import (
    list_available_models,
    record_model_proposal,
    record_convergence_decision,
)
from ml_agent.dataset import get_pos_label


def build_dispatch_table(dataset_name: str) -> dict[str, Callable[..., Any]]:
    """Builds one run's complete tool dispatch table for gemini_client.py.

    Given the active dataset's name, this resolves everything that's
    dataset- or run-specific exactly once, here, so nothing downstream
    ever needs to rediscover it:

      - one Trainer instance (the private model_ref -> fitted model
        store for this run; see Trainer's own docstring)
      - this dataset's pos_label, looked up via get_pos_label() and
        pre-filled into evaluate_model via functools.partial, so
        Gemini's dispatch never sees pos_label as an argument at all

    The returned dict maps each of the five tool names exactly as
    Gemini will return them to a ready-to-call callable:
      - Category A (list_available_models, train_model, evaluate_model)
        are real executions - train_model validates its own arguments
        against list_available_models()'s schema before touching sklearn.
      - Category B (record_model_proposal, record_convergence_decision)
        are pure structured-decision capture, stateless, wired in as-is.

    gemini_client.py's dispatch logic can then do
    dispatch_table[tool_name](**args) without knowing dataset_name,
    pos_label, or the Trainer instance exist.
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


# TODO (later, once gemini_client.py exists):
#   - the actual agent loop: call Gemini, read back which tool it picked,
#     look it up in dispatch_table, call it, feed the result back to Gemini
#   - iteration count + max-iterations guard
#   - the human-in-the-loop extension point, precisely at the handoff
#     between record_model_proposal's return and train_model's call