"""Gemini native function-calling tool definitions for ml-agent.

Two categories, deliberately kept distinct (see AGENTS.md once written):

  CATEGORY A - execution tools. Real scikit-learn side effects. These never
  see Gemini's "reasoning" directly - they receive plain arguments and run,
  the same way any ordinary function call would, regardless of who or what
  supplied those arguments.

  CATEGORY B - structured-decision tools. No execution happens here at all.
  Calling one of these just captures Gemini's chosen arguments as a dict -
  the tool exists purely to force free-form reasoning into parseable JSON.

The boundary between them is also where a human-in-the-loop confirmation
step can be inserted later (e.g. before train_model actually runs) without
touching either category's internals - see AGENTS.md "extension points".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# CATEGORY A - execution tools
# ---------------------------------------------------------------------------


def list_available_models() -> dict[str, Any]:
    """Returns supported model types and their tunable hyperparameter ranges.

    No parameters - this is a pure "what's on the menu" query. Gemini should
    call this before proposing a model, so that record_model_proposal's
    arguments stay within what train_model can actually accept.
    """
    return {
        "models": {
            "logistic_regression": {
                "sklearn_class": "LogisticRegression",
                "hyperparameters": {
                    "C": {
                        "type": "float",
                        "range": [0.001, 100.0],
                        "default": 1.0,
                        "description": (
                            "Inverse regularization strength; "
                            "smaller = stronger regularization."
                        ),
                    },
                    "class_weight": {
                        "type": "choice",
                        "options": [None, "balanced"],
                        "default": None,
                        "description": (
                            "'balanced' reweights classes inversely to "
                            "frequency - relevant for imbalanced data."
                        ),
                    },
                },
            },
            "random_forest": {
                "sklearn_class": "RandomForestClassifier",
                "hyperparameters": {
                    "n_estimators": {
                        "type": "int",
                        "range": [10, 500],
                        "default": 100,
                        "description": "Number of trees.",
                    },
                    "max_depth": {
                        "type": "int_or_null",
                        "range": [1, 50],
                        "default": None,
                        "description": "Max tree depth; null means unlimited.",
                    },
                    "class_weight": {
                        "type": "choice",
                        "options": [None, "balanced"],
                        "default": None,
                    },
                },
            },
            "svm": {
                "sklearn_class": "SVC",
                "hyperparameters": {
                    "C": {
                        "type": "float",
                        "range": [0.001, 100.0],
                        "default": 1.0,
                    },
                    "kernel": {
                        "type": "choice",
                        "options": ["linear", "rbf", "poly"],
                        "default": "rbf",
                    },
                    "class_weight": {
                        "type": "choice",
                        "options": [None, "balanced"],
                        "default": None,
                    },
                },
            },
        }
    }


def train_model(model_type: str, hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Fits `model_type` with `hyperparameters` on the current dataset.

    Real implementation lives in trainer.py - this is the tool-facing
    signature/contract only. Returns a model reference id, never the
    fitted object itself (Gemini only ever sees ids and numbers).
    """
    raise NotImplementedError("Wired up once trainer.py exists.")


def evaluate_model(model_ref: str, *, pos_label: int) -> dict[str, Any]:
    """Computes accuracy/precision/recall/f1/confusion_matrix for model_ref.

    `pos_label` is NOT part of the Gemini-visible tool schema (see the
    schema dict below - only `model_ref` appears there). It is supplied
    internally by whoever constructs this callable for dispatch (agent.py
    already knows which dataset is active for the whole run), via
    functools.partial or an equivalent closure. Gemini must never be asked
    to guess which class is positive - that's a documented dataset fact,
    not a judgment call.
    """
    raise NotImplementedError("Wired up once trainer.py exists.")


# ---------------------------------------------------------------------------
# CATEGORY B - structured-decision tools (no execution; pure capture)
# ---------------------------------------------------------------------------


def record_model_proposal(
    model_type: str, hyperparameters: dict[str, Any], reasoning: str
) -> dict[str, Any]:
    """Captures a proposed model + hyperparameters + reasoning as a dict.

    Nothing is trained here. This is the Category B -> Category A handoff
    point: agent.py reads this dict's model_type/hyperparameters and passes
    them, verbatim, as train_model's arguments. (A human-in-the-loop
    confirmation step would slot in right here, between this return value
    and the train_model call.)
    """
    return {
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "reasoning": reasoning,
    }


def record_convergence_decision(
    continue_iterating: bool, reasoning: str, next_step_hint: str = ""
) -> dict[str, Any]:
    """Captures the stop-or-continue decision + reasoning as a dict.

    Also pure capture, no execution. agent.py reads `continue_iterating`
    to decide whether to loop back to record_model_proposal or produce
    the final report.
    """
    return {
        "continue_iterating": continue_iterating,
        "reasoning": reasoning,
        "next_step_hint": next_step_hint,
    }


# ---------------------------------------------------------------------------
# Gemini tool-schema declarations (the JSON-schema side of each function
# above). gemini_client.py will hand this list to the API alongside the
# actual callables for dispatch.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_available_models",
        "description": (
            "Returns the set of model types currently supported by this "
            "system, along with their tunable hyperparameters and valid "
            "ranges/choices for each. Call this before proposing a model, "
            "so proposals stay within what can actually be trained."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "train_model",
        "description": (
            "Fits a model of the given type with the given hyperparameters "
            "on the current dataset. Returns a model reference id, not the "
            "fitted object itself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": "Must exactly match a name returned by list_available_models.",
                },
                "hyperparameters": {
                    "type": "object",
                    "description": (
                        "Key-value pairs of hyperparameter name to chosen "
                        "value. Keys and value ranges must match what "
                        "list_available_models reported for this model_type."
                    ),
                },
            },
            "required": ["model_type", "hyperparameters"],
        },
    },
    {
        "name": "evaluate_model",
        "description": (
            "Computes accuracy, precision, recall, f1, and the confusion "
            "matrix for a previously trained model, using the dataset's "
            "established positive class."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_ref": {
                    "type": "string",
                    "description": "The reference id returned by a prior train_model call.",
                }
            },
            "required": ["model_ref"],
        },
    },
    {
        "name": "record_model_proposal",
        "description": (
            "Use this to formally propose the next model to try. This does "
            "not train anything by itself - it records your choice and "
            "reasoning so it can be reviewed before training proceeds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": "Must exactly match a name from list_available_models.",
                },
                "hyperparameters": {
                    "type": "object",
                    "description": (
                        "Chosen hyperparameter values, within the ranges "
                        "list_available_models reported."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Why this model and these hyperparameters, given "
                        "the dataset characteristics and the optimization "
                        "target (e.g. recall)."
                    ),
                },
            },
            "required": ["model_type", "hyperparameters", "reasoning"],
        },
    },
    {
        "name": "record_convergence_decision",
        "description": (
            "Use this after reviewing evaluate_model's results to decide "
            "whether to stop iterating or try another model. This is a "
            "decision record, not an action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "continue_iterating": {
                    "type": "boolean",
                    "description": (
                        "True to propose another model, false to stop and "
                        "report the best one found so far."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Why this decision, referencing the specific metric "
                        "values observed."
                    ),
                },
                "next_step_hint": {
                    "type": "string",
                    "description": (
                        "Optional: a brief hint about what to try "
                        "differently next, if continue_iterating is true. "
                        "Empty string if not applicable."
                    ),
                },
            },
            "required": ["continue_iterating", "reasoning"],
        },
    },
]