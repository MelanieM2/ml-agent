# trainer.py — structural skeleton only, no real sklearn fitting/metrics yet

import uuid
from typing import Any
from ml_agent.tools import list_available_models

class Trainer:
    """Owns the model_ref -> fitted model mapping for one agent run.

    This is the encapsulated 'filing cabinet': one Trainer instance holds
    its own private dict of trained models. Nothing outside this class can
    read or mutate that dict directly — the only way in or out is through
    train_model() and evaluate_model() on this specific instance.
    """

    def __init__(self) -> None:
        # self._models is the actual "cabinet." Keys are UUID strings
        # (Gemini-facing model_ref values); values are small dicts holding
        # the fitted model plus enough metadata to make debugging sane.
        self._models: dict[str, dict[str, Any]] = {}

    def train_model(
        self, model_type: str, hyperparameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Fits model_type with hyperparameters, stores it, returns a ref.

        Returns only {"model_ref": ...} — Gemini is never shown the fitted
        object itself, only an id it can pass back into evaluate_model later.
        """
        schema = list_available_models()["models"]
        validate_hyperparameters(model_type, hyperparameters, schema)
        # ^ raises ValueError immediately if Gemini's arguments don't match
        #   the schema — nothing below this line runs on invalid input.

        # TODO (real logic, next session):
        #   1. instantiate the matching sklearn estimator
        #   2. call .fit(X_train, y_train)

        model_ref = uuid.uuid4().hex
        # Human-readable label for logs/debugging only — never used as a
        # lookup key, never shown to Gemini. Pure convenience.
        label = f"{model_type}_{model_ref[:6]}"

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
        """Looks up model_ref, computes metrics against pos_label.

        pos_label is keyword-only and never comes from Gemini's tool
        schema (see tools.py) — it's supplied internally by whichever
        binding mechanism agent.py uses (this is open question #2,
        still to be decided).
        """
        if model_ref not in self._models:
            # A clear, named error beats a bare KeyError surfacing deep
            # inside dispatch code with no context about which ref failed.
            raise ValueError(f"Unknown model_ref: {model_ref!r}")

        entry = self._models[model_ref]
        # TODO (real logic, next session):
        #   1. entry["model"].predict(X_test)
        #   2. accuracy_score / precision_score / recall_score / f1_score /
        #      confusion_matrix, all computed using pos_label
        #   3. return plain Python types only (no numpy scalars/arrays —
        #      Gemini's tool-result channel needs JSON-serializable output)

        return {
            "model_ref": model_ref,
            "model_type": entry["model_type"],
            "pos_label": pos_label,
            # real metric keys will go here once computation exists
        }
    

def validate_hyperparameters(
    model_type: str,
    hyperparameters: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Checks Gemini's proposed model_type/hyperparameters against the
    known schema returned by list_available_models(), before anything
    gets instantiated. Raises ValueError with a specific, named reason
    on any mismatch; returns None (silently) if everything checks out.

    `schema` is expected to be list_available_models()["models"] —
    the "models" sub-dict, not the full return value.

    This function never touches sklearn or self._models — it's a pure
    check, deliberately kept separate from Trainer so it's testable
    with zero setup (no Trainer instance needed at all).
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