# trainer.py — real sklearn fitting/evaluation logic

import uuid
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from ml_agent.tools import list_available_models

# Translates the string names in tools.py's schema (schema[model_type]
# ["sklearn_class"]) into the actual sklearn classes. One dict entry per
# supported model_type, so train_model itself never needs an if/elif
# chain -- adding a new model_type later means adding one line here and
# one entry in tools.py's schema, nothing else.
ESTIMATOR_REGISTRY: dict[str, type] = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "SVC": SVC,
}


class Trainer:
    """Owns the model_ref -> fitted model mapping for one agent run.

    This is the encapsulated 'filing cabinet': one Trainer instance holds
    its own private dict of trained models. Nothing outside this class can
    read or mutate that dict directly — the only way in or out is through
    train_model() and evaluate_model() on this specific instance.
    """

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] = {}

    def train_model(
        self,
        model_type: str,
        hyperparameters: dict[str, Any],
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict[str, Any]:
        """Fits model_type with hyperparameters, stores it, returns a ref.

        X_train/y_train are keyword-only, bound in via functools.partial
        in build_dispatch_table (agent.py) — Gemini only ever supplies
        model_type and hyperparameters.

        Returns only {"model_ref": ...} — Gemini is never shown the fitted
        object itself, only an id it can pass back into evaluate_model later.
        """
        schema = list_available_models()["models"]
        validate_hyperparameters(model_type, hyperparameters, schema)
        # ^ raises ValueError immediately if Gemini's arguments don't match
        #   the schema — nothing below this line runs on invalid input.

        # Look up which sklearn class this model_type maps to (a string,
        # per tools.py's schema), then find the actual class object.
        sklearn_class_name = schema[model_type]["sklearn_class"]
        estimator_class = ESTIMATOR_REGISTRY[sklearn_class_name]

        # Instantiate with Gemini's chosen hyperparameters unpacked as
        # keyword arguments (e.g. LogisticRegression(C=1.0, class_weight=None))
        # -- valid because validate_hyperparameters already confirmed every
        # key/value pair is legitimate for this model_type.
        estimator = estimator_class(**hyperparameters)
        estimator.fit(X_train, y_train)

        model_ref = uuid.uuid4().hex
        label = f"{model_type}_{model_ref[:6]}"

        self._models[model_ref] = {
            "model": estimator,  # the real fitted estimator, not None
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "label": label,
        }
        return {"model_ref": model_ref}

    def evaluate_model(
        self,
        model_ref: str,
        *,
        pos_label: int,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """Looks up model_ref, computes metrics against pos_label."""
        if model_ref not in self._models:
            raise ValueError(f"Unknown model_ref: {model_ref!r}")

        entry = self._models[model_ref]
        estimator = entry["model"]
        y_pred = estimator.predict(X_test)

        # average="binary" + pos_label tells sklearn exactly which class's
        # precision/recall/f1 to report -- correct for binary classification
        # where one class is meaningfully "the one we care about" (the rare
        # failure class for Climate Crashes; benign for Breast Cancer).
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, pos_label=pos_label, average="binary", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, pos_label=pos_label, average="binary", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, pos_label=pos_label, average="binary", zero_division=0
        )

        # Explicit labels=[other_class, pos_label] forces pos_label into the
        # second row/column consistently -- confusion_matrix's default
        # ascending-sort ([0, 1]) would silently assume class 0 is "negative,"
        # which is wrong for Breast Cancer (pos_label=1 is benign, the good
        # outcome, not 0). This keeps the matrix's meaning stable across
        # both datasets regardless of which class happens to be pos_label.
        other_label = 1 - pos_label
        matrix = confusion_matrix(y_test, y_pred, labels=[other_label, pos_label])

        return {
            "model_ref": model_ref,
            "model_type": entry["model_type"],
            "pos_label": pos_label,
            # Every value below is cast to a native Python type -- sklearn's
            # metric functions return numpy scalars (float64/int64), and
            # confusion_matrix returns a numpy ndarray. Gemini's tool-result
            # channel needs JSON-serializable output; numpy types raise
            # TypeError under json.dumps() (same concern flagged for
            # validate_split's counts in agent.py).
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": matrix.tolist(),
        }


def validate_hyperparameters(
    model_type: str,
    hyperparameters: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """(unchanged — see prior version)"""
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