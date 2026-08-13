# trainer.py — real sklearn fitting/evaluation logic

import uuid
import warnings
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
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Fits model_type with hyperparameters, stores it, returns a ref.

        X_train/y_train are keyword-only, bound in via functools.partial
        in build_dispatch_table (agent.py) -- Gemini only ever supplies
        model_type and hyperparameters.

        random_state is keyword-only, bound the same way, reusing the
        exact same seed value already used for the train/test split
        (the --random-state CLI flag) rather than introducing a second,
        separate one -- a deliberate single-shared-seed design, chosen
        over a two-seed alternative that would isolate split-variance
        from model-internal-variance, since this project's goal is a
        working agent, not a formal variance study. Passed directly into
        every estimator's constructor below. Deliberately NOT exposed as
        a Gemini-tunable hyperparameter in tools.py's schema: it's a
        reproducibility knob, not a modeling choice the agent should be
        making run to run.

        Of the three supported estimators, random_state is verified to
        change fit behavior for RandomForestClassifier only (bagging/
        feature sampling) -- previously left unseeded, which meant two
        runs with identical hyperparameters could produce different
        confusion matrices purely from an unseeded draw against numpy's
        global RNG; confirmed fixed via matching post-fix runs.
        LogisticRegression's lbfgs solver is a deterministic optimizer on
        a fixed convex objective regardless of seeding, so it was already
        reproducible before this parameter existed. SVC only uses
        random_state when probability=True (it seeds the internal
        cross-validation used for probability estimates); tools.py's
        schema never exposes probability as tunable, so it stays False
        here, making random_state currently a no-op for SVC specifically
        -- not independently verified to matter, unlike RandomForestClassifier.
        Passed uniformly to all three regardless, for constructor-interface
        consistency and forward-compatibility (e.g. if probability=True or
        a different SVM solver is ever exposed as tunable later), not
        because it's confirmed to affect every estimator equally today.
        Two narrower alternatives -- skip random_state for SVC
        specifically, or expose probability as a tunable SVM
        hyperparameter so the seed actually does something there --
        were considered and deliberately deferred rather than adopted.

        Returns {"model_ref": ..., "warnings": [...]} -- Gemini is never
        shown the fitted object itself, only an id it can pass back into
        evaluate_model later, plus a (possibly empty) list of any
        warnings raised during fit.

        "warnings" is always present in the return value, even when
        empty -- the same consistent-key-set-regardless-of-branch
        convention used for the per-iteration log in gemini_client.py.
        Captures every warning category raised during fit, not just
        ConvergenceWarning specifically -- a deliberate choice to keep
        this general-purpose rather than hardcoding around the one
        category actually observed so far. simplefilter("always") is
        required inside the catch_warnings block because a warning that
        already fired once earlier in the same process would otherwise
        be silently deduplicated by Python's default once-per-location
        behavior -- without it, a second run hitting the identical
        warning could go undetected.
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
        # key/value pair is legitimate for this model_type — PLUS
        # random_state, always, for every model_type, regardless of
        # whether that particular model's stochasticity would actually
        # matter for a given run. Safe to unpack alongside
        # **hyperparameters: schema never defines "random_state" as a
        # tunable param (see tools.py), so Gemini can never supply one
        # and collide with this.
        estimator = estimator_class(**hyperparameters, random_state=random_state)

        # Wrap ONLY the fit call — not the whole function — so warnings
        # from schema validation or anything else above/below this block
        # are never misattributed to the fit itself.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X_train, y_train)

        fit_warnings = [
            {"category": w.category.__name__, "message": str(w.message)}
            for w in caught
        ]

        model_ref = uuid.uuid4().hex
        label = f"{model_type}_{model_ref[:6]}"

        self._models[model_ref] = {
            "model": estimator,  # the real fitted estimator, not None
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "label": label,
        }
        return {"model_ref": model_ref, "warnings": fit_warnings}

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
    """Checks model_type and every hyperparameter against schema, raising
    ValueError on the first mismatch found.

    Kept as a standalone function rather than a Trainer method -- it's a
    pure function of its three arguments, with no instance state needed,
    so it can be tested directly without constructing a Trainer or
    fitting anything. Exists so a hallucinated model_type, or a
    hyperparameter value outside its documented range/choices, fails
    immediately with a specific, readable error, instead of failing deep
    inside whatever code instantiates the sklearn estimator -- a
    confusing failure far from its actual cause. schema is
    list_available_models()'s own return value, so this checks Gemini's
    choices against the exact same information Gemini itself was shown.

    Numeric ranges are treated as inclusive on both ends (low <= value
    <= high), matching the convention stated in list_available_models's
    own docstring -- "range" is a project-specific schema field, not a
    real JSON Schema keyword, so nothing enforces this convention except
    both sides agreeing to it explicitly.
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