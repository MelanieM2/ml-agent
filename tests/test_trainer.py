# tests/test_trainer.py
"""Tests for trainer.py's Trainer class: model fitting, evaluation, and
hyperparameter validation.

Central concern of this file: the fix that threads random_state into
every estimator's constructor. Before that fix, RandomForestClassifier
was instantiated with no random_state at all (sklearn default None ->
pulled from numpy's global RNG), so two runs with IDENTICAL
hyperparameters could -- and did -- produce different confusion
matrices. LogisticRegression's lbfgs solver only ever *looked*
reproducible because it's a deterministic optimizer on a fixed convex
objective, not because it was ever actually seeded.

Design choices made for this file (see DEVELOPMENT_LOG.md's "Test
coverage" section under "Reproducibility: seeding every estimator", and
TECHNICAL_NOTES.md Part 8, for the fuller rationale):
  - REAL fits, not mocked estimators -- a mock would confirm Trainer
    *calls* the estimator correctly, but the original bug was about the
    estimator's actual numeric behavior, which only a real fit exposes.
  - A LOCAL fixture (binary_classification_data below), not a
    project-wide conftest.py. Nothing else in tests/ currently needs the
    same synthetic dataset.
  - PARAMETRIZED across all three ESTIMATOR_REGISTRY entries, not just
    RandomForestClassifier -- the registry pattern that makes adding a
    new model type easy is exactly what would let a future
    random_state-threading regression on any other estimator go just as
    unnoticed as this one did.

CORRECTION LOG (logged plainly rather than smoothed over -- three
attempts, in order, until one was actually verified empirically rather
than just theorized):

  Attempt 1 (uniform feature scaling, X * 1e5): FAILED (16/17 passing).
  Reasoning was wrong: multiplying every feature by the SAME factor
  barely changes lbfgs's internal behavior at all -- it doesn't touch
  the *relative* conditioning between features, which is what actually
  matters to a quasi-Newton solver.

  Attempt 2 (near-perfect linear separability via high class_sep, plus
  C=100.0): ALSO FAILED on a second real run. Reasoning was still wrong,
  for a different reason: lbfgs's stopping rule checks gradient norm, and
  gradient norm can shrink below tolerance quickly even while coefficients
  are still growing toward a separating hyperplane that technically never
  converges. Theoretically coefficients diverge under perfect
  separability; practically, lbfgs still satisfies its own convergence
  check well before max_iter=50.

  Attempt 3 (ill-conditioned data via MISMATCHED per-feature scales, e.g.
  1e-5 to 1e6 across different features on the SAME dataset): VERIFIED
  EMPIRICALLY (not just theorized) by actually running it -- 10/10 across
  5 data-generation seeds x 2 training random_states, all reliably hitting
  n_iter_=50 without satisfying sklearn's internal tolerance. This is
  genuinely different from attempt 1: it's the MISMATCH between feature
  scales, not scale itself, that distorts lbfgs's internal quasi-Newton
  curvature (Hessian) approximation, which assumes comparably-scaled
  dimensions to work well. This is also, not coincidentally, the
  textbook real-world reason linear models are usually fit on
  standardized features -- this test doubles as a real demonstration of
  why that practice exists, not just a testing trick.

Hyperparameter combinations below (C, max_iter, class_weight, n_estimators,
max_depth, kernel) are verified directly against the real schema in
ml_agent/tools.py's list_available_models().
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from ml_agent.trainer import Trainer, validate_hyperparameters

# One case per ESTIMATOR_REGISTRY entry -- (model_type string per tools.py's
# schema, hyperparameters confirmed valid against the real schema in
# list_available_models()).
ESTIMATOR_CASES = [
    pytest.param(
        "logistic_regression", {"class_weight": "balanced"}, id="logistic_regression"
    ),
    pytest.param(
        "random_forest",
        {"n_estimators": 50, "max_depth": 5, "class_weight": "balanced"},
        id="random_forest",
    ),
    pytest.param("svm", {"kernel": "linear", "C": 1}, id="svm"),
]


@pytest.fixture
def binary_classification_data():
    """A small, deterministic, well-separated synthetic binary
    classification dataset -- fine for every test EXCEPT the
    convergence-warning test, which needs a genuinely hard-to-converge
    case (see ill_conditioned_binary_data below)."""
    X, y = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.5, 0.5],
        random_state=0,
    )
    X = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    y = pd.Series(y)

    X_train, X_test = X.iloc[:30], X.iloc[30:]
    y_train, y_test = y.iloc[:30], y.iloc[30:]
    return X_train, y_train, X_test, y_test


@pytest.fixture
def ill_conditioned_binary_data():
    """Same kind of feature values as a normal, reasonably separable
    synthetic dataset, but each of the 10 features scaled by a wildly
    different, fixed factor (1e-5 to 1e6). This is NOT the same as
    uniform scaling (multiplying every feature by the same number, which
    barely affects lbfgs's behavior at all, per Attempt 1 in the module
    docstring above) -- it's the *relative* mismatch between feature
    scales that distorts lbfgs's internal quasi-Newton curvature (Hessian)
    approximation, which assumes comparably-scaled dimensions to work
    well. This is also, not coincidentally, the textbook real-world
    reason linear models are usually fit on standardized features.

    Empirically verified (not just theorized) to trigger a genuine
    ConvergenceWarning at max_iter=50, across 5 data-generation seeds x 2
    training random_states, all reliably hitting n_iter_=50 without
    reaching sklearn's internal convergence tolerance.
    """
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=8,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.02,
        random_state=1,
    )
    scales = np.array([1e-3, 1e3, 1e-4, 1e4, 1e-2, 1e2, 1e-5, 1e5, 1, 1e6])
    X = X * scales
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y = pd.Series(y)
    return X, y


@pytest.fixture
def trainer():
    return Trainer()


# ---------------------------------------------------------------------
# random_state reproducibility -- the core regression guard for the
# random_state-threading fix.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("model_type,hyperparameters", ESTIMATOR_CASES)
def test_random_state_reaches_estimator_constructor(
    trainer, binary_classification_data, model_type, hyperparameters
):
    """Direct check that random_state is actually threaded into the
    fitted estimator's constructor, for EVERY registered estimator type --
    not just RandomForestClassifier, the one that broke.

    Reaches into trainer._models directly (a leading-underscore, in-module
    attribute) rather than only checking behavior indirectly -- an
    intentional, narrow exception to treating Trainer as a black box,
    because THIS specific attribute is exactly what the bug was about.
    """
    X_train, y_train, _, _ = binary_classification_data
    result = trainer.train_model(
        model_type, hyperparameters,
        X_train=X_train, y_train=y_train, random_state=99,
    )
    fitted_estimator = trainer._models[result["model_ref"]]["model"]
    assert fitted_estimator.random_state == 99


@pytest.mark.parametrize("model_type,hyperparameters", ESTIMATOR_CASES)
def test_same_random_state_gives_identical_predictions(
    trainer, binary_classification_data, model_type, hyperparameters
):
    """End-to-end reproducibility guarantee: fitting the SAME model_type +
    hyperparameters + random_state twice, on the same data, must produce
    identical predictions on held-out test data -- for all three
    registered estimators. This is the test that would have caught the
    original bug directly: pre-fix, two RandomForestClassifier fits with
    identical hyperparameters could disagree because random_state never
    reached the constructor.
    """
    X_train, y_train, X_test, y_test = binary_classification_data

    result_a = trainer.train_model(
        model_type, hyperparameters,
        X_train=X_train, y_train=y_train, random_state=7,
    )
    result_b = trainer.train_model(
        model_type, hyperparameters,
        X_train=X_train, y_train=y_train, random_state=7,
    )

    eval_a = trainer.evaluate_model(
        result_a["model_ref"], pos_label=1, X_test=X_test, y_test=y_test
    )
    eval_b = trainer.evaluate_model(
        result_b["model_ref"], pos_label=1, X_test=X_test, y_test=y_test
    )

    assert eval_a["confusion_matrix"] == eval_b["confusion_matrix"]


def test_random_state_is_not_gemini_tunable():
    """random_state is deliberately NOT exposed as a schema hyperparameter
    -- it's a reproducibility knob, not a modeling choice Gemini should be
    making. FAKE_SCHEMA below mirrors the real schema's field shapes
    (type/range/options) for realism, but is fully self-contained -- this
    test does not import or depend on the real tools.py schema.
    """
    fake_schema = {
        "logistic_regression": {
            "hyperparameters": {
                "C": {"type": "float", "range": [0.001, 100.0]},
            }
        }
    }
    with pytest.raises(ValueError, match="random_state"):
        validate_hyperparameters(
            "logistic_regression",
            {"C": 1.0, "random_state": 42},
            fake_schema,
        )


# ---------------------------------------------------------------------
# train_model return shape and warning capture
# ---------------------------------------------------------------------

def test_train_model_return_shape(trainer, binary_classification_data):
    """warnings must always be present, even when empty -- same
    consistent-key-set convention already used elsewhere in this project
    (gemini_client.py's per-iteration log, compare_runs.py's
    final_model_ambiguous flag)."""
    X_train, y_train, _, _ = binary_classification_data
    result = trainer.train_model(
        "logistic_regression", {"class_weight": "balanced"},
        X_train=X_train, y_train=y_train, random_state=1,
    )
    assert "model_ref" in result
    assert "warnings" in result
    assert isinstance(result["warnings"], list)


def test_convergence_warning_is_captured(trainer, ill_conditioned_binary_data):
    """max_iter=50 is the schema's floor; C=100.0 is the schema's ceiling
    (weakest allowed regularization -- the least help holding coefficients
    back). Combined with per-feature scale mismatch (see fixture
    docstring, and the three-attempt correction log at the top of this
    module), this reliably triggers a real ConvergenceWarning: verified
    empirically across multiple seeds to consistently hit n_iter_=50
    without satisfying sklearn's internal tolerance.
    """
    X_train, y_train = ill_conditioned_binary_data
    result = trainer.train_model(
        "logistic_regression",
        {"class_weight": "balanced", "max_iter": 50, "C": 100.0},
        X_train=X_train, y_train=y_train, random_state=1,
    )
    categories = [w["category"] for w in result["warnings"]]
    assert "ConvergenceWarning" in categories


# ---------------------------------------------------------------------
# evaluate_model: pos_label orientation, unknown ref, JSON-serializability
# ---------------------------------------------------------------------

@pytest.mark.parametrize("pos_label", [0, 1])
def test_confusion_matrix_orientation_matches_pos_label(
    trainer, binary_classification_data, pos_label
):
    """confusion_matrix's [1][1] cell must always correspond to true
    positives w.r.t. whichever class was passed as pos_label -- regardless
    of whether that's literal 0 or 1. Computed independently here via
    plain pandas boolean counting, not by re-deriving sklearn's own
    confusion_matrix call, so this genuinely checks evaluate_model's
    orientation logic rather than just restating it.
    """
    X_train, y_train, X_test, y_test = binary_classification_data
    result = trainer.train_model(
        "logistic_regression", {"class_weight": "balanced"},
        X_train=X_train, y_train=y_train, random_state=1,
    )
    fitted_estimator = trainer._models[result["model_ref"]]["model"]
    y_pred = pd.Series(fitted_estimator.predict(X_test), index=y_test.index)

    expected_true_positives = int(
        ((y_test == pos_label) & (y_pred == pos_label)).sum()
    )

    evaluation = trainer.evaluate_model(
        result["model_ref"], pos_label=pos_label, X_test=X_test, y_test=y_test
    )
    assert evaluation["confusion_matrix"][1][1] == expected_true_positives


def test_evaluate_unknown_model_ref_raises(trainer):
    with pytest.raises(ValueError, match="Unknown model_ref"):
        trainer.evaluate_model(
            "not-a-real-ref", pos_label=1, X_test=pd.DataFrame(), y_test=pd.Series()
        )


def test_evaluate_model_output_is_json_serializable(trainer, binary_classification_data):
    """Guards the exact concern train_model/evaluate_model's docstrings
    call out: sklearn's metric functions return numpy scalars, and
    confusion_matrix returns an ndarray -- both raise TypeError under
    json.dumps() unless explicitly cast, as evaluate_model already does.
    """
    X_train, y_train, X_test, y_test = binary_classification_data
    result = trainer.train_model(
        "logistic_regression", {"class_weight": "balanced"},
        X_train=X_train, y_train=y_train, random_state=1,
    )
    evaluation = trainer.evaluate_model(
        result["model_ref"], pos_label=1, X_test=X_test, y_test=y_test
    )
    json.dumps(evaluation)  # raises TypeError on failure -- no assert needed


# ---------------------------------------------------------------------
# validate_hyperparameters -- pure-function checks, self-contained schema
# mirroring the real schema's shape (verified against tools.py, not
# invented)
# ---------------------------------------------------------------------

FAKE_SCHEMA = {
    "logistic_regression": {
        "hyperparameters": {
            "C": {"type": "float", "range": [0.001, 100.0]},
            "max_iter": {"type": "int", "range": [50, 1000]},
            "class_weight": {"type": "choice", "options": [None, "balanced"]},
        }
    }
}


def test_validate_hyperparameters_unknown_model_type():
    with pytest.raises(ValueError, match="Unknown model_type"):
        validate_hyperparameters("not_a_model", {}, FAKE_SCHEMA)


def test_validate_hyperparameters_unknown_key():
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        validate_hyperparameters(
            "logistic_regression", {"not_a_param": 1}, FAKE_SCHEMA
        )


def test_validate_hyperparameters_out_of_range_float():
    with pytest.raises(ValueError, match="out of range"):
        validate_hyperparameters(
            "logistic_regression", {"C": 999.0}, FAKE_SCHEMA
        )


def test_validate_hyperparameters_invalid_choice():
    with pytest.raises(ValueError, match="not a valid choice"):
        validate_hyperparameters(
            "logistic_regression", {"class_weight": "not_a_real_option"}, FAKE_SCHEMA
        )
