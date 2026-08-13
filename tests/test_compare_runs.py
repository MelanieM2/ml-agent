"""Tests for compare_runs.py's summarize_run(): the "final model"
resolution logic, and specifically the fix for the real bug found in
result_log_2026_07_29_232959_breast_cancer.json (Gemini evaluated
random_forest LAST purely as a comparison point, but its own
convergence reasoning named an EARLIER logistic_regression evaluation
as the actual chosen model -- "last evaluated = final" was wrong).

Hand-built run_data dicts, not real files -- same "isolate the checkable
part" convention already used by validate_split/validate_hyperparameters/
summarize_run itself. Each test builds only the log entries the
function actually reads (tool_name/tool_args/result), skipping
unrelated fields real files have (timestamps, response_text) since
summarize_run() never looks at those.
"""
from __future__ import annotations

from ml_agent.compare_runs import summarize_run


def _train_entry(model_type: str, hyperparameters: dict, model_ref: str) -> dict:
    return {
        "tool_name": "train_model",
        "tool_args": {"model_type": model_type, "hyperparameters": hyperparameters},
        "result": {"model_ref": model_ref, "warnings": []},
    }


def _evaluate_entry(model_ref: str, model_type: str, metrics: dict) -> dict:
    return {
        "tool_name": "evaluate_model",
        "tool_args": {"model_ref": model_ref},
        "result": {"model_ref": model_ref, "model_type": model_type, **metrics},
    }


def _decision_entry(reasoning: str, continue_iterating: bool = False) -> dict:
    return {
        "tool_name": "record_convergence_decision",
        "tool_args": {},
        "result": {"reasoning": reasoning, "continue_iterating": continue_iterating},
    }


def test_best_by_target_wins_over_last_evaluated():
    """The exact breast_cancer scenario: three evaluations, target=recall,
    the best-recall model (0.9722) is NOT the last one evaluated (0.9583).
    final_model_type must be the best-recall one, and
    final_model_ambiguous must be True."""
    run_data = {
        "status": "converged",
        "iterations": 10,
        "config": {"dataset": "breast_cancer", "target": "recall", "random_state": 42},
        "log": [
            _train_entry("logistic_regression", {"class_weight": "balanced"}, "ref_A"),
            _evaluate_entry("ref_A", "logistic_regression", {
                "accuracy": 0.9474, "precision": 0.9583, "recall": 0.9583, "f1": 0.9583,
                "confusion_matrix": [[39, 3], [3, 69]],
            }),
            _train_entry(
                "logistic_regression",
                {"class_weight": "balanced", "max_iter": 500},
                "ref_B",
            ),
            _evaluate_entry("ref_B", "logistic_regression", {
                "accuracy": 0.9649, "precision": 0.9722, "recall": 0.9722, "f1": 0.9722,
                "confusion_matrix": [[40, 2], [2, 70]],
            }),
            _train_entry(
                "random_forest",
                {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"},
                "ref_C",
            ),
            _evaluate_entry("ref_C", "random_forest", {
                "accuracy": 0.9474, "precision": 0.9583, "recall": 0.9583, "f1": 0.9583,
                "confusion_matrix": [[39, 3], [3, 69]],
            }),
            _decision_entry("The max_iter=500 logistic regression is best."),
        ],
    }

    row = summarize_run(run_data, source_file="test.json")

    assert row["final_model_type"] == "logistic_regression"
    assert row["final_hyperparameters"] == {"class_weight": "balanced", "max_iter": 500}
    assert row["final_metrics"]["recall"] == 0.9722
    assert row["final_model_ambiguous"] is True


def test_last_evaluated_agrees_with_best_by_target_no_flag():
    """When the last-evaluated model IS also the best-by-target model,
    final_model_ambiguous must be False, not True -- the flag should
    only fire on a genuine disagreement."""
    run_data = {
        "status": "converged",
        "iterations": 4,
        "config": {"dataset": "climate", "target": "recall", "random_state": 42},
        "log": [
            _train_entry("random_forest", {"n_estimators": 100}, "ref_A"),
            _evaluate_entry("ref_A", "random_forest", {
                "accuracy": 0.9, "precision": 0.5, "recall": 0.4, "f1": 0.44,
                "confusion_matrix": [[90, 5], [3, 2]],
            }),
            _train_entry("svm", {"kernel": "linear", "C": 1}, "ref_B"),
            _evaluate_entry("ref_B", "svm", {
                "accuracy": 0.8, "precision": 0.3, "recall": 0.9, "f1": 0.45,
                "confusion_matrix": [[80, 15], [1, 4]],
            }),
            _decision_entry("SVM wins on recall."),
        ],
    }

    row = summarize_run(run_data, source_file="test.json")

    assert row["final_model_type"] == "svm"
    assert row["final_model_ambiguous"] is False


def test_no_target_falls_back_to_last_evaluated():
    """Older files with no config["target"] at all (pre-2026-07-29) must
    fall back to the original "last evaluated wins" behavior, and
    final_model_ambiguous must be None (nothing to compare against) --
    not False."""
    run_data = {
        "status": "converged",
        "iterations": 4,
        # No "config" key at all -- matches real pre-07-29 files.
        "log": [
            _train_entry("logistic_regression", {"C": 1}, "ref_A"),
            _evaluate_entry("ref_A", "logistic_regression", {
                "accuracy": 0.9, "precision": 0.8, "recall": 0.95, "f1": 0.87,
                "confusion_matrix": [[90, 5], [1, 4]],
            }),
            _train_entry("svm", {"kernel": "rbf"}, "ref_B"),
            _evaluate_entry("ref_B", "svm", {
                "accuracy": 0.6, "precision": 0.2, "recall": 1.0, "f1": 0.33,
                "confusion_matrix": [[60, 35], [0, 5]],
            }),
        ],
    }

    row = summarize_run(run_data, source_file="test.json")

    assert row["final_model_type"] == "svm"  # last evaluated, unchanged behavior
    assert row["dataset"] is None
    assert row["target"] is None
    assert row["final_model_ambiguous"] is None


def test_single_evaluation_never_ambiguous():
    """Only one evaluate_model call this run -- best-by-target and
    last-evaluated are trivially the same call, so ambiguous must be
    False, not True, even though target is known."""
    run_data = {
        "status": "converged",
        "iterations": 2,
        "config": {"dataset": "climate", "target": "recall", "random_state": 42},
        "log": [
            _train_entry("logistic_regression", {"C": 1}, "ref_A"),
            _evaluate_entry("ref_A", "logistic_regression", {
                "accuracy": 0.9, "precision": 0.8, "recall": 0.9, "f1": 0.85,
                "confusion_matrix": [[90, 5], [1, 4]],
            }),
        ],
    }

    row = summarize_run(run_data, source_file="test.json")

    assert row["final_model_ambiguous"] is False


def test_no_evaluations_at_all():
    """A run that hit max_iterations right after a record_model_proposal,
    before ever calling evaluate_model (the real
    result_log_2026_07_27_223458_climate.json shape for its FIRST
    proposal, and a hypothetical worst case for a run that trained
    nothing) -- final_model_type/metrics/hyperparameters must all be
    None, and ambiguous must be None (nothing to evaluate at all)."""
    run_data = {
        "status": "max_iterations_reached",
        "iterations": 1,
        "config": {"dataset": "climate", "target": "recall", "random_state": 42},
        "log": [
            {
                "tool_name": "record_model_proposal",
                "tool_args": {"model_type": "svm", "hyperparameters": {}, "reasoning": "..."},
                "result": {"model_type": "svm", "hyperparameters": {}, "reasoning": "..."},
            },
        ],
    }

    row = summarize_run(run_data, source_file="test.json")

    assert row["final_model_type"] is None
    assert row["final_hyperparameters"] is None
    assert row["final_metrics"] is None
    assert row["final_model_ambiguous"] is None
