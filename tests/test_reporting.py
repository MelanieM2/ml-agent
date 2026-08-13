"""Tests for reporting.py: filename auto-detect, single-vs-comparison
row loading, and the CSV/Markdown renderers -- including that the
final_model_ambiguous flag (compare_runs.py) actually surfaces in both
output formats.
"""
from __future__ import annotations

import csv
import json

import pytest

from ml_agent.reporting import (
    CSV_COLUMNS,
    detect_kind,
    load_rows,
    to_csv,
    to_markdown,
)


def test_detect_kind_result_log():
    assert detect_kind(__import__("pathlib").Path("result_log_2026_08_01_120000_climate.json")) == "single"


def test_detect_kind_comparison():
    assert detect_kind(__import__("pathlib").Path("comparison_2026_08_01_120000_climate.json")) == "comparison"


def test_detect_kind_unrecognized_raises():
    from pathlib import Path
    with pytest.raises(ValueError, match="Can't tell what kind"):
        detect_kind(Path("weird_name.json"))


def test_load_rows_single_run(tmp_path):
    """A result_log_*.json gets summarize_run() applied, becoming
    exactly ONE row -- not left as raw log data."""
    run_data = {
        "status": "converged",
        "iterations": 1,
        "config": {"dataset": "climate", "target": "recall", "random_state": 42},
        "log": [],
    }
    path = tmp_path / "result_log_2026_08_01_120000_climate.json"
    path.write_text(json.dumps(run_data))

    rows, kind = load_rows(path)

    assert kind == "single"
    assert len(rows) == 1
    assert rows[0]["source_file"] == path.name
    assert rows[0]["dataset"] == "climate"


def test_load_rows_comparison_loaded_directly(tmp_path):
    """A comparison_*.json is already a list of summarize_run() rows --
    load_rows() must NOT re-summarize it, just load the list as-is."""
    already_rows = [
        {"source_file": "a.json", "dataset": "climate", "final_model_type": "svm"},
        {"source_file": "b.json", "dataset": "climate", "final_model_type": "random_forest"},
    ]
    path = tmp_path / "comparison_2026_08_01_120000_climate.json"
    path.write_text(json.dumps(already_rows))

    rows, kind = load_rows(path)

    assert kind == "comparison"
    assert rows == already_rows  # loaded verbatim, not reshaped


def test_to_csv_writes_fixed_columns(tmp_path):
    """Every column in CSV_COLUMNS must appear in the header, in order,
    regardless of which fields a given row actually has populated."""
    rows = [{
        "source_file": "result_log_..._climate.json",
        "dataset": "climate",
        "target": "recall",
        "status": "converged",
        "iterations": 5,
        "elapsed_seconds": 10.5,
        "random_state": 42,
        "final_model_type": "logistic_regression",
        "final_hyperparameters": {"C": 1, "max_iter": 500},
        "final_metrics": {
            "accuracy": 0.9, "precision": 0.8, "recall": 0.95, "f1": 0.87,
            "confusion_matrix": [[90, 5], [1, 4]],
        },
        "final_model_ambiguous": True,
        "model_sequence": ["logistic_regression", "random_forest"],
        "warnings_encountered": [],
        "convergence_reasoning": "Logistic regression wins.",
    }]
    out_path = tmp_path / "export.csv"
    to_csv(rows, out_path)

    with out_path.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS
        written_row = next(reader)

    assert written_row["final_model_ambiguous"] == "True"
    # hyperparameters must be valid, double-quoted JSON, not a Python repr
    parsed_hp = json.loads(written_row["final_hyperparameters"])
    assert parsed_hp == {"C": 1, "max_iter": 500}
    assert written_row["recall"] == "0.95"


def test_to_markdown_single_run_shows_ambiguous_warning():
    rows = [{
        "source_file": "result_log_..._breast_cancer.json",
        "dataset": "breast_cancer",
        "target": "recall",
        "random_state": 42,
        "status": "converged",
        "iterations": 10,
        "elapsed_seconds": 11.3,
        "final_model_type": "logistic_regression",
        "final_hyperparameters": {"max_iter": 500, "class_weight": "balanced"},
        "final_metrics": {"accuracy": 0.9649, "precision": 0.9722, "recall": 0.9722, "f1": 0.9722},
        "final_model_ambiguous": True,
        "model_sequence": ["logistic_regression", "logistic_regression", "random_forest"],
        "warnings_encountered": [],
        "convergence_reasoning": "Logistic regression wins on recall.",
    }]

    markdown = to_markdown(rows, "single", source_file="result_log_..._breast_cancer.json")

    assert "logistic_regression" in markdown
    assert "⚠️" in markdown  # the ambiguity note must render


def test_to_markdown_single_run_no_warning_when_not_ambiguous():
    rows = [{
        "source_file": "result_log_..._climate.json",
        "dataset": "climate", "target": "recall", "random_state": 42,
        "status": "converged", "iterations": 4, "elapsed_seconds": 5.0,
        "final_model_type": "svm",
        "final_hyperparameters": {"kernel": "linear"},
        "final_metrics": {"accuracy": 0.8, "precision": 0.3, "recall": 0.9, "f1": 0.45},
        "final_model_ambiguous": False,
        "model_sequence": ["svm"],
        "warnings_encountered": [],
        "convergence_reasoning": "SVM is best.",
    }]

    markdown = to_markdown(rows, "single", source_file="result_log_..._climate.json")

    assert "⚠️" not in markdown


def test_to_markdown_comparison_is_a_table_with_flag_column():
    rows = [
        {
            "source_file": "a.json", "dataset": "climate", "target": "recall",
            "final_model_type": "svm", "final_hyperparameters": {"kernel": "linear"},
            "final_metrics": {"accuracy": 0.8, "precision": 0.3, "recall": 0.9, "f1": 0.45},
            "final_model_ambiguous": False, "iterations": 4, "status": "converged",
        },
        {
            "source_file": "b.json", "dataset": "climate", "target": "recall",
            "final_model_type": "logistic_regression", "final_hyperparameters": {"C": 1},
            "final_metrics": {"accuracy": 0.9, "precision": 0.8, "recall": 0.95, "f1": 0.87},
            "final_model_ambiguous": True, "iterations": 6, "status": "converged",
        },
    ]

    markdown = to_markdown(rows, "comparison", source_file="comparison_..._climate.json")

    assert markdown.count("|") > 10  # a real table, not prose
    assert markdown.count("⚠️") == 2  # once in the header, once for row b only
