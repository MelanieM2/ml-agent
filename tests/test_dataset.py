"""Tests for dataset.py -- loading, registry dispatch, and inspection.

Network-dependent behavior (fetch_openml) is not exercised directly;
instead we test the transformation logic against a mocked return
value, keeping the suite fast and independent of OpenML availability.
"""

import pandas as pd
import pytest
from sklearn.utils import Bunch

from ml_agent.dataset import (
    load_dataset,
    load_breast_cancer_dataset,
    inspect_dataset,
    DATASET_LOADERS,
)


def test_breast_cancer_loader_shape():
    X, y = load_breast_cancer_dataset()
    assert X.shape[0] == 569
    assert X.shape[1] == 30
    assert len(y) == 569


def test_load_dataset_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("not_a_real_dataset")


def test_load_dataset_dispatches_correctly():
    # Confirms the registry actually routes to the right function,
    # without needing to hit the network -- we check identity of the
    # callable, not the data itself.
    assert DATASET_LOADERS["breast_cancer"] is load_breast_cancer_dataset


def test_climate_crashes_drops_identifier_columns_and_remaps_labels(mocker):
    # Build a fake "raw" OpenML-shaped Bunch: 2 identifier columns +
    # 3 fake parameter columns, target as OpenML-style string labels.
    fake_data = pd.DataFrame({
        "study_id": [1, 1, 2],
        "run_id": [1, 2, 1],
        "param_a": [0.1, 0.2, 0.3],
        "param_b": [0.4, 0.5, 0.6],
        "param_c": [0.7, 0.8, 0.9],
    })
    fake_target = pd.Series(["2", "1", "2"])  # 1=failure, 2=success
    fake_bunch = Bunch(data=fake_data, target=fake_target)

    mocker.patch(
        "ml_agent.dataset.fetch_openml",
        return_value=fake_bunch,
    )

    from ml_agent.dataset import load_climate_crashes_dataset
    X, y = load_climate_crashes_dataset()

    # The two identifier columns should be gone.
    assert list(X.columns) == ["param_a", "param_b", "param_c"]
    assert X.shape[1] == 3

    # Labels should be remapped: '1' (failure) -> 1, '2' (success) -> 0
    assert list(y) == [0, 1, 0]


def test_inspect_dataset_structure():
    X, y = load_breast_cancer_dataset()
    summary = inspect_dataset(X, y)

    assert summary["n_rows"] == 569
    assert summary["n_features"] == 30
    assert "target_distribution" in summary
    assert "feature_summary" in summary
    assert summary["missing_values"] == {}  # breast cancer has none


def test_inspect_dataset_flags_missing_values():
    X = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    summary = inspect_dataset(X, y)

    assert summary["missing_values"] == {"a": 1}