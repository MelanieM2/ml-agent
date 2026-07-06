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
    get_pos_label,       # CHANGED: new import -- covers the new accessor
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
    #
    # CHANGED: DATASET_LOADERS["breast_cancer"] is now a DatasetSpec,
    # not the bare loader function itself -- this is the one place the
    # DatasetSpec refactor actually breaks the old assertion. The fix
    # is a single ".loader" added to reach through to the callable.
    assert DATASET_LOADERS["breast_cancer"].loader is load_breast_cancer_dataset


# NEW: DatasetSpec carries pos_label and description alongside the
# loader now -- these two tests cover that new surface directly,
# since nothing in the tests above exercises it.
def test_dataset_loaders_have_pos_label_and_description():
    for name, spec in DATASET_LOADERS.items():
        assert isinstance(spec.pos_label, int)
        assert isinstance(spec.description, str) and spec.description != ""


def test_get_pos_label_matches_registry():
    assert get_pos_label("climate") == DATASET_LOADERS["climate"].pos_label
    assert get_pos_label("breast_cancer") == DATASET_LOADERS["breast_cancer"].pos_label
    # Documented values, not just internal consistency -- both are 1,
    # but for opposite reasons (see dataset.py's DatasetSpec comment):
    # climate's 1 = failure (rare class), breast_cancer's 1 = benign
    # (majority class). Pinning the actual values here, not just
    # "matches itself", catches an accidental edit to either constant.
    assert get_pos_label("climate") == 1
    assert get_pos_label("breast_cancer") == 1


def test_get_pos_label_unknown_name_raises():
    # NEW: mirrors test_load_dataset_unknown_name_raises, since
    # get_pos_label repeats the same validation pattern.
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_pos_label("not_a_real_dataset")


def test_climate_crashes_drops_identifier_columns_and_remaps_labels(mocker):
    # Build a fake "raw" OpenML-shaped Bunch: 2 identifier columns +
    # 3 fake parameter columns, target as OpenML-style string labels.
    #
    # UNCHANGED: this test imports load_climate_crashes_dataset directly
    # rather than going through DATASET_LOADERS, so the DatasetSpec
    # refactor doesn't touch it at all.
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