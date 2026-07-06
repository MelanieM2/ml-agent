"""Dataset loading and inspection utilities.

Provides a small registry of loadable datasets and a dataset-agnostic
inspection function that produces the summary handed to Gemini as
context: shape, dtypes, target balance, missing values, and light
feature statistics. Deliberately never includes raw row-level data,
since inspection is meant to inform model *proposals*, not give the
agent (or the external API call) a full data dump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.utils import Bunch



def load_breast_cancer_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer Wisconsin dataset (sklearn built-in).

    Used as a fast, dependency-free fallback for debugging the agent
    pipeline in isolation from network calls.
    """
    bunch = cast(Bunch, load_breast_cancer(as_frame=True)) # bunch = load_breast_cancer(as_frame=True)
    X = bunch.data
    y = bunch.target
    return X, y


def load_climate_crashes_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Climate Model Simulation Crashes dataset (OpenML id 1467).

    Binary classification: predict whether a given combination of 18
    POP2 ocean-model parameters causes a numerical simulation crash.
    Fetched from OpenML rather than bundled locally, so this call
    requires network access the first time (sklearn caches it after).

    The raw OpenML source includes two extra leading columns (Latin
    hypercube study ID and simulation run ID) that are NOT physical
    parameters -- they're experiment bookkeeping, explicitly flagged
    by the dataset's own maintainers as unfit for prediction (see
    OpenML id 40994's changelog). We drop them here.

    OpenML also encodes the target as string labels '1'/'2' rather
    than a meaningful 0/1. We remap to standard binary semantics:
    1 = crash/failure (the rarer, positive class -- confirmed by
    matching label '1's count of 46 against the documented "46 of
    540 simulations failed"), 0 = success. This makes the positive
    class explicit and consistent with sklearn's recall/precision
    conventions, where the positive class is conventionally 1.
    """
    bunch = cast(Bunch, fetch_openml(data_id=1467, as_frame=True, parser="auto"))
    X = bunch.data.iloc[:, 2:]  # drop columns 0-1: study ID, run ID
    y = bunch.target.map({"1": 1, "2": 0}).astype(int)
    return X, y


# Each registry entry pairs a loader with dataset-level facts that must
# never be guessed or re-derived at evaluation time - in particular
# pos_label, which identifies the positive class for precision/recall.
# This is a documented fact about the dataset (established once, via
# label cross-checking against the dataset's own documentation), not a
# judgment call - so it is never exposed to Gemini as a tool argument.
# It is read internally, per dataset, by whichever code computes
# evaluation metrics (see tools.py's evaluate_model).
@dataclass(frozen=True)
class DatasetSpec:
    """One registry entry: how to load a dataset, plus facts about it
    that must never be guessed at evaluation time."""
    loader: Callable[[], tuple[pd.DataFrame, pd.Series]]
    pos_label: int
    description: str


# Registry mapping a short name to its DatasetSpec. This is the
# extension point: adding a new dataset later means writing one loader
# function and adding one line here -- nothing else in the file changes.
DATASET_LOADERS: dict[str, DatasetSpec] = {
    "climate": DatasetSpec(
        loader=load_climate_crashes_dataset,
        pos_label=1,
        description="1 = simulation failure (rare class, ~8.5%)",
    ),
    "breast_cancer": DatasetSpec(
        loader=load_breast_cancer_dataset,
        pos_label=1,
        description="1 = benign (majority class) - opposite sense from climate",
    ),
}


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Dispatch to the correct loader by name.

    Raises a clear error for unknown names rather than a confusing
    KeyError deep in a dict lookup.
    """
    if name not in DATASET_LOADERS:
        available = ", ".join(DATASET_LOADERS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_LOADERS[name].loader()


def get_pos_label(name: str) -> int:
    """Look up the documented positive-class label for a dataset by name.

    Kept as a separate accessor (rather than requiring every call site
    to reach into DATASET_LOADERS[name].pos_label directly) so agent.py
    and evaluate_model's construction have one obvious place to get this
    fact from, matching the load_dataset pattern above.
    """
    if name not in DATASET_LOADERS:
        available = ", ".join(DATASET_LOADERS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_LOADERS[name].pos_label


def inspect_dataset(X: pd.DataFrame, y: pd.Series) -> dict:
    """Produce a compact, LLM-friendly summary of a dataset.

    This is the single function whose output becomes part of the
    context sent to Gemini. It intentionally never includes raw rows.
    """
    target_counts = y.value_counts()
    target_distribution = {
        str(label): {
            "count": int(count),
            "percentage": round(100 * count / len(y), 2),
        }
        for label, count in target_counts.items()
    }

    numeric_cols = X.select_dtypes(include="number").columns
    feature_summary = {
        col: {
            "dtype": str(X[col].dtype),
            "mean": round(float(X[col].mean()), 4) if col in numeric_cols else None,
            "std": round(float(X[col].std()), 4) if col in numeric_cols else None,
            "min": round(float(X[col].min()), 4) if col in numeric_cols else None,
            "max": round(float(X[col].max()), 4) if col in numeric_cols else None,
        }
        for col in X.columns
    }

    return {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target_distribution": target_distribution,
        "missing_values": {
            col: int(count) for col, count in X.isnull().sum().items() if count > 0
        },
        "feature_summary": feature_summary,
    }


if __name__ == "__main__":
    # Quick manual sanity check -- not a substitute for tests/test_dataset.py
    X, y = load_dataset("breast_cancer")
    summary = inspect_dataset(X, y)
    print(f"Shape: {summary['n_rows']} rows, {summary['n_features']} features")
    print(f"Target distribution: {summary['target_distribution']}")
    print(f"Missing values: {summary['missing_values'] or 'none'}")
