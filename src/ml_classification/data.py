"""Data loading and splitting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_iris_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Return the Iris features and string-labelled target."""
    iris = load_iris(as_frame=True)
    y = iris.target.map(dict(enumerate(iris.target_names))).rename("species")
    return iris.data, y


def load_csv(path: str | Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV file and separate the target column.

    Rows with a missing target are dropped. Non-numeric feature columns are
    one-hot encoded so that any tabular file works out of the box.
    """
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")
    df = df.dropna(subset=[target])
    y = df[target]
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    if X.empty:
        raise ValueError("No feature columns left after removing the target.")
    return X, y


def split_data(X, y, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split with a fixed seed."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
