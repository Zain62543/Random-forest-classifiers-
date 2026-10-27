"""Model definitions and hyperparameter grids."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_models(seed: int = 42) -> dict[str, Pipeline]:
    """Three baseline classifiers wrapped in pipelines (imputation included)."""
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(random_state=seed)),
        ]),
    }


def param_grids() -> dict[str, dict[str, list]]:
    """Hyperparameter search spaces, keyed by model name."""
    return {
        "Logistic Regression": {"classifier__C": [0.1, 1, 10]},
        "Decision Tree": {
            "classifier__max_depth": [None, 3, 5],
            "classifier__min_samples_split": [2, 5],
            "classifier__criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 3, 5],
            "classifier__min_samples_split": [2, 5],
        },
    }
