"""Cross-validation, tuning and held-out evaluation."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


def _cv(seed: int, n_splits: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_validate_models(models, X, y, seed: int = 42, n_splits: int = 5) -> pd.DataFrame:
    """Mean/std CV accuracy for each model, sorted best first."""
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=_cv(seed, n_splits), scoring="accuracy")
        rows.append({"Model": name, "CV Mean Accuracy": scores.mean(), "CV Std": scores.std()})
    return pd.DataFrame(rows).sort_values("CV Mean Accuracy", ascending=False, ignore_index=True)


def tune_models(models, grids, X, y, seed: int = 42, n_splits: int = 5):
    """Grid-search every model. Returns (best_estimators, summary_frame)."""
    best, rows = {}, []
    for name, model in models.items():
        search = GridSearchCV(model, grids[name], cv=_cv(seed, n_splits), scoring="accuracy", n_jobs=-1)
        search.fit(X, y)
        best[name] = search.best_estimator_
        rows.append({
            "Model": name,
            "Best CV Accuracy": search.best_score_,
            "Best Params": {k.split("__", 1)[-1]: v for k, v in search.best_params_.items()},
        })
    return best, pd.DataFrame(rows)


def holdout_report(models, X_test, y_test) -> pd.DataFrame:
    """Held-out metrics for already-fitted models."""
    rows = []
    for name, model in models.items():
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Test Accuracy": accuracy_score(y_test, pred),
            "Macro Precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_test, pred, average="macro", zero_division=0),
            "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0),
        })
    return pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False, ignore_index=True)
