"""Figure helpers (static PNG output, headless-safe)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(model, X_test, y_test, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_feature_importance(model, feature_names, path: Path, title: str, top: int = 15) -> Path:
    clf = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    s = pd.Series(clf.feature_importances_, index=list(feature_names)).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(s) + 1)))
    s.plot(kind="barh", ax=ax, color="#2a6f97")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_model_comparison(results: pd.DataFrame, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=results, x="Model", y="Test Accuracy", color="#2a6f97", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
