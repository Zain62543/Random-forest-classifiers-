"""Reproducible tabular classification toolkit (Random Forest, Logistic Regression, Decision Tree)."""

from .data import load_csv, load_iris_frame, split_data
from .evaluate import cross_validate_models, holdout_report, tune_models
from .models import build_models, param_grids

__version__ = "0.2.0"
__all__ = [
    "build_models",
    "cross_validate_models",
    "holdout_report",
    "load_csv",
    "load_iris_frame",
    "param_grids",
    "split_data",
    "tune_models",
]
