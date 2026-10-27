import json

import numpy as np
import pandas as pd
import pytest

from ml_classification import (
    build_models,
    cross_validate_models,
    holdout_report,
    load_csv,
    load_iris_frame,
    param_grids,
    split_data,
    tune_models,
)
from ml_classification.cli import main


def test_iris_loads_with_string_labels():
    X, y = load_iris_frame()
    assert X.shape == (150, 4)
    assert set(y.unique()) == {"setosa", "versicolor", "virginica"}


def test_split_is_stratified_and_reproducible():
    X, y = load_iris_frame()
    a = split_data(X, y, 0.2, 42)
    b = split_data(X, y, 0.2, 42)
    assert a[0].index.equals(b[0].index)
    assert len(a[1]) == 30
    assert a[3].value_counts().nunique() == 1  # 10/10/10


def test_param_grids_cover_all_models():
    assert set(build_models()) == set(param_grids())


def test_cross_validation_table_is_sorted_and_sane():
    X, y = load_iris_frame()
    table = cross_validate_models(build_models(), X, y)
    assert list(table["CV Mean Accuracy"]) == sorted(table["CV Mean Accuracy"], reverse=True)
    assert table["CV Mean Accuracy"].min() > 0.85


def test_tuning_then_holdout_accuracy():
    X, y = load_iris_frame()
    Xtr, Xte, ytr, yte = split_data(X, y)
    best, tuned = tune_models(build_models(), param_grids(), Xtr, ytr)
    report = holdout_report(best, Xte, yte)
    assert len(tuned) == 3
    assert report["Test Accuracy"].max() >= 0.9


def test_load_csv_encodes_categoricals_and_drops_missing_target(tmp_path):
    df = pd.DataFrame({
        "size": [1.0, 2.0, np.nan, 4.0],
        "colour": ["r", "g", "r", "b"],
        "label": ["a", "b", None, "a"],
    })
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    X, y = load_csv(path, "label")
    assert len(X) == len(y) == 3
    assert "colour_g" in X.columns and "label" not in X.columns


def test_load_csv_rejects_unknown_target(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not found"):
        load_csv(path, "missing")


def test_cli_end_to_end_writes_outputs(tmp_path):
    main(["--out", str(tmp_path)])
    for name in ["summary.json", "test_results.csv", "cv_baseline.csv"]:
        assert (tmp_path / name).exists()
    for name in ["model_comparison.png", "confusion_matrix.png", "feature_importance.png"]:
        assert (tmp_path / "figures" / name).stat().st_size > 1000
    assert json.loads((tmp_path / "summary.json").read_text())["dataset"] == "Iris"
