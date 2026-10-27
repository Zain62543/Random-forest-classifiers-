"""Command line interface: benchmark classifiers on Iris or any CSV file.

Examples
--------
    ml-classify                                   # Iris demo
    ml-classify --csv data.csv --target label     # your own data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_csv, load_iris_frame, split_data
from .evaluate import cross_validate_models, holdout_report, tune_models
from .models import build_models, param_grids
from .plots import save_confusion_matrix, save_feature_importance, save_model_comparison


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ml-classify", description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", type=Path, help="CSV file to analyse (default: built-in Iris dataset)")
    p.add_argument("--target", help="Target column name (required with --csv)")
    p.add_argument("--out", type=Path, default=Path("reports"), help="Output directory (default: reports)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--folds", type=int, default=5)
    return p


def run(args: argparse.Namespace) -> dict:
    if args.csv:
        if not args.target:
            raise SystemExit("--target is required when using --csv")
        X, y = load_csv(args.csv, args.target)
        label = args.csv.stem
    else:
        X, y = load_iris_frame()
        label = "Iris"

    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)
    models = build_models(args.seed)

    cv_table = cross_validate_models(models, X_train, y_train, args.seed, args.folds)
    best, tuned = tune_models(models, param_grids(), X_train, y_train, args.seed, args.folds)
    final = holdout_report(best, X_test, y_test)

    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    top = final.iloc[0]["Model"]
    save_model_comparison(final, fig_dir / "model_comparison.png", f"{label}: tuned test accuracy")
    save_confusion_matrix(best["Random Forest"], X_test, y_test, fig_dir / "confusion_matrix.png",
                          f"{label}: Random Forest confusion matrix")
    save_feature_importance(best["Random Forest"], X.columns, fig_dir / "feature_importance.png",
                            f"{label}: Random Forest feature importance")

    cv_table.to_csv(args.out / "cv_baseline.csv", index=False)
    final.to_csv(args.out / "test_results.csv", index=False)
    summary = {
        "dataset": label,
        "n_samples": len(X),
        "n_features": int(X.shape[1]),
        "seed": args.seed,
        "best_model_on_test": top,
        "tuned": json.loads(tuned.to_json(orient="records")),
        "test": json.loads(final.to_json(orient="records")),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n== {label}: baseline {args.folds}-fold CV ==\n{cv_table.round(4).to_string(index=False)}")
    print(f"\n== Held-out test results (tuned) ==\n{final.round(4).to_string(index=False)}")
    print(f"\nOutputs written to {args.out}/")
    return summary


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
