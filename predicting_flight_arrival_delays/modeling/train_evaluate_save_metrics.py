"""Training on train sets and metrics calculation on valid sets"""

import json
from pathlib import Path

import dagshub
from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import typer

from predicting_flight_arrival_delays.config import (
    DAGSHUB_REPO_NAME,
    DAGSHUB_REPO_OWNER,
    ENCODING,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    RESAMPLE_METHODS,
)
from predicting_flight_arrival_delays.data.features import VARIANTS, build_xy
from predicting_flight_arrival_delays.data.transform import (
    Transformer,
    align_columns,
    encode_categoricals,
    resample_training_data,
    sparse_column_order,
    to_sparse_matrix,
)
from predicting_flight_arrival_delays.modeling import evaluate, train
from predicting_flight_arrival_delays.modeling.train import BUILDERS, HYPERPARAMS
from predicting_flight_arrival_delays.utils import (
    get_dvc_data_hash,
    get_git_dirty,
    safe_relative_path,
)

app = typer.Typer()


def prepare_fold(
    train_df: pd.DataFrame,
    encoding: str,
    test_df: pd.DataFrame | None = None,
    validation_df: pd.DataFrame | None = None,
    resample: str = "none",
) -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame | None, pd.Series | None,
    pd.DataFrame | None, pd.Series | None,
]:
    """Fit the transformer on train, transform every split, then resample train only.

    Args:
        train_df: Raw training dataframe for this fold.
        test_df: Raw test dataframe for this fold. Optional.
        encoding: Either "onehot" or "native"; determines how categoricals are
            encoded, and which resample methods are compatible.
        validation_df: Raw validation dataframe for this fold. Optional.
        resample: Training-set rebalancing strategy. One of "none", "undersample",
            "oversample", "smote" ("smote" requires encoding="onehot"). Defaults to
            "none" (no rebalancing).

    Returns:
        Tuple of (X_fit, y_fit, X_val, y_val, X_test, y_test, transformer,
        feature_columns, feature_means).
        X_fit/y_fit reflect the resampling; the other splits are untouched.
        X_val/y_val are (None, None) when validation_df is not given.
        X_test/y_test are (None, None) when test_df is not given.
        For encoding="onehot", X_fit/X_val/X_test are scipy.sparse CSR matrices
        (column names are lost); for "native" they remain DataFrames.
    """
    X_fit, y_fit = build_xy(train_df)

    transformer = Transformer(encoding=encoding).fit(X_fit, y_fit)
    X_fit = transformer.transform(X_fit)

    transformer.select_features(X_fit, y_fit)
    X_fit = transformer.apply_selection(X_fit)

    cat_cols = [c for c in transformer.categorical_columns if c in X_fit.columns]
    X_fit = encode_categoricals(X_fit, cat_cols, encoding)

    X_val, y_val = None, None
    if validation_df is not None:
        X_val, y_val = build_xy(validation_df)
        X_val = transformer.transform(X_val)
        X_val = transformer.apply_selection(X_val)
        X_val = encode_categoricals(X_val, cat_cols, encoding)
        X_fit, X_val = align_columns(X_fit, X_val, encoding)
        if encoding == "onehot":
            X_val = to_sparse_matrix(X_val)

    X_test, y_test = None, None
    if test_df is not None:
        X_test, y_test = build_xy(test_df)
        X_test = transformer.transform(X_test)
        X_test = transformer.apply_selection(X_test)
        X_test = encode_categoricals(X_test, cat_cols, encoding)
        X_fit, X_test = align_columns(X_fit, X_test, encoding)
        if encoding == "onehot":
            X_test = to_sparse_matrix(X_test)



    feature_columns = (
        sparse_column_order(X_fit) if encoding == "onehot" else list(X_fit.columns)
    )


    feature_means = (
        X_fit[feature_columns].mean().astype(float).to_dict()
        if encoding == "onehot"
        else {}
    )

    if encoding == "onehot":
        X_fit = to_sparse_matrix(X_fit)

    X_fit, y_fit = resample_training_data(X_fit, y_fit, resample, encoding)

    return (
        X_fit, y_fit, X_val, y_val, X_test, y_test,
        transformer, feature_columns, feature_means,
    )


def train_and_evaluate(
    folds_dir: list[Path],
    variant: str,
    model: str,
    config: str,
    resample: str = "none",
) -> dict[str, float]:
    """Run the full fit/validate cycle across every fold, and average the results.

    For each fold: prepares the data (with resampling on train only), fits the
    estimator (with LightGBM early stopping if X_val/y_val are supported), evaluates on the validation set,
    and logs everything to the active MLflow run.

    Args:
        folds_dir: One directory per fold, each containing train/validation
            parquet files.
        variant: Feature set variant name (e.g. 'all', 'noweather', 'nocarrier'); only
            used for logging.
        model: Algorithm key.
        config: Hyperparameter configuration key for that model.
        resample: Training-set rebalancing strategy.

    Returns:
        Per-metric averages across folds, plus "roc_auc_val_std" (their standard
        deviation), giving a sense of how stable the estimate is across folds.
    """
    encoding = ENCODING[model]
    fold_metrics = []

    for index, fold in enumerate(folds_dir):
        train_df = pd.read_parquet(fold / "train.parquet")
        validation_df = pd.read_parquet(fold / "validation.parquet")

        X_fit, y_fit, X_val, y_val, _, _, _, _, _ = prepare_fold(
            train_df, encoding, validation_df=validation_df, resample=resample
        )

        estimator = train.train(
            X_fit, y_fit, model, config, False, X_val=X_val, y_val=y_val
        )

        metrics_val = evaluate.evaluate(X_val, y_val, estimator)

        combined = {f"{k}_val": v for k, v in metrics_val.items()}

        mlflow.log_metrics({f"fold_{k}": v for k, v in combined.items()}, step=index)

        fold_metrics.append(combined)
        logger.info(
            f"{variant}/{model} fold {index}: "
            f"val ROC-AUC {combined['roc_auc_val']:.3f}, val PR-AUC {combined['pr_auc_val']:.3f} | "
        )

    if not fold_metrics:
        raise RuntimeError(f"No usable folds for {variant}/{model}")

    averages = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    averages["roc_auc_val_std"] = float(np.std([m["roc_auc_val"] for m in fold_metrics]))
    return averages


@app.command()
def run(
    variant: str = typer.Option(..., help="Which feature set variant to train"),
    model: str = typer.Option(..., help="Which algorithm to train"),
    config: str = typer.Option("default", help="Which hyperparameter set to use"),
    data_path: Path = typer.Option(PROCESSED_DATA_DIR / "selection"),
    experiment: str = typer.Option("flight-delay-v3", help="MLflow experiment name"),
    resample: str = typer.Option(
        "none",
        help="Training rebalancing strategy: 'none', 'undersample', 'oversample', "
        "'smote' ('smote' only valid with one-hot encoded models).",
    ),
    repo_owner: str = typer.Option(DAGSHUB_REPO_OWNER, help="DagsHub repository owner"),
    repo_name: str = typer.Option(DAGSHUB_REPO_NAME, help="DagsHub repository name"),
) -> None:
    """Cross-validate one model/config over every fold of a variant, and log averaged results.

    Args:
        variant: Feature set variant to train.
        model: Algorithm name key to train.
        config: Hyperparameter configuration key to use.
        data_path: Root directory containing one subfolder per variant, each with train/validation parquet files.
        experiment: MLflow experiment name to log this run under.
        resample: Training-set rebalancing strategy.
        repo_owner: DagsHub repository owner, used to route MLflow tracking.
        repo_name: DagsHub repository name, used to route MLflow tracking.
    """
    try:
        if variant not in VARIANTS:
            raise SystemExit(f"Unknown variant '{variant}'. Available: {list(VARIANTS)}")
        if model not in BUILDERS:
            raise SystemExit(f"Unknown model '{model}'. Available: {list(BUILDERS)}")
        if config not in HYPERPARAMS[model]:
            raise SystemExit(
                f"Unknown config '{config}' for {model}. Available: {list(HYPERPARAMS[model])}"
            )
        if resample not in RESAMPLE_METHODS:
            raise SystemExit(
                f"Unknown resample method '{resample}'. Available: {list(RESAMPLE_METHODS)}"
            )
        if resample == "smote" and ENCODING[model] != "onehot":
            raise SystemExit(
                f"resample='smote' requires one-hot encoding, but '{model}' uses "
                f"'{ENCODING[model]}'. Use 'undersample' or 'oversample' instead."
            )

        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        variant_dir = data_path / variant
        if not variant_dir.exists():
            raise FileNotFoundError(
                f"Variant directory not found for '{variant}': {safe_relative_path(variant_dir)}."
            )

        val_fold_dirs = [
            d for d in variant_dir.glob("fold_*")
            if d.is_dir() and (d / "validation.parquet").exists()
        ]
        if not val_fold_dirs:
            raise FileNotFoundError(
                f"No validation fold found at {safe_relative_path(variant_dir)}"
            )

        mlflow.enable_system_metrics_logging()
        mlflow.set_experiment(experiment)

        with mlflow.start_run(run_name=f"{variant}__{model}__{config}"):
            mlflow.log_params({
                "variant": variant,
                "algorithm": model,
                "config": config,
                "encoding": ENCODING[model],
                "calibrated": False,
                "resample": resample,
                "final": False,
                "n_folds": len(val_fold_dirs),
                "dvc_data_hash": get_dvc_data_hash(data_path),
                **{f"hp_{k}": v for k, v in HYPERPARAMS[model][config].items()},
            })
            metrics = train_and_evaluate(
                val_fold_dirs, variant, model, config, resample
            )
            mlflow.set_tag("git_dirty", get_git_dirty())
            mlflow.log_metrics(metrics)

        out_path = METRICS_DIR / "selection" / variant / f"{model}__{config}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({**metrics, "resample": resample}, indent=2))
    except Exception as e:
        logger.exception(f"An error occurred while training and evaluating the model: {e}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()