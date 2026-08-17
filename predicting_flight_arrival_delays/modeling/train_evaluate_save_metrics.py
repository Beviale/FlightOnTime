"""Training, model selection and metrics calculation"""

import json
from pathlib import Path
import dagshub
import mlflow
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from predicting_flight_arrival_delays.config import ENCODING, METRICS_DIR, PROCESSED_DATA_DIR, RESAMPLE_METHODS
from predicting_flight_arrival_delays.data.features import VARIANTS, build_xy
from predicting_flight_arrival_delays.data.transform import (
    Transformer,
    align_columns,
    encode_categoricals,
    resample_training_data,
)
from predicting_flight_arrival_delays.modeling import evaluate, train
from predicting_flight_arrival_delays.modeling.train import BUILDERS, HYPERPARAMS
from predicting_flight_arrival_delays.utils import safe_relative_path

app = typer.Typer()



def choose_threshold(y_true: pd.Series, y_prob: np.ndarray, beta: float = 1.0) -> float:
    """Find the threshold that maximises the F-beta score.

    Args:
        y_true: Observed labels.
        y_prob: Predicted probabilities.
        beta: the beta in F-beta score.

    Returns:
        The cutoff with the highest F-beta score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]

    b2 = beta**2
    denom = b2 * precision + recall
    f = np.where(denom > 0, (1 + b2) * precision * recall / np.where(denom > 0, denom, 1), 0)
    return float(thresholds[int(np.argmax(f))])


def compare_betas(
    y_true: pd.Series,
    y_prob: np.ndarray,
    betas: tuple[float, ...] = (1.0, 1.5, 2.0),
) -> pd.DataFrame:
    """Report the operating point each beta would produce.

    Args:
        y_true: Observed labels.
        y_prob: Predicted probabilities.
        betas: Beta values to compare.

    Returns:
        One row per beta, with the chosen threshold, alert rate, precision and recall.
    """
    rows = []
    for beta in betas:
        thr = choose_threshold(y_true, y_prob, beta=beta)
        pred = (y_prob >= thr).astype(int)
        rows.append({
            "beta": beta,
            "threshold": thr,
            "alert_rate": float(pred.mean()),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
        })
    return pd.DataFrame(rows)


def prepare_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    encoding: str,
    validation_df: pd.DataFrame | None = None,
    resample: str = "none",
) -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame | None, pd.Series | None,
    pd.DataFrame, pd.Series,
]:
    """Fit the transformer on train, transform every split, then resample train only.

    Args:
        train_df: Raw training dataframe for this fold.
        test_df: Raw test dataframe for this fold.
        encoding: Either "onehot" or "native"; determines how categoricals are
            encoded, and which resample methods are compatible.
        validation_df: Raw validation dataframe for this fold. Optional; omit when
            no held-out validation split applies.
        resample: Training-set rebalancing strategy. One of "none", "undersample",
            "oversample", "smote" ("smote" requires encoding="onehot"). Defaults to
            "none" (no rebalancing).

    Returns:
        Tuple of (X_fit, y_fit, X_val, y_val, X_test, y_test).
        X_fit/y_fit reflect the resampling; the other splits are untouched. X_val/
        y_val are (None, None) when validation_df is not given.
    """
    X_fit, y_fit = build_xy(train_df)
    X_test, y_test = build_xy(test_df)

    transformer = Transformer().fit(X_fit, y_fit)
    X_fit = transformer.transform(X_fit)
    X_test = transformer.transform(X_test)

    transformer.select_features(X_fit, y_fit)
    X_fit = transformer.apply_selection(X_fit)
    X_test = transformer.apply_selection(X_test)

    cat_cols = [c for c in transformer.categorical_columns if c in X_fit.columns]
    X_fit = encode_categoricals(X_fit, cat_cols, encoding)
    X_test = encode_categoricals(X_test, cat_cols, encoding)
    X_fit, X_test = align_columns(X_fit, X_test, encoding)

    X_val, y_val = None, None
    if validation_df is not None:
        X_val, y_val = build_xy(validation_df)
        X_val = transformer.transform(X_val)
        X_val = transformer.apply_selection(X_val)
        X_val = encode_categoricals(X_val, cat_cols, encoding)
        X_fit, X_val = align_columns(X_fit, X_val, encoding)

    X_fit, y_fit = resample_training_data(X_fit, y_fit, resample, encoding)

    return X_fit, y_fit, X_val, y_val, X_test, y_test


def train_and_evaluate(
    folds_dir: list[Path],
    variant: str,
    model: str,
    config: str,
    calibrate: bool = True,
    resample: str = "none",
) -> dict[str, float]:
    """Run the full fit/validate/test cycle across every fold, and average the results.

    For each fold: prepares the data (with resampling on train only), fits the
    estimator (with LightGBM early stopping if X_val/y_val are supported), picks the
    F1-optimal threshold on the validation set, evaluates on the held-out test set,
    and logs everything to the active MLflow run.

    Args:
        folds_dir: One directory per fold, each containing train/validation/test
            parquet files.
        variant: Feature set variant name (e.g. 'all', 'noweather', 'nocarrier'); only
            used for logging.
        model: Algorithm key.
        config: Hyperparameter configuration key for that model.
        calibrate: Whether to wrap the estimator in isotonic calibration.
        resample: Training-set rebalancing strategy.

    Returns:
        Per-metric averages across folds, plus "roc_auc_std" (their standard
        deviation), giving a sense of how stable the estimate is across folds.
    """
    encoding = ENCODING[model]
    fold_metrics = []

    for index, fold in enumerate(folds_dir):
        train_df = pd.read_parquet(fold / "train.parquet")
        validation_df = pd.read_parquet(fold / "validation.parquet")
        test_df = pd.read_parquet(fold / "test.parquet")

        X_fit, y_fit, X_val, y_val, X_test, y_test = prepare_fold(
            train_df, test_df, encoding, validation_df=validation_df, resample=resample
        )

        estimator = train.train(
            X_fit, y_fit, model, config, calibrate, X_val=X_val, y_val=y_val
        )


        if index == 0:
            raw_estimator = estimator.estimator if hasattr(estimator, "estimator") else estimator
            if hasattr(raw_estimator, "booster_"):  # lightgbm
                importances = pd.Series(
                    raw_estimator.booster_.feature_importance(importance_type="gain"),
                    index=X_fit.columns,
                ).sort_values(ascending=False)
            elif hasattr(raw_estimator, "feature_importances_"):  
                importances = pd.Series(
                    raw_estimator.feature_importances_, index=X_fit.columns
                ).sort_values(ascending=False)
            else:
                importances = None

            if importances is not None:
                logger.info(f"Top feature importances for {variant}/{model}:\n{importances}")

        val_prob = estimator.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, val_prob)

        if index == 0:
            logger.info(
                f"threshold options for {variant}/{model}:\n"
                f"{compare_betas(y_val, val_prob)}"
            )

        metrics = evaluate.evaluate(X_test, y_test, estimator, threshold)

        mlflow.log_metrics({f"fold_{k}": v for k, v in metrics.items()}, step=index)

        fold_metrics.append(metrics)
        logger.info(
            f"{variant}/{model} fold {index}: "
            f"ROC-AUC {metrics['roc_auc']:.3f}, PR-AUC {metrics['pr_auc']:.3f}, "
            f"recall {metrics['recall']:.3f}, threshold {threshold:.3f}"
        )

    if not fold_metrics:
        raise RuntimeError(f"No usable folds for {variant}/{model}")

    averages = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    averages["roc_auc_std"] = float(np.std([m["roc_auc"] for m in fold_metrics]))
    return averages


@app.command()
def run(
    variant: str = typer.Option(..., help="Which feature set variant to train"),
    model: str = typer.Option(..., help="Which algorithm to train"),
    config: str = typer.Option("default", help="Which hyperparameter set to use"),
    data_path: Path = typer.Option(PROCESSED_DATA_DIR / "selection"),
    experiment: str = typer.Option("flight-delay", help="MLflow experiment name"),
    calibrate: bool = typer.Option(True, help="Wrap estimators in isotonic calibration"),
    resample: str = typer.Option(
        "none",
        help="Training rebalancing strategy: 'none', 'undersample', 'oversample', "
        "'smote' ('smote' only valid with one-hot encoded models).",
    ),
    repo_owner: str = typer.Option("Beviale", help="DagsHub repository owner"),
    repo_name: str = typer.Option("FlightOnTime", help="DagsHub repository name"),
) -> None:
    """Cross-validate one model/config over every fold of a variant, and log averaged results.

    Args:
        variant: Feature set variant to train.
        model: Algorithm name key to train.
        config: Hyperparameter configuration key to use.
        data_path: Root directory containing one subfolder per variant, each with train/validation/test parquet files.
        experiment: MLflow experiment name to log this run under.
        calibrate: Whether to wrap the estimator in isotonic calibration.
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
                "calibrated": calibrate,
                "resample": resample,
                "n_folds": len(val_fold_dirs),
                **{f"hp_{k}": v for k, v in HYPERPARAMS[model][config].items()},
            })
            metrics = train_and_evaluate(
                val_fold_dirs, variant, model, config, calibrate, resample
            )
            mlflow.log_metrics(metrics)

        out_path = METRICS_DIR / variant / f"{model}__{config}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({**metrics, "resample": resample}, indent=2))
    except Exception as e:
        logger.exception(f"An error occurred while training and evaluating the model: {e}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()