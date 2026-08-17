"""Model selection and registration.

Reads the metrics produced by each training run, picks the best algorithm for each
production variant, refits it on the whole dataset and registers it.
"""

import json
import re
from pathlib import Path

import dagshub
import joblib
import mlflow
import pandas as pd
import typer
from loguru import logger
from sklearn.base import BaseEstimator

from predicting_flight_arrival_delays.config import (
    ENCODING,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    PRODUCTION_VARIANTS,
)
from predicting_flight_arrival_delays.data.features import build_xy
from predicting_flight_arrival_delays.data.transform import (
    Transformer,
    encode_categoricals,
    resample_training_data,
)
from predicting_flight_arrival_delays.modeling.train import HYPERPARAMS
from predicting_flight_arrival_delays.modeling.train import train as train_model
from predicting_flight_arrival_delays.modeling.train_evaluate_save_metrics import (
    choose_threshold,
    prepare_fold,
)
from predicting_flight_arrival_delays.utils import register_model_bundle, safe_relative_path

app = typer.Typer()


def load_metrics(variant: str) -> dict[tuple[str, str], dict[str, float]]:
    """Read every metrics file written for one variant.

    Args:
        variant: Which variant to load results for.

    Returns:
        (algorithm, config) to its fold-averaged metrics.
    """
    variant_dir = METRICS_DIR / variant
    files = sorted(variant_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No metrics in {safe_relative_path(variant_dir)} - run training first.")

    return {tuple(f.stem.split("__")): json.loads(f.read_text()) for f in files}


def final_split_dir(variant: str) -> Path:
    """Locate the fold used for the final, deployed refit.

    Args:
        variant: Which variant to use.

    Returns:
        The path to that fold's directory.

    """
    variant_dir = PROCESSED_DATA_DIR / "final" / variant
    if not (variant_dir / "train.parquet").exists() or not (variant_dir / "validation.parquet").exists():
        raise FileNotFoundError(
            f"No train/validation split found under {safe_relative_path(variant_dir)}."
        )
    return variant_dir


def _fit_transform_all(
    df: pd.DataFrame,
    encoding: str,
    resample: str,
) -> tuple[pd.DataFrame, pd.Series, Transformer]:
    """Fit a fresh Transformer on the whole of df and return its encoded output.

    Args:
        df: The full dataset to fit and transform.
        encoding: Either "onehot" or "native".
        resample: Training-set rebalancing strategy.

    Returns:
        Tuple of (encoded X, y, the fitted Transformer).
    """
    X, y = build_xy(df)

    transformer = Transformer(encoding=encoding).fit(X, y)
    X = transformer.transform(X)

    transformer.select_features(X, y)
    X = transformer.apply_selection(X)

    cat_cols = [c for c in transformer.categorical_columns if c in X.columns]
    X = encode_categoricals(X, cat_cols, encoding)

    X, y = resample_training_data(X, y, resample, encoding)

    return X, y, transformer


def fit_final(
    variant: str,
    algorithm: str,
    config: str,
    resample: str,
    calibrate: bool = True,
) -> tuple[BaseEstimator, Transformer, pd.DataFrame, float]:
    """Refit the chosen configuration for deployment.

    Stage 1 fits on train.parquet and picks the operating threshold on validation.parquet.
    Stage 2 refits from scratch on train+validation combined with the threshold kept fixed.

    Args:
        variant: Which feature set variant to use.
        algorithm: Which estimator to train.
        config: Which named hyperparameter set to use.
        resample: Training-set rebalancing strategy.
        calibrate: Whether to calibrate predicted probabilities.

    Returns:
        Tuple of (final model, its transformer, the final fitted feature matrix, and
        the operating threshold chosen in Stage 1).
    """
    encoding = ENCODING[algorithm]

    fold_dir = final_split_dir(variant)
    train_df = pd.read_parquet(fold_dir / "train.parquet")
    validation_df = pd.read_parquet(fold_dir / "validation.parquet")

    # --- Stage 1: threshold selection, on data never used to fit this model ---
    X_fit, y_fit, X_val, y_val, _, _ = prepare_fold(
        train_df, validation_df, encoding, validation_df, resample,
    )
    model_temp = train_model(
        X_fit, y_fit, algorithm, config, calibrate, X_val=X_val, y_val=y_val
    )
    raw_estimator = model_temp.estimator if hasattr(model_temp, "estimator") else model_temp
    best_n_estimators = getattr(raw_estimator, "best_iteration_", None)    
    threshold = choose_threshold(y_val, model_temp.predict_proba(X_val)[:, 1])

    # --- Stage 2: refit from scratch on the whole fold, threshold kept fixed ---
    full_df = pd.concat([train_df, validation_df], ignore_index=True)
    X_full, y_full, transformer = _fit_transform_all(full_df, encoding, resample)
    if best_n_estimators is not None:
        config_stage2 = {**HYPERPARAMS[algorithm][config], "n_estimators": best_n_estimators}
        model = train_model(X_full, y_full, algorithm, config, calibrate, json_config=config_stage2)
    else:
        model = train_model(X_full, y_full, algorithm, config, calibrate)

    return model, transformer, X_full, threshold


def register_winner(
    variant: str,
    algorithm: str,
    config: str,
    resample: str,
    metrics: dict[str, float],
    calibrate: bool,
    alias: str | None = None,
    models_path: Path | None = None,
) -> None:
    """Refit the winning configuration; register to MLflow, optionally save locally.

    Args:
        variant: Which production variant is being registered.
        algorithm: The winning estimator for that variant.
        config: The winning hyperparameter configuration.
        resample: Training-set rebalancing strategy.
        metrics: Cross-validated (fold-averaged) metrics to attach to the run.
        calibrate: Whether to calibrate predicted probabilities.
        alias: If given, promote this version under this alias immediately (e.g."champion").
        models_path: If given, also save model/transformer/columns locally under
            models_path/variant/algorithm__config. Optional; off by default - MLflow registration is the
            intended way to persist this model, local saving is mainly for
            debugging/inspection.
    """
    with mlflow.start_run(run_name=f"{variant}__final__{algorithm}"):
        model, transformer, X, threshold = fit_final(
            variant, algorithm, config, resample, calibrate
        )

        mlflow.log_params({
            "variant": variant,
            "algorithm": algorithm,
            "config": config,
            "encoding": ENCODING[algorithm],
            "calibrated": calibrate,
            "resample": resample,
            "final": True,
            "n_features": X.shape[1],
            **{f"hp_{k}": v for k, v in HYPERPARAMS[algorithm][config].items()},
        })
        mlflow.log_metrics(metrics)
        mlflow.log_metric("operating_threshold", threshold)

        register_model_bundle(
            model=model,
            transformer=transformer,
            columns=list(X.columns),
            registered_model_name=f"flight-delay-{variant}",
            signature_sample=X.head(100),
            alias=alias,
        )
        logger.success(
            f"Registered flight-delay-{variant} ({X.shape[1]} features)"
            + (f", promoted as '{alias}'" if alias else "")
        )

        if models_path is not None:
            save_dir = models_path / variant / f"{algorithm}__{config}"
            save_dir.mkdir(parents=True, exist_ok=True)

            model_file = save_dir / "model.joblib"
            transformer_file = save_dir / "transformer.joblib"
            columns_file = save_dir / "columns.json"

            joblib.dump(model, model_file)
            transformer.save(transformer_file)
            columns_file.write_text(json.dumps(list(X.columns)))

            logger.info(f"Model also saved locally to {safe_relative_path(model_file)}")
            logger.info(f"Transformer also saved locally to {safe_relative_path(transformer_file)}")
            logger.info(f"Training columns also saved locally to {safe_relative_path(columns_file)}")


@app.command()
def run(
    experiment: str = typer.Option("flight-delay", help="MLflow experiment name"),
    calibrate: bool = typer.Option(True, help="Wrap estimators in isotonic calibration"),
    alias: str | None = typer.Option(
        "champion",
        help="Alias to promote each variant's winning model version under.",
    ),
    models_path: Path | None = typer.Option(
        None,
        help="If given, also save model/transformer/columns locally under "
        "models_path/variant/algorithm__config/. None (the default) disables "
        "local saving -- MLflow registration is the intended way to persist the "
        "winning model.",
    ),
    repo_owner: str = typer.Option("beviale", help="DagsHub repository owner"),
    repo_name: str = typer.Option("FlightOnTime", help="DagsHub repository name"),
) -> None:
    """Pick the best algorithm per production variant and register it.

    Args:
        experiment: MLflow experiment name to log the final runs under.
        calibrate: Whether to wrap estimators in isotonic calibration.
        alias: Alias to promote the winning version under (e.g. "champion").
        models_path: If given, also save model/transformer/columns locally. None (the default) disables it.
        repo_owner: DagsHub repository owner, used to route MLflow tracking.
        repo_name: DagsHub repository name, used to route MLflow tracking.
    """
    try:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_experiment(experiment)

        for variant in PRODUCTION_VARIANTS:
            candidates = load_metrics(variant)
            algorithm, config = max(candidates, key=lambda k: candidates[k]["pr_auc"])

            winner_metrics = dict(candidates[(algorithm, config)])
            resample = winner_metrics.pop("resample")

            fold_dir = final_split_dir(variant)
            validation_df = pd.read_parquet(fold_dir / "validation.parquet")
            baseline = validation_df["IsDelayed"].mean()

            if winner_metrics["pr_auc"] <= baseline:
                logger.error(
                    f"{variant}: best PR-AUC {winner_metrics['pr_auc']:.3f} "
                    f"does not beat the majority-class baseline {baseline:.3f} -- not registering"
                )
                continue

            logger.success(
                f"{variant}: winner is {algorithm}/{config} "
                f"(PR-AUC {winner_metrics['pr_auc']:.3f}, resample={resample})"
            )
            register_winner(
                variant, algorithm, config, resample,
                winner_metrics, calibrate, alias=alias, models_path=models_path,
            )
    except Exception as e:
        logger.exception(f"An error occurred while selecting the winner model: {e}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()