"""Model training.
"""

import json
from pathlib import Path
from typing import Any

import dagshub
import joblib
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from loguru import logger
import mlflow
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
import typer
import yaml

from predicting_flight_arrival_delays.config import ENCODING, HYPERPARAMS_PATH, MODELS_DIR, SEED
from predicting_flight_arrival_delays.data.features import VARIANTS, build_xy
from predicting_flight_arrival_delays.data.transform import (
    Transformer,
    align_columns,
    encode_categoricals,
)
from predicting_flight_arrival_delays.utils import register_model_bundle, safe_relative_path

app = typer.Typer()

BUILDERS: dict[str, Any] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "lightgbm": LGBMClassifier,
}
EARLY_STOPPING_KEY = "early_stopping_rounds"


def _load_hyperparams(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load the algorithm -> config -> params mapping from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Nested dict: HYPERPARAMS[algorithm][config] -> kwargs for that estimator.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"No hyperparameters file found at {path}")
    with open(path) as f:
        hyperparams = yaml.safe_load(f)

    for configs in hyperparams.values():
        for params in configs.values():
            params["random_state"] = SEED
    return hyperparams


HYPERPARAMS = _load_hyperparams(HYPERPARAMS_PATH)


def _build_model(algorithm: str, config: str, json_config: dict | None = None) -> tuple[BaseEstimator, int | None]:
    """Instantiate an unfitted, uncalibrated estimator for the given algorithm/config.

    Args:
        algorithm: Key identifying the algorithm in BUILDERS.
        config: Key identifying the hyperparameter configuration in HYPERPARAMS. Optional.
        json_config: JSON configuration. Optional. If not None, overrides the config parameter.

        Note: At least one of config or json_config must not be None.

    Returns:
        Tuple of:
            - An unfitted, uncalibrated scikit-learn compatible estimator.
            - The configured early_stopping_rounds, or None if absent/not lightgbm.
    """
    if json_config is None:
        params = dict(HYPERPARAMS[algorithm][config])
    else:
        params = json_config
   
    early_stopping_rounds = params.pop(EARLY_STOPPING_KEY, None)
    if algorithm != "lightgbm":
        early_stopping_rounds = None

    model = BUILDERS[algorithm](**params)
    return model, early_stopping_rounds

def train(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    model: str,
    config: str,
    calibrate: bool,
    json_config: dict | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> BaseEstimator:
    """Fit a single estimator on already-transformed features, then calibrate it.

    Args:
        X_fit: Training features, already transformed/encoded.
        y_fit: Training target, aligned with X_fit.
        model: Key identifying the algorithm in BUILDERS.
        config: Key identifying the hyperparameter configuration in HYPERPARAMS.
        calibrate: Whether to calibrate the fitted estimator with isotonic regression.
        json_config: Complete hyperparameter dict. Optional.
        X_val: Validation features. Used for LightGBM early stopping, and as the
            calibration set when given. Optional.
        y_val: Validation target, aligned with X_val. Optional.

    Returns:
        The fitted estimator. Bare if calibrate=False; a CalibratedClassifierCV
        otherwise (fit on X_val if given, via internal 5-fold CV if not).
    """
    estimator, early_stopping_rounds = _build_model(model, config, json_config)

    fit_kwargs: dict[str, Any] = {}
    if model == "lightgbm" and early_stopping_rounds is not None and X_val is not None and y_val is not None:
        fit_kwargs["eval_X"] = X_val
        fit_kwargs["eval_y"] = y_val
        fit_kwargs["callbacks"] = [
            early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            log_evaluation(period=0),
        ]

    if not calibrate:
        estimator.fit(X_fit, y_fit, **fit_kwargs)
        return estimator

    if X_val is not None and y_val is not None:
        estimator.fit(X_fit, y_fit, **fit_kwargs)
        calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="isotonic")
        calibrated.fit(X_val, y_val)
        return calibrated

    logger.info(
        "calibrate=True but no X_val/y_val given: calibrating via 5-fold internal "
        "cross-validation on the full training set instead."
    )
    calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv=5)
    calibrated.fit(X_fit, y_fit)
    return calibrated


def train_with_transformer(
    train_df: pd.DataFrame,
    variant: str,
    model: str,
    config: str,
    calibrate: bool,
    evaluation_df: pd.DataFrame | None = None,
) -> tuple[Transformer, BaseEstimator, pd.DataFrame]:
    """Fit a Transformer, then fit the estimator on the transformed data.

    If evaluation_df is given, the features are transformed with the same (train-fitted)
    Transformer and used for LightGBM early stopping; ignored for other algorithms.

    Args:
        train_df: Training matrix.
        variant: feature variant to use.
        model: Name of the algorithm to use (key in BUILDERS and ENCODING).
        config: Named hyperparameter configuration set to apply.
        calibrate: Whether to calibrate the model's output probabilities.
        evaluation_df: Validation matrix. Optional.

    Returns:
        Tuple of (fitted Transformer, fitted estimator, the final encoded X_fit).
    """
    X_fit, y_fit = build_xy(train_df, variant)
    encoding_type = ENCODING.get(model, "onehot")

    transformer = Transformer(encoding=encoding_type)
    X_fit = transformer.fit_transform(X_fit)

    transformer.select_features(X_fit, y_fit)
    X_fit = transformer.apply_selection(X_fit)

    cat_cols = [c for c in transformer.categorical_columns if c in X_fit.columns]
    X_fit = encode_categoricals(X_fit, cat_cols, encoding_type)

    X_val = None
    y_val = None
    if evaluation_df is not None:
        X_val, y_val = build_xy(evaluation_df, variant)
        X_val = transformer.transform(X_val)
        X_val = transformer.apply_selection(X_val)
        X_val = encode_categoricals(X_val, cat_cols, encoding_type)
        X_fit, X_val = align_columns(X_fit, X_val, encoding_type)

    estimator = train(X_fit, y_fit, model, config, calibrate, X_val=X_val, y_val=y_val)

    return transformer, estimator, X_fit


@app.command()
def run(
    variant: str = typer.Option(..., help="Which feature set variant to train"),
    model: str = typer.Option(..., help="Which algorithm to train"),
    config: str = typer.Option("default", help="Which hyperparameter set to use"),
    train_path: Path = typer.Option(..., help="Path to training parquet file"),
    validation_path: Path | None = typer.Option(
        None, help="Path to validation parquet file, used for LightGBM early stopping"
    ),
    calibrate: bool = typer.Option(True, help="Wrap estimators in isotonic calibration"),
    models_path: Path | None = typer.Option(
        MODELS_DIR, help="Directory to save trained model and transformer locally. None disables local saving."
    ),
    experiment: str | None = typer.Option(
        None,
        help="MLflow experiment name.",
    ),
    registered_model_name: str | None = typer.Option(
        None,
        help="MLflow registered model name.",
    ),
    alias: str | None = typer.Option(
        None,
        help="MLflow alias.",
    ),
    repo_owner: str = typer.Option("beviale", help="DagsHub repository owner"),
    repo_name: str = typer.Option("FlightOnTime", help="DagsHub repository name"),
) -> tuple[BaseEstimator, Transformer, list[str]]:
    """Train one model/config; optionally save locally and/or register to MLflow.

    Local saving and MLflow registration are independent: either, both, or
    neither can be active in a single call, controlled respectively by
    models_path and by experiment/registered_model_name.

    Args:
        variant: Feature set variant.
        model: Algorithm key.
        config: Hyperparameter config key.
        train_path: Training Parquet file.
        validation_path: Validation Parquet file, for LightGBM early stopping and for calibration. None
            disables it.
        calibrate: Whether to wrap the estimator in isotonic calibration.
        models_path: Local root save directory. None disables local saving.
        experiment: MLflow experiment name. None (or registered_model_name=None)
            disables MLflow registration.
        registered_model_name: MLflow registered model name. None (or
            experiment=None) disables MLflow registration.
        alias: Optional alias to promote this version under immediately (e.g.
            "champion"). None (the default) leaves the version unpromoted.
        repo_owner: DagsHub repository owner, used to route MLflow tracking.
        repo_name: DagsHub repository name, used to route MLflow tracking.

    Returns:
        (fitted estimator, fitted Transformer, training columns).
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

        if not train_path.exists():
            raise FileNotFoundError(f"No dataframe file found at {safe_relative_path(train_path)}")

        train_df = pd.read_parquet(train_path)
        if train_df.empty:
            raise ValueError(f"Loaded dataframe at {safe_relative_path(train_path)} is empty")


        validation_df = None
        if validation_path is not None:
            if not validation_path.exists():
                raise FileNotFoundError(f"No dataframe file found at {safe_relative_path(validation_path)}")
            validation_df = pd.read_parquet(validation_path)
            if validation_df.empty:
                raise ValueError(f"Loaded dataframe at {safe_relative_path(validation_path)} is empty")


        logger.info(f"Training {model} ({config}) for variant '{variant}'...")
        transformer, estimator, X_fit = train_with_transformer(
            train_df, variant, model, config, calibrate, validation_df
        )
        columns = list(estimator.feature_names_in_)

        if models_path is not None:
            save_dir = models_path / variant / f"{model}__{config}"
            save_dir.mkdir(parents=True, exist_ok=True)

            model_file = save_dir / "model.joblib"
            transformer_file = save_dir / "transformer.joblib"
            columns_file = save_dir / "columns.json"

            joblib.dump(estimator, model_file)
            joblib.dump(transformer, transformer_file)
            columns_file.write_text(json.dumps(columns))

            logger.info(f"Model saved to {safe_relative_path(model_file)}")
            logger.info(f"Transformer saved to {safe_relative_path(transformer_file)}")
            logger.info(f"Training columns saved to {safe_relative_path(columns_file)}")

        if experiment is not None and registered_model_name is not None:

            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            mlflow.set_experiment(experiment)

            with mlflow.start_run(run_name=f"{variant}__{model}__{config}"):
                mlflow.log_params({
                    "variant": variant, "algorithm": model, "config": config,
                    "encoding": ENCODING[model], "calibrated": calibrate,
                })
                register_model_bundle(
                    model=estimator,
                    transformer=transformer,
                    columns=columns,
                    registered_model_name=registered_model_name,
                    signature_sample=X_fit.head(100),
                    alias=alias,
                )
            logger.info(f"Registered to MLflow as '{registered_model_name}'")

        return estimator, transformer, columns
    except Exception as e:
        logger.exception(f"An error occurred while training the model: {e}")
        raise typer.Exit(code=1) from None

if __name__ == "__main__":
    app()