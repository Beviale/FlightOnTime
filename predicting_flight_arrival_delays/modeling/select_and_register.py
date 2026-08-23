"""Model selection and registration.

Reads the metrics produced by each training run (train_evaluate_save_metrics.py),
picks the best algorithm/config per production variant using validation-set PR-AUC,
then for that winner:

  1. For every walk-forward fold: fits on that fold's train, calibrates and picks
     the operating threshold on that fold's validation, and evaluates on that
     fold's test. These per-fold scores are averaged.
  2. Using only the last (most recent, widest-training-window) fold: fits on
     train+validation combined, then calibrates and picks an operating
     threshold on that fold's test. This is the model that gets registered.
    Step 1's averaged metrics are what gets logged/reported for it.
"""

import json
from pathlib import Path

import dagshub
import joblib
from loguru import logger
import mlflow
from mlflow.data.pandas_dataset import from_pandas
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
import typer

from predicting_flight_arrival_delays.config import (
    DAGSHUB_REPO_NAME,
    DAGSHUB_REPO_OWNER,
    ENCODING,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    PRODUCTION_VARIANTS,
)
from predicting_flight_arrival_delays.data.features import build_xy
from predicting_flight_arrival_delays.data.transform import (
    Transformer,
    align_columns,
    encode_categoricals,
    resample_training_data,
    sparse_column_order,
    to_sparse_matrix,
)
from predicting_flight_arrival_delays.modeling import evaluate
from predicting_flight_arrival_delays.modeling.train import HYPERPARAMS
from predicting_flight_arrival_delays.modeling.train import train as train_model
from predicting_flight_arrival_delays.modeling.train_evaluate_save_metrics import prepare_fold
from predicting_flight_arrival_delays.utils import (
    get_dvc_data_hash,
    get_git_dirty,
    register_model_bundle,
    safe_relative_path,
)

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


def load_metrics(variant: str) -> dict[tuple[str, str], dict[str, float]]:
    """Read every metrics file written for one variant.

    Args:
        variant: Which variant to load results for.

    Returns:
        (algorithm, config) to its fold-averaged metrics.
    """
    variant_dir = METRICS_DIR / "selection" / variant
    files = sorted(variant_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No metrics in {safe_relative_path(variant_dir)} - run training first.")

    return {tuple(f.stem.split("__")): json.loads(f.read_text()) for f in files}


def all_fold_dirs(variant: str) -> list[Path]:
    """List every walk-forward fold's directory, in order.

    Args:
        variant: Which variant to use.

    Returns:
        Fold directories, sorted by fold number.

    Raises:
        FileNotFoundError: If no fold directories are found, or any is missing
            train/validation/test.
    """
    variant_dir = PROCESSED_DATA_DIR / "selection" / variant
    fold_dirs = sorted(
        (d for d in variant_dir.glob("fold_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not fold_dirs:
        raise FileNotFoundError(f"No folds found under {safe_relative_path(variant_dir)}.")

    required = ["train.parquet", "validation.parquet", "test.parquet"]
    for d in fold_dirs:
        missing = [f for f in required if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f"Fold {safe_relative_path(d)} is missing {missing}.")
    return fold_dirs


def register_winner(
    variant: str,
    algorithm: str,
    config: str,
    resample: str,
    calibrate: bool,
    alias: str | None = None,
    models_path: Path | None = None,
) -> None:
    """Score the winner, then refit it and register it.

    Args:
        variant: Which production variant is being registered.
        algorithm: The winning estimator for that variant.
        config: The winning hyperparameter configuration.
        resample: Training-set rebalancing strategy.
        calibrate: Whether to calibrate predicted probabilities.
        alias: If given, promote this version under this alias immediately (e.g."champion").
        models_path: If given, also save model/transformer/columns locally under
            models_path/variant/algorithm__config. Optional; off by default - MLflow registration is the
            intended way to persist this model, local saving is mainly for
            debugging/inspection.
    """
    encoding = ENCODING[algorithm]
    all_folds = all_fold_dirs(variant)
    fold_dir = all_folds[-1]  # the last fold: used for Step 2, the model registration
    train_df = pd.read_parquet(fold_dir / "train.parquet")
    validation_df = pd.read_parquet(fold_dir / "validation.parquet")
    test_df = pd.read_parquet(fold_dir / "test.parquet")

    with mlflow.start_run(run_name=f"{variant}__final__{algorithm}"):
        # --- Step 1: evaluation, averaged across every fold
        per_fold_metrics = []
        per_fold_baseline = []
        for fold in all_folds:
            fold_train_df = pd.read_parquet(fold / "train.parquet")
            fold_validation_df = pd.read_parquet(fold / "validation.parquet")
            fold_test_df = pd.read_parquet(fold / "test.parquet")

            X_fit_f, y_fit_f, X_val_f, y_val_f, X_test_f, y_test_f = prepare_fold(
                fold_train_df, encoding, test_df=fold_test_df,
                validation_df=fold_validation_df, resample=resample,
            )
            fold_model = train_model(
                X_fit_f, y_fit_f, algorithm, config, calibrate, X_val=X_val_f, y_val=y_val_f
            )
            fold_threshold = choose_threshold(y_val_f, fold_model.predict_proba(X_val_f)[:, 1], 1.2)
            per_fold_metrics.append(
                evaluate.evaluate(X_test_f, y_test_f, fold_model, fold_threshold, 1.2)
            )
            per_fold_baseline.append(float(y_test_f.mean()))

        metrics = {
            k: float(np.mean([m[k] for m in per_fold_metrics]))
            for k in per_fold_metrics[0]
        }
        metrics["roc_auc_std"] = float(np.std([m["roc_auc"] for m in per_fold_metrics]))

        out_path = METRICS_DIR / "winner" / variant / f"{algorithm}__{config}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({**metrics, "resample": resample}, indent=2))

        baseline = float(np.mean(per_fold_baseline))

        if metrics["pr_auc"] <= baseline:
            logger.error(
                f"{variant}: {algorithm}/{config} test PR-AUC {metrics['pr_auc']:.3f} "
                f"does not beat the random-guess PR-AUC baseline {baseline:.3f} - not registering"
            )
            return

        # --- Step 2: the model that is registered.
        full_train_df = pd.concat([train_df, validation_df], ignore_index=True)
        X_full, y_full = build_xy(full_train_df)

        transformer = Transformer(encoding=encoding).fit(X_full, y_full)
        X_full = transformer.transform(X_full)
        transformer.select_features(X_full, y_full)
        X_full = transformer.apply_selection(X_full)
        cat_cols = [c for c in transformer.categorical_columns if c in X_full.columns]
        X_full = encode_categoricals(X_full, cat_cols, encoding)

        X_test2, y_test2 = build_xy(test_df)
        X_test2 = transformer.transform(X_test2)
        X_test2 = transformer.apply_selection(X_test2)
        X_test2 = encode_categoricals(X_test2, cat_cols, encoding)
        X_full, X_test2 = align_columns(X_full, X_test2, encoding)
        if encoding == "onehot":
            X_test2 = to_sparse_matrix(X_test2)

        X_full, y_full = resample_training_data(X_full, y_full, resample, encoding)

        feature_columns = (
            sparse_column_order(X_full) if encoding == "onehot" else list(X_full.columns)
        )

        if encoding == "onehot":
            X_full = to_sparse_matrix(X_full)

        model = train_model(
            X_full, y_full, algorithm, config, calibrate, X_val=X_test2, y_val=y_test2
        )
        final_threshold = choose_threshold(y_test2, model.predict_proba(X_test2)[:, 1], 1.2)

        dataset = from_pandas(
            pd.concat([full_train_df, test_df], ignore_index=True),
            source=str(fold_dir),
            name=f"{variant}_final_train",
            digest=get_dvc_data_hash(PROCESSED_DATA_DIR / "selection"),
        )
        mlflow.log_input(dataset)
        mlflow.set_tag("git_dirty", get_git_dirty())

        n_features = X_full.shape[1]

        mlflow.log_params({
            "variant": variant,
            "algorithm": algorithm,
            "config": config,
            "encoding": encoding,
            "calibrated": calibrate,
            "resample": resample,
            "final": True,
            "n_features": n_features,
            **{f"hp_{k}": v for k, v in HYPERPARAMS[algorithm][config].items()},
        })
        mlflow.log_metrics(metrics)
        mlflow.log_metric("operating_threshold", final_threshold)

        register_model_bundle(
            model=model,
            transformer=transformer,
            columns=feature_columns,
            registered_model_name=f"flight-delay-{variant}",
            signature_sample=X_full[:100].toarray() if hasattr(X_full, "toarray") else X_full.head(100),
            alias=alias,
        )
        logger.success(
            f"Registered flight-delay-{variant} ({n_features} features) "
            f"test PR-AUC {metrics['pr_auc']:.3f}"
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
            columns_file.write_text(json.dumps(feature_columns))

            logger.info(f"Model also saved locally to {safe_relative_path(model_file)}")
            logger.info(f"Transformer also saved locally to {safe_relative_path(transformer_file)}")
            logger.info(f"Training columns also saved locally to {safe_relative_path(columns_file)}")


@app.command()
def run(
    experiment: str = typer.Option("flight-delay-v2", help="MLflow experiment name"),
    calibrate: bool = typer.Option(True, help="Wrap estimators in isotonic calibration"),
    alias: str | None = typer.Option(
        "champion",
        help="Alias to promote each variant's winning model version under.",
    ),
    models_path: Path | None = typer.Option(
        None,
        help="If given, also save model/transformer/columns locally under "
        "models_path/variant/algorithm__config/. None (the default) disables "
        "local saving - MLflow registration is the intended way to persist the "
        "winning model.",
    ),
    repo_owner: str = typer.Option(DAGSHUB_REPO_OWNER, help="DagsHub repository owner"),
    repo_name: str = typer.Option(DAGSHUB_REPO_NAME, help="DagsHub repository name"),
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
            algorithm, config = max(candidates, key=lambda k: candidates[k]["pr_auc_val"])
            winner = candidates[(algorithm, config)]
            resample = winner["resample"]

            logger.success(
                f"{variant}: winner is {algorithm}/{config} "
                f"(validation PR-AUC {winner['pr_auc_val']:.3f}, resample={resample})"
            )
            register_winner(
                variant, algorithm, config, resample,
                calibrate, alias=alias, models_path=models_path,
            )
    except Exception as e:
        logger.exception(f"An error occurred while selecting the winner model: {e}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()