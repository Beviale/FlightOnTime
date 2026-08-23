"""Model evaluation.
"""

import json
from pathlib import Path

import dagshub
import joblib
from loguru import logger
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import typer

from predicting_flight_arrival_delays.config import (
    DAGSHUB_REPO_NAME,
    DAGSHUB_REPO_OWNER,
    METRICS_DIR,
)
from predicting_flight_arrival_delays.data.features import build_xy
from predicting_flight_arrival_delays.data.transform import (
    Transformer,
    align_to_training_columns,
    encode_categoricals,
)
from predicting_flight_arrival_delays.utils import (
    get_run_params,
    load_model_bundle,
    safe_relative_path,
)

app = typer.Typer()


def evaluate(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    estimator: BaseEstimator,
    threshold: float | None = None,
    fbeta: float = 1.0,
    ) -> dict[str, float]:
    """Evaluate a trained model on an evaluation dataset.

    Args:
        X_test: Evaluation features, already transformed/encoded exactly as the
            estimator was trained on.
        y_test: Evaluation target, aligned with X_test.
        estimator: Trained scikit-learn or LightGBM model instance.
        threshold: Decision threshold for binary classification. 
            Default to None. If None the threshold-based metrics are not computed.
        fbeta: The beta in the F-beta score. Default to 1.0.    
    Returns:
        Dictionary containing evaluated classification metrics.
    """
    y_prob = estimator.predict_proba(X_test)[:, 1]
    if threshold is not None:
        y_pred = (y_prob >= threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "brier": float(brier_score_loss(y_test, y_prob)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            f"f{fbeta}": float(fbeta_score(y_test, y_pred, beta=fbeta, zero_division=0)),
            "alert_rate": float(y_pred.mean()),
            "threshold": float(threshold),
        }
    else:
         return {
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "brier": float(brier_score_loss(y_test, y_prob)),
        }


def _evaluate_on_dataframe(
    evaluate_df: pd.DataFrame,
    variant: str,
    estimator: BaseEstimator,
    transformer: Transformer,
    training_columns: list[str] | None,
    threshold: float,
) -> dict[str, float]:
    """Prepare evaluate_df exactly as the model was trained, then evaluate it.

    Args:
        evaluate_df: Raw evaluation rows, features + target.
        variant: Feature set variant this model was trained for.
        estimator: Fitted model to evaluate.
        transformer: The Transformer fitted alongside estimator.
        training_columns: Exact column names/order the model was fitted on, for
            reindexing. Optional; if None, evaluate_df's columns are used as-is,
            with a warning.
        threshold: Decision threshold for classification metrics.

    Returns:
        Dictionary of evaluation metrics.
    """
    X_test, y_test = build_xy(evaluate_df, variant)
    X_test = transformer.transform(X_test)
    X_test = transformer.apply_selection(X_test)
    cat_cols = [c for c in transformer.categorical_columns if c in X_test.columns]
    X_test = encode_categoricals(X_test, cat_cols, transformer.encoding)

    if training_columns is not None:
        X_test = align_to_training_columns(X_test, training_columns, transformer)
    else:
        logger.warning(
            "No training column list available; assuming evaluate_df already "
            "covers every category seen in training."
        )

    return evaluate(X_test, y_test, estimator, threshold)


@app.command()
def evaluate_from_local_path(
    model_path: Path = typer.Option(
        ..., help="Path to saved model directory (models_path/variant/model__config)"
    ),
    evaluate_df_path: Path = typer.Option(
        ..., help="Path to evaluation Parquet dataset"
    ),
    output_path: Path = typer.Option(
        METRICS_DIR, help="Base output directory for JSON metrics"
    ),
    threshold: float = typer.Option(
        0.5, help="Decision threshold for classification metrics"
    ),
) -> dict[str, float]:
    """Load a locally saved model artifact, evaluate it, and write metrics JSON.

    model_path must contain "model.joblib", "transformer.joblib" and "columns.json".

    Args:
        model_path: Path to the saved model directory.
        evaluate_df_path: Path to the evaluation Parquet dataset.
        output_path: Base directory where metrics JSON will be stored. Defaults to
            METRICS_DIR.
        threshold: Threshold used for converting probabilities to binary
            predictions. Defaults to 0.5.

    Returns:
        Dictionary containing computed evaluation metrics.
    """
    try:
        if not evaluate_df_path.exists():
            raise FileNotFoundError(f"Evaluation file not found at {safe_relative_path(evaluate_df_path)}")

        evaluate_df = pd.read_parquet(evaluate_df_path)
        if evaluate_df.empty:
            raise ValueError(f"Evaluation dataframe at {safe_relative_path(evaluate_df_path)} is empty")

        model_file = model_path / "model.joblib"
        transformer_file = model_path / "transformer.joblib"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at {safe_relative_path(model_file)}")
        if not transformer_file.exists():
            raise FileNotFoundError(f"Transformer file not found at {safe_relative_path(transformer_file)}")

        try:
            model, config = model_path.name.split("__")
            variant = model_path.parent.name
        except ValueError as err:
            raise ValueError(
                f"Folder structure '{safe_relative_path(model_path)}' does not match expected pattern '.../<variant>/<model>__<config>'"
            ) from err

        estimator = joblib.load(model_file)
        transformer: Transformer = joblib.load(transformer_file)

        logger.info(f"Evaluating {variant}/{model}__{config} on {evaluate_df_path.name}...")

        columns_path = model_path / "columns.json"
        training_columns = json.loads(columns_path.read_text()) if columns_path.exists() else None

        metrics = _evaluate_on_dataframe(
            evaluate_df, variant, estimator, transformer, training_columns, threshold
        )

        out_path = output_path / variant / f"{model}__{config}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        logger.success(
            f"{variant}/{model}__{config}: ROC-AUC {metrics['roc_auc']:.3f}, "
            f"PR-AUC {metrics['pr_auc']:.3f}, Brier {metrics['brier']:.3f} -> {out_path}"
        )
        return metrics
    except Exception as e:
        logger.exception(f"An error occurred while evaluating the model: {e}")
        raise typer.Exit(code=1) from None


@app.command()
def evaluate_from_mlflow(
    registered_model_name: str = typer.Option(
        ..., help="MLflow registered model name (e.g. 'flight-delay-all')"
    ),
    evaluate_df_path: Path = typer.Option(
        ..., help="Path to evaluation Parquet dataset"
    ),
    stage: str = typer.Option(
        "None", help="Registry stage or alias to resolve (e.g. 'champion'). "
        "'None' resolves to the latest registered version."
    ),
    output_path: Path = typer.Option(
        METRICS_DIR, help="Base output directory for JSON metrics"
    ),
    threshold: float = typer.Option(
        0.5, help="Decision threshold for classification metrics"
    ),
    repo_owner: str = typer.Option(DAGSHUB_REPO_OWNER, help="DagsHub repository owner"),
    repo_name: str = typer.Option(DAGSHUB_REPO_NAME, help="DagsHub repository name"),
) -> dict[str, float]:
    """Load a model straight from the MLflow registry, evaluate it, write metrics JSON.

    Args:
        registered_model_name: Name the model version was registered under.
        evaluate_df_path: Path to the evaluation Parquet dataset.
        stage: Registry stage or alias to resolve (e.g. "champion").
        output_path: Base directory where metrics JSON will be stored.
        threshold: Threshold used for converting probabilities to binary predictions.
        repo_owner: DagsHub repository owner, used to route MLflow tracking.
        repo_name: DagsHub repository name, used to route MLflow tracking.
 
    Returns:
        Dictionary containing computed evaluation metrics.
    """
    try:
        if not evaluate_df_path.exists():
            raise FileNotFoundError(f"Evaluation file not found at {evaluate_df_path}")
    
        evaluate_df = pd.read_parquet(evaluate_df_path)
        if evaluate_df.empty:
            raise ValueError(f"Evaluation dataframe at {evaluate_df_path} is empty")
    
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
        logger.info(f"Loading '{registered_model_name}' (stage='{stage}') from MLflow registry...")
        estimator, transformer, training_columns, run_id = load_model_bundle(
            registered_model_name, stage=stage
        )
    
        variant = get_run_params(run_id)["variant"]

        logger.info(
            f"Evaluating {registered_model_name} (stage='{stage}') on {evaluate_df_path.name}..."
        )
        metrics = _evaluate_on_dataframe(
            evaluate_df, variant, estimator, transformer, training_columns, threshold
        )
    
        out_path = output_path / variant / f"{registered_model_name}__{stage}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
        logger.success(
            f"{registered_model_name} (stage='{stage}'): ROC-AUC {metrics['roc_auc']:.3f}, "
            f"PR-AUC {metrics['pr_auc']:.3f}, Brier {metrics['brier']:.3f} -> {out_path}"
        )
    
        return metrics
    except Exception as e:
        logger.exception(f"An error occurred while evaluating the model: {e}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()