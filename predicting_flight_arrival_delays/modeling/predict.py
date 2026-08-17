"""Inference: score flights with a registered model.

Loads a model from the MLflow registry together with the transformer it was trained
with, so live requests are shaped exactly as the training data was. Both the model
and the transformer/column list it needs are recovered entirely from the registry --
no dependency on the local filesystem of the machine that ran training (see
load_transformer/load_columns).

Which variant is used depends on the input: flights carrying weather features go to
"all", flights without them to "noweather".
"""

import json
from functools import lru_cache
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import pandas as pd
import typer
from loguru import logger

from predicting_flight_arrival_delays.config import INTERIM_DATA_DIR, WEATHER_COLUMNS
from predicting_flight_arrival_delays.data.features import get_feature_columns
from predicting_flight_arrival_delays.data.transform import Transformer, encode_categoricals

app = typer.Typer()

DEFAULT_STAGE = "None"

# load_model/load_transformer/load_columns below are @lru_cache'd for the
# lifetime of this process. If the "champion" alias is repointed to a new
# version by select_and_register.py while this process is still running, the
# cache would keep serving the OLD version -- lru_cache has no way to notice an
# alias was repointed, since the cache key ("champion") never changes. This
# project's answer is operational, not code: select_and_register.py triggers a
# restart of the inference process after every successful registration (see its
# --restart-webhook-url), and a fresh process starts with an empty cache. If
# this service is ever run without that restart step, the cache can go stale.


def _registered_run_id(variant: str, stage: str = DEFAULT_STAGE) -> str:
    """Resolve the run that produced a registered model version.

    models:/<name>/<stage> only serves what mlflow.sklearn.log_model itself wrote
    (MLmodel, the pickled estimator, etc.). Artifacts logged separately into the
    same artifact_path (transformer.joblib, columns.json -- see
    modeling/select.py::register_winner) are NOT included when downloading through
    that URI, even though they live in the same run's "model/" folder. Downloading
    via runs:/<run_id>/model/... instead reaches the actual run artifact store,
    where those files are present.

    Args:
        variant: Which production variant to resolve.
        stage: Registry stage or version alias.

    Returns:
        The run id that logged that model version.

    Raises:
        FileNotFoundError: If no registered version exists for that variant.
    """
    client = mlflow.MlflowClient()
    name = f"flight-delay-{variant}"

    if stage not in (DEFAULT_STAGE, ""):
        return client.get_model_version_by_alias(name, stage).run_id

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise FileNotFoundError(f"No registered versions found for {name}")
    return max(versions, key=lambda v: int(v.version)).run_id


@lru_cache(maxsize=4)
def load_model(variant: str, stage: str = DEFAULT_STAGE):
    """Load a registered model from the MLflow registry.

    Cached, so repeated calls in a long-running service do not re-download it.

    Args:
        variant: Which production variant to load.
        stage: Registry stage or version alias to resolve.

    Returns:
        The loaded estimator.
    """
    uri = f"models:/flight-delay-{variant}/{stage}"
    logger.info(f"Loading {uri}")
    return mlflow.sklearn.load_model(uri)


@lru_cache(maxsize=4)
def load_transformer(variant: str, stage: str = DEFAULT_STAGE) -> Transformer:
    """Download and load the transformer bundled with a registered model version.

    Args:
        variant: Which production variant to load.
        stage: Registry stage or version alias.

    Returns:
        The fitted transformer.
    """
    run_id = _registered_run_id(variant, stage)
    path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/model/transformer_{variant}.joblib"
    )
    return Transformer.load(Path(path))


@lru_cache(maxsize=4)
def load_columns(variant: str, stage: str = DEFAULT_STAGE) -> tuple[str, ...]:
    """Download the exact column order the registered model was fitted on.

    Written by modeling/select.py::register_winner. Needed because the encoded
    feature set of a live batch can otherwise differ from what the model expects --
    e.g. a batch missing a category present in training produces one fewer one-hot
    column, and column order itself must match for the underlying numpy array the
    estimator sees. See align_to_training_columns, which uses this to fix both.

    Args:
        variant: Which production variant to load.
        stage: Registry stage or version alias.

    Returns:
        The fitted column names, in training order.
    """
    run_id = _registered_run_id(variant, stage)
    path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/model/columns_{variant}.json"
    )
    return tuple(json.loads(Path(path).read_text()))


def align_to_training_columns(
    X: pd.DataFrame,
    variant: str,
    transformer: Transformer,
) -> pd.DataFrame:
    """Reindex encoded features to exactly the columns the model was fitted on.

    Two problems this fixes, both silent otherwise:
    - Missing one-hot columns: a live batch may not contain every category seen in
      training, producing fewer dummy columns than the model expects. Filled with
      0 -- "this category is absent", the correct value for a one-hot column that
      never got created.
    - Column order: scikit-learn estimators read the underlying array positionally;
      even same-named columns in a different order can silently misalign features.

    For "native" (category dtype) encoding, this mainly re-establishes column
    order; categorical columns are re-cast after reindexing since a
    reindex-introduced fill value can otherwise coerce the dtype away from
    "category".

    Args:
        X: Encoded features for a live batch, from encode_categoricals.
        variant: Which production variant this batch is being prepared for.
        transformer: The transformer that produced X, used to know which columns
            are categorical (for the dtype re-cast described above).

    Returns:
        X reindexed to the training-time columns, in the training-time order.
    """
    training_columns = load_columns(variant)
    X = X.reindex(columns=list(training_columns), fill_value=0)

    if transformer.encoding == "native":
        for col in transformer.categorical_columns:
            if col in X.columns:
                X[col] = X[col].astype("category")

    return X


def has_weather(df: pd.DataFrame) -> pd.Series:
    """Flag rows that carry usable weather features.

    Args:
        df: Flights to check.

    Returns:
        A boolean Series, True where every weather column is present and non-null.
    """
    present = [c for c in WEATHER_COLUMNS if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    return df[present].notna().all(axis=1)


def prepare_for_inference(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Apply the fitted transformation to raw flights, ready for the model.

    Args:
        df: Flights with the same columns the pipeline produces.
        variant: Which production variant to prepare for.

    Returns:
        The transformed feature matrix, column-aligned to what the model expects.
    """
    transformer = load_transformer(variant)

    X = df[get_feature_columns(df, variant)]
    X = transformer.transform(X)
    X = transformer.apply_selection(X)

    cat_cols = [c for c in transformer.categorical_columns if c in X.columns]
    X = encode_categoricals(X, cat_cols, transformer.encoding)

    return align_to_training_columns(X, variant, transformer)


def predict_variant(df: pd.DataFrame, variant: str, threshold: float) -> pd.DataFrame:
    """Score a set of flights with one variant.

    Args:
        df: Flights to score.
        variant: Which production variant to use.
        threshold: Cutoff turning the probability into a delayed/on-time label.

    Returns:
        One row per flight, with the predicted probability, the label and the variant
        that produced them.
    """
    X = prepare_for_inference(df, variant)
    model = load_model(variant)

    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "delay_probability": proba,
            "is_delayed": (proba >= threshold).astype(int),
            "variant": variant,
        },
        index=df.index,
    )


def predict(
    df: pd.DataFrame,
    threshold_all: float,
    threshold_noweather: float,
) -> pd.DataFrame:
    """Score flights, routing each one to the variant its input supports.

    Args:
        df: Flights to score.
        threshold_all: Operating threshold for the `all` model.
        threshold_noweather: Operating threshold for the `noweather` model.

    Returns:
        One row per input flight, in the original order.

    Raises:
        ValueError: If df is empty.
    """
    if df.empty:
        raise ValueError("No flights to score: df is empty.")

    with_weather = has_weather(df)
    logger.info(
        f"{int(with_weather.sum())} flights with weather, "
        f"{int((~with_weather).sum())} without"
    )

    parts = []
    if with_weather.any():
        parts.append(predict_variant(df[with_weather], "all", threshold_all))
    if (~with_weather).any():
        parts.append(predict_variant(df[~with_weather], "noweather", threshold_noweather))

    return pd.concat(parts).reindex(df.index)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    input_path: Path = typer.Argument(..., help="Parquet or CSV of flights to score"),
    output_path: Path = typer.Option(INTERIM_DATA_DIR / "predictions.parquet"),
    threshold_all: float = typer.Option(0.5, help="Operating threshold for `all`"),
    threshold_noweather: float = typer.Option(0.5, help="Operating threshold for `noweather`"),
    repo_owner: str = typer.Option("se4ai2526-uniba", help="DagsHub repository owner"),
    repo_name: str = typer.Option("FlightOnTime", help="DagsHub repository name"),
) -> None:
    """Score a batch of flights and write the predictions to disk.

    Args:
        input_path: Parquet or CSV file with flights to score.
        output_path: Where to write the predictions parquet.
        threshold_all: Operating threshold for the `all` model.
        threshold_noweather: Operating threshold for the `noweather` model.
        repo_owner: DagsHub repository owner, used to route MLflow tracking.
        repo_name: DagsHub repository name, used to route MLflow tracking.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If input_path is empty.
    """
    import dagshub

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    if not input_path.exists():
        raise FileNotFoundError(f"No dataframe file found at {input_path}")

    df = (
        pd.read_parquet(input_path)
        if input_path.suffix == ".parquet"
        else pd.read_csv(input_path)
    )
    logger.info(f"Scoring {len(df)} flights from {input_path}")

    predictions = predict(df, threshold_all, threshold_noweather)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)
    logger.success(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    app()