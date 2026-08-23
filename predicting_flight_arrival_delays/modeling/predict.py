"""Inference: score flights with a registered model.

Loads a model from the MLflow registry together with the transformer it was trained
with, so live requests are shaped exactly as the training data was. Both the model
and the transformer/column list it needs are recovered entirely from the registry -
no dependency on the local filesystem of the machine that ran training.

Which variant is used depends on the input: flights carrying weather features go to
"all", flights without them to "noweather".
"""

from functools import lru_cache
from pathlib import Path

import dagshub
from loguru import logger
import pandas as pd
import typer

from predicting_flight_arrival_delays.config import (
    DAGSHUB_REPO_NAME,
    DAGSHUB_REPO_OWNER,
    INTERIM_DATA_DIR,
)
from predicting_flight_arrival_delays.data.features import (
    WEATHER_COLUMNS_DESTINATION,
    WEATHER_COLUMNS_ORIGIN,
    get_feature_columns,
)
from predicting_flight_arrival_delays.data.transform import (
    Transformer,
    align_to_training_columns,
    encode_categoricals,
)
from predicting_flight_arrival_delays.utils import load_model_bundle

app = typer.Typer()

DEFAULT_STAGE = "None"


WEATHER_FEATURE_COLUMNS = WEATHER_COLUMNS_ORIGIN + WEATHER_COLUMNS_DESTINATION


@lru_cache(maxsize=4)
def _load_bundle(variant: str, stage: str = DEFAULT_STAGE):
    """Load and cache a variant's full model bundle from the registry.

    A single cached call point for model, transformer and training columns
    together.

    Args:
        variant: Which production variant to load.
        stage: Registry stage or version alias to resolve.

    Returns:
        (estimator, transformer, columns, run_id) - see load_model_bundle.
    """
    logger.info(f"Loading flight-delay-{variant} ({stage})")
    return load_model_bundle(f"flight-delay-{variant}", stage=stage)


def has_weather(df: pd.DataFrame) -> pd.Series:
    """Flag rows that carry usable weather features.

    Args:
        df: Flights to check.

    Returns:
        A boolean Series, True where every weather column is present and non-null.
    """
    missing = [c for c in WEATHER_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return pd.Series(False, index=df.index)
    return df[WEATHER_FEATURE_COLUMNS].notna().all(axis=1)


def prepare_for_inference(
    df: pd.DataFrame,
    variant: str,
    transformer: Transformer,
    training_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Apply the fitted transformation to raw flights, ready for the model.

    Args:
        df: Flights with the same columns the pipeline produces.
        variant: Which production variant to prepare for.
        transformer: That variant's fitted transformer, from _load_bundle.
        training_columns: That variant's fitted column order, from _load_bundle.

    Returns:
        The transformed feature matrix, column-aligned to what the model expects.
    """
    X = df[get_feature_columns(df, variant)]
    X = transformer.transform(X)
    X = transformer.apply_selection(X)

    cat_cols = [c for c in transformer.categorical_columns if c in X.columns]
    X = encode_categoricals(X, cat_cols, transformer.encoding)

    return align_to_training_columns(X, training_columns, transformer)


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
    model, transformer, training_columns, _ = _load_bundle(variant)
    X = prepare_for_inference(df, variant, transformer, training_columns)

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
    repo_owner: str = typer.Option(DAGSHUB_REPO_OWNER, help="DagsHub repository owner"),
    repo_name: str = typer.Option(DAGSHUB_REPO_NAME, help="DagsHub repository name"),
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