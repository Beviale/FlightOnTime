"""The registered champions, loaded from the MLflow registry, scored on real flights.

One bundle per production varian:

    flight-delay-all        @champion
    flight-delay-noweather  @champion

That is the artefact production would serve, complete with the transformer and
the column list that 'register_model_bundle' stored alongside it, so these tests
exercise the same thing 'predict.py' loads.

Two things have to be in place, and each skips with its own message when it is
not: credentials for the registry, and the processed folds pulled from DVC.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import dagshub
import pandas as pd
import pyarrow.parquet as pq
import pytest

from predicting_flight_arrival_delays.config import (
    DAGSHUB_REPO_NAME,
    DAGSHUB_REPO_OWNER,
    PROCESSED_DATA_DIR,
    PRODUCTION_VARIANTS,
    TARGET,
)
from predicting_flight_arrival_delays.modeling.predict import prepare_for_inference
from predicting_flight_arrival_delays.utils import get_run_params, load_model_bundle

# The alias select_and_register promotes each winner under.
ALIAS = "champion"


SAMPLE_ROWS = 2000


def _credentials_available() -> bool:
    """Whether the registry can be reached without prompting anyone."""
    if os.environ.get("DAGSHUB_USER_TOKEN") or os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        return True
    return (Path.home() / ".dagshub" / "tokens").exists()


def _find_last_fold(variant: str):
    """The most recent walk-forward fold's directory, or None if absent."""
    variant_dir = PROCESSED_DATA_DIR / "selection" / variant
    if not variant_dir.is_dir():
        return None

    folds = sorted(
        (d for d in variant_dir.glob("fold_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[1]),
    )
    return folds[-1] if folds else None


def _read_sample(path, n_rows: int) -> pd.DataFrame:
    """Read up to n_rows spread across the file's row groups."""
    parquet = pq.ParquetFile(path)
    stride = max(1, parquet.metadata.num_row_groups // 8)

    frames, taken = [], 0
    for index in range(0, parquet.metadata.num_row_groups, stride):
        frames.append(parquet.read_row_group(index).to_pandas())
        taken += len(frames[-1])
        if taken >= n_rows:
            break

    return pd.concat(frames, ignore_index=True).head(n_rows)


@pytest.fixture(scope="session")
def registry():
    """Point MLflow at DagsHub, once per session."""
    if not _credentials_available():
        pytest.skip(
            "no DagsHub credentials found; set DAGSHUB_USER_TOKEN or run 'dagshub login' "
            "to test against the registry"
        )

    try:
        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    except Exception as error:
        pytest.skip(f"could not reach the MLflow registry: {error}")


def _load_bundle(variant: str, registry):
    """Load one variant's champion and the held-out flights to score it on."""
    fold_dir = _find_last_fold(variant)
    if fold_dir is None or not (fold_dir / "test.parquet").exists():
        pytest.skip(f"no walk-forward folds for '{variant}' - run `dvc pull`")

    registered_name = f"flight-delay-{variant}"
    try:
        estimator, transformer, columns, run_id = load_model_bundle(registered_name, stage=ALIAS)
    except Exception as error:
        pytest.skip(f"no '{ALIAS}' version of {registered_name}: {error}")

    try:
        params = get_run_params(run_id)
        trained_as = f"{params.get('algorithm')}__{params.get('config')}"
    except Exception:
        trained_as = "unknown"

    sample = _read_sample(fold_dir / "test.parquet", SAMPLE_ROWS)

    def score(rows: pd.DataFrame):
        """Score flights the way production does, and return probabilities."""
        X = prepare_for_inference(rows, variant, transformer, tuple(columns))
        return estimator.predict_proba(X)[:, 1]

    return SimpleNamespace(
        name=f"{registered_name}@{ALIAS} ({trained_as})",
        variant=variant,
        estimator=estimator,
        transformer=transformer,
        columns=tuple(columns),
        sample=sample,
        carries_weather="PrecipitationOrigin" in sample.columns,
        base_rate=float(sample[TARGET].mean()),
        score=score,
    )


@pytest.fixture(scope="session", params=PRODUCTION_VARIANTS)
def model(request, registry):
    """One registered champion per production variant."""
    return _load_bundle(request.param, registry)


@pytest.fixture(scope="session")
def sample(model):
    """Held-out flights from the model's own variant and fold."""
    return model.sample


@pytest.fixture
def weather_model(model):
    """The subset of models whose variant actually carries weather features."""
    if not model.carries_weather:
        pytest.skip(f"variant '{model.variant}' has no weather features")
    return model
