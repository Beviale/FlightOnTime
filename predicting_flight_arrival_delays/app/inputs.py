"""Which columns a request actually has to carry.

The pipeline produces several columns, but a released model rarely reads all of
them: feature selection drops the constant, the redundant and the uninformative ones
before training. Asking a caller for a column the model then throws away is noise, so
the request is defined by what the served models consume rather than by what the
pipeline happens to produce. This set is recovered from the bundle itself, deterministically.


The request covers both served models at once: a column is asked for if either of
them reads it. The two are selected separately and need not agree, and which one
answers is only decided once the forecast has been tried - so asking for the union is
what lets the fallback be served with the same information as the main model.
"""

import pandas as pd

from predicting_flight_arrival_delays.app.schema import FlightRequest
from predicting_flight_arrival_delays.data.transform import OTHER

# Every column a caller could be asked for: the manual-entry request, in full.
CANDIDATE_INPUTS = list(FlightRequest.model_fields)

# Needed to reach the forecast, whatever the model makes of them: where the airports
# are, and when the flight is at each end of the route.
FORECAST_INPUTS = frozenset(
    {
        "FlightDate",
        "DepTimeDecimal",
        "CRSElapsedTime",
        "OriginAirportID",
        "DestAirportID",
    }
)


def contributes(column: str, transformer, model_columns: set[str]) -> bool:
    """Whether a raw column leaves any trace in what the model reads.

    Args:
        column: A raw pipeline column.
        transformer: The fitted transformer the model was trained with.
        model_columns: The model's input columns, from columns.json.

    Returns:
        True if the column survives into the model, directly or through a column
        derived from it.
    """
    if column in model_columns:
        return True

    kept = transformer.category_keep.get(column)
    if kept and {f"{column}_{value}" for value in set(kept) | {OTHER}} & model_columns:
        return True

    derived = f"{column}DelayRate"
    return derived in model_columns and column in transformer.rate_columns()


def contributing_columns(bundle, candidates: list[str]) -> set[str]:
    """Every candidate column the bundle's model actually reads.

    Args:
        bundle: A loaded model bundle.
        candidates: The raw columns to test.

    Returns:
        Those that contribute to the model's input.
    """
    model_columns = set(bundle.columns)
    return {c for c in candidates if contributes(c, bundle.transformer, model_columns)}


def required_inputs(bundles: dict, candidates: list[str]) -> set[str]:
    """The columns a request must carry to be scored by whichever model answers.

    Args:
        bundles: The loaded bundles, keyed by variant.
        candidates: The raw columns a caller could be asked for.

    Returns:
        Every column any served model reads, plus those the forecast lookup needs
        regardless. Empty of model columns if nothing is loaded, in which case the
        request is not held to a model it does not have.
    """
    required = FORECAST_INPUTS & set(candidates)
    for bundle in bundles.values():
        required |= contributing_columns(bundle, candidates)
    return required


def approximated_inputs(flight, bundle, floor: float) -> list[str]:
    """Which of the columns this model leans on most the caller did not send.

    Args:
        flight: The request as it arrived.
        bundle: The model that answered it.
        floor: The share of the model's total weight a column has to carry before
            its absence is worth mentioning.

    Returns:
        Those the request left out, in the model's own order of importance. Empty
        when the request was complete, or when the version registered no ranking.
    """
    ranked = [column for column, share in bundle.importance.items() if share >= floor]
    if not ranked:
        return []

    sent = {name for name in flight.supplied() if getattr(flight, name, None) is not None}
    return [column for column in ranked if column not in sent]


def complete_frame(df: pd.DataFrame, transformer) -> pd.DataFrame:
    """Restore the columns the transformer expects but the request did not carry.

    Args:
        df: The feature frame as the request produced it.
        transformer: The fitted transformer about to receive it.

    Returns:
        A copy carrying every column the transformer will reach for.
    """
    df = df.copy()
    derived = {f"{c}DelayRate" for c in transformer.rate_columns() if c in df.columns}

    for column in transformer.numeric_columns:
        if column not in df.columns and column not in derived:
            df[column] = transformer.impute_values.get(column, 0.0)

    for column in transformer.category_keep:
        if column not in df.columns:
            df[column] = OTHER

    return df
