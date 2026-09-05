"""Scoring against the bundles held in application state.
"""

import pandas as pd

from predicting_flight_arrival_delays.app.inputs import complete_frame
from predicting_flight_arrival_delays.app.utils import Bundle
from predicting_flight_arrival_delays.modeling.predict import has_weather, prepare_for_inference


class ModelUnavailableError(RuntimeError):
    """Raised when the variant a flight needs is not loaded."""


def prepared_matrix(df: pd.DataFrame, bundle: Bundle) -> pd.DataFrame:
    """Build the matrix this model actually reads, for these flights.

    Args:
        df: Flights, already completed with whatever the request left out.
        bundle: The variant to prepare for.

    Returns:
        The transformed, encoded, column-aligned matrix.
    """
    return prepare_for_inference(
        df, bundle.variant, bundle.transformer, tuple(bundle.columns)
    )


def score_variant(df: pd.DataFrame, bundle: Bundle, threshold: float) -> pd.DataFrame:
    """Score a set of flights with one loaded variant.

    Args:
        df: Flights, as the request produced them.
        bundle: The variant to score with.
        threshold: Cutoff turning the probability into a delayed/on-time label.

    Returns:
        One row per flight: the probability, the label, the variant and the threshold
        that produced them.
    """
    df = complete_frame(df, bundle.transformer)
    X = prepared_matrix(df, bundle)
    probability = bundle.model.predict_proba(X)[:, 1]

    return pd.DataFrame(
        {
            "delay_probability": probability,
            "is_delayed": (probability >= threshold).astype(int),
            "variant": bundle.variant,
            "threshold": threshold,
        },
        index=df.index,
    )


def score(
    df: pd.DataFrame,
    bundles: dict[str, Bundle],
    threshold: float | None = None,
) -> pd.DataFrame:
    """Score flights, routing each to the variant its input supports.

    Args:
        df: Flights.
        bundles: The variants loaded at startup, keyed by variant.
        threshold: One cutoff for whichever variant answers.

    Returns:
        One row per input flight, in the original order.

    Raises:
        ValueError: If df is empty.
        ModelUnavailableError: If a flight needs a variant that is not loaded.
    """
    if df.empty:
        raise ValueError("No flights to score: the feature frame is empty.")

    with_weather = has_weather(df)
    routing = [("all", with_weather), ("noweather", ~with_weather)]

    parts = []
    for variant, mask in routing:
        if not mask.any():
            continue

        bundle = bundles.get(variant)
        if bundle is None:
            raise ModelUnavailableError(
                f"The '{variant}' model is not loaded, and {int(mask.sum())} of the "
                f"{len(df)} flights in this request need it. Has it been registered yet?"
            )

        cutoff = bundle.threshold if threshold is None else threshold
        parts.append(score_variant(df[mask], bundle, cutoff))

    return pd.concat(parts).reindex(df.index)
