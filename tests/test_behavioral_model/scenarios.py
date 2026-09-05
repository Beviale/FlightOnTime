"""Perturbations built from the data's own observed range."""

import pandas as pd

# Present in every production variant.
CONGESTION_LEVERS = ["OriginCongestion", "DestCongestion"]

# Present in 'all'; absent from 'noweather' by design.
WEATHER_LEVERS = [
    "PrecipitationOrigin",
    "SnowfallOrigin",
    "WindSpeed10mOrigin",
    "WindGusts10mOrigin",
    "PrecipitationDest",
    "SnowfallDest",
    "WindSpeed10mDest",
    "WindGusts10mDest",
]


def present(sample: pd.DataFrame, levers: list[str]) -> list[str]:
    """The subset of levers this variant's feature set actually carries."""
    return [c for c in levers if c in sample.columns]


def pin(sample: pd.DataFrame, columns: list[str], how: str) -> pd.DataFrame:
    """Flatten every listed column to its own minimum or maximum.

    Args:
        sample: Flights to perturb; left untouched.
        columns: Columns to pin; any not present are skipped.
        how: "min" for the mildest conditions recorded, "max" for the worst.

    Returns:
        A copy with those columns flattened.
    """
    perturbed = sample.copy()
    for column in present(sample, columns):
        perturbed[column] = getattr(sample[column], how)()
    return perturbed


def levers_for(sample: pd.DataFrame) -> list[str]:
    """Every lever available in this sample, congestion and weather alike."""
    return present(sample, CONGESTION_LEVERS + WEATHER_LEVERS)


def calm(sample: pd.DataFrame) -> pd.DataFrame:
    """The same flights under the mildest conditions the data records."""
    return pin(sample, levers_for(sample), "min")


def storm(sample: pd.DataFrame) -> pd.DataFrame:
    """The same flights under the worst conditions the data records."""
    return pin(sample, levers_for(sample), "max")
