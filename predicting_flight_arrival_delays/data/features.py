"""Feature set definitions and model variants.

The preprocessing stage produces a single dataset containing every column. Model
variants are not separate files: they are different views of that same dataset,
built here by dropping the columns each variant is meant to ignore.

Three variants:

    all        full feature set
    noweather  weather features removed
    nocarrier  carrier identity removed - analysis only, to check whether the
               carrier carries real signal or mostly proxies the routes and
               airports it serves
"""

import pandas as pd
from loguru import logger
from predicting_flight_arrival_delays.config import(
TARGET,
SERVICE_COLUMNS,
WEATHER_COLUMNS
)

CARRIER_COLUMNS = ["ReportingAirline", "FlightNumberReportingAirline", "OriginCarrier", "DestCarrier"]
WEATHER_COLUMNS_ORIGIN = [col + "Origin" for col in WEATHER_COLUMNS]
WEATHER_COLUMNS_DESTINATION = [col + "Dest" for col in WEATHER_COLUMNS]
NOWEATHER_DROP = WEATHER_COLUMNS_ORIGIN + WEATHER_COLUMNS_DESTINATION + ["LeadDays"]


VARIANTS: dict[str, list[str]] = {
    "all": [],
    "noweather": NOWEATHER_DROP,
    "nocarrier": CARRIER_COLUMNS,
}

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame, variant: str = "all") -> list[str]:
    """List the columns a given variant feeds to the model.

    Three groups are excluded: service columns the pipeline needs but the model must not
    see, the columns that define the variant itself, and the target.

    Args:
        df: Preprocessed flights.
        variant: Which variant to build the feature list for.

    Returns:
        The remaining column names, in their original order.

    Raises:
        ValueError: If the variant is not one of those defined in VARIANTS.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Available: {list(VARIANTS)}")

    excluded = set(SERVICE_COLUMNS) | set(VARIANTS[variant]) | {TARGET}
    return [c for c in df.columns if c not in excluded]


def select_features_variant(
    df: pd.DataFrame,
    variant: str = "all",
    drop_missing_weather: bool = True,
) -> pd.DataFrame:
    """Select the feature columns and target for one variant, as a single DataFrame.

    Args:
        df: Preprocessed flights, with features already built and weather joined.
        variant: Which feature set to build.
        drop_missing_weather: Whether to drop rows with unmatched weather. Applies to
            any variant that keeps weather columns ("all" and "nocarrier").

    Returns:
        The feature columns plus the target, as one DataFrame.
    """
    feature_cols = get_feature_columns(df, variant)

    if drop_missing_weather:
        weather_present = [
            c for c in WEATHER_COLUMNS_ORIGIN + WEATHER_COLUMNS_DESTINATION
            if c in feature_cols
        ]
        if weather_present:
            before = len(df)
            df = df.dropna(subset=weather_present)
            dropped = before - len(df)
            if dropped:
                logger.info(
                    f"variant '{variant}': dropped {dropped} flights ({dropped / before:.2%}) "
                    f"with unmatched weather - these are served by 'noweather' in production."
                )

    df = df[feature_cols + [TARGET]]
    logger.info(f"variant '{variant}': {df.shape[0]} rows, {len(feature_cols)} features")
    return df


def build_xy(
    df: pd.DataFrame,
    variant: str | None = None,
    drop_missing_weather: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a preprocessed dataframe into features and target, optionally for one variant.

    Args:
        df: Preprocessed flights.
        variant: Which feature set to build. If None, no columns are dropped.
        drop_missing_weather: Whether to drop rows with unmatched weather. Applies to
            any variant that keeps weather columns ("all" and "nocarrier").

    Returns:
        The feature matrix and the target, aligned on the same rows.
        """
    if variant is not None:
        df = select_features_variant(df, variant, drop_missing_weather)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]  

    if variant is None:
        logger.info(f"Dataframe size: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y


def describe_variants(df: pd.DataFrame) -> None:
    """Log how many features each variant sees.

    Args:
        df: Preprocessed flights.
    """
    for variant in VARIANTS:
        cols = get_feature_columns(df, variant)
        logger.info(f"{variant:>10}: {len(cols)} features")