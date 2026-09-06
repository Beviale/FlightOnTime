"""Turn API requests into the feature frame the model reads.

Manual entry hands over every feature except the weather, already in the shape the
pipeline produces, so there is little to build: the request becomes a frame as it
stands. Only two things are added, both of which the caller cannot supply.

    LeadDays   how far ahead the request is being made. It is not a property of the
               flight but of the moment it is asked about, and it has to mean the
               same thing it meant in training - the age of the forecast being read.
    weather    the forecast at the origin around departure and at the destination
               around arrival, fetched live.
"""

from datetime import UTC, date, datetime

from loguru import logger
import pandas as pd

from predicting_flight_arrival_delays.app.enrichment.reference import (
    get_airport,
    load_airports_table,
)
from predicting_flight_arrival_delays.app.enrichment.weather_live import weather_at
from predicting_flight_arrival_delays.app.schema import FlightRequest
from predicting_flight_arrival_delays.config import DATE_COLUMN, MAX_LEAD_DAYS, WEATHER_COLUMNS
from predicting_flight_arrival_delays.data.features import (
    WEATHER_COLUMNS_DESTINATION,
    WEATHER_COLUMNS_ORIGIN,
)
from predicting_flight_arrival_delays.data.preprocess import add_utc_columns

FEATURE_FRAME_COLUMNS = [
    *FlightRequest.model_fields,
    "LeadDays",
    *WEATHER_COLUMNS_ORIGIN,
    *WEATHER_COLUMNS_DESTINATION,
]

NUMERIC_WEATHER_COLUMNS = [
    c
    for c in WEATHER_COLUMNS_ORIGIN + WEATHER_COLUMNS_DESTINATION
    if not c.startswith("WeatherCode")
]

WEATHER_OK = "ok"
WEATHER_BEYOND_HORIZON = "beyond_forecast_horizon"
WEATHER_UNAVAILABLE = "unavailable"
WEATHER_UNKNOWN_AIRPORT = "unknown_airport"


def lead_days(flight_date: date, today: date) -> int:
    """How old the forecast for this flight would be.

    Clipped to the range the model was trained on: a flight in the past reads the
    freshest forecast, one beyond the horizon has no forecast at all and is handled
    by the caller.

    Args:
        flight_date: When the flight departs.
        today: The day the request is made.

    Returns:
        The lead time in days, 0 to MAX_LEAD_DAYS.
    """
    return max(0, min((flight_date - today).days, MAX_LEAD_DAYS))


def attach_weather(df: pd.DataFrame, requests: list[FlightRequest], today: date) -> list[str]:
    """Fill the weather columns in place, one live forecast per airport and day.

    Args:
        df: The frame being built, already carrying DepUtcHour and ArrUtcHour.
        requests: The flights being scored, aligned with df.
        today: The day the request is made.

    Returns:
        One status per row: why the weather is there, or why it is not.
    """
    for column in WEATHER_COLUMNS_ORIGIN + WEATHER_COLUMNS_DESTINATION:
        df[column] = None

    cache: dict = {}
    statuses = []

    for index, request in zip(df.index, requests, strict=True):
        if (request.FlightDate - today).days > MAX_LEAD_DAYS:
            statuses.append(WEATHER_BEYOND_HORIZON)
            continue

        origin_airport = get_airport(request.OriginAirportID)
        dest_airport = get_airport(request.DestAirportID)
        if origin_airport is None or dest_airport is None:
            statuses.append(WEATHER_UNKNOWN_AIRPORT)
            continue

        origin = weather_at(origin_airport, df.at[index, "DepUtcHour"], cache)
        dest = weather_at(dest_airport, df.at[index, "ArrUtcHour"], cache)
        if origin is None or dest is None:
            statuses.append(WEATHER_UNAVAILABLE)
            continue

        for column in WEATHER_COLUMNS:
            df.at[index, f"{column}Origin"] = origin[column]
            df.at[index, f"{column}Dest"] = dest[column]
        statuses.append(WEATHER_OK)

    logger.info(f"Weather resolved for {statuses.count(WEATHER_OK)} of {len(statuses)} flights")
    return statuses


def build_feature_frame(
    requests: list[FlightRequest], today: date | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the feature frame for a list of requests.

    Args:
        requests: The flights to score.
        today: The day the request is made; defaults to the current date. Injectable
            so tests can pin the forecast lead time.

    Returns:
        The feature frame - the columns the request carried, plus the lead time and
        the weather - and one weather status per row.

    Raises:
        ValueError: If requests is empty.
    """
    if not requests:
        raise ValueError("No flights to score: the request carries none.")

    today = today or datetime.now(UTC).date()

    df = pd.DataFrame([r.model_dump(exclude_unset=True) for r in requests])
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df["LeadDays"] = [lead_days(r.FlightDate, today) for r in requests]

    df["DepHour"] = df["DepTimeDecimal"].astype(int)
    df = add_utc_columns(df, load_airports_table())

    statuses = attach_weather(df, requests, today)

    for column in NUMERIC_WEATHER_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df[[c for c in FEATURE_FRAME_COLUMNS if c in df.columns]], statuses
