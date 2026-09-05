"""Weather forecast lookup at inference time.

A forecast that cannot be retrieved - the service is down, the flight is beyond the forecast horizon, 
the hour is missing from the series - comes back as None, which leaves the weather columns empty
and sends the flight to the 'noweather' model.
"""

from datetime import timedelta

from loguru import logger
import pandas as pd

from predicting_flight_arrival_delays.app.enrichment.reference import Airport
from predicting_flight_arrival_delays.config import (
    FORECAST_TIMEOUT_SECONDS,
    FORECAST_URL,
    WEATHER_COLUMNS,
    WEATHER_MODEL,
    WEATHER_VARS,
)
from predicting_flight_arrival_delays.utils import fetch, to_pascal_case

WINDOW_DAYS = 1


def fetch_forecast(latitude: float, longitude: float, start: str, end: str) -> pd.DataFrame:
    """Fetch the hourly forecast series for one location and date range.

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        start: First date to fetch, YYYY-MM-DD.
        end: Last date to fetch, YYYY-MM-DD.

    Returns:
        Hourly rows indexed by UTC timestamp, with the columns named in
        WEATHER_COLUMNS.

    Raises:
        requests.HTTPError: If the forecast service rejects the request.
        KeyError: If the response carries no hourly series.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(WEATHER_VARS),
        "models": WEATHER_MODEL,
        "timezone": "UTC",
    }

    hourly = fetch(FORECAST_URL, params, timeout=FORECAST_TIMEOUT_SECONDS)["hourly"]
    series = pd.DataFrame(hourly).rename(columns={c: to_pascal_case(c) for c in hourly})
    series["Time"] = pd.to_datetime(series["Time"], utc=True)
    return series.set_index("Time")


def weather_at(
    airport: Airport,
    when_utc: pd.Timestamp,
    cache: dict[tuple[str, str], pd.DataFrame | None] | None = None,
) -> dict[str, float | str] | None:
    """Read the forecast for one airport at one hour.

    Args:
        airport: Which airport to read the forecast for.
        when_utc: The UTC hour wanted, as produced by preprocess.add_utc_columns.
        cache: Optional memo shared across a batch, so flights through the same
            airport on the same day cost one call rather than one each.

    Returns:
        One value per entry in WEATHER_COLUMNS, or None if the forecast could not
        be retrieved for that hour.
    """
    if not airport.locatable or pd.isna(when_utc):
        return None

    day = pd.Timestamp(when_utc).tz_convert("UTC").normalize()
    start = (day - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    end = (day + timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")

    key = (airport.iata, start)
    cache = {} if cache is None else cache

    if key not in cache:
        try:
            cache[key] = fetch_forecast(airport.latitude, airport.longitude, start, end)
        except Exception as e:
            logger.warning(f"Forecast unavailable for {airport.iata} {start}..{end}: {e}")
            cache[key] = None

    series = cache[key]
    if series is None:
        return None

    hour = pd.Timestamp(when_utc).tz_convert("UTC").floor("h")
    if hour not in series.index:
        logger.warning(f"Forecast for {airport.iata} does not cover {hour}")
        return None

    row = series.loc[hour]
    if row[WEATHER_COLUMNS].isna().any():
        logger.warning(f"Forecast for {airport.iata} at {hour} is incomplete")
        return None

    values: dict[str, float | str] = {c: float(row[c]) for c in WEATHER_COLUMNS}
    # Training stores the weather code as the string of an integer, not a number.
    values["WeatherCode"] = str(int(row["WeatherCode"]))
    return values
