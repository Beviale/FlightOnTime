"""Preprocessing pipeline, PRE-SPLIT stage.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import typer
from loguru import logger
import holidays
from predicting_flight_arrival_delays.config import (
EXTERNAL_DATA_DIR, 
DATE_COLUMN, 
INTERIM_DATA_DIR, 
MAX_LEAD_DAYS, 
KEEP_COLUMNS, 
FULL_LEAD_COVERAGE_START, 
RAW_DATA_DIR,
WEATHER_COLUMNS,
SEED,
)
FULL_LEAD_COVERAGE_START = pd.Timestamp(FULL_LEAD_COVERAGE_START)
from predicting_flight_arrival_delays.data.weather import load_weather
app = typer.Typer()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# US federal holidays
US_HOLIDAYS = holidays.UnitedStates()


# ---------------------------------------------------------------------------
# 1. Cleaning
# ---------------------------------------------------------------------------

def load_and_clean(path: Path) -> pd.DataFrame:
    """Load one BTS CSV, drop unusable flights and unusable columns.

    Flights dropped: cancelled, diverted, or with no arrival outcome recorded -
    none of these have a valid target.
    Columns dropped: post-departure fields, diversion columns, and redundant identifiers.

    Args:
        path: The path to the file containing the flight data.

    Returns:
        The DataFrame clenaed.
    """
    needed = KEEP_COLUMNS + ["Cancelled", "Diverted", "ArrDel15"]

    df = pd.read_csv(path, usecols=lambda c: c in needed)
    n_raw = len(df)

    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)].copy()
    df = df.dropna(subset=["ArrDel15"])
    logger.info(f"{path.name}: {n_raw} rows -> {len(df)} after dropping cancelled/diverted/no-target")

    df["IsDelayed"] = df["ArrDel15"].astype(int)
    df = df.drop(columns=["Cancelled", "Diverted", "ArrDel15"])

    return df


# ---------------------------------------------------------------------------
# 2. Temporal features
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features and local scheduled time.

    CRSDepTime/CRSArrTime are ints in HHMM form (e.g. 1430). BTS writes midnight as
    2400, normalised to 0 here.

    Time of day is kept in two forms: the hour alone (used later to match the hourly
    weather series) and a continuous decimal hour (14:55 -> 14.917).

    Args:
        df: Flights.

    Returns:
        The same DataFrame with "IsWeekend", "DepHour". "DepTimeDecimal", "ArrHour", and "ArrTimeDecimal" added.
    """
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    for time_col, prefix in [("CRSDepTime", "Dep"), ("CRSArrTime", "Arr")]:
        raw = df[time_col].replace(2400, 0)
        hours = raw // 100
        minutes = raw % 100
        df[f"{prefix}Hour"] = hours.astype(int)        
        df[f"{prefix}TimeDecimal"] = hours + minutes / 60  

    return df


# ---------------------------------------------------------------------------
# 3. Holday features
# ---------------------------------------------------------------------------
def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add US federal holiday features to each flight.

    Two columns are produced:

        - IsHoliday: 1 if the flight date is itself a federal holiday.
        - DaysToNearestHoliday: signed distance in days to the closest holiday -
        negative before it, positive after. Zero on the holiday itself.

    Args:
        df: Flights with a datetime.

    Returns:
        The same DataFrame with "IsHoliday" and "DaysToNearestHoliday" added.
    """
    dates = df[DATE_COLUMN].dt.date

    df["IsHoliday"] = dates.map(lambda d: d in US_HOLIDAYS).astype(int)

    holiday_dates = sorted(
        d for d in US_HOLIDAYS[df[DATE_COLUMN].min():df[DATE_COLUMN].max()]
    )
    holiday_series = pd.Series(pd.to_datetime(holiday_dates))

    def _days_to_nearest(d):
        if holiday_series.empty:
            return pd.NA
        diffs = (holiday_series - pd.Timestamp(d)).dt.days
        return diffs.loc[diffs.abs().idxmin()]

    unique_dates = pd.Series(df[DATE_COLUMN].unique())
    mapping = {d: _days_to_nearest(d) for d in unique_dates}
    df["DaysToNearestHoliday"] = df[DATE_COLUMN].map(mapping)

    return df


# ---------------------------------------------------------------------------
# 4. Lead time assignment
# ---------------------------------------------------------------------------

def assign_lead_days(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Assign one forecast lead time per flight.

    Simulates how far in advance a user would have requested the prediction.

    Flights before FULL_LEAD_COVERAGE_START are forced to lead time 0, since older
    lead times are not available from the weather source for that period.

    Args:
        df: Flights with a datetime.

    Returns:
        The same DataFrame with "LeadDays" added.
    """
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, MAX_LEAD_DAYS + 1, size=len(df))

    before_coverage = df[DATE_COLUMN] < FULL_LEAD_COVERAGE_START
    df["LeadDays"] = np.where(before_coverage, 0, sampled)

    n_forced = int(before_coverage.sum())
    logger.info(
        f"lead_days: {n_forced} flights forced to 0 (before {FULL_LEAD_COVERAGE_START.date()}), "
        f"{len(df) - n_forced} sampled uniformly 0-{MAX_LEAD_DAYS}"
    )
    return df


# ---------------------------------------------------------------------------
# 5. UTC conversion
# ---------------------------------------------------------------------------

def add_utc_columns(df: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Convert local scheduled times to UTC hourly timestamps, at origin and destination.

    Args:
        df: Flights.
        airports: Airport reference table, providing latitude, longitude and IANA timezone name for each airport.
    Returns:
        The same DataFrame with "DepUtcHour" and "ArrUtcHour"` added.
    """
    tz_map = airports.set_index("AirportId")["Timezone"].to_dict()
 
    # --- Departure: local origin time -> UTC ---
    tz_series = df["OriginAirportID"].map(tz_map)
    local_naive = df[DATE_COLUMN].dt.normalize() + pd.to_timedelta(df["DepHour"], unit="h")
 
    dep_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for tz_name, idx in tz_series.groupby(tz_series).groups.items():
        dep_utc.loc[idx] = (
            local_naive.loc[idx]
            .dt.tz_localize(tz_name, ambiguous="NaT", nonexistent="NaT")
            .dt.tz_convert("UTC")
        )
    df["DepUtcHour"] = dep_utc
 
    # --- Arrival: departure UTC + scheduled duration, floored to the hour.
    df["ArrUtcHour"] = (
        dep_utc + pd.to_timedelta(df["CRSElapsedTime"], unit="minute")
    ).dt.floor("h")
 
    n_missing = df["DepUtcHour"].isna().sum() + df["ArrUtcHour"].isna().sum()
    if n_missing:
        logger.warning(f"{n_missing} timestamps could not be converted (unknown airport or DST edge)")
 
    in_flights = set(df["OriginAirportID"])
    in_table = set(airports["AirportId"])
    missing_airports = in_flights - in_table
    if missing_airports:
        logger.warning(
            f"{len(missing_airports)} airports missing from airports.csv: "
            f"{sorted(missing_airports)}"
        )
    return df


# ---------------------------------------------------------------------------
# Join weather to flights
# ---------------------------------------------------------------------------

def join_weather_to_flights(flights: pd.DataFrame, weather_dir: Path) -> pd.DataFrame:
    """Attach the weather at both origin and destination to each flight.

    Args:
        flights: Preprocessed flights, already carrying OriginAirportID,
            DestAirportID, DepUtcHour, ArrUtcHour and LeadDays.
        weather_dir: Directory of (airport, lead time) parquet files.

    Returns:
        The same flights with weather features added for both ends of the route.
    """
    weather = load_weather(weather_dir)

    for airport_col, hour_col, suffix in [
        ("OriginAirportID", "DepUtcHour", "Origin"),
        ("DestAirportID", "ArrUtcHour", "Dest"),
    ]:
        side = weather.rename(columns={v: f"{v}{suffix}" for v in WEATHER_COLUMNS})
        flights = flights.merge(
            side,
            left_on=[airport_col, hour_col, "LeadDays"],
            right_on=["AirportId", "Time", "LeadDays"],
            how="left",
        ).drop(columns=["AirportId", "Time"])

        del side
    return flights


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def prepare_flights(
    bts_path: Path = typer.Argument(RAW_DATA_DIR),
    airports_path: Path = typer.Option(EXTERNAL_DATA_DIR / "airports.csv"),
    output_path: Path = typer.Option(INTERIM_DATA_DIR / "flights_features.parquet"),
):
    """Clean the flights and build the calendar, schedule and holiday features.

    Args:
        bts_path: A single BTS CSV, or a directory searched recursively for CSVs.
        airports_path: Airport reference table with coordinates and timezones.
        output_path: Where the prepared flights are written.
    """
    try:
        paths = sorted(bts_path.rglob("*.csv")) if bts_path.is_dir() else [bts_path]
        if not paths:
            raise SystemExit(f"No CSV found at {bts_path}")

        airports = pd.read_csv(airports_path)
        df = pd.concat([load_and_clean(p) for p in paths], ignore_index=True)
        df = add_temporal_features(df)
        df = add_holiday_features(df)
        df = assign_lead_days(df)
        df = add_utc_columns(df, airports)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.success(f"Saved {len(df)} flights to {output_path}")
    except Exception as e:
        logger.exception(f"An error occurred while preprocess the flights: {e}")
        raise typer.Exit(code=1)

@app.command()
def join_weather(
    flights_path: Path = typer.Option(INTERIM_DATA_DIR / "flights_features.parquet"),
    weather_dir: Path = typer.Option(EXTERNAL_DATA_DIR / "weather"),
    output_path: Path = typer.Option(INTERIM_DATA_DIR / "flights_preprocessed.parquet"),
):
    """Attach the downloaded weather series to the prepared flights.

    Args:
        flights_path: Flights produced by prepare-flights.
        weather_dir: Directory of (airport, lead time) parquet files.
        output_path: Where the joined result is written.
    """
    try:
        df = pd.read_parquet(flights_path)
        df = join_weather_to_flights(df, weather_dir)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.success(f"Saved {len(df)} flights with weather to {output_path}")
    except Exception as e:
        logger.exception(f"An error occurred while attaching the weather to the flights: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()