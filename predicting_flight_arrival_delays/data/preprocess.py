"""Preprocessing pipeline, PRE-SPLIT stage.
"""

from pathlib import Path

import holidays
from loguru import logger
import numpy as np
import pandas as pd
import typer

from predicting_flight_arrival_delays.config import (
    DATE_COLUMN,
    EXTERNAL_DATA_DIR,
    FULL_LEAD_COVERAGE_START,
    INTERIM_DATA_DIR,
    KEEP_COLUMNS,
    MAX_LEAD_DAYS,
    RAW_DATA_DIR,
    SEED,
    WEATHER_COLUMNS,
)

FULL_LEAD_COVERAGE_START = pd.Timestamp(FULL_LEAD_COVERAGE_START)
from predicting_flight_arrival_delays.data.weather import load_weather
from predicting_flight_arrival_delays.utils import to_pascal_case

app = typer.Typer()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# US federal holidays
US_HOLIDAYS = holidays.UnitedStates()


# ---------------------------------------------------------------------------
# 1 Add congestione features
# ---------------------------------------------------------------------------
def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add each flight's scheduling congestion at both its origin and destination airport.

    Args:
        df: Dataframe containing flights data with Origin, Dest, CRSDepTime and
            CRSArrTime present, and Cancelled/Diverted not yet filtered out.

    Returns:
        The same DataFrame with "OriginCongestion" and "DestCongestion" added.
    """
    dep_hour = (df["CRSDepTime"].replace(2400, 0) // 100).astype(int)
    df["OriginCongestion"] = df.groupby(
        [df[DATE_COLUMN], df["Origin"], dep_hour]
    )["Origin"].transform("count")

    arr_hour = (df["CRSArrTime"].replace(2400, 0) // 100).astype(int)
    df["DestCongestion"] = df.groupby(
        [df[DATE_COLUMN], df["Dest"], arr_hour]
    )["Dest"].transform("count")
    return df


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------

def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable flights.

    Flights dropped: cancelled, diverted, or with no arrival outcome recorded -
    none of these have a valid target.

    Args:
        df: The Dataframe containing all the flights data.

    Returns:
        The DataFrame clenaed.
    """
    n_raw = len(df)

    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)].copy()
    df = df.dropna(subset=["ArrDel15"])
    logger.info(f"{n_raw} rows -> {len(df)} after dropping cancelled/diverted/no-target")

    df["IsDelayed"] = df["ArrDel15"].astype(int)
    df = df.drop(columns=["Cancelled", "Diverted", "ArrDel15"])

    return df


# ---------------------------------------------------------------------------
# 3. Temporal features
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
        The same DataFrame with "DepHour", "DepTimeDecimal", "ArrHour", and "ArrTimeDecimal" added.
    """
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    for time_col, prefix in [("CRSDepTime", "Dep"), ("CRSArrTime", "Arr")]:
        raw = df[time_col].replace(2400, 0)
        hours = raw // 100
        minutes = raw % 100
        df[f"{prefix}Hour"] = hours.astype(int)        
        df[f"{prefix}TimeDecimal"] = hours + minutes / 60  

    return df


# ---------------------------------------------------------------------------
# 4. Holiday features
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


    start = df[DATE_COLUMN].min() - pd.DateOffset(years=1)
    end = df[DATE_COLUMN].max() + pd.DateOffset(years=1)
    holiday_dates = sorted(d for d in US_HOLIDAYS[start:end])
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
# 5. Aircraft Schedule Features
# ---------------------------------------------------------------------------
def add_aircraft_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add each flight's position in its aircraft's schedule for that day.

    Args:
        df: Flights with TailNumber, DATE_COLUMN and CRSDepTime.

    Returns:
        The same DataFrame with "AircraftDailyLegs" and "LegPosition" added.
    """
    df = df.sort_values(["TailNumber", DATE_COLUMN, "CRSDepTime"])
    grouped = df.groupby(["TailNumber", DATE_COLUMN])
    df["AircraftDailyLegs"] = grouped["TailNumber"].transform("count")
    df["LegPosition"] = grouped.cumcount() + 1
    return df


# ---------------------------------------------------------------------------
# 6. Lead time assignment
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
# 7. UTC conversion
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
# 8. Add Turnaround features
# ---------------------------------------------------------------------------

def add_turnaround_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add each flight's scheduled turnaround time since its aircraft's previous leg.

    Args:
        df: Flights with TailNumber, DepUtcHour and ArrUtcHour.

    Returns:
        The same DataFrame with "ScheduledTurnaround" added (minutes; NaN
        only for an aircraft's first-ever appearance in the dataset, left to
        the Transformer's median imputation).
    """
    df = df.sort_values(["TailNumber", "DepUtcHour"])
    prev_arr = df.groupby("TailNumber")["ArrUtcHour"].shift(1)
    df["ScheduledTurnaround"] = (df["DepUtcHour"] - prev_arr).dt.total_seconds() / 60
    return df

# ---------------------------------------------------------------------------
# 9. Carrier-airport interaction features
# ---------------------------------------------------------------------------
def add_carrier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add each flight's airport-carrier combination, for historical delay rates.

    Args:
        df: Flights with OriginAirportID, DestAirportID and ReportingAirline.

    Returns:
        The same DataFrame with "OriginCarrier" and "DestCarrier" added.
    """
    df["OriginCarrier"] = df["OriginAirportID"].astype(str) + df["ReportingAirline"]
    df["DestCarrier"] = df["DestAirportID"].astype(str) + df["ReportingAirline"]
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
        side = weather.rename(
            columns={
                **{v: f"{v}{suffix}" for v in WEATHER_COLUMNS},
                "AirportId": airport_col,
                "Time": hour_col,
            },
            copy=False,
        )
        flights = flights.merge(side, on=[airport_col, hour_col, "LeadDays"], how="left")

        del side

    for col in ["WeatherCodeOrigin", "WeatherCodeDest"]:
        if col in flights.columns:
            codes = flights[col].astype("Int64")
            flights[col] = codes.astype(str).mask(codes.isna())

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
    """Clean the flights and build the carrier, calendar, schedule, and holiday features.

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
        needed = KEEP_COLUMNS + ["Cancelled", "Diverted", "ArrDel15"]
        df = pd.concat(
            [pd.read_csv(p, usecols=lambda c: c in needed) for p in paths],
            ignore_index=True,
        )        
        df.columns = [to_pascal_case(c) for c in df.columns]
        df = add_congestion_features(df)  
        df = load_and_clean(df)       
        df = add_temporal_features(df)
        df = add_holiday_features(df)
        df = add_aircraft_schedule_features(df) 
        df = assign_lead_days(df)
        df = add_utc_columns(df, airports)
        df = add_turnaround_features(df) 
        df = add_carrier_features(df)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.success(f"Saved {len(df)} flights to {output_path}")
    except Exception as e:
        logger.exception(f"An error occurred while preprocess the flights: {e}")
        raise typer.Exit(code=1) from None

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
        df = join_weather_to_flights(pd.read_parquet(flights_path), weather_dir)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.success(f"Saved {len(df)} flights with weather to {output_path}")
    except Exception as e:
        logger.exception(f"An error occurred while attaching the weather to the flights: {e}")
        raise typer.Exit(code=1) from None

if __name__ == "__main__":
    app()