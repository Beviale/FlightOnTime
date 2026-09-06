"""Weather retrieval from Open-Meteo, at variable forecast lead times.

Two endpoints are used:

- Historical Forecast API -> lead time 0 (the freshest forecast issued for that date).
- Previous Runs API       -> lead times 1.. (forecast issued N days ahead).

Coverage caveat (verified empirically against the API): full variable coverage on the
Previous Runs endpoint was only confirmed from around December 2024 onward. Before
that, only temperature is populated for GFS. preprocess.assign_lead_days therefore
forces lead 0 for earlier dates.

Airports are keyed on AirportID (stable over time).

One API call covers an entire date range for one (airport, lead time) pair, so the
number of calls is (airports x lead times actually needed).
"""

from collections import defaultdict
from pathlib import Path
import time

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from predicting_flight_arrival_delays.config import (
    DATE_COLUMN,
    EXTERNAL_DATA_DIR,
    HISTORICAL_FORECAST_URL,
    INTERIM_DATA_DIR,
    MAX_LEAD_DAYS,
    PREVIOUS_RUNS_URL,
    WEATHER_MODEL,
    WEATHER_VARS,
)
from predicting_flight_arrival_delays.utils import fetch, to_pascal_case

app = typer.Typer()

# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_series(
    latitude: float, longitude: float, start: str, end: str, lead_days: int
) -> pd.DataFrame:
    """Fetch the full hourly weather series for one airport, date range and lead time.

    Lead time 0 comes from the Historical Forecast API, while
    lead times 1 and above come from the Previous Runs API.

    Args:
        latitude: Airport latitude.
        longitude: Airport longitude.
        start: First date to fetch, YYYY-MM-DD.
        end: Last date to fetch, YYYY-MM-DD.
        lead_days: How many days ahead the forecast was issued, 0 to MAX_LEAD_DAYS.

    Returns:
        Hourly rows with columns Time (UTC), LeadDays and the weather variables.

    Raises:
        ValueError: If lead_days falls outside the supported range.
    """
    if not 0 <= lead_days <= MAX_LEAD_DAYS:
        raise ValueError(f"LeadDays must be 0..{MAX_LEAD_DAYS}, got {lead_days}")

    if lead_days == 0:
        url = HISTORICAL_FORECAST_URL
        hourly_vars = WEATHER_VARS
        rename = {}
    else:
        url = PREVIOUS_RUNS_URL
        hourly_vars = [f"{v}_previous_day{lead_days}" for v in WEATHER_VARS]
        rename = {f"{v}_previous_day{lead_days}": v for v in WEATHER_VARS}

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(hourly_vars),
        "models": WEATHER_MODEL,
        "timezone": "UTC",
    }

    hourly = fetch(url, params)["hourly"]
    df = pd.DataFrame(hourly)
    if rename:
        df = df.rename(columns=rename)

    df = df.rename(columns={c: to_pascal_case(c) for c in df.columns})

    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df["LeadDays"] = lead_days
    return df


def download_weather(
    requests_map: dict[int, set[int]],
    airports: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_dir: Path,
    sleep: float = 0.3,
) -> None:
    """Download the weather series for every (airport, lead time) combination needed.

    One file per combination, named {airport_id}_lead{n}.parquet. Files already on disk
    are skipped, so an interrupted run can be restarted with the same arguments and
    picks up where it stopped.

    A single request covers the whole date range for one combination, since Open-Meteo
    returns an entire hourly series at once. The number of calls is therefore driven by
    how many combinations exist.

    Args:
        requests_map: Airport id to the set of lead times needed for it.
        airports: Airport reference table providing Latitude and Longitude per AirportID.
        start_date: First date to fetch, YYYY-MM-DD.
        end_date: Last date to fetch, YYYY-MM-DD.
        output_dir: Where the parquet files are written.
        sleep: Pause between requests, in seconds. Keeps the request rate under the free tier's per-minute limit.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    coords = airports.set_index("AirportId")[["Latitude", "Longitude"]].to_dict("index")

    combos = [(aid, lead) for aid, leads in requests_map.items() for lead in sorted(leads)]
    logger.info(f"{len(combos)} (airport, lead) combinations; range {start_date} .. {end_date}")

    n_done = n_skipped = n_failed = 0
    for airport_id, lead in tqdm(combos, desc="Weather"):
        out_path = output_dir / f"{airport_id}_lead{lead}.parquet"
        if out_path.exists():
            logger.warning(f"{airport_id}_lead{lead}.parquet already downloaded")
            n_skipped += 1
            continue

        if airport_id not in coords:
            logger.warning(f"AirportId {airport_id} not in airports.csv -- skipping")
            n_failed += 1
            continue

        c = coords[airport_id]
        try:
            series = fetch_series(c["Latitude"], c["Longitude"], start_date, end_date, lead)
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.warning(f"airport {airport_id}, lead {lead}: fetch failed ({e})")
            n_failed += 1
            continue

        series.insert(0, "AirportId", airport_id)
        series.to_parquet(out_path, index=False)
        n_done += 1
        time.sleep(sleep)

    logger.success(
        f"Weather in {output_dir} -- {n_done} downloaded, {n_skipped} already present, "
        f"{n_failed} failed"
    )


# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------


def load_weather(weather_dir: Path) -> pd.DataFrame:
    """Load every downloaded weather file into a single table.

    Args:
        weather_dir: Directory of parquet files.

    Returns:
        All weather rows in one DataFrame, with columns AirportID, Time (UTC hourly
        timestamps), LeadDays and the variables listed in WEATHER_VARS.

    Raises:
        FileNotFoundError: If the directory holds no parquet files, which means the
            download step has not run yet.
    """
    files = sorted(weather_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No weather files in {weather_dir} -- run download_weather first."
        )
    logger.info(f"Loading {len(files)} weather files from {weather_dir}")

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        for column in frame.columns:
            if frame[column].dtype == "float64":
                frame[column] = frame[column].astype("float32")
        if "WeatherCode" in frame.columns:
            frame["WeatherCode"] = frame["WeatherCode"].astype("float32")
        frames.append(frame)

    weather = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Weather series: {len(weather)} rows, "
        f"{weather.memory_usage(deep=True).sum() / 1024**3:.2f} GB in memory"
    )
    return weather


# ---------------------------------------------------------------------------
# Build request
# ---------------------------------------------------------------------------


def build_weather_requests(df: pd.DataFrame) -> dict[int, set[int]]:
    """Map each airport_id to the set of lead times actually needed for it.

    Args:
        df: Flights. with the lead days column.

    Returns:
        dictionary mapping each airport id to the set of lead times needed for it.
    """
    requests_map: dict[int, set[int]] = defaultdict(set)

    for airport_col in ["OriginAirportID", "DestAirportID"]:
        pairs = df[[airport_col, "LeadDays"]].drop_duplicates()
        for airport_id, lead in pairs.itertuples(index=False):
            requests_map[int(airport_id)].add(int(lead))

    n_combos = sum(len(v) for v in requests_map.values())
    logger.info(
        f"{len(requests_map)} airports, {n_combos} (airport, lead_days) combinations needed"
    )
    return dict(requests_map)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


@app.command()
def run(
    flights_path: Path = typer.Option(INTERIM_DATA_DIR / "flights_features.parquet"),
    airports_path: Path = typer.Option(EXTERNAL_DATA_DIR / "airports.csv"),
    output_dir: Path = typer.Option(EXTERNAL_DATA_DIR / "weather"),
    sleep: float = typer.Option(0.3),
):
    """Download every weather series the prepared flights need.

    Args:
        flights_path (Path): Path to the input Parquet file containing preprocessed flight data.
        airports_path (Path): Path to the reference CSV table containing airport metadata.
        output_dir (Path): Directory where the new weather Parquet files will be stored.
        sleep (float): Delay in seconds between individual API requests to avoid rate limits.
    """
    try:
        df = pd.read_parquet(flights_path)
        airports = pd.read_csv(airports_path)

        requests_map = build_weather_requests(df)
        download_weather(
            requests_map=requests_map,
            airports=airports,
            start_date=str(df[DATE_COLUMN].min().date()),
            end_date=str(df[DATE_COLUMN].max().date()),
            output_dir=output_dir,
            sleep=sleep,
        )
    except Exception as e:
        logger.exception(f"An error occurred while setting up the download weather process: {e}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
