"""Build the airport identity table used by the auto-lookup path.

A caller names a flight with IATA codes. The model needs more: BTS identifies each
airport by AirportID and locates it with CityName and State. No
flight-schedule service returns these three fields. AeroDataBox, for example, gives
a municipality and an ISO country code, not a US state and a BTS id.

The table is built from the prepared flights. Every known airport appears there,
as an origin on some flights and as a destination on
others, carrying the same identity either way. Each row holds one IATA code, one AirportID, and
the city and state recorded under that id.

The key is the id, so a IATA code can appear on more than one row. BTS reassigns codes,
and an airport that took one over from another gets a row of its own. Picking
between those rows belongs to the serving path: app/enrichment/identity.py reads
airports.csv for the ids a code lists, and settles the choice with the coordinates
the schedule service returns.
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from predicting_flight_arrival_delays.config import (
    DATE_COLUMN,
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
)

AIRPORT_IDENTITY_CSV = EXTERNAL_DATA_DIR / "airport_identity.csv"

app = typer.Typer()

SIDES = [
    ("Origin", "OriginAirportID", "OriginCityName", "OriginState"),
    ("Dest", "DestAirportID", "DestCityName", "DestState"),
]


PLACE = ["AirportId", "CityName", "State"]


def _latest_place(flights: pd.DataFrame, airport_ids: list[int]) -> pd.DataFrame:
    """The city and state on each of these airports' most recent flight.

    Args:
        flights: Prepared flights, with both ends of the route and the date.
        airport_ids: The ids to look up.

    Returns:
        One row per id, indexed on it: CityName, State and the date they come from.
    """
    frames = []
    for _, id_col, city_col, state_col in SIDES:
        side = flights.loc[
            flights[id_col].isin(airport_ids), [id_col, city_col, state_col, DATE_COLUMN]
        ].copy()
        side.columns = PLACE + [DATE_COLUMN]
        frames.append(_last_flight(side))

    return _last_flight(pd.concat(frames, ignore_index=True)).set_index("AirportId")


def _last_flight(flights: pd.DataFrame) -> pd.DataFrame:
    """Keep each airport's most recent row, whole."""
    return flights.sort_values(DATE_COLUMN).drop_duplicates("AirportId", keep="last")


def _keep_latest_place(
    flights: pd.DataFrame, identity: pd.DataFrame, renamed: list[int]
) -> pd.DataFrame:
    """Settle each disagreeing id on the city and state of its most recent flight.

    Args:
        flights: Prepared flights, for the dates.
        identity: The folded table, still carrying every city and state.
        renamed: The ids that carry more than one.

    Returns:
        The table with one city and state per id.
    """
    latest = _latest_place(flights, renamed)

    for airport_id in renamed:
        seen = identity.loc[identity["AirportId"] == airport_id, ["CityName", "State"]]
        kept = latest.loc[airport_id]
        logger.warning(
            f"Airport {airport_id} appears in more than one place: "
            + ", ".join(
                f"{row.CityName!r} ({row.State})" for row in seen.drop_duplicates().itertuples()
            )
            + f". Keeping {kept.CityName!r} ({kept.State}), from its most recent "
            f"flight on {kept.FlightDate:%Y-%m-%d}."
        )

    identity = identity.copy()
    renaming = identity["AirportId"].isin(renamed)
    for column in ("CityName", "State"):
        identity.loc[renaming, column] = identity.loc[renaming, "AirportId"].map(latest[column])

    return identity.drop_duplicates()


def build_identity(flights: pd.DataFrame) -> pd.DataFrame:
    """Fold both ends of every route into one airport table.

    Args:
        flights: Prepared flights, with both ends of the route and the date.

    Returns:
        One row per (code, id) pair: Iata, AirportId, CityName, State.
    """
    KEYS = ["Iata", "AirportId", "CityName", "State"]

    frames = []
    for iata_col, id_col, city_col, state_col in SIDES:
        side = flights[[iata_col, id_col, city_col, state_col]].copy()
        side.columns = KEYS
        frames.append(side.drop_duplicates())

    identity = pd.concat(frames, ignore_index=True).drop_duplicates()

    places = identity.groupby("AirportId")[["CityName", "State"]].nunique().max(axis=1)
    renamed = sorted(places[places > 1].index)
    if renamed:
        identity = _keep_latest_place(flights, identity, renamed)

    logger.info(f"{len(identity)} airports recovered from the prepared flights")
    return identity.sort_values("Iata").reset_index(drop=True)


@app.command()
def run(
    flights_path: Path = typer.Option(
        INTERIM_DATA_DIR / "flights_features.parquet",
        help="Flights produced by preprocess prepare-flights.",
    ),
    output_path: Path = typer.Option(
        AIRPORT_IDENTITY_CSV,
        help="Where the identity table is written.",
    ),
) -> None:
    """Write the airport identity table the auto-lookup path resolves IATA codes with.

    Args:
        flights_path: Flights produced by preprocess prepare-flights.
        output_path: Where the table is written.

    Raises:
        FileNotFoundError: If the input is missing.
    """
    try:
        if not flights_path.exists():
            raise FileNotFoundError(f"Missing input: {flights_path}")

        columns = [c for side in SIDES for c in side] + [DATE_COLUMN]
        identity = build_identity(pd.read_parquet(flights_path, columns=columns))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        identity.to_csv(output_path, index=False)
        logger.success(f"Saved {len(identity)} airports to {output_path}")
    except Exception as e:
        logger.exception(f"An error occurred while creating the airport identity table: {e}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
