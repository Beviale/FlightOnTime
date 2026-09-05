"""Airport coordinates and timezone, read from the pipeline's own airport table.

The caller names the airport in full - id, code, city, state - so nothing here feeds
the model. What the service still needs is where the airport physically is: the
coordinates the forecast is requested for, and the timezone that turns a local
scheduled hour into the UTC hour the forecast is read at. Both are already in
airports.csv, built by the build_airports stage and used by the weather stage during
training, so serving reads the same table.

Airports are keyed on AirportID, the way the weather stage keys them, because the id
is stable over time while a IATA code can be reassigned.
"""

from dataclasses import dataclass
from functools import lru_cache

from loguru import logger
import pandas as pd

from predicting_flight_arrival_delays.config import EXTERNAL_DATA_DIR

AIRPORTS_CSV = EXTERNAL_DATA_DIR / "airports.csv"


@dataclass(frozen=True)
class Airport:
    """Where one airport is, and what time it keeps."""

    iata: str
    airport_id: int
    latitude: float
    longitude: float
    timezone: str

    @property
    def locatable(self) -> bool:
        """Whether this airport can be matched to a weather forecast."""
        return bool(self.timezone) and pd.notna(self.latitude) and pd.notna(self.longitude)


@lru_cache(maxsize=1)
def load_airports_table() -> pd.DataFrame:
    """Read airports.csv once per process.

    Returns:
        The airport table, with AirportId, Iata, Latitude, Longitude and Timezone.

    Raises:
        FileNotFoundError: If the table has not been built or pulled.
    """
    if not AIRPORTS_CSV.exists():
        raise FileNotFoundError(
            f"Airport table not found at {AIRPORTS_CSV}. Run the build_airports stage "
            "(or dvc pull) before serving."
        )

    table = pd.read_csv(AIRPORTS_CSV)
    logger.info(f"Loaded {len(table)} airports from {AIRPORTS_CSV}")
    return table


@lru_cache(maxsize=1)
def load_airports() -> dict[int, Airport]:
    """Index the airport table by BTS AirportID.

    Returns:
        Every known airport, keyed on its AirportID.
    """
    return {
        int(row.AirportId): Airport(
            iata=row.Iata,
            airport_id=int(row.AirportId),
            latitude=float(row.Latitude),
            longitude=float(row.Longitude),
            timezone=row.Timezone,
        )
        for row in load_airports_table().itertuples()
    }


def get_airport(airport_id: int) -> Airport | None:
    """Resolve one BTS AirportID.

    An airport absent from the table is not an error: the flight can still be scored,
    it simply cannot be matched to a forecast, so it is served without weather.

    Args:
        airport_id: BTS numeric airport id.

    Returns:
        The matching airport, or None if the table does not carry it.
    """
    airport = load_airports().get(int(airport_id))
    if airport is None:
        logger.warning(
            f"Airport {airport_id} is not in {AIRPORTS_CSV.name} - "
            "no coordinates to request a forecast for"
        )
    return airport
