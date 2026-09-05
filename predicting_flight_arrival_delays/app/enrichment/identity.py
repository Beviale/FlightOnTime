"""How BTS names an airport, resolved from an IATA code.

The model was trained on BTS's own naming - a numeric AirportID, a city written as
"New York, NY", a two-letter state - and no flight-schedule service returns any of
it. AeroDataBox gives a municipality and an ISO country code, which is a different
thing.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt

from loguru import logger
import pandas as pd

from predicting_flight_arrival_delays.app.enrichment.reference import (
    get_airport,
    load_airports_table,
)
from predicting_flight_arrival_delays.config import EXTERNAL_DATA_DIR

AIRPORT_IDENTITY_CSV = EXTERNAL_DATA_DIR / "airport_identity.csv"

EARTH_RADIUS_KM = 6371.0


MAX_MATCH_KM = 5.0


class UnknownAirportError(LookupError):
    """Raised when a code reaches no airport the model has flown."""


class AmbiguousAirportError(LookupError):
    """Raised when a code lists several airports and nothing tells them apart."""


@dataclass(frozen=True)
class AirportIdentity:
    """One airport, named the way BTS names it."""

    iata: str
    airport_id: int
    city_name: str
    state: str


@lru_cache(maxsize=1)
def load_identity() -> dict[int, AirportIdentity]:
    """Read the identity table into an id-keyed lookup, once per process.

    Returns:
        Every airport the model has flown, keyed on its BTS AirportID.

    Raises:
        FileNotFoundError: If the table has not been built.
    """
    if not AIRPORT_IDENTITY_CSV.exists():
        raise FileNotFoundError(
            f"Airport identity table not found at {AIRPORT_IDENTITY_CSV}. Run "
            "predicting_flight_arrival_delays/data/build_airport_identity.py first."
        )

    table = pd.read_csv(AIRPORT_IDENTITY_CSV)
    identity = {
        int(row.AirportId): AirportIdentity(
            iata=str(row.Iata).strip().upper(),
            airport_id=int(row.AirportId),
            city_name=row.CityName,
            state=row.State,
        )
        for row in table.itertuples()
    }
    logger.info(f"Loaded {len(identity)} airport identities from {AIRPORT_IDENTITY_CSV.name}")
    return identity


@lru_cache(maxsize=1)
def load_codes() -> dict[str, tuple[int, ...]]:
    """Index the reference table by IATA code.

    Returns:
        Every code airports.csv lists, mapped to the airport ids under it. Almost
        always one; more than one means BTS has reassigned that code.
    """
    grouped: dict[str, list[int]] = {}
    for row in load_airports_table().itertuples():
        grouped.setdefault(str(row.Iata).strip().upper(), []).append(int(row.AirportId))
    return {code: tuple(ids) for code, ids in grouped.items()}


def _distance_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Great-circle distance between two points, in kilometres."""
    lat_a, lon_a, lat_b, lon_b = map(radians, (lat_a, lon_a, lat_b, lon_b))
    haversine = (
        sin((lat_b - lat_a) / 2) ** 2
        + cos(lat_a) * cos(lat_b) * sin((lon_b - lon_a) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * asin(sqrt(haversine))


def _by_location(code: str, candidates: tuple[int, ...], location: dict) -> int:
    """Pick the candidate the reported location belongs to, and check that it does.

    Args:
        code: The IATA code, for the messages.
        candidates: The airport ids listed under it.
        location: AeroDataBox's {"lat": ..., "lon": ...} for the airport the flight
            actually uses.

    Returns:
        The id of the airport the location belongs to.

    Raises:
        AmbiguousAirportError: If several airports share the code and the location
            cannot separate them.
        UnknownAirportError: If the closest is further than MAX_MATCH_KM, or if the
            lone candidate cannot be measured against at all.
    """
    lat, lon = location.get("lat"), location.get("lon")
    if lat is None or lon is None:
        raise _unmeasurable(code, candidates, "the location carries no coordinates")

    measurable = [
        (airport_id, _distance_km(lat, lon, airport.latitude, airport.longitude))
        for airport_id in candidates
        if (airport := get_airport(airport_id)) is not None and airport.locatable
    ]
    if not measurable:
        raise _unmeasurable(code, candidates, "none of them has coordinates on file")

    airport_id, distance = min(measurable, key=lambda pair: pair[1])
    if distance > MAX_MATCH_KM:
        listed = f"lists {list(candidates)}, and the closest" if len(candidates) > 1 else "is"
        raise UnknownAirportError(
            f"'{code}' {listed} {distance:.1f} km from where the flight actually "
            f"departs - further than the {MAX_MATCH_KM} km a match is allowed. "
            "This is not that airport."
        )

    logger.info(f"'{code}' resolved to {airport_id}, {distance:.1f} km from the reported location")
    return airport_id


def _unmeasurable(code: str, candidates: tuple[int, ...], why: str) -> LookupError:
    """The error for an airport that cannot be measured against a location.

    Args:
        code: The IATA code, for the message.
        candidates: The airport ids listed under it.
        why: What stopped the measurement.

    Returns:
        The error to raise - ambiguous where several airports share the code, unknown
        where the single one simply cannot be confirmed.
    """
    if len(candidates) > 1:
        return AmbiguousAirportError(
            f"'{code}' lists {list(candidates)} and {why}, so they cannot be told apart."
        )
    return UnknownAirportError(
        f"'{code}' cannot be confirmed: {why}, so there is no way to tell whether "
        f"airport {candidates[0]} is the one the flight departs from."
    )


def get_identity(iata: str, location: dict | None) -> AirportIdentity:
    """Resolve one IATA code to the identity the model was trained on.

    Args:
        iata: Airport code, case-insensitive.
        location: AeroDataBox's {"lat": ..., "lon": ...} for that airport.

    Returns:
        How BTS names that airport, under the code the caller asked for.

    Raises:
        UnknownAirportError: If the code is not in the reference table, if the single
            airport it reaches cannot be confirmed against the location or sits further
            than MAX_MATCH_KM from it, or if it is one the model has never flown.
        AmbiguousAirportError: If the code lists several airports and the location
            cannot separate them.
    """
    code = iata.strip().upper()

    candidates = load_codes().get(code)
    if not candidates:
        raise UnknownAirportError(f"Unknown airport '{iata}': no such code in the airport table.")

    if location is None:
        raise _unmeasurable(code, candidates, "no location was reported for it")

    airport_id = _by_location(code, candidates, location)

    place = load_identity().get(airport_id)
    if place is None:
        raise UnknownAirportError(
            f"BTS has recorded no flight at airport {airport_id} ('{code}'), so the model "
            "has never seen it."
        )

    return replace(place, iata=code)
