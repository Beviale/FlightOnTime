"""Flight schedules from AeroDataBox, for the auto-lookup path.

Three calls, each answering something the caller cannot know:

    find_flight        the timetable entry - route, scheduled times, distance,
                       coordinates - for one flight number on one date.
    count_movements    how many flights share an airport and an hour, which is what
                       the congestion features count.
    aircraft_rotation  every leg an aircraft flies on a day, from which its position
                       in the rotation and its turnaround follow.
"""

from datetime import date
import os
import time

from loguru import logger
import pandas as pd
import requests

from predicting_flight_arrival_delays.config import (
    AERODATABOX_HOST,
    AERODATABOX_TIMEOUT_SECONDS,
    DOMESTIC_COUNTRY_CODES,
)

# 204 is how the service says "no such flight that day" - not an error.
NO_CONTENT = 204
# The plan limits requests per second, and one prediction makes three or four calls
# back to back. A rejected call is retried rather than failing the whole request.
TOO_MANY_REQUESTS = 429
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 1.5


class ScheduleUnavailableError(RuntimeError):
    """Raised when the schedule service cannot answer for a flight."""


class FlightNotFoundError(LookupError):
    """Raised when no flight matches the number, date and origin given."""


def _headers() -> dict[str, str]:
    """Build the RapidAPI headers.

    Returns:
        Host and key headers.

    Raises:
        ScheduleUnavailableError: If no API key is configured.
    """
    key = os.environ.get("RAPIDAPI_KEY", "").strip()
    if not key:
        raise ScheduleUnavailableError(
            "RAPIDAPI_KEY is not set - the auto-lookup path cannot reach the schedule service."
        )
    return {"x-rapidapi-host": AERODATABOX_HOST, "x-rapidapi-key": key}


def _get(path: str, params: dict | None = None) -> object | None:
    """Call one AeroDataBox endpoint.

    Args:
        path: Endpoint path, starting with a slash.
        params: Query parameters.

    Returns:
        The decoded body, or None when the service answers 204 (nothing scheduled).

    Raises:
        ScheduleUnavailableError: If the service errors, times out or returns
            something that is not JSON.
    """
    url = f"https://{AERODATABOX_HOST}{path}"
    headers = _headers()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                url, headers=headers, params=params or {}, timeout=AERODATABOX_TIMEOUT_SECONDS
            )
        except requests.RequestException as e:
            raise ScheduleUnavailableError(f"Schedule service unreachable: {e}") from e

        if response.status_code != TOO_MANY_REQUESTS:
            break

        if attempt == MAX_ATTEMPTS:
            raise ScheduleUnavailableError(
                f"Schedule service rate-limited {MAX_ATTEMPTS} attempts at {path}"
            )
        wait = BACKOFF_SECONDS * attempt
        logger.warning(f"Rate-limited on {path}; retrying in {wait:.1f}s")
        time.sleep(wait)

    if response.status_code == NO_CONTENT:
        return None
    if not response.ok:
        raise ScheduleUnavailableError(
            f"Schedule service answered {response.status_code} for {path}"
        )
    try:
        return response.json()
    except ValueError as e:
        raise ScheduleUnavailableError(f"Schedule service returned no JSON for {path}") from e


def find_flight(marketing_carrier: str, number: int, flight_date: date, origin: str) -> dict:
    """Fetch one flight's timetable entry.

    Args:
        marketing_carrier: The code the flight is sold under, e.g. "AA".
        number: Flight number.
        flight_date: Date of departure, local to the origin.
        origin: IATA code of the departure airport.

    Returns:
        The matching leg, as the service returns it.

    Raises:
        FlightNotFoundError: If nothing matches the number, date and origin.
        ScheduleUnavailableError: If the service cannot be reached.
    """
    code = f"{marketing_carrier}{number}"
    legs = _get(
        f"/flights/number/{code}/{flight_date.isoformat()}",
        {"dateLocalRole": "Departure", "withAircraftImage": "false", "withLocation": "false"},
    )
    if not legs:
        raise FlightNotFoundError(f"No flight {code} on {flight_date}.")

    matches = [
        leg
        for leg in legs
        if leg.get("departure", {}).get("airport", {}).get("iata") == origin
        and leg.get("arrival", {}).get("airport", {}).get("iata")
    ]
    if not matches:
        flown = sorted(
            {leg.get("departure", {}).get("airport", {}).get("iata") for leg in legs} - {None}
        )
        raise FlightNotFoundError(
            f"Flight {code} on {flight_date} does not depart from {origin}. "
            f"That day it departs from: {', '.join(flown) or 'nowhere the service knows'}."
        )

    if len(matches) > 1:
        logger.warning(f"{code} on {flight_date} has {len(matches)} legs from {origin}; taking the first")
    return matches[0]


def _is_domestic(airport: dict) -> bool:
    """Whether an airport is US soil, the way BTS counts it.

    Args:
        airport: An airport object from the schedule service.

    Returns:
        True if its ISO country code is the US or one of its territories.
    """
    return (airport or {}).get("countryCode", "").lower() in DOMESTIC_COUNTRY_CODES


def count_movements(airport: str, when_local: pd.Timestamp, arriving: bool) -> int:
    """Count the flights sharing an airport and a scheduled local hour.

    This is the quantity the congestion features hold, rebuilt from the live schedule.
    The window is the one clock hour the flight is scheduled in; 
    cancelled flights are included, because the training count was taken before cancellations
    were filtered out.

    Only flights with both ends on US soil are counted - BTS records nothing else.

    Args:
        airport: IATA code of the airport to count at.
        when_local: The scheduled local time of the flight, whose hour is used.
        arriving: Count arrivals rather than departures.

    Returns:
        How many flights share that airport and hour.

    Raises:
        ScheduleUnavailableError: If the service cannot be reached.
    """
    hour = pd.Timestamp(when_local).floor("h")
    window = f"{hour.strftime('%Y-%m-%dT%H:00')}/{hour.strftime('%Y-%m-%dT%H:59')}"
    direction = "Arrival" if arriving else "Departure"

    body = _get(
        f"/flights/airports/iata/{airport}/{window}",
        {
            "direction": direction,
            "withCancelled": "true",
            "withCodeshared": "false",
            "withCargo": "false",
            "withPrivate": "false",
            "withLocation": "false",
        },
    )
    if not body:
        return 0

    movements = body.get("arrivals" if arriving else "departures", [])
    
    domestic = [m for m in movements if _is_domestic((m.get("movement") or {}).get("airport"))]

    logger.info(
        f"{airport} {hour:%Y-%m-%d %H}h {direction.lower()}s: "
        f"{len(domestic)} domestic of {len(movements)}"
    )
    return len(domestic)


def aircraft_rotation(registration: str, flight_date: date) -> list[dict]:
    """Fetch every leg one aircraft flies on a day.

    Answers only for a day whose aircraft assignments have been published - in
    practice the day itself and the past. For a future date the service returns
    nothing, and the rotation features cannot be built at all.

    Args:
        registration: Aircraft tail number.
        flight_date: The day to fetch.

    Returns:
        The legs, in the order the service returns them. Empty if unknown.

    Raises:
        ScheduleUnavailableError: If the service cannot be reached.
    """
    legs = _get(
        f"/flights/reg/{registration}/{flight_date.isoformat()}", {"dateLocalRole": "Departure"}
    )
    if not legs:
        logger.info(f"No published rotation for {registration} on {flight_date}")
        return []
    return legs
