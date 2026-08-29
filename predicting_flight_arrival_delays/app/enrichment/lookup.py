"""Fill in a flight the caller only named.

Auto-lookup asks for six things - the two airports, the date, the number, the code the
flight is sold under and the code that operates it - and recovers the rest. What it
produces is not a feature frame of its own but a FlightRequest, the same object manual
entry produces, so from that point on both paths run through exactly the same code and
cannot drift apart.

Where each field of that FlightRequest comes from - which is not the same as what the
caller sends, because one of the six never becomes a field at all:

    from the caller     the airports, the date, the number, the operating carrier.
                        The marketing code is the sixth thing asked for and appears
                        nowhere below: it exists to find the flight, and the model
                        was never trained on it.
    from the schedule   scheduled times, block time, distance, and - only for a flight
                        already under way - the aircraft's rotation
    from the timetable  how many flights share each airport and hour
    computed            the calendar breakdown, the distance group, the two
                        airport-carrier combinations
    from a table        each airport's BTS id, city and state

Three columns are left empty on a future flight: an aircraft is assigned close to
departure, so its rotation cannot be known in advance. The fitted Transformer fills
them with its training medians.
"""

from datetime import date

from loguru import logger
import pandas as pd

from predicting_flight_arrival_delays.app.enrichment.aerodatabox import (
    FlightNotFoundError,
    aircraft_rotation,
    count_movements,
    find_flight,
)
from predicting_flight_arrival_delays.app.enrichment.identity import get_identity
from predicting_flight_arrival_delays.app.schema import FlightLookupRequest, FlightRequest
from predicting_flight_arrival_delays.config import DATE_COLUMN
from predicting_flight_arrival_delays.data.preprocess import add_holiday_features

DISTANCE_GROUP_MILES = 250
MAX_DISTANCE_GROUP = 11
# BTS numbers the week from Monday as 1; python numbers it from Monday as 0.
BTS_WEEKDAY_OFFSET = 1


def distance_group(distance: float) -> int:
    """Bucket a distance the way BTS does.

    Args:
        distance: Distance in miles.

    Returns:
        The group number, 1 to 11.
    """
    return min(int(distance // DISTANCE_GROUP_MILES) + 1, MAX_DISTANCE_GROUP)


def decimal_hour(local: str) -> float:
    """Read a scheduled local time as the decimal hour the model was trained on.

    Args:
        local: Local time as the schedule service writes it, "YYYY-MM-DD HH:MM±TT:TT".

    Returns:
        The hour with minutes as a fraction, 14:30 becoming 14.5.
    """
    stamp = pd.Timestamp(local[:16])
    return stamp.hour + stamp.minute / 60


def holiday_features(dates: list[date]) -> pd.DataFrame:
    """Build the two calendar features the pipeline derives from a flight date.
    Args:
        dates: One date per flight.

    Returns:
        A frame with IsHoliday and DaysToNearestHoliday, aligned with dates.
    """
    frame = pd.DataFrame({DATE_COLUMN: pd.to_datetime(pd.Series(dates))})
    return add_holiday_features(frame)[["IsHoliday", "DaysToNearestHoliday"]]


def rotation_features(leg: dict, flight_date: date) -> dict[str, float | None]:
    """Work out where a flight sits in its aircraft's day.

    Args:
        leg: The flight's own timetable entry.
        flight_date: The day being scored.

    Returns:
        AircraftDailyLegs, LegPosition and ScheduledTurnaround, any of which may be
        None.
    """
    empty = {"AircraftDailyLegs": None, "LegPosition": None, "ScheduledTurnaround": None}

    registration = (leg.get("aircraft") or {}).get("reg")
    if not registration:
        logger.info("No aircraft assigned yet - the rotation features stay empty")
        return empty

    legs = aircraft_rotation(registration, flight_date)
    if not legs:
        return empty

    def departure_of(entry: dict) -> pd.Timestamp:
        return pd.Timestamp(entry.get("departure", {}).get("scheduledTime", {}).get("utc"))

    def arrival_of(entry: dict) -> pd.Timestamp:
        return pd.Timestamp(entry.get("arrival", {}).get("scheduledTime", {}).get("utc"))

    ordered = sorted(legs, key=departure_of)
    ours = departure_of(leg)

    position = next(
        (i + 1 for i, entry in enumerate(ordered) if departure_of(entry) == ours), None
    )
    if position is None:
        logger.warning(f"{registration} rotation does not contain this leg; features left empty")
        return empty

 
    turnaround = None
    if position > 1:
        previous = arrival_of(ordered[position - 2]).floor("h")
        turnaround = (ours.floor("h") - previous).total_seconds() / 60

    return {
        "AircraftDailyLegs": len(ordered),
        "LegPosition": position,
        "ScheduledTurnaround": turnaround,
    }


def to_flight_request(lookup: FlightLookupRequest, calendar: pd.Series) -> FlightRequest:
    """Recover one named flight in full.

    Args:
        lookup: What the caller supplied.
        calendar: That flight's IsHoliday and DaysToNearestHoliday.

    Returns:
        The flight as manual entry would have described it.

    Raises:
        FlightNotFoundError: If the flight, or one of its airports, is not known.
        ScheduleUnavailableError: If the schedule service cannot be reached.
    """
    leg = find_flight(lookup.MarketingCarrier, lookup.FlightNumber, lookup.FlightDate, lookup.Origin)

    arrival_iata = leg["arrival"]["airport"]["iata"]
    if arrival_iata != lookup.Dest:
        raise FlightNotFoundError(
            f"{lookup.MarketingCarrier}{lookup.FlightNumber} leaves {lookup.Origin} for "
            f"{arrival_iata} on {lookup.FlightDate}, not for {lookup.Dest}."
        )


    origin = get_identity(lookup.Origin, leg["departure"]["airport"].get("location"))
    dest = get_identity(lookup.Dest, leg["arrival"]["airport"].get("location"))
    departure_local = leg["departure"]["scheduledTime"]["local"]
    arrival_local = leg["arrival"]["scheduledTime"]["local"]

    block_minutes = (
        pd.Timestamp(leg["arrival"]["scheduledTime"]["utc"])
        - pd.Timestamp(leg["departure"]["scheduledTime"]["utc"])
    ).total_seconds() / 60
    miles = leg["greatCircleDistance"]["mile"]

    return FlightRequest(
        FlightDate=lookup.FlightDate,
        Month=lookup.FlightDate.month,
        DayofMonth=lookup.FlightDate.day,
        DayOfWeek=lookup.FlightDate.weekday() + BTS_WEEKDAY_OFFSET,
        IsHoliday=int(calendar["IsHoliday"]),
        DaysToNearestHoliday=int(calendar["DaysToNearestHoliday"]),
        ReportingAirline=lookup.ReportingAirline,
        FlightNumberReportingAirline=lookup.FlightNumber,
        OriginAirportID=origin.airport_id,
        Origin=origin.iata,
        OriginCityName=origin.city_name,
        OriginState=origin.state,
        DestAirportID=dest.airport_id,
        Dest=dest.iata,
        DestCityName=dest.city_name,
        DestState=dest.state,
        # Keyed on the id, as add_carrier_features builds it for training.
        OriginCarrier=f"{origin.airport_id}{lookup.ReportingAirline}",
        DestCarrier=f"{dest.airport_id}{lookup.ReportingAirline}",
        DepTimeDecimal=decimal_hour(departure_local),
        ArrTimeDecimal=decimal_hour(arrival_local),
        CRSElapsedTime=block_minutes,
        Distance=miles,
        DistanceGroup=distance_group(miles),
        OriginCongestion=count_movements(origin.iata, pd.Timestamp(departure_local[:16]), False),
        DestCongestion=count_movements(dest.iata, pd.Timestamp(arrival_local[:16]), True),
        **rotation_features(leg, lookup.FlightDate),
    )


def resolve(lookups: list[FlightLookupRequest]) -> list[FlightRequest]:
    """Recover every named flight in full.

    Args:
        lookups: The flights the caller named.

    Returns:
        One fully described flight per request, in the order given.

    Raises:
        ValueError: If lookups is empty.
        FlightNotFoundError: If any flight is not known.
        ScheduleUnavailableError: If the schedule service cannot be reached.
    """
    if not lookups:
        raise ValueError("No flights to look up: the request carries none.")

    calendar = holiday_features([lookup.FlightDate for lookup in lookups])
    return [
        to_flight_request(lookup, calendar.iloc[position])
        for position, lookup in enumerate(lookups)
    ]
