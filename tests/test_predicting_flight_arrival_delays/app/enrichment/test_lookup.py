"""Tests for predicting_flight_arrival_delays.app.enrichment.lookup."""

from datetime import date

import pytest

from predicting_flight_arrival_delays.app.enrichment import lookup as lookup_module
from predicting_flight_arrival_delays.app.enrichment.aerodatabox import FlightNotFoundError
from predicting_flight_arrival_delays.app.enrichment.lookup import (
    decimal_hour,
    distance_group,
    resolve,
    rotation_features,
)
from predicting_flight_arrival_delays.app.schema import FlightLookupRequest


FLIGHT_DATE = date(2026, 8, 25)


AIRPORT_LOCATIONS = {
    "DFW": {"lat": 32.8972, "lon": -97.0378},
    "LBB": {"lat": 33.6617, "lon": -101.8186},
}


def a_lookup(**overrides) -> FlightLookupRequest:
    fields = {
        "FlightDate": FLIGHT_DATE,
        "MarketingCarrier": "AA",
        "ReportingAirline": "MQ",
        "FlightNumber": 3500,
        "Origin": "DFW",
        "Dest": "LBB",
    }
    return FlightLookupRequest(**(fields | overrides))


def a_leg(dest: str = "LBB", registration: str | None = None) -> dict:
    entry = {
        "departure": {
            "airport": {"iata": "DFW", "location": AIRPORT_LOCATIONS["DFW"]},
            "scheduledTime": {"utc": "2026-08-25 12:00Z", "local": "2026-08-25 07:00-05:00"},
        },
        "arrival": {
            "airport": {"iata": dest, "location": AIRPORT_LOCATIONS.get(dest)},
            "scheduledTime": {"utc": "2026-08-25 13:17Z", "local": "2026-08-25 08:17-05:00"},
        },
        "greatCircleDistance": {"mile": 281.95},
    }
    if registration:
        entry["aircraft"] = {"reg": registration}
    return entry


@pytest.fixture
def schedule(monkeypatch):

    def build(leg=None, congestion=(62, 1), rotation=()):
        counts = list(congestion)
        monkeypatch.setattr(lookup_module, "find_flight",
                            lambda *args, **kwargs: leg if leg is not None else a_leg())
        monkeypatch.setattr(lookup_module, "count_movements",
                            lambda *args, **kwargs: counts.pop(0))
        monkeypatch.setattr(lookup_module, "aircraft_rotation",
                            lambda *args, **kwargs: list(rotation))

    return build


class TestDistanceGroup:
    @pytest.mark.parametrize(
        ("distance", "expected"), [(31, 1), (249, 1), (250, 2), (2500, 11), (5095, 11)]
    )
    def test_matches_the_bts_buckets(self, distance, expected):
        assert distance_group(distance) == expected


class TestDecimalHour:
    @pytest.mark.parametrize(
        ("local", "expected"),
        [("2026-08-25 07:00-05:00", 7.0), ("2026-08-25 08:17-05:00", 8 + 17 / 60),
         ("2026-08-25 00:00-05:00", 0.0), ("2026-08-25 23:30-05:00", 23.5)],
    )
    def test_reads_the_local_clock_not_the_offset(self, local, expected):
        assert decimal_hour(local) == pytest.approx(expected)


class TestRotationFeatures:
    def rotation_leg(self, departure: str, arrival: str) -> dict:
        return {
            "departure": {"scheduledTime": {"utc": departure}},
            "arrival": {"scheduledTime": {"utc": arrival}},
        }

    def test_without_an_assigned_aircraft_everything_stays_empty(self, schedule):
        """A future flight has no aircraft yet, so there is no rotation to read."""
        schedule()

        features = rotation_features(a_leg(), FLIGHT_DATE)

        assert set(features.values()) == {None}

    def test_the_position_and_the_count_come_from_the_day_s_legs(self, schedule):
        schedule(rotation=[
            self.rotation_leg("2026-08-25 09:00Z", "2026-08-25 11:00Z"),
            self.rotation_leg("2026-08-25 12:00Z", "2026-08-25 13:17Z"),
            self.rotation_leg("2026-08-25 15:00Z", "2026-08-25 17:00Z"),
        ])

        features = rotation_features(a_leg(registration="N123AA"), FLIGHT_DATE)

        assert features["AircraftDailyLegs"] == 3
        assert features["LegPosition"] == 2

    def test_the_turnaround_is_measured_from_the_previous_arrival(self, schedule):
        schedule(rotation=[
            self.rotation_leg("2026-08-25 09:00Z", "2026-08-25 11:00Z"),
            self.rotation_leg("2026-08-25 12:00Z", "2026-08-25 13:17Z"),
        ])

        features = rotation_features(a_leg(registration="N123AA"), FLIGHT_DATE)

        assert features["ScheduledTurnaround"] == 60

    def test_the_turnaround_is_floored_the_way_training_floors_it(self, schedule):
        """Training rounds both ends down to the hour, so it only ever saw whole
        hours; a truer figure would be a value the model has never seen."""
        schedule(rotation=[
            self.rotation_leg("2026-08-25 09:00Z", "2026-08-25 11:40Z"),
            self.rotation_leg("2026-08-25 12:20Z", "2026-08-25 13:17Z"),
        ])
        leg = a_leg(registration="N123AA")
        leg["departure"]["scheduledTime"]["utc"] = "2026-08-25 12:20Z"

        features = rotation_features(leg, FLIGHT_DATE)

        assert features["ScheduledTurnaround"] == 60

    def test_the_first_leg_of_the_day_has_no_turnaround(self, schedule):
        schedule(rotation=[self.rotation_leg("2026-08-25 12:00Z", "2026-08-25 13:17Z")])

        features = rotation_features(a_leg(registration="N123AA"), FLIGHT_DATE)

        assert features["LegPosition"] == 1
        assert features["ScheduledTurnaround"] is None

    def test_an_unpublished_rotation_leaves_everything_empty(self, schedule):
        schedule(rotation=[])

        assert set(rotation_features(a_leg(registration="N123AA"), FLIGHT_DATE).values()) == {None}


class TestResolve:
    def test_a_named_flight_comes_back_fully_described(self, schedule):
        schedule()

        flight = resolve([a_lookup()])[0]

        assert flight.Origin == "DFW"
        assert flight.OriginAirportID == 11298
        assert flight.OriginCityName == "Dallas/Fort Worth, TX"
        assert flight.OriginState == "TX"

    def test_the_operating_carrier_is_the_one_the_caller_gave(self, schedule):
        """The schedule service reports the marketing brand; BTS records the operator,
        and that is what the model was trained on."""
        schedule()

        flight = resolve([a_lookup()])[0]

        assert flight.ReportingAirline == "MQ"
        assert (flight.OriginCarrier, flight.DestCarrier) == ("11298MQ", "12896MQ")

    def test_the_schedule_supplies_the_times_and_the_distance(self, schedule):
        schedule()

        flight = resolve([a_lookup()])[0]

        assert flight.DepTimeDecimal == 7.0
        assert flight.CRSElapsedTime == 77
        assert flight.Distance == 281.95
        assert flight.DistanceGroup == 2

    def test_the_calendar_is_derived_from_the_date(self, schedule):
        schedule()

        flight = resolve([a_lookup()])[0]

        assert flight.Month == 8
        assert flight.DayOfWeek == 2  # a Tuesday, and BTS counts Monday as 1

    def test_the_congestion_counts_reach_the_flight(self, schedule):
        schedule(congestion=(62, 1))

        flight = resolve([a_lookup()])[0]

        assert (flight.OriginCongestion, flight.DestCongestion) == (62, 1)

    def test_a_flight_that_goes_somewhere_else_is_refused(self, schedule):
        schedule(leg=a_leg(dest="SPI"))

        with pytest.raises(FlightNotFoundError, match="SPI"):
            resolve([a_lookup()])

    def test_every_field_the_model_may_read_is_sent(self, schedule):
        schedule()

        flight = resolve([a_lookup()])[0]

        assert flight.supplied() == set(type(flight).model_fields)

    def test_an_empty_request_is_rejected(self):
        with pytest.raises(ValueError, match="none"):
            resolve([])

    def test_a_batch_keeps_its_order(self, schedule):
        schedule(congestion=(62, 1, 40, 2))

        flights = resolve([a_lookup(FlightNumber=3500), a_lookup(FlightNumber=3501)])

        assert [f.FlightNumberReportingAirline for f in flights] == [3500, 3501]
