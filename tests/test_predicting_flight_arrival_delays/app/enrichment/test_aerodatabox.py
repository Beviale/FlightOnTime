"""Tests for predicting_flight_arrival_delays.app.enrichment.aerodatabox."""

from datetime import date

import pytest

from predicting_flight_arrival_delays.app.enrichment import aerodatabox
from predicting_flight_arrival_delays.app.enrichment.aerodatabox import (
    FlightNotFoundError,
    ScheduleUnavailableError,
    count_movements,
    find_flight,
)

FLIGHT_DATE = date(2026, 8, 25)


def leg(origin: str, dest: str | None, hour: int = 7) -> dict:
    """One timetable entry, as the service returns it."""
    return {
        "number": "AA 3500",
        "departure": {
            "airport": {"iata": origin, "countryCode": "us"},
            "scheduledTime": {"utc": f"2026-08-25 {hour + 5:02d}:00Z",
                              "local": f"2026-08-25 {hour:02d}:00-05:00"},
        },
        "arrival": {
            "airport": {"iata": dest, "countryCode": "us"} if dest else {},
            "scheduledTime": {"utc": "2026-08-25 13:17Z", "local": "2026-08-25 08:17-05:00"},
        },
        "greatCircleDistance": {"mile": 281.95},
    }


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    """A key must be present, or every call refuses before reaching the network."""
    monkeypatch.setenv("RAPIDAPI_KEY", "test-key")


@pytest.fixture
def responses(monkeypatch):

    def build(*queue):
        calls = []
        remaining = list(queue)

        def fake_get(url, headers=None, params=None, timeout=None):
            calls.append((url, params))
            return remaining.pop(0) if remaining else FakeResponse(200, [])

        monkeypatch.setattr(aerodatabox.requests, "get", fake_get)
        monkeypatch.setattr(aerodatabox.time, "sleep", lambda seconds: None)
        return calls

    return build


class TestFindFlight:
    def test_the_origin_picks_the_leg_meant(self, responses):
        responses(FakeResponse(200, [leg("LBB", "DFW"), leg("DFW", "LBB")]))

        found = find_flight("AA", 3500, FLIGHT_DATE, "DFW")

        assert found["departure"]["airport"]["iata"] == "DFW"

    def test_a_leg_with_no_arrival_airport_is_discarded(self, responses):
        responses(FakeResponse(200, [leg("DFW", None), leg("DFW", "LBB")]))

        found = find_flight("AA", 3500, FLIGHT_DATE, "DFW")

        assert found["arrival"]["airport"]["iata"] == "LBB"

    def test_a_day_with_no_such_flight_is_reported_as_not_found(self, responses):
        responses(FakeResponse(204))

        with pytest.raises(FlightNotFoundError, match="AA3500"):
            find_flight("AA", 3500, FLIGHT_DATE, "DFW")

    def test_the_error_names_where_the_flight_does_depart_from(self, responses):
        responses(FakeResponse(200, [leg("LBB", "DFW")]))

        with pytest.raises(FlightNotFoundError, match="LBB"):
            find_flight("AA", 3500, FLIGHT_DATE, "DFW")

    def test_the_marketing_code_is_what_is_searched(self, responses):
        """The service is indexed on the code the flight is sold under."""
        calls = responses(FakeResponse(200, [leg("DFW", "LBB")]))

        find_flight("AA", 3500, FLIGHT_DATE, "DFW")

        assert "/flights/number/AA3500/2026-08-25" in calls[0][0]


class TestRateLimiting:
    def test_a_rejected_call_is_retried(self, responses):
        calls = responses(FakeResponse(429), FakeResponse(200, [leg("DFW", "LBB")]))

        find_flight("AA", 3500, FLIGHT_DATE, "DFW")

        assert len(calls) == 2

    def test_giving_up_says_so(self, responses):
        responses(*[FakeResponse(429)] * aerodatabox.MAX_ATTEMPTS)

        with pytest.raises(ScheduleUnavailableError, match="rate-limited"):
            find_flight("AA", 3500, FLIGHT_DATE, "DFW")

    def test_any_other_error_is_not_retried(self, responses):
        calls = responses(FakeResponse(500))

        with pytest.raises(ScheduleUnavailableError, match="500"):
            find_flight("AA", 3500, FLIGHT_DATE, "DFW")
        assert len(calls) == 1


class TestCountMovements:
    def departures(self, *country_codes) -> FakeResponse:
        return FakeResponse(200, {
            "departures": [
                {"movement": {"airport": {"iata": "XXX", "countryCode": cc}}}
                for cc in country_codes
            ]
        })

    def test_only_flights_with_both_ends_on_us_soil_are_counted(self, responses):
        responses(self.departures("us", "us", "gb", "fr", "mx"))

        assert count_movements("JFK", "2026-08-25 08:30", arriving=False) == 2

    def test_the_us_territories_count_as_domestic(self, responses):
        responses(self.departures("us", "pr", "vi"))

        assert count_movements("JFK", "2026-08-25 08:30", arriving=False) == 3

    def test_the_window_is_the_one_hour_the_flight_sits_in(self, responses):
        calls = responses(self.departures("us"))

        count_movements("JFK", "2026-08-25 08:47", arriving=False)

        assert "/flights/airports/iata/JFK/2026-08-25T08:00/2026-08-25T08:59" in calls[0][0]

    def test_cancelled_flights_are_asked_for(self, responses):
        calls = responses(self.departures("us"))

        count_movements("JFK", "2026-08-25 08:00", arriving=False)

        assert calls[0][1]["withCancelled"] == "true"

    def test_codeshares_are_excluded(self, responses):
        calls = responses(self.departures("us"))

        count_movements("JFK", "2026-08-25 08:00", arriving=False)

        assert calls[0][1]["withCodeshared"] == "false"

    def test_arrivals_are_read_from_their_own_key(self, responses):
        responses(FakeResponse(200, {
            "arrivals": [{"movement": {"airport": {"iata": "ATL", "countryCode": "us"}}}],
            "departures": [],
        }))

        assert count_movements("LBB", "2026-08-25 08:00", arriving=True) == 1

    def test_an_hour_with_nothing_scheduled_counts_zero(self, responses):
        responses(FakeResponse(204))

        assert count_movements("LBB", "2026-08-25 03:00", arriving=False) == 0


class TestMissingKey:
    def test_a_missing_api_key_is_reported_before_any_call(self, monkeypatch):
        monkeypatch.delenv("RAPIDAPI_KEY", raising=False)

        with pytest.raises(ScheduleUnavailableError, match="RAPIDAPI_KEY"):
            find_flight("AA", 3500, FLIGHT_DATE, "DFW")
