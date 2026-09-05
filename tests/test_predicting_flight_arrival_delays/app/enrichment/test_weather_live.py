"""Tests for predicting_flight_arrival_delays.app.enrichment.weather_live."""

import pandas as pd
import pytest

from predicting_flight_arrival_delays.app.enrichment import weather_live
from predicting_flight_arrival_delays.app.enrichment.reference import Airport, get_airport
from predicting_flight_arrival_delays.config import WEATHER_COLUMNS

DEPARTURE = pd.Timestamp("2026-03-12 18:00", tz="UTC")


@pytest.fixture
def jfk() -> Airport:
    return get_airport(12478)


@pytest.fixture
def series() -> pd.DataFrame:
    index = pd.date_range("2026-03-11", "2026-03-13 23:00", freq="h", tz="UTC", name="Time")
    values = {
        "Temperature2m": 11.5,
        "Precipitation": 0.2,
        "Snowfall": 0.0,
        "WindSpeed10m": 14.0,
        "WindGusts10m": 22.0,
        "WeatherCode": 3.0,
    }
    return pd.DataFrame({c: v for c, v in values.items()}, index=index)


@pytest.fixture
def stub_fetch(monkeypatch, series):
    def build(result=None):
        calls = []

        def fake_fetch_forecast(latitude, longitude, start, end):
            calls.append((latitude, longitude, start, end))
            if isinstance(result, Exception):
                raise result
            return series if result is None else result

        monkeypatch.setattr(weather_live, "fetch_forecast", fake_fetch_forecast)
        return calls

    return build


class TestWeatherAt:
    def test_it_returns_every_column_the_join_produced_in_training(self, jfk, stub_fetch):
        stub_fetch()

        assert set(weather_live.weather_at(jfk, DEPARTURE)) == set(WEATHER_COLUMNS)

    def test_the_weather_code_comes_back_as_the_string_training_stored(self, jfk, stub_fetch):
        stub_fetch()

        assert weather_live.weather_at(jfk, DEPARTURE)["WeatherCode"] == "3"

    def test_a_window_around_the_flight_date_is_requested(self, jfk, stub_fetch):
        calls = stub_fetch()

        weather_live.weather_at(jfk, DEPARTURE)

        _, _, start, end = calls[0]
        assert (start, end) == ("2026-03-11", "2026-03-13")

    def test_a_second_lookup_on_the_same_day_is_served_from_the_cache(self, jfk, stub_fetch):
        calls = stub_fetch()
        cache: dict = {}

        weather_live.weather_at(jfk, DEPARTURE, cache)
        weather_live.weather_at(jfk, DEPARTURE + pd.Timedelta(hours=3), cache)

        assert len(calls) == 1

    def test_a_failed_lookup_is_not_retried_within_the_same_batch(self, jfk, stub_fetch):
        calls = stub_fetch(result=TimeoutError("service down"))
        cache: dict = {}

        assert weather_live.weather_at(jfk, DEPARTURE, cache) is None
        assert weather_live.weather_at(jfk, DEPARTURE, cache) is None
        assert len(calls) == 1

    def test_an_unreachable_service_degrades_instead_of_raising(self, jfk, stub_fetch):
        stub_fetch(result=ConnectionError("no route to host"))

        assert weather_live.weather_at(jfk, DEPARTURE) is None

    def test_an_hour_the_series_does_not_cover_degrades(self, jfk, stub_fetch):
        stub_fetch(result=pd.DataFrame())

        assert weather_live.weather_at(jfk, DEPARTURE) is None

    def test_an_incomplete_row_is_treated_as_no_forecast(self, jfk, stub_fetch, series):
        """The 'all' model was never trained to see a partial weather block."""
        gapped = series.copy()
        gapped.loc[DEPARTURE, "WindGusts10m"] = None
        stub_fetch(result=gapped)

        assert weather_live.weather_at(jfk, DEPARTURE) is None

    def test_a_missing_hour_is_not_asked_about(self, jfk, stub_fetch):
        """add_utc_columns leaves NaT where a timezone could not be resolved."""
        calls = stub_fetch()

        assert weather_live.weather_at(jfk, pd.NaT) is None
        assert calls == []
