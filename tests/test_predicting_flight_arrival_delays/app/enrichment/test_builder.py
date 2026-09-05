"""Tests for predicting_flight_arrival_delays.app.enrichment.builder."""

from datetime import date

import pandas as pd
import pytest

from predicting_flight_arrival_delays.app.enrichment.builder import (
    FEATURE_FRAME_COLUMNS,
    WEATHER_BEYOND_HORIZON,
    WEATHER_OK,
    WEATHER_UNAVAILABLE,
    WEATHER_UNKNOWN_AIRPORT,
    build_feature_frame,
    lead_days,
)
from predicting_flight_arrival_delays.app.schema import FlightRequest
from predicting_flight_arrival_delays.config import DATE_COLUMN, MAX_LEAD_DAYS
from predicting_flight_arrival_delays.data.features import (
    WEATHER_COLUMNS_DESTINATION,
    WEATHER_COLUMNS_ORIGIN,
)
from predicting_flight_arrival_delays.modeling.predict import has_weather

WEATHER_FEATURES = WEATHER_COLUMNS_ORIGIN + WEATHER_COLUMNS_DESTINATION


class TestLeadDays:
    def test_counts_the_days_to_departure(self):
        assert lead_days(date(2026, 3, 12), date(2026, 3, 10)) == 2

    def test_is_clipped_to_the_range_the_model_was_trained_on(self):
        assert lead_days(date(2026, 6, 1), date(2026, 3, 10)) == MAX_LEAD_DAYS

    def test_a_past_flight_reads_the_freshest_forecast(self):
        assert lead_days(date(2026, 3, 1), date(2026, 3, 10)) == 0


class TestBuildFeatureFrame:
    def test_carries_exactly_the_columns_the_pipeline_produces(
        self, flight_request, today, stub_forecast
    ):
        stub_forecast()

        frame, _ = build_feature_frame([flight_request], today=today)

        assert list(frame.columns) == FEATURE_FRAME_COLUMNS

    def test_what_the_caller_sent_reaches_the_model_unchanged(
        self, flight_request, today, stub_forecast
    ):
        """The service adds to the request; it does not recompute it."""
        stub_forecast()

        frame, _ = build_feature_frame([flight_request], today=today)

        sent = flight_request.model_dump()
        for column, value in sent.items():
            if column == DATE_COLUMN:
                continue
            assert frame.loc[0, column] == value

    def test_the_lead_time_is_the_one_field_the_caller_does_not_set(
        self, flight_request, today, stub_forecast
    ):
        stub_forecast()

        frame, _ = build_feature_frame([flight_request], today=today)

        assert "LeadDays" not in flight_request.model_dump()
        assert frame.loc[0, "LeadDays"] == 2

    def test_a_resolved_forecast_fills_both_ends_of_the_route(
        self, flight_request, today, stub_forecast
    ):
        stub_forecast()

        frame, statuses = build_feature_frame([flight_request], today=today)

        assert statuses == [WEATHER_OK]
        assert has_weather(frame).all()

    def test_the_forecast_is_read_at_the_flight_s_own_hours(
        self, flight_request, today, stub_forecast
    ):
        calls = stub_forecast()

        build_feature_frame([flight_request], today=today)

        (origin_iata, departure), (dest_iata, arrival) = calls
        assert (origin_iata, dest_iata) == ("JFK", "LAX")
        assert departure == pd.Timestamp("2026-03-12 18:00", tz="UTC")
        assert arrival == pd.Timestamp("2026-03-13 00:00", tz="UTC")

    def test_a_flight_beyond_the_horizon_never_calls_the_forecast_service(
        self, body, today, stub_forecast
    ):
        calls = stub_forecast()
        request = FlightRequest(**body(days_ahead=30))

        frame, statuses = build_feature_frame([request], today=today)

        assert statuses == [WEATHER_BEYOND_HORIZON]
        assert calls == []
        assert not has_weather(frame).any()

    def test_an_unreachable_forecast_service_leaves_the_flight_without_weather(
        self, flight_request, today, stub_forecast
    ):
        stub_forecast(result=None)

        frame, statuses = build_feature_frame([flight_request], today=today)

        assert statuses == [WEATHER_UNAVAILABLE]
        assert not has_weather(frame).any()

    def test_an_airport_with_no_known_location_is_scored_without_weather(
        self, body, today, stub_forecast
    ):
        stub_forecast()
        request = FlightRequest(**body(OriginAirportID=99999))

        frame, statuses = build_feature_frame([request], today=today)

        assert statuses == [WEATHER_UNKNOWN_AIRPORT]
        assert not has_weather(frame).any()

    def test_the_weather_code_stays_the_string_training_stored(
        self, flight_request, today, stub_forecast
    ):
        stub_forecast()

        frame, _ = build_feature_frame([flight_request], today=today)

        assert frame.loc[0, "WeatherCodeOrigin"] == "3"

    def test_missing_weather_is_a_numeric_gap_not_a_category(
        self, flight_request, today, stub_forecast
    ):
        stub_forecast(result=None)

        frame, _ = build_feature_frame([flight_request], today=today)

        for column in WEATHER_FEATURES:
            if not column.startswith("WeatherCode"):
                assert pd.api.types.is_numeric_dtype(frame[column])

    def test_a_batch_keeps_one_row_per_request_in_order(self, body, today, stub_forecast):
        stub_forecast()
        requests = [FlightRequest(**body(FlightNumberReportingAirline=n)) for n in (1, 2, 3)]

        frame, statuses = build_feature_frame(requests, today=today)

        assert list(frame["FlightNumberReportingAirline"]) == [1, 2, 3]
        assert len(statuses) == 3

    def test_an_empty_request_is_rejected(self, today):
        with pytest.raises(ValueError, match="none"):
            build_feature_frame([], today=today)
