"""Tests for predicting_flight_arrival_delays.app.schema."""

import pytest
from pydantic import ValidationError

from predicting_flight_arrival_delays.app.enrichment.builder import FEATURE_FRAME_COLUMNS
from predicting_flight_arrival_delays.app.inputs import FORECAST_INPUTS
from predicting_flight_arrival_delays.app.schema import FlightRequest
from predicting_flight_arrival_delays.data.features import (
    WEATHER_COLUMNS_DESTINATION,
    WEATHER_COLUMNS_ORIGIN,
)

WEATHER_FEATURES = set(WEATHER_COLUMNS_ORIGIN + WEATHER_COLUMNS_DESTINATION)
SERVICE_SUPPLIED = WEATHER_FEATURES | {"LeadDays"}


class TestFlightRequest:
    def test_it_asks_for_every_feature_the_service_cannot_supply(self):
        """Manual entry is the caller's half of the feature frame: all of it but the
        weather, and but the lead time of the forecast."""
        expected = set(FEATURE_FRAME_COLUMNS) - SERVICE_SUPPLIED

        assert set(FlightRequest.model_fields) == expected

    def test_it_asks_for_nothing_the_service_supplies(self):
        assert not set(FlightRequest.model_fields) & SERVICE_SUPPLIED

    def test_a_complete_body_is_accepted(self, body):
        assert FlightRequest(**body()).Origin == "JFK"

    def test_codes_are_uppercased(self, body):
        """The training vocabulary is uppercase, so 'jfk' would land in OTHER."""
        request = FlightRequest(**body(Origin="jfk", OriginCarrier=" 12478aa "))

        assert (request.Origin, request.OriginCarrier) == ("JFK", "12478AA")

    @pytest.mark.parametrize("field", sorted(FORECAST_INPUTS))
    def test_a_field_the_forecast_needs_is_always_required(self, body, field):
        payload = body()
        del payload[field]

        with pytest.raises(ValidationError):
            FlightRequest(**payload)

    @pytest.mark.parametrize("field", ["Origin", "OriginCongestion", "AircraftDailyLegs"])
    def test_any_other_field_is_optional_here(self, body, field):
        """Whether it is really needed depends on the loaded model, which the schema
        cannot know; app.inputs decides it at request time."""
        payload = body()
        del payload[field]

        assert getattr(FlightRequest(**payload), field) is None

    def test_a_field_left_out_is_not_the_same_as_one_sent_null(self, body):
        omitted = FlightRequest(**{k: v for k, v in body().items() if k != "LegPosition"})
        explicit = FlightRequest(**body(LegPosition=None))

        assert "LegPosition" not in omitted.supplied()
        assert "LegPosition" in explicit.supplied()

    def test_a_missing_previous_leg_is_sent_as_null(self, body):
        assert FlightRequest(**body(ScheduledTurnaround=None)).ScheduledTurnaround is None

    @pytest.mark.parametrize(
        ("field", "value"), [("Month", 4), ("DistanceGroup", 3), ("OriginCarrier", "12478DL")]
    )
    def test_a_derived_field_is_taken_as_sent(self, body, field, value):
        assert getattr(FlightRequest(**body(**{field: value})), field) == value

    @pytest.mark.parametrize("hour", [-0.5, 24.0, 25.3])
    def test_a_decimal_hour_outside_the_day_is_rejected(self, body, hour):
        with pytest.raises(ValidationError):
            FlightRequest(**body(DepTimeDecimal=hour))

    def test_a_negative_distance_is_rejected(self, body):
        with pytest.raises(ValidationError):
            FlightRequest(**body(Distance=-100))

    def test_a_holiday_flag_outside_zero_and_one_is_rejected(self, body):
        with pytest.raises(ValidationError):
            FlightRequest(**body(IsHoliday=2))

    def test_an_aircraft_cannot_fly_a_zeroth_leg(self, body):
        with pytest.raises(ValidationError):
            FlightRequest(**body(LegPosition=0))
