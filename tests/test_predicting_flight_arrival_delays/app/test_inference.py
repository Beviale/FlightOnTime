"""Tests for predicting_flight_arrival_delays.app.inference."""

import pytest

from predicting_flight_arrival_delays.app.enrichment.builder import build_feature_frame
from predicting_flight_arrival_delays.app.inference import ModelUnavailableError, score
from predicting_flight_arrival_delays.app.schema import FlightRequest

WITH_WEATHER = 2
WITHOUT_WEATHER = 30


@pytest.fixture
def frame(body, today, stub_forecast):
    """Factory for a scored-ready frame, one row per lead time asked for."""
    stub_forecast()

    def build(*days_ahead: int):
        requests = [
            FlightRequest(**body(days_ahead=d, FlightNumberReportingAirline=n + 1))
            for n, d in enumerate(days_ahead)
        ]
        return build_feature_frame(requests, today=today)[0]

    return build


class TestScore:
    def test_a_flight_with_weather_is_served_by_the_full_model(self, bundles, frame):
        assert list(score(frame(WITH_WEATHER), bundles)["variant"]) == ["all"]

    def test_a_flight_without_weather_falls_back(self, bundles, frame):
        """Beyond the forecast horizon there is nothing for the full model to read."""
        assert list(score(frame(WITHOUT_WEATHER), bundles)["variant"]) == ["noweather"]

    def test_a_mixed_batch_is_split_between_the_two_models(self, bundles, frame):
        scored = score(frame(WITH_WEATHER, WITHOUT_WEATHER, 1), bundles)

        assert list(scored["variant"]) == ["all", "noweather", "all"]

    def test_the_original_order_survives_the_split(self, bundles, frame):
        """Rows are routed by mask and concatenated back, so order is not free."""
        rows = frame(WITHOUT_WEATHER, WITH_WEATHER)

        scored = score(rows, bundles)

        assert list(scored.index) == list(rows.index)
        assert list(scored["variant"]) == ["noweather", "all"]

    def test_each_model_applies_the_threshold_it_was_released_with(self, bundles, frame):
        scored = score(frame(WITH_WEATHER, WITHOUT_WEATHER), bundles)

        assert list(scored["threshold"]) == [
            bundles["all"].threshold,
            bundles["noweather"].threshold,
        ]

    def test_an_override_replaces_the_released_threshold(self, bundles, frame):
        """The stub scores 0.80, so a cutoff above it must flip the label."""
        rows = frame(WITH_WEATHER)

        assert score(rows, bundles)["is_delayed"].iloc[0] == 1
        assert score(rows, bundles, threshold=0.9)["is_delayed"].iloc[0] == 0

    def test_a_missing_variant_names_itself(self, bundles, frame):
        """A registry that carries only one variant should say which is absent."""
        with pytest.raises(ModelUnavailableError, match="noweather"):
            score(frame(WITHOUT_WEATHER), {"all": bundles["all"]})

    def test_an_unused_variant_is_not_required(self, bundles, frame):
        """Serving only 'all' is fine as long as no flight needs the other one."""
        scored = score(frame(WITH_WEATHER), {"all": bundles["all"]})

        assert scored["variant"].iloc[0] == "all"

    def test_an_empty_frame_is_rejected(self, bundles, frame):
        with pytest.raises(ValueError, match="empty"):
            score(frame(WITH_WEATHER).iloc[0:0], bundles)

    def test_the_probability_is_a_usable_number(self, bundles, frame):
        probability = score(frame(WITH_WEATHER), bundles)["delay_probability"].iloc[0]

        assert 0.0 <= probability <= 1.0
