"""What the service reports about itself.

The counters live in the process-wide Prometheus registry and never reset, so every
assertion here is on a delta: read the sample, do the thing, read it again.
"""

import pytest

from predicting_flight_arrival_delays.app import monitoring


def sample(metric, suffix: str = "_total", **labels) -> float:
    """The current value of one labelled sample, or zero if it has never been set.

    Args:
        metric: The Counter or Histogram to read.
        suffix: Which sample of a multi-sample metric to take.
        **labels: The label set identifying the series.

    Returns:
        The value Prometheus would expose right now.
    """
    for family in metric.collect():
        for point in family.samples:
            if point.name.endswith(suffix) and point.labels == labels:
                return point.value
    return 0.0


def a_result(**overrides) -> dict:
    """One scored flight, in the shape the endpoints hand to record_scoring."""
    return {
        "variant": "all",
        "weather": "resolved",
        "is_delayed": 0,
        "delay_probability": 0.21,
        "approximated": [],
    } | overrides


class TestRecordScoring:
    def test_the_request_is_counted_once_whatever_the_batch(self):
        before = sample(monitoring.requests_total, endpoint="/batch-predictions", kind="batch")

        monitoring.record_scoring("/batch-predictions", "batch", [a_result()] * 9, 0.1)

        after = sample(monitoring.requests_total, endpoint="/batch-predictions", kind="batch")
        assert after - before == 1

    def test_every_flight_in_it_is_counted(self):
        before = sample(monitoring.flights_scored_total, variant="all", weather="resolved")

        monitoring.record_scoring("/batch-predictions", "batch", [a_result()] * 9, 0.1)

        after = sample(monitoring.flights_scored_total, variant="all", weather="resolved")
        assert after - before == 9

    def test_the_variant_that_answered_is_recorded(self):
        """The whole point of the metric: a fallback answer still returns 200."""
        before = sample(
            monitoring.flights_scored_total, variant="noweather", weather="beyond_horizon"
        )

        monitoring.record_scoring(
            "/predictions",
            "single",
            [a_result(variant="noweather", weather="beyond_horizon")],
            0.02,
        )

        after = sample(
            monitoring.flights_scored_total, variant="noweather", weather="beyond_horizon"
        )
        assert after - before == 1

    @pytest.mark.parametrize(("is_delayed", "outcome"), [(1, "delayed"), (0, "on_time")])
    def test_the_call_is_recorded_on_either_side_of_the_threshold(self, is_delayed, outcome):
        before = sample(monitoring.outcomes_total, outcome=outcome)

        monitoring.record_scoring(
            "/predictions", "single", [a_result(is_delayed=is_delayed)], 0.02
        )

        assert sample(monitoring.outcomes_total, outcome=outcome) - before == 1

    def test_the_probability_is_observed_per_flight(self):
        before = sample(monitoring.delay_probability, suffix="_count")

        monitoring.record_scoring("/batch-predictions", "batch", [a_result()] * 4, 0.1)

        assert sample(monitoring.delay_probability, suffix="_count") - before == 4

    def test_only_a_batch_reports_its_size(self):
        """A single flight is not a batch of one; counting it would flatten the histogram."""
        before = sample(monitoring.batch_size, suffix="_count")

        monitoring.record_scoring("/predictions", "single", [a_result()], 0.02)

        assert sample(monitoring.batch_size, suffix="_count") == before

    def test_the_scoring_time_is_kept_per_endpoint(self):
        before = sample(monitoring.scoring_seconds, suffix="_count", endpoint="/explanations")

        monitoring.record_scoring("/explanations", "single", [a_result()], 0.4)

        after = sample(monitoring.scoring_seconds, suffix="_count", endpoint="/explanations")
        assert after - before == 1

    def test_a_flight_missing_a_column_the_model_leans_on_is_flagged(self):
        before = sample(monitoring.approximated_total)

        monitoring.record_scoring(
            "/predictions",
            "single",
            [a_result(approximated=["OriginCongestion"]), a_result()],
            0.02,
        )

        assert sample(monitoring.approximated_total) - before == 1


class TestRecordError:
    def test_the_reason_is_kept_apart_from_the_endpoint(self):
        before = sample(
            monitoring.errors_total, endpoint="/predictions", reason="model_unavailable"
        )

        monitoring.record_error("/predictions", "model_unavailable")

        after = sample(
            monitoring.errors_total, endpoint="/predictions", reason="model_unavailable"
        )
        assert after - before == 1


class TestTheMetricsEndpoint:
    def test_it_survives_the_gradio_mount(self):
        """Gradio takes over '/', so a route added after it would never be reached."""
        from predicting_flight_arrival_delays.app.main import app

        assert "/metrics" in {getattr(route, "path", None) for route in app.routes}
