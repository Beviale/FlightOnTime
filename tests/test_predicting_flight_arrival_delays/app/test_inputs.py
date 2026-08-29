"""Tests for predicting_flight_arrival_delays.app.inputs."""

import pandas as pd
import pytest

from predicting_flight_arrival_delays.app.enrichment.builder import build_feature_frame
from predicting_flight_arrival_delays.app.inference import score
from predicting_flight_arrival_delays.app.inputs import (
    FORECAST_INPUTS,
    complete_frame,
    contributes,
    contributing_columns,
    required_inputs,
)
from predicting_flight_arrival_delays.app.schema import FlightRequest
from predicting_flight_arrival_delays.data.transform import OTHER

CANDIDATES = list(FlightRequest.model_fields)


class Stub:
    def __init__(self, category_keep=None, delay_rate_columns=(), numeric_columns=(),
                 impute_values=None):  # noqa: D107
        self.category_keep = category_keep or {}
        self.delay_rate_columns = list(delay_rate_columns)
        self.wide_columns_ = []
        self.numeric_columns = list(numeric_columns)
        self.impute_values = impute_values or {}

    def rate_columns(self):
        """As the real Transformer does: named columns first, then the wide ones."""
        return self.delay_rate_columns + [
            c for c in self.wide_columns_ if c not in self.delay_rate_columns
        ]


class TestContributes:
    def test_a_numeric_feature_appears_under_its_own_name(self):
        assert contributes("Distance", Stub(), {"Distance", "Month"})

    def test_a_one_hot_ancestor_is_traced_through_its_categories(self):
        """The names are rebuilt from the categories the transformer kept, not
        guessed from a prefix."""
        transformer = Stub(category_keep={"Dest": {"JFK", "SEA"}})

        assert contributes("Dest", transformer, {"Dest_JFK", "Dest_SEA"})

    def test_a_single_surviving_category_is_enough(self):
        transformer = Stub(category_keep={"Dest": {"JFK", "SEA", "MIA"}})

        assert contributes("Dest", transformer, {"Dest_SEA"})

    def test_the_other_bucket_counts_as_a_surviving_category(self):
        transformer = Stub(category_keep={"Dest": {"JFK"}})

        assert contributes("Dest", transformer, {f"Dest_{OTHER}"})

    def test_a_delay_rate_source_counts_even_when_the_column_itself_was_dropped(self):
        """Origin is the usual case: gone as a feature, still needed to look up the
        historical rate that replaced it."""
        transformer = Stub(delay_rate_columns=["Origin"])

        assert contributes("Origin", transformer, {"OriginDelayRate"})

    def test_a_column_with_no_trace_does_not_contribute(self):
        transformer = Stub(category_keep={"Dest": {"JFK"}}, delay_rate_columns=["Origin"])

        assert not contributes("DestCongestion", transformer, {"Dest_JFK", "OriginDelayRate"})

    def test_a_similar_name_is_not_mistaken_for_an_ancestor(self):
        """DestCarrier_12892AA must not make Dest look like it survived."""
        transformer = Stub(category_keep={"Dest": {"JFK", "LAX"}})

        assert not contributes("Dest", transformer, {"DestCarrier_12892AA", "DestState_CA"})


class TestContributingColumns:
    def test_it_reads_a_real_fitted_bundle(self, bundles):
        """Fitted as training fits, so the selection it reflects is a real one."""
        contributing = contributing_columns(bundles["all"], CANDIDATES)

        assert contributing
        assert contributing < set(CANDIDATES)

    def test_the_weather_model_and_the_fallback_need_not_agree(self, bundles):
        """They are selected separately, which is why the request takes the union."""
        for variant in ("all", "noweather"):
            assert contributing_columns(bundles[variant], CANDIDATES)


class TestRequiredInputs:
    def test_it_is_the_union_of_what_the_served_models_read(self, bundles):
        union = contributing_columns(bundles["all"], CANDIDATES) | contributing_columns(
            bundles["noweather"], CANDIDATES
        )

        assert required_inputs(bundles, CANDIDATES) >= union

    def test_the_forecast_inputs_are_required_whatever_the_models_make_of_them(self, bundles):
        """Without them there is nowhere and no hour to ask a forecast about."""
        assert FORECAST_INPUTS <= required_inputs(bundles, CANDIDATES)

    def test_serving_one_model_asks_for_less_than_serving_both(self, bundles):
        one = required_inputs({"all": bundles["all"]}, CANDIDATES)

        assert one <= required_inputs(bundles, CANDIDATES)

    def test_with_no_model_loaded_only_the_forecast_inputs_are_held_to(self):
        assert required_inputs({}, CANDIDATES) == FORECAST_INPUTS


class TestCompleteFrame:
    def test_a_missing_numeric_column_comes_back_as_its_training_median(self):
        transformer = Stub(numeric_columns=["Distance"], impute_values={"Distance": 900.0})

        completed = complete_frame(pd.DataFrame({"Month": [3]}), transformer)

        assert completed["Distance"].iloc[0] == 900.0

    def test_a_missing_categorical_column_comes_back_as_other(self):
        transformer = Stub(category_keep={"Dest": {"JFK"}})

        completed = complete_frame(pd.DataFrame({"Month": [3]}), transformer)

        assert completed["Dest"].iloc[0] == OTHER

    def test_a_column_the_request_carried_is_left_alone(self):
        transformer = Stub(numeric_columns=["Distance"], impute_values={"Distance": 900.0})

        completed = complete_frame(pd.DataFrame({"Distance": [2475.0]}), transformer)

        assert completed["Distance"].iloc[0] == 2475.0

    def test_a_delay_rate_column_is_not_put_back(self):
        """It is numeric and fitted, but the transformer derives it on every call -
        restoring it here would leave the frame carrying it twice."""
        transformer = Stub(
            numeric_columns=["OriginDelayRate"],
            delay_rate_columns=["Origin"],
            impute_values={"OriginDelayRate": 0.5},
        )

        completed = complete_frame(pd.DataFrame({"Origin": ["JFK"]}), transformer)

        assert "OriginDelayRate" not in completed.columns

    def test_a_delay_rate_is_put_back_when_its_source_was_not_sent(self):
        transformer = Stub(
            numeric_columns=["DestCarrierDelayRate"],
            delay_rate_columns=["DestCarrier"],
            impute_values={"DestCarrierDelayRate": 0.2},
        )

        completed = complete_frame(pd.DataFrame({"Origin": ["JFK"]}), transformer)

        assert completed["DestCarrierDelayRate"].iloc[0] == 0.2

    def test_the_caller_s_frame_is_not_modified(self):
        transformer = Stub(numeric_columns=["Distance"], impute_values={"Distance": 900.0})
        df = pd.DataFrame({"Month": [3]})

        complete_frame(df, transformer)

        assert list(df.columns) == ["Month"]

    def test_a_partial_request_still_scores(self, bundles, body, today, stub_forecast):
        """The whole point: what the models do not read need not be sent."""
        stub_forecast()
        required = required_inputs(bundles, CANDIDATES)
        payload = {k: v for k, v in body().items() if k in required}
        frame, _ = build_feature_frame([FlightRequest(**payload)], today=today)

        scored = score(frame, bundles)

        assert scored["variant"].iloc[0] == "all"
        assert 0.0 <= scored["delay_probability"].iloc[0] <= 1.0


@pytest.mark.parametrize("variant", ["all", "noweather"])
def test_the_columns_a_model_reads_are_a_subset_of_what_can_be_asked(bundles, variant):
    """A model reading something the request cannot carry would be unservable."""
    assert contributing_columns(bundles[variant], CANDIDATES) <= set(CANDIDATES)
