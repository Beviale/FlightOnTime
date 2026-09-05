"""Invariance tests: changes that must not move a prediction."""

import numpy as np
import pytest
from thresholds import MAX_FLIGHT_NUMBER_EFFECT

from predicting_flight_arrival_delays.config import SERVICE_COLUMNS


class TestDeterminism:
    def test_the_same_input_always_scores_the_same(self, model, sample):
        first = model.score(sample)
        second = model.score(sample)

        np.testing.assert_allclose(first, second, err_msg=model.name)

    def test_scoring_does_not_consume_the_input(self, model, sample):
        """prepare_for_inference must leave the caller's frame untouched."""
        before = sample.copy()
        model.score(sample)

        assert sample.equals(before), model.name


class TestBatchIndependence:
    def test_row_order_does_not_change_a_prediction(self, model, sample):
        """A flight's score is a property of the flight, not of its neighbours."""
        straight = model.score(sample)
        reversed_scores = model.score(sample.iloc[::-1])

        np.testing.assert_allclose(straight, reversed_scores[::-1], err_msg=model.name)

    def test_a_flight_scores_the_same_alone_as_in_a_crowd(self, model, sample):
        """Production serves one flight at a time; batching must not shift it."""
        in_batch = model.score(sample)[0]
        alone = model.score(sample.head(1))[0]

        assert alone == pytest.approx(in_batch, abs=1e-9), model.name

    def test_a_single_airport_batch_is_not_mistaken_for_another(self, model, sample):
        """The batch that broke: one airport means most dummy columns are absent."""
        airport = sample["Origin"].iloc[0]
        selected = sample["Origin"].to_numpy() == airport

        together = model.score(sample)[selected]
        apart = model.score(sample[selected])

        np.testing.assert_allclose(together, apart, atol=1e-9, err_msg=model.name)

    def test_splitting_a_batch_in_half_changes_nothing(self, model, sample):
        half = len(sample) // 2
        whole = model.score(sample)
        halves = np.concatenate([model.score(sample.iloc[:half]), model.score(sample.iloc[half:])])

        np.testing.assert_allclose(whole, halves, atol=1e-9, err_msg=model.name)

    def test_the_dataframe_index_carries_no_signal(self, model, sample):
        reindexed = sample.copy()
        reindexed.index = np.arange(500_000, 500_000 + len(reindexed))

        np.testing.assert_allclose(model.score(sample), model.score(reindexed), err_msg=model.name)


class TestIrrelevantFeatures:
    def test_the_flight_number_barely_moves_the_score(self, model, sample):
        low = sample.copy()
        low["FlightNumberReportingAirline"] = 100
        high = sample.copy()
        high["FlightNumberReportingAirline"] = 9999

        shift = np.abs(model.score(high) - model.score(low)).mean()

        assert shift < MAX_FLIGHT_NUMBER_EFFECT, (
            f"{model.name}: flight number moved the score by {shift:.3f}"
        )

    def test_no_service_column_reached_the_feature_space(self, model):
        leaked = [c for c in model.columns if c.split("_")[0] in set(SERVICE_COLUMNS)]

        assert leaked == [], f"{model.name}: {leaked}"

    def test_the_target_never_became_a_feature(self, model):
        assert not any("IsDelayed" in c for c in model.columns), model.name
