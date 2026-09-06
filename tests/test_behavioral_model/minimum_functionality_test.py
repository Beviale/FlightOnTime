"""Minimum functionality tests: the cases the model must not get wrong."""

import numpy as np
from scenarios import calm, storm
from thresholds import MAX_CALIBRATION_GAP, MIN_PREDICTION_SPREAD, MIN_SEPARATION_RATIO


def _pin(sample, **values):
    perturbed = sample.copy()
    for column, value in values.items():
        perturbed[column] = value
    return perturbed


class TestObviousCases:
    def test_the_worst_conditions_beat_the_mildest(self, model, sample):
        mildest = model.score(calm(sample)).mean()
        worst = model.score(storm(sample)).mean()

        assert worst > mildest, f"{model.name}: {worst:.3f} against {mildest:.3f}"

    def test_the_two_cases_are_separated(self, model, sample):
        mildest = model.score(calm(sample)).mean()
        worst = model.score(storm(sample)).mean()

        assert worst / mildest > MIN_SEPARATION_RATIO, (
            f"{model.name}: worst {worst:.3f} is only {worst / mildest:.2f}x mildest {mildest:.3f}"
        )

    def test_the_worst_conditions_raise_most_individual_flights(self, model, sample):
        """Not just the average: the shift must be broad, not driven by a few rows."""
        raised = (model.score(storm(sample)) > model.score(calm(sample))).mean()

        assert raised > 0.5, f"{model.name}: only {raised:.1%} of flights moved up"


class TestOutputContract:
    def test_one_probability_per_flight(self, model, sample):
        assert model.score(sample).shape == (len(sample),), model.name

    def test_every_probability_is_a_probability(self, model, sample):
        scores = model.score(sample)

        assert np.isfinite(scores).all(), model.name
        assert ((scores >= 0.0) & (scores <= 1.0)).all(), model.name

    def test_a_single_flight_can_be_scored(self, model, sample):
        """This is the shape production actually serves."""
        scores = model.score(sample.head(1))

        assert scores.shape == (1,), model.name
        assert 0.0 <= scores[0] <= 1.0, model.name

    def test_the_model_does_not_answer_the_same_thing_every_time(self, model, sample):
        """A constant predictor would satisfy every bound above."""
        spread = model.score(sample).std()

        assert spread > MIN_PREDICTION_SPREAD, (
            f"{model.name}: predictions are effectively constant ({spread:.4f})"
        )

    def test_the_average_prediction_is_near_the_observed_rate(self, model, sample):
        """Calibration is fitted before registration, so on held-out flights the
        mean predicted probability should land near the rate actually seen."""
        predicted = model.score(sample).mean()

        assert abs(predicted - model.base_rate) < MAX_CALIBRATION_GAP, (
            f"{model.name}: predicts {predicted:.3f} against {model.base_rate:.3f}"
        )


class TestUnseenInputs:
    def test_an_airport_never_seen_in_training_is_still_scored(self, model, sample):
        """Unseen categories fold into OTHER rather than breaking inference."""
        unknown = _pin(sample.head(50), Origin="ZZZ", Dest="QQQ")

        scores = model.score(unknown)

        assert np.isfinite(scores).all(), model.name
        assert ((scores >= 0.0) & (scores <= 1.0)).all(), model.name

    def test_an_unseen_carrier_is_still_scored(self, model, sample):
        unknown = _pin(sample.head(50), ReportingAirline="ZZ")

        assert np.isfinite(model.score(unknown)).all(), model.name

    def test_a_missing_turnaround_is_imputed_not_fatal(self, model, sample):
        """An aircraft's first appearance genuinely has no previous leg."""
        gap = sample.head(50).copy()
        gap["ScheduledTurnaround"] = np.nan

        assert np.isfinite(model.score(gap)).all(), model.name

    def test_an_unseen_weather_code_is_still_scored(self, weather_model, sample):
        unknown = _pin(sample.head(50), WeatherCodeOrigin="99", WeatherCodeDest="99")

        assert np.isfinite(weather_model.score(unknown)).all(), weather_model.name
