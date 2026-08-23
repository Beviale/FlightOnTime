"""Directional tests: changes that must move a prediction the right way.
"""

import numpy as np
from scenarios import CONGESTION_LEVERS, WEATHER_LEVERS, calm, pin, present, storm
from thresholds import (
    MIN_CONGESTION_EFFECT,
    MIN_SEPARATION_RATIO,
    MIN_WEATHER_EFFECT,
)


def _group_shift(model, sample, levers):
    """Mean probability with the group at its minimum, then at its maximum."""
    columns = present(sample, levers)
    return (
        model.score(pin(sample, columns, "min")).mean(),
        model.score(pin(sample, columns, "max")).mean(),
    )


class TestCongestionRaisesRisk:
    """Congestion is in every production variant; weather is not."""

    def test_a_busier_airport_does_not_lower_the_odds(self, model, sample):
        quiet, busy = _group_shift(model, sample, CONGESTION_LEVERS)

        assert busy > quiet, f"{model.name}: congestion lowered the delay probability"

    def test_congestion_moves_the_odds_materially(self, model, sample):
        """A flat response would mean the features were dropped by selection."""
        quiet, busy = _group_shift(model, sample, CONGESTION_LEVERS)

        assert busy - quiet > MIN_CONGESTION_EFFECT, (
            f"{model.name}: congestion moved the mean by only {busy - quiet:.4f}"
        )


class TestWeatherRaisesRisk:
    def test_worse_weather_does_not_lower_the_odds(self, weather_model, sample):
        mild, rough = _group_shift(weather_model, sample, WEATHER_LEVERS)

        assert rough > mild, f"{weather_model.name}: bad weather lowered the probability"

    def test_weather_moves_the_odds_materially(self, weather_model, sample):
        mild, rough = _group_shift(weather_model, sample, WEATHER_LEVERS)

        assert rough - mild > MIN_WEATHER_EFFECT, (
            f"{weather_model.name}: weather moved the mean by only {rough - mild:.4f}"
        )

    def test_weather_is_not_simply_ignored(self, weather_model, sample):
        """The whole point of the 'all' variant is that weather adds something
        the 'noweather' feature set cannot provide."""
        mild, rough = _group_shift(weather_model, sample, WEATHER_LEVERS)

        assert rough / mild > 1.0 + MIN_WEATHER_EFFECT, weather_model.name


class TestCombinedConditions:
    def test_the_worst_conditions_beat_the_mildest(self, model, sample):
        assert model.score(storm(sample)).mean() > model.score(calm(sample)).mean(), model.name

    def test_the_gap_is_wider_than_either_group_alone(self, model, sample):
        """Congestion and weather must not cancel each other out."""
        mildest = model.score(calm(sample)).mean()
        worst = model.score(storm(sample)).mean()

        quiet, busy = _group_shift(model, sample, CONGESTION_LEVERS)

        assert worst - mildest >= busy - quiet, model.name


class TestProbabilitiesStayWellFormed:
    def test_absurd_inputs_still_yield_a_probability(self, model, sample):
        """Nothing in the pipeline clips inputs, so the model must not produce
        NaN or step outside [0, 1] when handed a nonsense value."""
        perturbed = sample.copy()
        perturbed["OriginCongestion"] = 10_000
        perturbed["Distance"] = 99_999

        scores = model.score(perturbed)

        assert np.isfinite(scores).all(), model.name
        assert ((scores >= 0.0) & (scores <= 1.0)).all(), model.name

    def test_absurd_weather_still_yields_a_probability(self, weather_model, sample):
        perturbed = sample.copy()
        perturbed["PrecipitationOrigin"] = 500.0
        perturbed["Temperature2mOrigin"] = -300.0

        scores = weather_model.score(perturbed)

        assert np.isfinite(scores).all(), weather_model.name
        assert ((scores >= 0.0) & (scores <= 1.0)).all(), weather_model.name

    def test_the_separation_survives_absurd_values(self, model, sample):
        """Out-of-range inputs saturate a tree model rather than inverting it."""
        mildest = model.score(calm(sample)).mean()
        worst = model.score(storm(sample)).mean()

        assert worst / mildest > MIN_SEPARATION_RATIO, model.name
