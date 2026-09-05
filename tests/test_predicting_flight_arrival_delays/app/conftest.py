"""Fixtures for the serving layer."""

from datetime import date, timedelta

import numpy as np
import pytest

from predicting_flight_arrival_delays.app.schema import FlightRequest
from predicting_flight_arrival_delays.app.utils import Bundle
from predicting_flight_arrival_delays.data.features import build_xy, select_features_variant
from predicting_flight_arrival_delays.data.transform import Transformer, encode_categoricals

JFK_LAX = {
    "Month": 3,
    "DayofMonth": 12,
    "DayOfWeek": 4,
    "FlightDate": "2026-03-12",
    "IsHoliday": 0,
    "DaysToNearestHoliday": -12,
    "ReportingAirline": "AA",
    "FlightNumberReportingAirline": 100,
    "OriginAirportID": 12478,
    "Origin": "JFK",
    "OriginCityName": "New York, NY",
    "OriginState": "NY",
    "DestAirportID": 12892,
    "Dest": "LAX",
    "DestCityName": "Los Angeles, CA",
    "DestState": "CA",
    "OriginCarrier": "12478AA",
    "DestCarrier": "12892AA",
    "DepTimeDecimal": 14.5,
    "ArrTimeDecimal": 17.92,
    "CRSElapsedTime": 385.0,
    "Distance": 2475.0,
    "DistanceGroup": 10,
    "OriginCongestion": 42,
    "DestCongestion": 37,
    "AircraftDailyLegs": 4,
    "LegPosition": 2,
    "ScheduledTurnaround": 55.0,
}


class StubModel:
    """A model that reports the variant it belongs to, through its probability."""

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, X):
        p = np.full(len(X), self.probability)
        return np.column_stack([1 - p, p])


@pytest.fixture(scope="session")
def make_bundle(make_flights):
    """Factory for a Bundle whose transformer is fitted the way training fits it."""
    flights = make_flights(400)

    def build(variant: str, probability: float, threshold: float) -> Bundle:
        df = select_features_variant(flights, variant)
        X, y = build_xy(df)

        transformer = Transformer(min_category_count=5, encoding="onehot")
        X_fit = transformer.fit_transform(X, y)
        transformer.select_features(X_fit, y)
        X_fit = transformer.apply_selection(X_fit)

        cat_cols = [c for c in transformer.categorical_columns if c in X_fit.columns]
        X_fit = encode_categoricals(X_fit, cat_cols, "onehot")

        return Bundle(
            variant=variant,
            model=StubModel(probability),
            transformer=transformer,
            columns=list(X_fit.columns),
            run_id=f"run-{variant}",
            threshold=threshold,

            params={
                "variant": variant,
                "algorithm": "lightgbm",
                "config": "default",
                "encoding": "onehot",
                "resample": "none",
                "calibrated": "True",
                "n_features": str(len(X_fit.columns)),
                "hp_learning_rate": "0.05",
                "hp_num_leaves": "31",
            },
            metrics={
                "pr_auc": 0.42,
                "roc_auc": 0.71,
                "brier": 0.15,
                "operating_threshold": threshold,
            },
        )

    return build


@pytest.fixture(scope="session")
def bundles(make_bundle) -> dict[str, Bundle]:
    return {
        "all": make_bundle("all", 0.80, 0.40),
        "noweather": make_bundle("noweather", 0.20, 0.60),
    }


@pytest.fixture
def today() -> date:
    return date(2026, 3, 10)


@pytest.fixture
def body(today):

    def build(days_ahead: int = 2, base: date | None = None, **overrides) -> dict:
        flight_date = (base or today) + timedelta(days=days_ahead)
        dated = {
            "FlightDate": flight_date.isoformat(),
            "Month": flight_date.month,
            "DayofMonth": flight_date.day,
            "DayOfWeek": flight_date.weekday() + 1,
        }
        return JFK_LAX | dated | overrides

    return build


@pytest.fixture
def flight_request(body) -> FlightRequest:
    return FlightRequest(**body())


@pytest.fixture
def forecast() -> dict[str, float | str]:
    return {
        "Temperature2m": 11.5,
        "Precipitation": 0.2,
        "Snowfall": 0.0,
        "WindSpeed10m": 14.0,
        "WindGusts10m": 22.0,
        "WeatherCode": "3",
    }


@pytest.fixture
def stub_forecast(monkeypatch, forecast):

    unset = object()

    def build(result=unset):
        calls = []

        def fake_weather_at(airport, when_utc, cache=None):
            calls.append((airport.iata, when_utc))
            return forecast if result is unset else result

        monkeypatch.setattr(
            "predicting_flight_arrival_delays.app.enrichment.builder.weather_at",
            fake_weather_at,
        )
        return calls

    return build
