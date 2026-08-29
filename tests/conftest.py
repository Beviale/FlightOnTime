"""Fixtures shared by the whole test suite.

The synthetic dataset built here mimics the output of the preprocessing stage
(`data/preprocess.py`), i.e. what 'data/split_data.py' reads and what every
downstream module expects: flight identity, calendar/schedule features, weather
at both ends of the route, the service columns the pipeline needs but the model
never sees, and the target.

The target is drawn from a logit that rises with origin precipitation,
origin congestion and destination wind, so the several unit tests that fit a real estimator
have something learnable to work with and can assert that fitting produced a model rather than noise.
"""

import numpy as np
import pandas as pd
import pytest

from predicting_flight_arrival_delays.config import TARGET, WEATHER_COLUMNS, SEED, DATE_COLUMN

ORIGINS = ["ATL", "DFW", "ORD", "LAX"]
DESTS = ["JFK", "SEA", "MIA"]
AIRLINES = ["AA", "DL", "UA"]

# BTS gives every airport a numeric id alongside its code, and keys the historical
# delay rates on it: a code can be reassigned to another airport, an id cannot.
AIRPORT_IDS = {
    "ATL": 10397, "DFW": 11298, "ORD": 13930, "LAX": 12892,
    "JFK": 12478, "SEA": 14747, "MIA": 13303,
}

# The coefficients that make the target learnable for the unit tests that fit.
INTERCEPT = -1.2
PRECIPITATION_WEIGHT = 1.4
CONGESTION_WEIGHT = 0.05
WIND_WEIGHT = 0.02


def _weather_block(rng: np.random.Generator, n: int, suffix: str) -> dict:
    """Build one side (origin or destination) of the weather feature block."""
    return {
        f"Temperature2m{suffix}": rng.normal(12.0, 8.0, n),
        f"Precipitation{suffix}": rng.gamma(1.0, 0.5, n),
        f"Snowfall{suffix}": rng.gamma(0.5, 0.2, n),
        f"WindSpeed10m{suffix}": rng.normal(15.0, 5.0, n),
        f"WindGusts10m{suffix}": rng.normal(25.0, 7.0, n),
        f"WeatherCode{suffix}": rng.choice(["0", "3", "61"], n),
    }


@pytest.fixture(scope="session")
def make_flights():
    """Factory for synthetic preprocessed flights.

    Session-scoped so the behavioural tests, which need a much larger sample to
    clear the Transformer's 1000-row category floor, can build one themselves.
    """

    def build(n_rows: int = 300, seed: int = SEED) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = n_rows

        origin = rng.choice(ORIGINS, n)
        dest = rng.choice(DESTS, n)
        airline = rng.choice(AIRLINES, n)
        dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(np.arange(n) % 90, unit="D")

        data = {
            DATE_COLUMN: dates,
            "Origin": origin,
            "Dest": dest,
            "OriginAirportID": [AIRPORT_IDS[code] for code in origin],
            "DestAirportID": [AIRPORT_IDS[code] for code in dest],
            "OriginCarrier": [f"{AIRPORT_IDS[o]}{a}" for o, a in zip(origin, airline)],
            "DestCarrier": [f"{AIRPORT_IDS[d]}{a}" for d, a in zip(dest, airline)],
            "ReportingAirline": airline,
            "FlightNumberReportingAirline": rng.integers(100, 999, n),
            "Distance": rng.normal(1200, 400, n),
            "DistanceGroup": rng.integers(1, 11, n),
            "DayOfWeek": (np.arange(n) % 7) + 1,
            "Month": dates.month,
            "DepTimeDecimal": rng.uniform(5, 23, n),
            "ArrTimeDecimal": rng.uniform(5, 23, n),
            "OriginCongestion": rng.integers(1, 30, n),
            "DestCongestion": rng.integers(1, 30, n),
            "IsHoliday": rng.integers(0, 2, n),
            "DaysToNearestHoliday": rng.integers(-15, 16, n),
            "AircraftDailyLegs": rng.integers(1, 6, n),
            "LegPosition": rng.integers(1, 6, n),
            "ScheduledTurnaround": rng.normal(90, 30, n),
            "LeadDays": rng.integers(0, 6, n),
            # --- service columns: needed by the pipeline, never fed to the model
            "CRSDepTime": rng.integers(500, 2359, n),
            "CRSArrTime": rng.integers(500, 2359, n),
            "DepHour": rng.integers(5, 24, n),
            "ArrHour": rng.integers(5, 24, n),
            "TailNumber": rng.choice([f"N{i}AA" for i in range(20)], n),
        }
        data.update(_weather_block(rng, n, "Origin"))
        data.update(_weather_block(rng, n, "Dest"))

        df = pd.DataFrame(data)

        logit = (
            INTERCEPT
            + PRECIPITATION_WEIGHT * df["PrecipitationOrigin"]
            + CONGESTION_WEIGHT * df["OriginCongestion"]
            + WIND_WEIGHT * df["WindSpeed10mDest"]
        )
        probability = 1 / (1 + np.exp(-logit))
        df[TARGET] = (rng.uniform(size=n) < probability).astype(int)

        df.loc[df.index[:5], "ScheduledTurnaround"] = np.nan
        return df

    return build


@pytest.fixture
def flights_df(make_flights) -> pd.DataFrame:
    """A synthetic preprocessed flights dataset, 300 rows over 90 days."""
    return make_flights()


@pytest.fixture
def weather_suffixes() -> list[str]:
    """Every weather feature name produced by the join, both route ends."""
    return [c + side for side in ("Origin", "Dest") for c in WEATHER_COLUMNS]


class StubEstimator:
    """A predict_proba-only estimator, so metric tests stay deterministic."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.probabilities, self.probabilities])


@pytest.fixture
def stub_estimator():
    """Factory returning a StubEstimator for a given probability vector."""
    return StubEstimator
