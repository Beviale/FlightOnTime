"""Great Expectations suite for data/interim/flights_preprocessed.parquet"""

from datetime import datetime

import great_expectations as gx
import pytest
from util import SAMPLE_ROWS, failures, load_parquet_sample, show_results, validate

from predicting_flight_arrival_delays.config import (
    DATE_COLUMN,
    INTERIM_DATA_DIR,
    MAX_LEAD_DAYS,
    WEATHER_COLUMNS,
)

DATA_PATH = INTERIM_DATA_DIR / "flights_preprocessed.parquet"

WEATHER_ORIGIN = [c + "Origin" for c in WEATHER_COLUMNS]
WEATHER_DEST = [c + "Dest" for c in WEATHER_COLUMNS]


EXPECTED_COLUMNS = [
    "Month",
    "DayOfWeek",
    DATE_COLUMN,
    "ReportingAirline",
    "TailNumber",
    "FlightNumberReportingAirline",
    "OriginAirportID",
    "Origin",
    "OriginCityName",
    "OriginState",
    "DestAirportID",
    "Dest",
    "DestCityName",
    "DestState",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "DistanceGroup",
    "OriginCongestion",
    "DestCongestion",
    "IsDelayed",
    "DepHour",
    "DepTimeDecimal",
    "ArrHour",
    "ArrTimeDecimal",
    "IsHoliday",
    "DaysToNearestHoliday",
    "AircraftDailyLegs",
    "LegPosition",
    "LeadDays",
    "DepUtcHour",
    "ArrUtcHour",
    "ScheduledTurnaround",
    "OriginCarrier",
    "DestCarrier",
    *WEATHER_ORIGIN,
    *WEATHER_DEST,
]

NEVER_NULL = [
    DATE_COLUMN,
    "Origin",
    "Dest",
    "ReportingAirline",
    "TailNumber",
    "OriginAirportID",
    "DestAirportID",
    "CRSDepTime",
    "CRSArrTime",
    "Distance",
    "IsDelayed",
    "LeadDays",
    "DaysToNearestHoliday",
    "OriginCongestion",
    "DestCongestion",
    "AircraftDailyLegs",
    "LegPosition",
]


def build_expectations() -> list:
    """The full expectation suite for the preprocessed flights."""
    expectations = [
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1),
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=EXPECTED_COLUMNS, exact_match=True
        ),
    ]

    expectations += [
        gx.expectations.ExpectColumnValuesToNotBeNull(column=column) for column in NEVER_NULL
    ]

    # --- Target
    expectations += [
        gx.expectations.ExpectColumnValuesToBeInSet(column="IsDelayed", value_set=[0, 1]),
        gx.expectations.ExpectColumnMeanToBeBetween(
            column="IsDelayed", min_value=0.05, max_value=0.50
        ),
    ]

    # --- Flight identity
    expectations += [
        gx.expectations.ExpectColumnValuesToMatchRegex(column="Origin", regex=r"^[A-Z]{3}$"),
        gx.expectations.ExpectColumnValuesToMatchRegex(column="Dest", regex=r"^[A-Z]{3}$"),
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column="ReportingAirline", regex=r"^[A-Z0-9]{2}$"
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="OriginAirportID", min_value=1, max_value=99999
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="DestAirportID", min_value=1, max_value=99999
        ),
    ]

    # --- Airport-carrier pairs
    expectations += [
        gx.expectations.ExpectColumnValuesToMatchRegex(column=column, regex=r"^\d{5}[A-Z0-9]{2}$")
        for column in ("OriginCarrier", "DestCarrier")
    ]

    # --- Schedule
    expectations += [
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="CRSDepTime", min_value=1, max_value=2400
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="CRSArrTime", min_value=1, max_value=2400
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="CRSElapsedTime", min_value=1, max_value=1440
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Distance", min_value=1, max_value=6000
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="DistanceGroup", min_value=1, max_value=11
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(column="Month", min_value=1, max_value=12),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="DayOfWeek", min_value=1, max_value=7
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(column="DepHour", min_value=0, max_value=23),
        gx.expectations.ExpectColumnValuesToBeBetween(column="ArrHour", min_value=0, max_value=23),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="DepTimeDecimal", min_value=0, max_value=24
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="ArrTimeDecimal", min_value=0, max_value=24
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column=DATE_COLUMN,
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2027, 1, 1),
        ),
    ]

    # --- Derived features
    expectations += [
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="OriginCongestion", min_value=1, max_value=2000
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="DestCongestion", min_value=1, max_value=2000
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="AircraftDailyLegs", min_value=1, max_value=30
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="LegPosition", min_value=1, max_value=30
        ),
        # A leg cannot be the fifth of three.
        gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="AircraftDailyLegs", column_B="LegPosition", or_equal=True
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(column="IsHoliday", value_set=[0, 1]),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="DaysToNearestHoliday", min_value=-366, max_value=366
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="LeadDays", min_value=0, max_value=MAX_LEAD_DAYS
        ),
    ]

    # --- Weather. Nulls are expected here and GX ignores them: an unmatched
    # forecast is exactly what the 'noweather' variant exists to serve.
    for side in (WEATHER_ORIGIN, WEATHER_DEST):
        temperature, precipitation, snowfall, wind, gusts, code = side
        expectations += [
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=temperature, min_value=-60, max_value=60
            ),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=precipitation, min_value=0, max_value=500
            ),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=snowfall, min_value=0, max_value=500
            ),
            gx.expectations.ExpectColumnValuesToBeBetween(column=wind, min_value=0, max_value=200),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=gusts, min_value=0, max_value=300
            ),
            gx.expectations.ExpectColumnValuesToMatchRegex(column=code, regex=r"^\d+$"),
        ]

    return expectations


@pytest.mark.skipif(
    not DATA_PATH.exists(), reason="flights_preprocessed.parquet not pulled from DVC"
)
def test_preprocessed_flights_meet_expectations():
    """Validate a sample; the script below validates the whole file."""
    df = load_parquet_sample(DATA_PATH, n_rows=SAMPLE_ROWS)
    result = validate(df, build_expectations(), "flights_processed")

    assert result.success, "\n".join(failures(result))


if __name__ == "__main__":
    import pandas as pd

    full = pd.read_parquet(DATA_PATH)
    show_results(validate(full, build_expectations(), "flights_processed"))
