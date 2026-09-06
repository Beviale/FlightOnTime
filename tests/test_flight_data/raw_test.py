"""Great Expectations suite for the raw BTS On-Time Performance extracts."""

import great_expectations as gx
import pandas as pd
import pytest
from util import failures, show_results, validate

from predicting_flight_arrival_delays.config import (
    DATE_COLUMN,
    KEEP_COLUMNS,
    MAX_BLOCK_MINUTES,
    RAW_DATA_DIR,
)

SAMPLE_ROWS = 200_000

REQUIRED_COLUMNS = KEEP_COLUMNS + ["Cancelled", "Diverted", "ArrDel15"]


def find_raw_csvs() -> list:
    """Every monthly extract on disk, oldest first, empty if none were pulled."""
    if not RAW_DATA_DIR.exists():
        return []
    return sorted(RAW_DATA_DIR.rglob("*.csv"))


def load_raw(path, n_rows: int | None = SAMPLE_ROWS) -> pd.DataFrame:
    """Read a BTS extract, keeping only the columns the pipeline uses."""
    return pd.read_csv(
        path, usecols=lambda c: c in REQUIRED_COLUMNS, nrows=n_rows, low_memory=False
    )


def build_expectations() -> list:
    """The expectation suite for a raw monthly extract."""
    expectations = [gx.expectations.ExpectTableRowCountToBeBetween(min_value=1)]

    expectations += [
        gx.expectations.ExpectColumnToExist(column=column) for column in REQUIRED_COLUMNS
    ]

    # --- The flags load_and_clean filters on, and the target it derives.
    expectations += [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Cancelled"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Diverted"),
        gx.expectations.ExpectColumnValuesToBeInSet(column="Cancelled", value_set=[0, 1]),
        gx.expectations.ExpectColumnValuesToBeInSet(column="Diverted", value_set=[0, 1]),
        gx.expectations.ExpectColumnValuesToBeInSet(column="ArrDel15", value_set=[0, 1]),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="ArrDel15",
            row_condition="Cancelled == 0 and Diverted == 0",
            condition_parser="pandas",
        ),
    ]

    # --- Identity
    expectations += [
        gx.expectations.ExpectColumnValuesToNotBeNull(column=DATE_COLUMN),
        gx.expectations.ExpectColumnValuesToMatchRegex(column="Origin", regex=r"^[A-Z]{3}$"),
        gx.expectations.ExpectColumnValuesToMatchRegex(column="Dest", regex=r"^[A-Z]{3}$"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Tail_Number", mostly=0.95),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="OriginAirportID", min_value=1, max_value=99999
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="DestAirportID", min_value=1, max_value=99999
        ),
    ]

    # --- Schedule. add_temporal_features splits these as HHMM, so anything
    # outside 1..2400 would silently produce an impossible hour.
    expectations += [
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="CRSDepTime", min_value=1, max_value=2400
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="CRSArrTime", min_value=1, max_value=2400
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="CRSElapsedTime", min_value=1, max_value=MAX_BLOCK_MINUTES, mostly=0.9999
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Distance", min_value=1, max_value=6000
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(column="Month", min_value=1, max_value=12),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="DayOfWeek", min_value=1, max_value=7
        ),
    ]

    return expectations


@pytest.mark.parametrize("path", find_raw_csvs(), ids=lambda path: path.parent.name)
def test_raw_flights_meet_expectations(path):
    month = path.parent.name
    result = validate(load_raw(path), build_expectations(), f"flights_raw::{month}")

    assert result.success, "\n".join(failures(result))


if __name__ == "__main__":
    paths = find_raw_csvs()
    if not paths:
        raise SystemExit(f"No raw CSV found under {RAW_DATA_DIR} -- run `dvc pull` first.")

    for path in paths:
        month = path.parent.name
        print(f"\n=== {month} ===")
        full = load_raw(path, n_rows=None)
        show_results(validate(full, build_expectations(), f"flights_raw::{month}"))
