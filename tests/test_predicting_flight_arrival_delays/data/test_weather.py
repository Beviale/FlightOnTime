"""Tests for predicting_flight_arrival_delays.data.weather."""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from predicting_flight_arrival_delays.config import (
    HISTORICAL_FORECAST_URL,
    MAX_LEAD_DAYS,
    PREVIOUS_RUNS_URL,
    WEATHER_MODEL,
    DATE_COLUMN,
)
from predicting_flight_arrival_delays.data import weather as weather_module
from predicting_flight_arrival_delays.data.weather import (
    build_weather_requests,
    download_weather,
    fetch_series,
    load_weather,
)

runner = CliRunner()

HOURS = ["2025-03-01T00:00", "2025-03-01T01:00"]


@pytest.fixture
def captured_calls(monkeypatch):
    """Replace the HTTP layer, recording what each call asked for."""
    calls = []

    def fake_fetch(url, params, timeout=180):
        calls.append({"url": url, "params": params})
        prefix = ""
        if url == PREVIOUS_RUNS_URL:
            lead = params["hourly"].split("_previous_day")[1][0]
            prefix = f"_previous_day{lead}"
        return {
            "hourly": {
                "time": HOURS,
                f"temperature_2m{prefix}": [3.0, 4.0],
                f"precipitation{prefix}": [0.0, 0.2],
                f"snowfall{prefix}": [0.0, 0.0],
                f"wind_speed_10m{prefix}": [11.0, 12.0],
                f"wind_gusts_10m{prefix}": [21.0, 22.0],
                f"weather_code{prefix}": [0, 61],
            }
        }

    monkeypatch.setattr(weather_module, "fetch", fake_fetch)
    return calls


class TestFetchSeries:
    def test_lead_zero_uses_the_historical_endpoint(self, captured_calls):
        """Lead 0 is the freshest forecast issued for that date."""
        fetch_series(33.6, -84.4, "2025-03-01", "2025-03-01", 0)

        assert captured_calls[0]["url"] == HISTORICAL_FORECAST_URL

    def test_lead_above_zero_uses_the_previous_runs_endpoint(self, captured_calls):
        fetch_series(33.6, -84.4, "2025-03-01", "2025-03-01", 3)

        assert captured_calls[0]["url"] == PREVIOUS_RUNS_URL

    def test_previous_run_variables_are_requested_by_lead(self, captured_calls):
        fetch_series(33.6, -84.4, "2025-03-01", "2025-03-01", 3)

        assert "temperature_2m_previous_day3" in captured_calls[0]["params"]["hourly"]

    def test_request_carries_the_coordinates_and_range(self, captured_calls):
        fetch_series(33.6, -84.4, "2025-03-01", "2025-03-05", 0)
        params = captured_calls[0]["params"]

        assert params["latitude"] == 33.6
        assert params["longitude"] == -84.4
        assert params["start_date"] == "2025-03-01"
        assert params["end_date"] == "2025-03-05"
        assert params["models"] == WEATHER_MODEL
        assert params["timezone"] == "UTC"

    def test_columns_come_back_pascal_cased(self, captured_calls):
        df = fetch_series(33.6, -84.4, "2025-03-01", "2025-03-01", 0)

        assert {"Time", "Temperature2m", "WindSpeed10m", "WeatherCode"} <= set(df.columns)

    def test_previous_run_suffix_is_stripped(self, captured_calls):
        """Both endpoints must produce the same column names downstream."""
        lead0 = fetch_series(33.6, -84.4, "2025-03-01", "2025-03-01", 0)
        lead3 = fetch_series(33.6, -84.4, "2025-03-01", "2025-03-01", 3)

        assert list(lead0.columns) == list(lead3.columns)

    def test_time_is_parsed_as_utc(self, captured_calls):
        df = fetch_series(33.6, -84.4, "2025-03-01", "2025-03-01", 0)

        assert str(df["Time"].dt.tz) == "UTC"

    def test_lead_days_is_recorded_on_every_row(self, captured_calls):
        df = fetch_series(33.6, -84.4, "2025-03-01", "2025-03-01", 4)

        assert (df["LeadDays"] == 4).all()

    @pytest.mark.parametrize("lead", [-1, MAX_LEAD_DAYS + 1])
    def test_lead_outside_the_supported_range_is_rejected(self, captured_calls, lead):
        with pytest.raises(ValueError, match="LeadDays must be"):
            fetch_series(33.6, -84.4, "2025-03-01", "2025-03-01", lead)


class TestBuildWeatherRequests:
    def test_both_route_ends_contribute(self):
        df = pd.DataFrame(
            {
                "OriginAirportID": [10397, 10397],
                "DestAirportID": [12892, 13930],
                "LeadDays": [0, 2],
            }
        )
        requests_map = build_weather_requests(df)

        assert requests_map[10397] == {0, 2}
        assert requests_map[12892] == {0}
        assert requests_map[13930] == {2}

    def test_duplicate_combinations_are_collapsed(self):
        df = pd.DataFrame(
            {
                "OriginAirportID": [10397] * 5,
                "DestAirportID": [12892] * 5,
                "LeadDays": [1, 1, 1, 1, 1],
            }
        )
        requests_map = build_weather_requests(df)

        assert requests_map == {10397: {1}, 12892: {1}}

    def test_an_airport_on_both_ends_gets_one_entry(self):
        df = pd.DataFrame(
            {"OriginAirportID": [10397], "DestAirportID": [10397], "LeadDays": [3]}
        )

        assert build_weather_requests(df) == {10397: {3}}

    def test_keys_and_values_are_plain_ints(self):
        """They become filenames, so numpy scalars would render awkwardly."""
        df = pd.DataFrame(
            {"OriginAirportID": [10397], "DestAirportID": [12892], "LeadDays": [1]}
        )
        requests_map = build_weather_requests(df)

        assert all(type(k) is int for k in requests_map)
        assert all(type(v) is int for leads in requests_map.values() for v in leads)


class TestDownloadWeather:
    @pytest.fixture
    def airports(self):
        return pd.DataFrame(
            {
                "AirportId": [10397, 12892],
                "Latitude": [33.6, 33.9],
                "Longitude": [-84.4, -118.4],
            }
        )

    def test_one_file_per_airport_and_lead(self, tmp_path, airports, captured_calls):
        download_weather({10397: {0, 1}, 12892: {0}}, airports, "2025-03-01", "2025-03-02",
                         tmp_path / "weather", sleep=0)
        written = sorted(p.name for p in (tmp_path / "weather").glob("*.parquet"))

        assert written == ["10397_lead0.parquet", "10397_lead1.parquet", "12892_lead0.parquet"]

    def test_airport_id_is_the_first_column(self, tmp_path, airports, captured_calls):
        download_weather({10397: {0}}, airports, "2025-03-01", "2025-03-02",
                         tmp_path / "weather", sleep=0)
        df = pd.read_parquet(tmp_path / "weather" / "10397_lead0.parquet")

        assert df.columns[0] == "AirportId"
        assert (df["AirportId"] == 10397).all()

    def test_existing_files_are_not_refetched(self, tmp_path, airports, captured_calls):
        """An interrupted download can be restarted with the same arguments."""
        download_weather({10397: {0}}, airports, "2025-03-01", "2025-03-02",
                         tmp_path / "weather", sleep=0)
        calls_after_first = len(captured_calls)

        download_weather({10397: {0}}, airports, "2025-03-01", "2025-03-02",
                         tmp_path / "weather", sleep=0)

        assert len(captured_calls) == calls_after_first

    def test_airport_missing_from_the_reference_table_is_skipped(
        self, tmp_path, airports, captured_calls
    ):
        download_weather({99999: {0}}, airports, "2025-03-01", "2025-03-02",
                         tmp_path / "weather", sleep=0)

        assert list((tmp_path / "weather").glob("*.parquet")) == []

    def test_a_failing_request_does_not_stop_the_rest(
        self, tmp_path, airports, monkeypatch, captured_calls
    ):
        def flaky_fetch(url, params, timeout=180):
            if params["latitude"] == 33.6:
                raise ValueError("upstream is down")
            return {
                "hourly": {
                    "time": HOURS,
                    "temperature_2m": [3.0, 4.0],
                    "precipitation": [0.0, 0.2],
                    "snowfall": [0.0, 0.0],
                    "wind_speed_10m": [11.0, 12.0],
                    "wind_gusts_10m": [21.0, 22.0],
                    "weather_code": [0, 61],
                }
            }

        monkeypatch.setattr(weather_module, "fetch", flaky_fetch)
        download_weather({10397: {0}, 12892: {0}}, airports, "2025-03-01", "2025-03-02",
                         tmp_path / "weather", sleep=0)

        assert [p.name for p in (tmp_path / "weather").glob("*.parquet")] == [
            "12892_lead0.parquet"
        ]


class TestRunCommand:
    @pytest.fixture
    def inputs(self, tmp_path):
        flights = pd.DataFrame(
            {
                "OriginAirportID": [10397, 10397],
                "DestAirportID": [12892, 12892],
                "LeadDays": [0, 2],
                DATE_COLUMN: pd.to_datetime(["2025-03-01", "2025-03-04"]),
            }
        )
        flights_path = tmp_path / "flights.parquet"
        flights.to_parquet(flights_path, index=False)

        airports_path = tmp_path / "airports.csv"
        pd.DataFrame(
            {
                "AirportId": [10397, 12892],
                "Latitude": [33.6, 33.9],
                "Longitude": [-84.4, -118.4],
            }
        ).to_csv(airports_path, index=False)

        return flights_path, airports_path

    def test_one_file_per_needed_combination(self, tmp_path, inputs, captured_calls):
        flights_path, airports_path = inputs
        output_dir = tmp_path / "weather"

        result = runner.invoke(
            weather_module.app,
            [
                "--flights-path", str(flights_path),
                "--airports-path", str(airports_path),
                "--output-dir", str(output_dir),
                "--sleep", "0",
            ],
        )

        assert result.exit_code == 0, result.output
        assert sorted(p.name for p in output_dir.glob("*.parquet")) == [
            "10397_lead0.parquet",
            "10397_lead2.parquet",
            "12892_lead0.parquet",
            "12892_lead2.parquet",
        ]

    def test_the_date_range_is_taken_from_the_flights(self, tmp_path, inputs, captured_calls):
        """One request covers the whole period, so the range must span the data."""
        flights_path, airports_path = inputs

        runner.invoke(
            weather_module.app,
            [
                "--flights-path", str(flights_path),
                "--airports-path", str(airports_path),
                "--output-dir", str(tmp_path / "weather"),
                "--sleep", "0",
            ],
        )

        assert captured_calls[0]["params"]["start_date"] == "2025-03-01"
        assert captured_calls[0]["params"]["end_date"] == "2025-03-04"

    def test_a_missing_flights_file_exits_with_an_error_code(self, tmp_path, inputs):
        _, airports_path = inputs

        result = runner.invoke(
            weather_module.app,
            [
                "--flights-path", str(tmp_path / "absent.parquet"),
                "--airports-path", str(airports_path),
                "--output-dir", str(tmp_path / "weather"),
            ],
        )

        assert result.exit_code == 1


class TestLoadWeather:
    def test_every_file_is_concatenated(self, tmp_path):
        directory = tmp_path / "weather"
        directory.mkdir()
        for airport_id in (10397, 12892):
            pd.DataFrame(
                {
                    "AirportId": [airport_id, airport_id],
                    "Time": pd.to_datetime(HOURS, utc=True),
                    "LeadDays": [0, 0],
                    "Temperature2m": [1.0, 2.0],
                }
            ).to_parquet(directory / f"{airport_id}_lead0.parquet", index=False)

        out = load_weather(directory)

        assert len(out) == 4
        assert set(out["AirportId"]) == {10397, 12892}

    def test_measurements_are_downcast_to_float32(self, tmp_path):
        """The whole series is held in memory for both sides of the join; double
        precision on temperatures and millimetres costs a gigabyte for nothing."""
        directory = tmp_path / "weather"
        directory.mkdir()
        pd.DataFrame(
            {
                "AirportId": [10397, 10397],
                "Time": pd.to_datetime(HOURS, utc=True),
                "LeadDays": [0, 0],
                "Temperature2m": [1.0, 2.0],
                "WindSpeed10m": [10.0, 12.0],
                "WeatherCode": [0, 61],
            }
        ).to_parquet(directory / "10397_lead0.parquet", index=False)

        out = load_weather(directory)

        assert out["Temperature2m"].dtype == np.float32
        assert out["WindSpeed10m"].dtype == np.float32
        assert out["WeatherCode"].dtype == np.float32

    def test_files_disagreeing_on_the_code_type_concatenate_cleanly(self, tmp_path):
        """The API returns a code for most hours and nothing for some, so the
        column is an integer in some downloaded files and a float in others.
        Left alone, concatenating them promotes the whole series to float64."""
        directory = tmp_path / "weather"
        directory.mkdir()

        def write(name, codes):
            pd.DataFrame(
                {
                    "AirportId": [10397, 10397],
                    "Time": pd.to_datetime(HOURS, utc=True),
                    "LeadDays": [0, 0],
                    "Temperature2m": [1.0, 2.0],
                    "WeatherCode": codes,
                }
            ).to_parquet(directory / name, index=False)

        write("a_lead0.parquet", pd.Series([0, 61], dtype="int64"))
        write("b_lead0.parquet", pd.Series([3.0, np.nan], dtype="float64"))

        out = load_weather(directory)

        assert out["WeatherCode"].dtype == np.float32
        assert out["WeatherCode"].isna().sum() == 1

    def test_a_missing_code_survives_the_conversion_to_a_label(self, tmp_path):
        """join_weather_to_flights turns the code into a string label; a null
        one has to reach that step as a null, not as an unconvertible float."""
        directory = tmp_path / "weather"
        directory.mkdir()
        pd.DataFrame(
            {
                "AirportId": [10397, 10397],
                "Time": pd.to_datetime(HOURS, utc=True),
                "LeadDays": [0, 0],
                "Temperature2m": [1.0, 2.0],
                "WeatherCode": pd.Series([61.0, np.nan], dtype="float64"),
            }
        ).to_parquet(directory / "10397_lead0.parquet", index=False)

        codes = load_weather(directory)["WeatherCode"].astype("Int64")
        labels = codes.astype(str).mask(codes.isna())

        assert labels.tolist()[0] == "61"
        assert pd.isna(labels.tolist()[1])

    def test_the_join_keys_keep_their_type(self, tmp_path):
        """Downcasting a merge key would force pandas to upcast it back, copying
        the column, and risks a silent mismatch with the flights side."""
        directory = tmp_path / "weather"
        directory.mkdir()
        pd.DataFrame(
            {
                "AirportId": [10397, 10397],
                "Time": pd.to_datetime(HOURS, utc=True),
                "LeadDays": [0, 0],
                "Temperature2m": [1.0, 2.0],
            }
        ).to_parquet(directory / "10397_lead0.parquet", index=False)

        out = load_weather(directory)

        assert out["AirportId"].dtype == np.int64
        assert out["LeadDays"].dtype == np.int64

    def test_index_is_reset(self, tmp_path):
        directory = tmp_path / "weather"
        directory.mkdir()
        for name in ("a", "b"):
            pd.DataFrame({"AirportId": [1, 2]}).to_parquet(
                directory / f"{name}.parquet", index=False
            )

        assert list(load_weather(directory).index) == [0, 1, 2, 3]

    def test_empty_directory_means_the_download_never_ran(self, tmp_path):
        directory = tmp_path / "weather"
        directory.mkdir()

        with pytest.raises(FileNotFoundError, match="run download_weather first"):
            load_weather(directory)
