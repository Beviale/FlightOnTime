"""Tests for predicting_flight_arrival_delays.data.preprocess."""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from predicting_flight_arrival_delays.config import DATE_COLUMN, MAX_LEAD_DAYS
from predicting_flight_arrival_delays.data import preprocess as preprocess_module
from predicting_flight_arrival_delays.data.preprocess import (
    FULL_LEAD_COVERAGE_START,
    add_aircraft_schedule_features,
    add_carrier_features,
    add_congestion_features,
    add_holiday_features,
    add_temporal_features,
    add_turnaround_features,
    add_utc_columns,
    assign_lead_days,
    join_weather_to_flights,
    load_and_clean,
)

runner = CliRunner()


class TestAddCongestionFeatures:
    @pytest.fixture
    def scheduled(self):
        """Three flights leaving ATL in the 08:00 hour, one in the 09:00 hour."""
        return pd.DataFrame(
            {
                DATE_COLUMN: pd.to_datetime(["2025-03-01"] * 4 + ["2025-03-02"]),
                "Origin": ["ATL", "ATL", "ATL", "ATL", "ATL"],
                "Dest": ["JFK", "JFK", "LAX", "JFK", "JFK"],
                "CRSDepTime": [800, 830, 845, 905, 800],
                "CRSArrTime": [1100, 1130, 1400, 1205, 1100],
            }
        )

    def test_origin_congestion_counts_the_departure_hour(self, scheduled):
        out = add_congestion_features(scheduled)

        assert list(out["OriginCongestion"]) == [3, 3, 3, 1, 1]

    def test_dest_congestion_counts_the_arrival_hour(self, scheduled):
        out = add_congestion_features(scheduled)

        assert list(out["DestCongestion"]) == [2, 2, 1, 1, 1]

    def test_congestion_does_not_cross_days(self, scheduled):
        """The last row is a different date, so it shares nobody's hour."""
        out = add_congestion_features(scheduled)

        assert out["OriginCongestion"].iloc[-1] == 1

    def test_midnight_is_normalised(self):
        """BTS writes midnight as 2400; 2400 and 0 are the same hour."""
        df = pd.DataFrame(
            {
                DATE_COLUMN: pd.to_datetime(["2025-03-01"] * 2),
                "Origin": ["ATL", "ATL"],
                "Dest": ["JFK", "JFK"],
                "CRSDepTime": [2400, 15],
                "CRSArrTime": [300, 315],
            }
        )
        out = add_congestion_features(df)

        assert list(out["OriginCongestion"]) == [2, 2]


class TestLoadAndClean:
    @pytest.fixture
    def raw(self):
        return pd.DataFrame(
            {
                "Cancelled": [0, 1, 0, 0, 0],
                "Diverted": [0, 0, 1, 0, 0],
                "ArrDel15": [0.0, 0.0, 0.0, 1.0, np.nan],
                "Origin": ["ATL"] * 5,
            }
        )

    def test_cancelled_diverted_and_targetless_rows_go(self, raw):
        """None of these three has a usable arrival outcome."""
        out = load_and_clean(raw)

        assert len(out) == 2

    def test_target_is_built_from_arrdel15(self, raw):
        out = load_and_clean(raw)

        assert list(out["IsDelayed"]) == [0, 1]
        assert out["IsDelayed"].dtype == int

    def test_source_columns_are_removed(self, raw):
        """Keeping ArrDel15 would leak the target straight into the features."""
        out = load_and_clean(raw)

        assert not {"Cancelled", "Diverted", "ArrDel15"} & set(out.columns)

    def test_input_is_not_mutated(self, raw):
        before = raw.copy()
        load_and_clean(raw)

        pd.testing.assert_frame_equal(raw, before)


class TestAddTemporalFeatures:
    @pytest.fixture
    def scheduled(self):
        return pd.DataFrame(
            {
                DATE_COLUMN: ["2025-03-01", "2025-03-01", "2025-03-01"],
                "CRSDepTime": [1455, 2400, 0],
                "CRSArrTime": [1730, 30, 100],
            }
        )

    def test_flight_date_becomes_datetime(self, scheduled):
        out = add_temporal_features(scheduled)

        assert pd.api.types.is_datetime64_any_dtype(out[DATE_COLUMN])

    def test_hour_is_the_integer_part(self, scheduled):
        out = add_temporal_features(scheduled)

        assert list(out["DepHour"]) == [14, 0, 0]

    def test_decimal_time_carries_the_minutes(self, scheduled):
        """14:55 becomes 14.917, so the model sees time as a continuous quantity."""
        out = add_temporal_features(scheduled)

        assert out["DepTimeDecimal"].iloc[0] == pytest.approx(14 + 55 / 60)

    def test_midnight_2400_is_normalised_to_zero(self, scheduled):
        out = add_temporal_features(scheduled)

        assert out["DepHour"].iloc[1] == 0
        assert out["DepTimeDecimal"].iloc[1] == pytest.approx(0.0)

    def test_arrival_columns_are_built_too(self, scheduled):
        out = add_temporal_features(scheduled)

        assert list(out["ArrHour"]) == [17, 0, 1]
        assert out["ArrTimeDecimal"].iloc[0] == pytest.approx(17.5)


class TestAddHolidayFeatures:
    @pytest.fixture
    def around_july_fourth(self):
        return pd.DataFrame(
            {DATE_COLUMN: pd.to_datetime(["2025-07-01", "2025-07-04", "2025-07-06", "2025-12-25"])}
        )

    def test_holidays_are_flagged(self, around_july_fourth):
        out = add_holiday_features(around_july_fourth)

        assert list(out["IsHoliday"]) == [0, 1, 0, 1]

    def test_distance_is_zero_on_the_holiday(self, around_july_fourth):
        out = add_holiday_features(around_july_fourth)

        assert out["DaysToNearestHoliday"].iloc[1] == 0

    def test_distance_is_signed(self, around_july_fourth):
        """The sign says which side of the holiday the flight falls on."""
        out = add_holiday_features(around_july_fourth)
        before, after = out["DaysToNearestHoliday"].iloc[0], out["DaysToNearestHoliday"].iloc[2]

        assert abs(before) == 3
        assert abs(after) == 2
        assert np.sign(before) != np.sign(after)

    def test_same_date_gets_the_same_value(self):
        """The distance is computed per unique date, then mapped back to every row."""
        df = pd.DataFrame({DATE_COLUMN: pd.to_datetime(["2025-07-01"] * 3 + ["2025-07-06"])})
        out = add_holiday_features(df)

        assert out["DaysToNearestHoliday"].iloc[:3].nunique() == 1
        assert out["DaysToNearestHoliday"].nunique() == 2

    def test_the_calendar_is_not_cut_to_the_data(self):
        """August carries no federal holiday. The nearest is Labor Day, which no
        window derived from these two flights would ever contain."""
        df = pd.DataFrame({DATE_COLUMN: pd.to_datetime(["2025-08-05", "2025-08-06"])})
        out = add_holiday_features(df)

        assert list(out["IsHoliday"]) == [0, 0]
        assert list(out["DaysToNearestHoliday"]) == [27, 26]  # Labor Day, 2025-09-01

    def test_a_holiday_just_past_the_last_flight_is_still_seen(self):
        """Independence Day is days ahead of these flights but outside their range;
        measuring against Juneteenth instead is wrong in both size and sign."""
        df = pd.DataFrame({DATE_COLUMN: pd.to_datetime(["2025-06-28", "2025-06-30"])})
        out = add_holiday_features(df)

        assert list(out["DaysToNearestHoliday"]) == [6, 4]  # 2025-07-04

    def test_a_holiday_just_before_the_first_flight_is_still_seen(self):
        df = pd.DataFrame({DATE_COLUMN: pd.to_datetime(["2025-07-07", "2025-07-08"])})
        out = add_holiday_features(df)

        assert list(out["DaysToNearestHoliday"]) == [-3, -4]  # 2025-07-04

    def test_the_two_holiday_columns_never_contradict_each_other(self):
        """IsHoliday reads the full calendar, so the distance must agree with it."""
        df = pd.DataFrame({DATE_COLUMN: pd.to_datetime(["2025-07-01", "2025-07-04"])})
        out = add_holiday_features(df)

        assert out["IsHoliday"].iloc[1] == 1
        assert out["DaysToNearestHoliday"].iloc[1] == 0

    def test_the_distance_is_numeric(self):
        """An object column would be treated as categorical by the Transformer
        and one-hot encoded instead of scaled."""
        df = pd.DataFrame({DATE_COLUMN: pd.to_datetime(["2025-08-05"])})
        out = add_holiday_features(df)

        assert pd.api.types.is_numeric_dtype(out["DaysToNearestHoliday"])


class TestAddAircraftScheduleFeatures:
    @pytest.fixture
    def legs(self):
        return pd.DataFrame(
            {
                "TailNumber": ["N1", "N1", "N1", "N2", "N1"],
                DATE_COLUMN: pd.to_datetime(
                    ["2025-03-01", "2025-03-01", "2025-03-01", "2025-03-01", "2025-03-02"]
                ),
                "CRSDepTime": [1200, 800, 1600, 900, 700],
            }
        )

    def test_daily_leg_count_is_per_aircraft_and_day(self, legs):
        out = add_aircraft_schedule_features(legs)
        n1_day1 = out[(out["TailNumber"] == "N1") & (out[DATE_COLUMN] == "2025-03-01")]

        assert set(n1_day1["AircraftDailyLegs"]) == {3}

    def test_leg_position_follows_the_schedule(self, legs):
        """Position is 1-based and ordered by scheduled departure time."""
        out = add_aircraft_schedule_features(legs)
        n1_day1 = out[(out["TailNumber"] == "N1") & (out[DATE_COLUMN] == "2025-03-01")]

        assert list(n1_day1["CRSDepTime"]) == [800, 1200, 1600]
        assert list(n1_day1["LegPosition"]) == [1, 2, 3]

    def test_a_new_day_restarts_the_count(self, legs):
        out = add_aircraft_schedule_features(legs)
        n1_day2 = out[(out["TailNumber"] == "N1") & (out[DATE_COLUMN] == "2025-03-02")]

        assert list(n1_day2["AircraftDailyLegs"]) == [1]
        assert list(n1_day2["LegPosition"]) == [1]


class TestAssignLeadDays:
    @pytest.fixture
    def dated(self):
        dates = pd.to_datetime(["2024-06-01", "2024-11-30"] + ["2025-03-01"] * 48)
        return pd.DataFrame({DATE_COLUMN: dates})

    def test_lead_days_stay_in_range(self, dated):
        out = assign_lead_days(dated)

        assert out["LeadDays"].between(0, MAX_LEAD_DAYS).all()

    def test_dates_before_coverage_are_forced_to_zero(self, dated):
        """The weather source has no older lead times for that period."""
        out = assign_lead_days(dated)
        early = out[out[DATE_COLUMN] < FULL_LEAD_COVERAGE_START]

        assert (early["LeadDays"] == 0).all()
        assert len(early) == 2

    def test_assignment_is_reproducible(self, dated):
        first = assign_lead_days(dated.copy())["LeadDays"].tolist()
        second = assign_lead_days(dated.copy())["LeadDays"].tolist()

        assert first == second

    def test_a_different_seed_changes_the_draw(self, dated):
        default = assign_lead_days(dated.copy())["LeadDays"].tolist()
        other = assign_lead_days(dated.copy(), seed=7)["LeadDays"].tolist()

        assert default != other


class TestAddUtcColumns:
    @pytest.fixture
    def airports(self):
        return pd.DataFrame(
            {
                "AirportId": [10397, 12892],
                "Timezone": ["America/New_York", "America/Los_Angeles"],
            }
        )

    @pytest.fixture
    def flights(self):
        return pd.DataFrame(
            {
                DATE_COLUMN: pd.to_datetime(["2025-03-01", "2025-03-01"]),
                "OriginAirportID": [10397, 12892],
                "DepHour": [8, 8],
                "CRSElapsedTime": [120, 90],
            }
        )

    def test_departure_is_converted_from_local_time(self, flights, airports):
        """08:00 in New York is 13:00 UTC in March; in Los Angeles it is 16:00."""
        out = add_utc_columns(flights, airports)

        assert out["DepUtcHour"].iloc[0] == pd.Timestamp("2025-03-01 13:00", tz="UTC")
        assert out["DepUtcHour"].iloc[1] == pd.Timestamp("2025-03-01 16:00", tz="UTC")

    def test_arrival_is_departure_plus_scheduled_duration(self, flights, airports):
        out = add_utc_columns(flights, airports)

        assert out["ArrUtcHour"].iloc[0] == pd.Timestamp("2025-03-01 15:00", tz="UTC")

    def test_arrival_is_floored_to_the_hour(self, flights, airports):
        """The weather series is hourly, so arrival must land on an hour boundary."""
        out = add_utc_columns(flights, airports)

        assert (out["ArrUtcHour"].dt.minute == 0).all()

    def test_unknown_airport_yields_no_timestamp(self, flights, airports):
        flights.loc[0, "OriginAirportID"] = 99999
        out = add_utc_columns(flights, airports)

        assert pd.isna(out["DepUtcHour"].iloc[0])


class TestAddTurnaroundFeatures:
    def test_turnaround_is_measured_from_the_previous_leg(self):
        df = pd.DataFrame(
            {
                "TailNumber": ["N1", "N1", "N2"],
                "DepUtcHour": pd.to_datetime(
                    ["2025-03-01 08:00", "2025-03-01 13:00", "2025-03-01 09:00"], utc=True
                ),
                "ArrUtcHour": pd.to_datetime(
                    ["2025-03-01 11:00", "2025-03-01 16:00", "2025-03-01 12:00"], utc=True
                ),
            }
        )
        out = add_turnaround_features(df).sort_index()

        assert out["ScheduledTurnaround"].iloc[1] == pytest.approx(120.0)

    def test_first_leg_of_an_aircraft_has_no_turnaround(self):
        """A NaN here is left to the Transformer's median imputation."""
        df = pd.DataFrame(
            {
                "TailNumber": ["N1", "N2"],
                "DepUtcHour": pd.to_datetime(["2025-03-01 08:00", "2025-03-01 09:00"], utc=True),
                "ArrUtcHour": pd.to_datetime(["2025-03-01 11:00", "2025-03-01 12:00"], utc=True),
            }
        )
        out = add_turnaround_features(df)

        assert out["ScheduledTurnaround"].isna().all()

    def test_turnaround_never_crosses_aircraft(self):
        df = pd.DataFrame(
            {
                "TailNumber": ["N1", "N2"],
                "DepUtcHour": pd.to_datetime(["2025-03-01 08:00", "2025-03-01 12:00"], utc=True),
                "ArrUtcHour": pd.to_datetime(["2025-03-01 11:00", "2025-03-01 14:00"], utc=True),
            }
        )
        out = add_turnaround_features(df)

        assert out.loc[out["TailNumber"] == "N2", "ScheduledTurnaround"].isna().all()


class TestAddCarrierFeatures:
    @pytest.fixture
    def two_flights(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "OriginAirportID": [10397, 11298],
                "DestAirportID": [12478, 12892],
                "Origin": ["ATL", "DFW"],
                "Dest": ["JFK", "LAX"],
                "ReportingAirline": ["DL", "AA"],
            }
        )

    def test_airport_and_carrier_are_concatenated(self, two_flights):
        out = add_carrier_features(two_flights)

        assert list(out["OriginCarrier"]) == ["10397DL", "11298AA"]
        assert list(out["DestCarrier"]) == ["12478DL", "12892AA"]

    def test_the_airport_half_is_the_id_and_not_the_code(self, two_flights):
        """Two airports that held the same code in turn must not share a pair, for
        the reason the airport delay rates are keyed on the id."""
        same_code = pd.DataFrame(
            {
                "OriginAirportID": [10423, 16440],
                "DestAirportID": [12478, 12478],
                "Origin": ["AUS", "AUS"],
                "Dest": ["JFK", "JFK"],
                "ReportingAirline": ["AA", "AA"],
            }
        )

        out = add_carrier_features(same_code)

        assert out["OriginCarrier"].nunique() == 2


class TestPrepareFlightsCommand:
    """The whole PRE-split feature chain, driven through the CLI."""

    @pytest.fixture
    def bts_csv(self, tmp_path):
        """A BTS On-Time Performance extract, with its original column names."""
        directory = tmp_path / "raw" / "2025_03"
        directory.mkdir(parents=True)
        pd.DataFrame(
            {
                DATE_COLUMN: ["2025-03-01"] * 4 + ["2025-03-02"] * 2,
                "OriginAirportID": [10397, 12892, 10397, 12892, 10397, 12892],
                "DestAirportID": [12892, 10397, 12892, 10397, 12892, 10397],
                "Origin": ["ATL", "LAX", "ATL", "LAX", "ATL", "LAX"],
                "Dest": ["LAX", "ATL", "LAX", "ATL", "LAX", "ATL"],
                "Reporting_Airline": ["DL", "AA", "DL", "AA", "DL", "AA"],
                "Flight_Number_Reporting_Airline": [1, 2, 3, 4, 5, 6],
                "Tail_Number": ["N1", "N2", "N1", "N2", "N1", "N2"],
                "CRSDepTime": [800, 830, 1400, 1430, 900, 930],
                "CRSArrTime": [1100, 1130, 1700, 1730, 1200, 1230],
                "CRSElapsedTime": [300, 300, 300, 300, 300, 300],
                "Distance": [1946, 1946, 1946, 1946, 1946, 1946],
                "DistanceGroup": [8, 8, 8, 8, 8, 8],
                "DayofMonth": [1, 1, 1, 1, 2, 2],
                "DayOfWeek": [6, 6, 6, 6, 7, 7],
                "Month": [3, 3, 3, 3, 3, 3],
                "OriginState": ["GA", "CA", "GA", "CA", "GA", "CA"],
                "DestState": ["CA", "GA", "CA", "GA", "CA", "GA"],
                "OriginCityName": ["Atlanta"] * 3 + ["Los Angeles"] * 3,
                "DestCityName": ["Los Angeles"] * 3 + ["Atlanta"] * 3,
                "Cancelled": [0, 0, 1, 0, 0, 0],
                "Diverted": [0, 0, 0, 0, 0, 0],
                "ArrDel15": [0.0, 1.0, 0.0, 0.0, 1.0, None],
                "SomeColumnWeDoNotWant": ["x"] * 6,
            }
        ).to_csv(directory / "flights.csv", index=False)
        return tmp_path / "raw"

    @pytest.fixture
    def airports_csv(self, tmp_path):
        path = tmp_path / "airports.csv"
        pd.DataFrame(
            {
                "AirportId": [10397, 12892],
                "Iata": ["ATL", "LAX"],
                "Latitude": [33.64, 33.94],
                "Longitude": [-84.43, -118.41],
                "Timezone": ["America/New_York", "America/Los_Angeles"],
            }
        ).to_csv(path, index=False)
        return path

    @pytest.fixture
    def prepared(self, tmp_path, bts_csv, airports_csv):
        output_path = tmp_path / "flights_features.parquet"
        result = runner.invoke(
            preprocess_module.app,
            [
                "prepare-flights",
                str(bts_csv),
                "--airports-path",
                str(airports_csv),
                "--output-path",
                str(output_path),
            ],
        )
        assert result.exit_code == 0, result.output
        return pd.read_parquet(output_path)

    def test_unusable_flights_are_gone(self, prepared):
        """One cancelled and one without an outcome, out of six."""
        assert len(prepared) == 4

    def test_column_names_are_pascal_cased(self, prepared):
        assert "ReportingAirline" in prepared.columns
        assert "TailNumber" in prepared.columns
        assert "FlightNumberReportingAirline" in prepared.columns

    def test_unwanted_source_columns_are_never_read(self, prepared):
        assert "SomeColumnWeDoNotWant" not in prepared.columns

    def test_every_feature_stage_contributed(self, prepared):
        expected = {
            "OriginCongestion",
            "DestCongestion",
            "IsDelayed",
            "DepHour",
            "DepTimeDecimal",
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
        }
        assert expected <= set(prepared.columns)

    def test_a_single_csv_file_works_too(self, tmp_path, bts_csv, airports_csv):
        """bts_path may be one file rather than a directory to walk."""
        output_path = tmp_path / "one_file.parquet"

        result = runner.invoke(
            preprocess_module.app,
            [
                "prepare-flights",
                str(bts_csv / "2025_03" / "flights.csv"),
                "--airports-path",
                str(airports_csv),
                "--output-path",
                str(output_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(pd.read_parquet(output_path)) == 4

    def test_an_empty_directory_exits_with_an_error_code(self, tmp_path, airports_csv):
        empty = tmp_path / "empty"
        empty.mkdir()

        result = runner.invoke(
            preprocess_module.app,
            [
                "prepare-flights",
                str(empty),
                "--airports-path",
                str(airports_csv),
                "--output-path",
                str(tmp_path / "out.parquet"),
            ],
        )

        assert result.exit_code == 1

    def test_a_missing_airports_table_exits_with_an_error_code(self, tmp_path, bts_csv):
        result = runner.invoke(
            preprocess_module.app,
            [
                "prepare-flights",
                str(bts_csv),
                "--airports-path",
                str(tmp_path / "absent.csv"),
                "--output-path",
                str(tmp_path / "out.parquet"),
            ],
        )

        assert result.exit_code == 1


class TestJoinWeatherCommand:
    @pytest.fixture
    def flights_parquet(self, tmp_path):
        path = tmp_path / "flights_features.parquet"
        pd.DataFrame(
            {
                "OriginAirportID": [10397],
                "DestAirportID": [12892],
                "DepUtcHour": pd.to_datetime(["2025-03-01 13:00"], utc=True),
                "ArrUtcHour": pd.to_datetime(["2025-03-01 15:00"], utc=True),
                "LeadDays": [2],
            }
        ).to_parquet(path, index=False)
        return path

    @pytest.fixture
    def weather_dir(self, tmp_path):
        directory = tmp_path / "weather"
        directory.mkdir()
        for airport_id in (10397, 12892):
            pd.DataFrame(
                {
                    "AirportId": airport_id,
                    "Time": pd.to_datetime(["2025-03-01 13:00", "2025-03-01 15:00"], utc=True),
                    "LeadDays": 2,
                    "Temperature2m": [5.0, 7.0],
                    "Precipitation": [0.0, 1.5],
                    "Snowfall": [0.0, 0.0],
                    "WindSpeed10m": [10.0, 12.0],
                    "WindGusts10m": [20.0, 24.0],
                    "WeatherCode": [0, 61],
                }
            ).to_parquet(directory / f"{airport_id}_lead2.parquet", index=False)
        return directory

    def test_the_joined_dataset_is_written(self, tmp_path, flights_parquet, weather_dir):
        output_path = tmp_path / "flights_preprocessed.parquet"

        result = runner.invoke(
            preprocess_module.app,
            [
                "join-weather",
                "--flights-path",
                str(flights_parquet),
                "--weather-dir",
                str(weather_dir),
                "--output-path",
                str(output_path),
            ],
        )

        assert result.exit_code == 0, result.output
        joined = pd.read_parquet(output_path)
        assert joined["Temperature2mOrigin"].iloc[0] == pytest.approx(5.0)
        assert joined["Temperature2mDest"].iloc[0] == pytest.approx(7.0)

    def test_a_missing_weather_directory_exits_with_an_error_code(self, tmp_path, flights_parquet):
        result = runner.invoke(
            preprocess_module.app,
            [
                "join-weather",
                "--flights-path",
                str(flights_parquet),
                "--weather-dir",
                str(tmp_path / "absent"),
                "--output-path",
                str(tmp_path / "out.parquet"),
            ],
        )

        assert result.exit_code == 1


class TestJoinWeatherToFlights:
    @pytest.fixture
    def weather_dir(self, tmp_path):
        """Two airports, one lead time, two hourly slots each."""
        directory = tmp_path / "weather"
        directory.mkdir()
        for airport_id in (10397, 12892):
            times = pd.to_datetime(["2025-03-01 13:00", "2025-03-01 15:00"], utc=True)
            pd.DataFrame(
                {
                    "AirportId": airport_id,
                    "Time": times,
                    "LeadDays": 2,
                    "Temperature2m": [5.0, 7.0],
                    "Precipitation": [0.0, 1.5],
                    "Snowfall": [0.0, 0.0],
                    "WindSpeed10m": [10.0, 12.0],
                    "WindGusts10m": [20.0, 24.0],
                    "WeatherCode": [0, 61],
                }
            ).to_parquet(directory / f"{airport_id}_lead2.parquet", index=False)
        return directory

    @pytest.fixture
    def flights(self):
        return pd.DataFrame(
            {
                "OriginAirportID": [10397],
                "DestAirportID": [12892],
                "DepUtcHour": pd.to_datetime(["2025-03-01 13:00"], utc=True),
                "ArrUtcHour": pd.to_datetime(["2025-03-01 15:00"], utc=True),
                "LeadDays": [2],
            }
        )

    def test_both_route_ends_are_joined(self, flights, weather_dir):
        out = join_weather_to_flights(flights, weather_dir)

        assert out["Temperature2mOrigin"].iloc[0] == pytest.approx(5.0)
        assert out["Temperature2mDest"].iloc[0] == pytest.approx(7.0)

    def test_join_keys_are_dropped_afterwards(self, flights, weather_dir):
        out = join_weather_to_flights(flights, weather_dir)

        assert "AirportId" not in out.columns
        assert "Time" not in out.columns

    def test_weather_code_becomes_a_label(self, flights, weather_dir):
        """The code is a category, not a magnitude, so it is stored as a string."""
        out = join_weather_to_flights(flights, weather_dir)

        assert out["WeatherCodeOrigin"].iloc[0] == "0"
        assert out["WeatherCodeDest"].iloc[0] == "61"

    def test_the_weather_code_stays_categorical(self, flights, weather_dir):
        """object dtype is how the Transformer recognises a categorical column;
        a pandas string dtype would fall through both of its type filters."""
        out = join_weather_to_flights(flights, weather_dir)

        assert out["WeatherCodeOrigin"].dtype == object

    def test_an_unmatched_code_stays_missing(self, flights, weather_dir):
        """Stringifying it would create a second sentinel for 'unknown',
        competing with the Transformer's OTHER bucket and, worse, surviving the
        weather-completeness filter as if it were a real category."""
        flights.loc[0, "LeadDays"] = 5
        out = join_weather_to_flights(flights, weather_dir)

        assert out["WeatherCodeOrigin"].isna().all()
        assert out["WeatherCodeDest"].isna().all()

    def test_row_count_is_preserved(self, flights, weather_dir):
        out = join_weather_to_flights(flights, weather_dir)

        assert len(out) == len(flights)

    def test_unmatched_flights_keep_empty_weather(self, flights, weather_dir):
        """The left join is what 'noweather' exists to serve in production."""
        flights.loc[0, "LeadDays"] = 5
        out = join_weather_to_flights(flights, weather_dir)

        assert out["Temperature2mOrigin"].isna().all()
