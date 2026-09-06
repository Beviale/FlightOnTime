"""Tests for predicting_flight_arrival_delays.data.build_airports."""

import pandas as pd
import pytest
from typer.testing import CliRunner

from predicting_flight_arrival_delays.data import build_airports as build_airports_module
from predicting_flight_arrival_delays.data.build_airports import app, build

runner = CliRunner()

RAW_COLUMNS = [
    "AIRPORT_ID",
    "AIRPORT",
    "LATITUDE",
    "LONGITUDE",
    "AIRPORT_IS_LATEST",
    "AIRPORT_COUNTRY_CODE_ISO",
]


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    directory = tmp_path / "external" / "raw"
    directory.mkdir(parents=True)
    monkeypatch.setattr(build_airports_module, "EXTERNAL_RAW_DATA_DIR", directory)
    return directory


@pytest.fixture
def write_raw(raw_dir):
    """Write a T_MASTER_CORD-shaped CSV where the module expects to find it."""

    def build(rows):
        pd.DataFrame(rows).to_csv(raw_dir / "T_MASTER_CORD.csv", index=False)
        return raw_dir / "T_MASTER_CORD.csv"

    return build


@pytest.fixture
def us_airports():
    return {
        "AIRPORT_ID": [10397, 12892, 13930],
        "AIRPORT": ["ATL", "LAX", "ORD"],
        "LATITUDE": [33.64, 33.94, 41.98],
        "LONGITUDE": [-84.43, -118.41, -87.90],
        "AIRPORT_IS_LATEST": [1, 1, 1],
        "AIRPORT_COUNTRY_CODE_ISO": ["US", "US", "US"],
    }


class TestBuild:
    def test_the_header_is_what_the_rest_of_the_pipeline_reads(
        self, write_raw, us_airports, tmp_path
    ):
        """preprocess.add_utc_columns and weather.download_weather index this file
        by 'AirportId' and read 'Latitude'/'Longitude'/'Timezone' by name."""
        write_raw(us_airports)
        out = tmp_path / "airports.csv"

        build(out)

        assert list(pd.read_csv(out).columns) == [
            "AirportId",
            "Iata",
            "Latitude",
            "Longitude",
            "Timezone",
        ]

    def test_timezones_come_from_the_coordinates(self, write_raw, us_airports, tmp_path):
        write_raw(us_airports)
        out = tmp_path / "airports.csv"

        build(out)
        table = pd.read_csv(out).set_index("AirportId")

        assert table.loc[10397, "Timezone"] == "America/New_York"
        assert table.loc[12892, "Timezone"] == "America/Los_Angeles"

    def test_every_coordinate_resolves_to_some_zone(self, write_raw, us_airports, tmp_path):
        """TimezoneFinder falls back to an Etc/GMT offset over open water rather
        than returning nothing, so no row is ever left without a timezone."""
        rows = {k: v + [v[0]] for k, v in us_airports.items()}
        rows["AIRPORT_ID"][-1] = 77777
        rows["LATITUDE"][-1] = 0.0
        rows["LONGITUDE"][-1] = 0.0
        write_raw(rows)
        out = tmp_path / "airports.csv"

        build(out)
        table = pd.read_csv(out).set_index("AirportId")

        assert table.loc[77777, "Timezone"] == "Etc/GMT"
        assert table["Timezone"].notna().all()

    def test_non_us_airports_are_dropped(self, write_raw, us_airports, tmp_path):
        rows = {k: v + [v[0]] for k, v in us_airports.items()}
        rows["AIRPORT_ID"][-1] = 99999
        rows["AIRPORT_COUNTRY_CODE_ISO"][-1] = "CA"
        write_raw(rows)
        out = tmp_path / "airports.csv"

        build(out)

        assert 99999 not in set(pd.read_csv(out)["AirportId"])

    def test_superseded_records_are_dropped(self, write_raw, us_airports, tmp_path):
        """An airport code can be reassigned; only the current record counts."""
        rows = {k: v + [v[0]] for k, v in us_airports.items()}
        rows["AIRPORT_ID"][-1] = 88888
        rows["AIRPORT_IS_LATEST"][-1] = 0
        write_raw(rows)
        out = tmp_path / "airports.csv"

        build(out)

        assert 88888 not in set(pd.read_csv(out)["AirportId"])

    def test_unusable_coordinates_are_dropped(self, write_raw, us_airports, tmp_path):
        rows = {k: list(v) for k, v in us_airports.items()}
        rows["LATITUDE"][0] = "n/a"
        write_raw(rows)
        out = tmp_path / "airports.csv"

        build(out)

        assert 10397 not in set(pd.read_csv(out)["AirportId"])

    def test_duplicate_airport_ids_are_collapsed(self, write_raw, us_airports, tmp_path):
        rows = {k: v + [v[0]] for k, v in us_airports.items()}
        write_raw(rows)
        out = tmp_path / "airports.csv"

        build(out)
        table = pd.read_csv(out)

        assert len(table) == 3
        assert table["AirportId"].is_unique

    def test_the_output_directory_is_created(self, write_raw, us_airports, tmp_path):
        write_raw(us_airports)
        out = tmp_path / "deep" / "nested" / "airports.csv"

        build(out)

        assert out.exists()

    def test_a_missing_source_table_stops_the_build(self, raw_dir, tmp_path):
        with pytest.raises(SystemExit, match="run `download` first"):
            build(tmp_path / "airports.csv")

    def test_a_source_table_missing_columns_stops_the_build(self, write_raw, tmp_path):
        write_raw({"AIRPORT_ID": [1], "AIRPORT": ["ATL"]})

        with pytest.raises(SystemExit, match="Missing columns"):
            build(tmp_path / "airports.csv")


class TestRunCommand:
    @pytest.fixture
    def downloads(self, monkeypatch):
        calls = []
        monkeypatch.setattr(build_airports_module, "download", lambda: calls.append(1))
        return calls

    def test_the_source_is_fetched_by_default(
        self, raw_dir, downloads, us_airports, tmp_path, write_raw
    ):
        write_raw(us_airports)
        result = runner.invoke(app, ["--output-path", str(tmp_path / "airports.csv")])

        assert result.exit_code == 0, result.output
        assert len(downloads) == 1

    def test_an_existing_table_can_be_reused(
        self, raw_dir, downloads, us_airports, tmp_path, write_raw
    ):
        write_raw(us_airports)
        result = runner.invoke(
            app, ["--no-force-download", "--output-path", str(tmp_path / "airports.csv")]
        )

        assert result.exit_code == 0, result.output
        assert downloads == []

    def test_a_missing_table_is_fetched_even_without_force(self, raw_dir, downloads, tmp_path):
        result = runner.invoke(
            app, ["--no-force-download", "--output-path", str(tmp_path / "airports.csv")]
        )

        assert len(downloads) == 1
        assert result.exit_code == 1  # download is stubbed out, so the build finds nothing

    def test_a_failure_exits_with_an_error_code(self, raw_dir, downloads, tmp_path):
        result = runner.invoke(app, ["--output-path", str(tmp_path / "airports.csv")])

        assert result.exit_code == 1

    def test_an_unexpected_error_is_reported_not_raised(
        self, raw_dir, downloads, tmp_path, monkeypatch, write_raw, us_airports
    ):
        write_raw(us_airports)

        def boom(output_path):
            raise RuntimeError("timezone database unavailable")

        monkeypatch.setattr(build_airports_module, "build", boom)

        result = runner.invoke(app, ["--output-path", str(tmp_path / "airports.csv")])

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
