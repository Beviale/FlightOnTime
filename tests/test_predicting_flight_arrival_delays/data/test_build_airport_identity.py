"""Tests for predicting_flight_arrival_delays.data.build_airport_identity."""

from loguru import logger
import pandas as pd
import pytest
from typer.testing import CliRunner

from predicting_flight_arrival_delays.data.build_airport_identity import app, build_identity
from predicting_flight_arrival_delays.config import DATE_COLUMN

runner = CliRunner()


@pytest.fixture
def warnings_logged() -> list[str]:
    """Collect the warnings a call emits.
    """
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink)


def flights(rows) -> pd.DataFrame:
    """Build prepared-flight rows."""
    return pd.DataFrame(
        [(o[0], o[1], o[2], o[3], d[0], d[1], d[2], d[3], date) for date, o, d in rows],
        columns=[
            "Origin", "OriginAirportID", "OriginCityName", "OriginState",
            "Dest", "DestAirportID", "DestCityName", "DestState",
            DATE_COLUMN,
        ],
    ).astype({DATE_COLUMN: "datetime64[ns]"})


ATL = ("ATL", 10397, "Atlanta, GA", "GA")
LAX = ("LAX", 12892, "Los Angeles, CA", "CA")
ORD = ("ORD", 13930, "Chicago, IL", "IL")


class TestBuildIdentity:
    def test_both_ends_of_the_route_contribute(self):
        """An airport is an origin on some flights and a destination on others."""
        table = build_identity(flights([("2025-01-01", ATL, LAX)]))

        assert set(table["Iata"]) == {"ATL", "LAX"}

    def test_one_row_per_airport_however_many_flights(self):
        table = build_identity(
            flights([("2025-01-01", ATL, LAX)] * 500 + [("2025-02-01", LAX, ATL)] * 500)
        )

        assert len(table) == 2

    def test_the_columns_are_what_the_lookup_reads(self):
        table = build_identity(flights([("2025-01-01", ATL, LAX)]))

        assert list(table.columns) == ["Iata", "AirportId", "CityName", "State"]

    def test_the_identity_is_carried_over_intact(self):
        table = build_identity(flights([("2025-01-01", ATL, LAX)])).set_index("Iata")

        assert table.loc["ATL", "AirportId"] == 10397
        assert table.loc["ATL", "CityName"] == "Atlanta, GA"
        assert table.loc["ATL", "State"] == "GA"

    def test_rows_come_back_sorted_by_code(self):
        table = build_identity(flights([("2025-01-01", ORD, ATL), ("2025-01-02", LAX, ATL)]))

        assert list(table["Iata"]) == sorted(table["Iata"])
        assert list(table.index) == list(range(len(table)))


class TestACodeCoveringTwoAirportsIsKept:
    """BTS reassigns codes: 'AUS' names both the Austin airport that closed in 1999
    and the one that replaced it. The table keeps both rows and lets the serving
    path pick, because there the schedule service reports where the flight is."""

    def test_both_airports_survive(self):
        old_austin = ("AUS", 10423, "Austin, TX", "TX")
        new_austin = ("AUS", 16440, "Austin, TX", "TX")

        table = build_identity(
            flights([("2025-01-01", old_austin, LAX), ("2025-06-01", new_austin, LAX)])
        )

        assert sorted(table.loc[table["Iata"] == "AUS", "AirportId"]) == [10423, 16440]

    def test_a_clash_at_the_destination_end_is_kept_too(self):
        old_austin = ("AUS", 10423, "Austin, TX", "TX")
        new_austin = ("AUS", 16440, "Austin, TX", "TX")

        table = build_identity(
            flights([("2025-01-01", LAX, old_austin), ("2025-06-01", LAX, new_austin)])
        )

        assert (table["Iata"] == "AUS").sum() == 2


class TestARenamedPlaceKeepsTheLatestSpelling:
    """The id is the key, so it must carry exactly one city and state."""

    EARLY = ("DCA", 11278, "Washington, DC", "VA")
    LATE = ("DCA", 11278, "Washington, DC (Metropolitan Area)", "VA")

    def test_the_city_last_flown_under_wins(self):
        table = build_identity(
            flights([("2025-01-01", self.EARLY, LAX), ("2025-09-01", self.LATE, LAX)])
        ).set_index("Iata")

        assert table.loc["DCA", "CityName"] == "Washington, DC (Metropolitan Area)"

    def test_the_state_last_flown_under_wins(self):
        early = ("DCA", 11278, "Washington, DC", "VA")
        late = ("DCA", 11278, "Washington, DC", "DC")

        table = build_identity(
            flights([("2025-01-01", early, LAX), ("2025-09-01", late, LAX)])
        ).set_index("Iata")

        assert table.loc["DCA", "State"] == "DC"

    def test_the_order_the_rows_arrive_in_does_not_decide(self):
        """The date decides, not the position in the frame."""
        table = build_identity(
            flights([("2025-09-01", self.LATE, LAX), ("2025-01-01", self.EARLY, LAX)])
        ).set_index("Iata")

        assert table.loc["DCA", "CityName"] == "Washington, DC (Metropolitan Area)"

    def test_the_city_and_state_come_from_the_same_record(self):
        """They describe one place. Taking the latest of each on its own could pair
        a city with a state it was never recorded beside."""
        early = ("DCA", 11278, "Washington, DC", "VA")
        late = ("DCA", 11278, "Arlington, VA", "DC")

        table = build_identity(
            flights([("2025-01-01", early, LAX), ("2025-09-01", late, LAX)])
        ).set_index("Iata")

        assert (table.loc["DCA", "CityName"], table.loc["DCA", "State"]) == (
            "Arlington, VA",
            "DC",
        )

    def test_the_most_recent_flight_decides_not_the_most_frequent(self):
        """Five hundred flights under the old place against one under the new: the
        rule is recency, so the single later flight still wins."""
        table = build_identity(
            flights(
                [("2025-01-01", self.EARLY, LAX)] * 500
                + [("2025-09-01", self.LATE, LAX)]
            )
        ).set_index("Iata")

        assert table.loc["DCA", "CityName"] == "Washington, DC (Metropolitan Area)"

    def test_the_id_is_left_with_one_row(self):
        table = build_identity(
            flights([("2025-01-01", self.EARLY, LAX), ("2025-09-01", self.LATE, LAX)])
        )

        assert (table["AirportId"] == 11278).sum() == 1

    def test_a_rename_at_the_destination_end_is_settled_too(self):
        table = build_identity(
            flights([("2025-01-01", LAX, self.EARLY), ("2025-09-01", LAX, self.LATE)])
        ).set_index("Iata")

        assert table.loc["DCA", "CityName"] == "Washington, DC (Metropolitan Area)"

    def test_the_dropped_place_is_warned_about(self, warnings_logged):
        """Silently picking one of two names is how the wrong city reaches a caller."""
        build_identity(
            flights([("2025-01-01", self.EARLY, LAX), ("2025-09-01", self.LATE, LAX)])
        )

        warned = "".join(warnings_logged)
        assert "11278" in warned
        assert "Washington, DC" in warned
        assert "Washington, DC (Metropolitan Area)" in warned

    def test_nothing_is_warned_about_when_the_naming_holds(self, warnings_logged):
        build_identity(flights([("2025-01-01", ATL, LAX), ("2025-09-01", ATL, LAX)]))

        assert warnings_logged == []

    def test_the_same_place_throughout_is_fine(self):
        """The resolution must key on a real disagreement, not on repetition."""
        table = build_identity(flights([("2025-01-01", ATL, LAX), ("2025-09-01", ATL, LAX)]))

        assert (table["Iata"] == "ATL").sum() == 1

    def test_a_second_code_for_the_renamed_airport_survives(self):
        """Settling the naming must not collapse the codes: both still resolve."""
        early = ("DCA", 11278, "Washington, DC", "VA")
        late = ("WAS", 11278, "Washington, DC (Metropolitan Area)", "VA")

        table = build_identity(
            flights([("2025-01-01", early, LAX), ("2025-09-01", late, LAX)])
        ).set_index("Iata")

        assert table.loc["DCA", "AirportId"] == 11278
        assert table.loc["WAS", "AirportId"] == 11278
        assert table.loc["DCA", "CityName"] == "Washington, DC (Metropolitan Area)"


class TestSeveralCodesForOneAirport:
    def test_both_codes_are_kept(self):
        renamed = ("CHI", 13930, "Chicago, IL", "IL")

        table = build_identity(
            flights([("2025-01-01", ORD, LAX), ("2025-06-01", renamed, LAX)])
        ).set_index("Iata")

        assert table.loc["ORD", "AirportId"] == 13930
        assert table.loc["CHI", "AirportId"] == 13930


class TestRunCommand:
    @pytest.fixture
    def flights_parquet(self, tmp_path):
        path = tmp_path / "flights_features.parquet"
        flights([("2025-01-01", ATL, LAX), ("2025-02-01", ORD, ATL)]).to_parquet(
            path, index=False
        )
        return path

    def test_the_table_is_written(self, tmp_path, flights_parquet):
        output_path = tmp_path / "airport_identity.csv"

        result = runner.invoke(
            app,
            ["--flights-path", str(flights_parquet), "--output-path", str(output_path)],
        )

        assert result.exit_code == 0, result.output
        assert set(pd.read_csv(output_path)["Iata"]) == {"ATL", "LAX", "ORD"}

    def test_the_output_directory_is_created(self, tmp_path, flights_parquet):
        output_path = tmp_path / "deep" / "nested" / "airport_identity.csv"

        runner.invoke(
            app,
            ["--flights-path", str(flights_parquet), "--output-path", str(output_path)],
        )

        assert output_path.exists()

    def test_a_missing_input_exits_with_an_error_code(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "--flights-path", str(tmp_path / "absent.parquet"),
                "--output-path", str(tmp_path / "out.csv"),
            ],
        )

        assert result.exit_code == 1
