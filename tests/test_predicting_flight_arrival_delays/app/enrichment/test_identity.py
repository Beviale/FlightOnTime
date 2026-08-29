"""Tests for predicting_flight_arrival_delays.app.enrichment.identity."""

import pandas as pd
import pytest

from predicting_flight_arrival_delays.app.enrichment import identity as identity_module
from predicting_flight_arrival_delays.app.enrichment import reference as reference_module
from predicting_flight_arrival_delays.app.enrichment.identity import (
    MAX_MATCH_KM,
    AmbiguousAirportError,
    UnknownAirportError,
    get_identity,
)


MUELLER = (10423, "AUS", 30.298056, -97.701389)
BERGSTROM = (16440, "AUS", 30.194444, -97.670000)
ATLANTA = (10397, "ATL", 33.640444, -84.426944)
CHICAGO = (13930, "ORD", 41.978611, -87.904722)

NEAR_BERGSTROM = {"lat": 30.1945, "lon": -97.6699}
NEAR_MUELLER = {"lat": 30.2981, "lon": -97.7014}
NEAR_ATLANTA = {"lat": 33.6404, "lon": -84.4269}
NEAR_CHICAGO = {"lat": 41.9786, "lon": -87.9047}

CACHED = (
    identity_module.load_identity,
    identity_module.load_codes,
    reference_module.load_airports,
    reference_module.load_airports_table,
)


@pytest.fixture
def tables(tmp_path, monkeypatch):
    """Write the two tables and point the modules at them."""

    def build(reference, flown=None):
        """reference: rows of airports.csv. flown: the ids the model has seen."""
        flown = [row[0] for row in reference] if flown is None else flown

        airports_csv = tmp_path / "airports.csv"
        pd.DataFrame(
            [(i, code, lat, lon, "America/Chicago") for i, code, lat, lon in reference],
            columns=["AirportId", "Iata", "Latitude", "Longitude", "Timezone"],
        ).to_csv(airports_csv, index=False)

        identity_csv = tmp_path / "airport_identity.csv"
        pd.DataFrame(
            [
                (code, i, f"City {i}", "TX")
                for i, code, _, _ in reference
                if i in flown
            ],
            columns=["Iata", "AirportId", "CityName", "State"],
        ).to_csv(identity_csv, index=False)

        monkeypatch.setattr(reference_module, "AIRPORTS_CSV", airports_csv)
        monkeypatch.setattr(identity_module, "AIRPORT_IDENTITY_CSV", identity_csv)
        for cached in CACHED:
            cached.cache_clear()

    for cached in CACHED:
        cached.cache_clear()
    yield build
    for cached in CACHED:
        cached.cache_clear()


class TestOneAirportUnderTheCode:
    def test_the_code_resolves_to_its_id(self, tables):
        tables([ATLANTA])

        assert get_identity("ATL", NEAR_ATLANTA).airport_id == 10397

    def test_the_naming_comes_from_the_identity_table(self, tables):
        tables([ATLANTA])
        found = get_identity("ATL", NEAR_ATLANTA)

        assert found.city_name == "City 10397"
        assert found.state == "TX"

    def test_the_code_is_case_and_space_insensitive(self, tables):
        tables([ATLANTA])

        assert get_identity("  atl ", NEAR_ATLANTA).airport_id == 10397


class TestACodeThatLeadsNowhere:
    def test_a_code_absent_from_the_reference_table_is_refused(self, tables):
        tables([ATLANTA])

        with pytest.raises(UnknownAirportError, match="no such code"):
            get_identity("LAX", NEAR_ATLANTA)

    def test_an_airport_the_model_never_flew_is_refused(self, tables):
        tables([ATLANTA, CHICAGO], flown=[10397])

        with pytest.raises(UnknownAirportError, match="never seen"):
            get_identity("ORD", NEAR_CHICAGO)


class TestACodeListingTwoAirports:
    def test_the_nearest_candidate_wins(self, tables):
        tables([MUELLER, BERGSTROM])

        assert get_identity("AUS", NEAR_BERGSTROM).airport_id == 16440

    def test_the_other_location_picks_the_other_airport(self, tables):
        tables([MUELLER, BERGSTROM])

        assert get_identity("AUS", NEAR_MUELLER).airport_id == 10423

    def test_twelve_kilometres_is_enough_to_tell_them_apart(self, tables):
        """Close by any map, far apart for this."""
        tables([MUELLER, BERGSTROM])

        near = get_identity("AUS", NEAR_BERGSTROM)
        far = get_identity("AUS", NEAR_MUELLER)

        assert near.airport_id != far.airport_id

    def test_without_a_location_it_refuses_rather_than_guessing(self, tables):
        tables([MUELLER, BERGSTROM])

        with pytest.raises(AmbiguousAirportError, match="AUS"):
            get_identity("AUS", None)

    def test_the_error_names_the_candidates(self, tables):
        tables([MUELLER, BERGSTROM])

        with pytest.raises(AmbiguousAirportError, match="10423"):
            get_identity("AUS", None)

    def test_candidates_with_no_coordinates_cannot_be_separated(self, tables):
        """A location is useless if the reference table cannot place the candidates."""
        nowhere = [
            (10423, "AUS", float("nan"), float("nan")),
            (16440, "AUS", float("nan"), float("nan")),
        ]
        tables(nowhere)

        with pytest.raises(AmbiguousAirportError, match="coordinates"):
            get_identity("AUS", NEAR_BERGSTROM)

    def test_the_chosen_id_still_has_to_be_one_the_model_flew(self, tables):
        """Coordinates can land on the airport that closed; the model never saw it."""
        tables([MUELLER, BERGSTROM], flown=[16440])

        with pytest.raises(UnknownAirportError, match="never seen"):
            get_identity("AUS", NEAR_MUELLER)


class TestHowFarTheMatchIsAllowedToBe:

    FAR_FROM_BOTH = {"lat": 30.25, "lon": -97.55}

    def test_a_location_beyond_the_limit_is_refused_with_two_candidates(self, tables):
        tables([MUELLER, BERGSTROM])

        with pytest.raises(UnknownAirportError, match="not that airport"):
            get_identity("AUS", self.FAR_FROM_BOTH)

    def test_a_location_beyond_the_limit_is_refused_with_one_candidate(self, tables):
        tables([ATLANTA])

        with pytest.raises(UnknownAirportError, match="not that airport"):
            get_identity("ATL", {"lat": 41.9786, "lon": -87.9047})

    def test_the_error_says_how_far_and_how_far_was_allowed(self, tables):
        tables([ATLANTA])

        with pytest.raises(UnknownAirportError, match=f"{MAX_MATCH_KM} km"):
            get_identity("ATL", {"lat": 41.9786, "lon": -87.9047})

    def test_a_location_on_the_airport_passes(self, tables):
        tables([MUELLER, BERGSTROM])

        assert get_identity("AUS", NEAR_BERGSTROM).airport_id == BERGSTROM[0]

    def test_a_terminal_rather_than_the_reference_point_passes(self, tables):
        tables([ATLANTA])
        two_km_north = {"lat": 33.6584, "lon": -84.4269}

        assert get_identity("ATL", two_km_north).airport_id == 10397

    def test_a_lone_candidate_without_a_location_is_refused(self, tables):
        tables([ATLANTA])

        with pytest.raises(UnknownAirportError, match="cannot be confirmed"):
            get_identity("ATL", None)

    def test_the_location_cannot_be_left_out_by_accident(self, tables):
        tables([ATLANTA])

        with pytest.raises(TypeError):
            get_identity("ATL")

    def test_a_location_without_coordinates_is_refused(self, tables):
        tables([ATLANTA])

        with pytest.raises(UnknownAirportError, match="cannot be confirmed"):
            get_identity("ATL", {})

    def test_a_lone_candidate_with_no_coordinates_on_file_is_refused(self, tables):
        tables([(10397, "ATL", None, None)])

        with pytest.raises(UnknownAirportError, match="cannot be confirmed"):
            get_identity("ATL", NEAR_BERGSTROM)

    def test_several_candidates_that_cannot_be_measured_stay_ambiguous(self, tables):
        tables([(10423, "AUS", None, None), (16440, "AUS", None, None)])

        with pytest.raises(AmbiguousAirportError, match="cannot be told apart"):
            get_identity("AUS", NEAR_BERGSTROM)


class TestTheCallersCodeIsKept:
    def test_the_code_asked_for_is_the_one_returned(self, tables):
        tables([CHICAGO])

        assert get_identity("ord", NEAR_CHICAGO).iata == "ORD"
