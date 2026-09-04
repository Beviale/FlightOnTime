"""Tests for the prediction endpoints."""

from datetime import date, timedelta

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from fastapi.testclient import TestClient

from predicting_flight_arrival_delays.app.enrichment.aerodatabox import (
    FlightNotFoundError,
    ScheduleUnavailableError,
)
from predicting_flight_arrival_delays.app.enrichment.identity import UnknownAirportError
from predicting_flight_arrival_delays.app.inputs import CANDIDATE_INPUTS, required_inputs
from predicting_flight_arrival_delays.app.main import app
from predicting_flight_arrival_delays.app.schema import FlightRequest
from predicting_flight_arrival_delays.config import MAX_BATCH_SIZE


def serve(bundles: dict) -> TestClient:
    """Put the application in the state its lifespan would have left it in."""
    app.state.bundles = bundles
    app.state.required_inputs = required_inputs(bundles, CANDIDATE_INPUTS)
    return TestClient(app)


@pytest.fixture
def a_flight(body):
    """A valid request body, dated against the real calendar the app reads."""

    def build(days_ahead: int = 2, **overrides) -> dict:
        return body(days_ahead=days_ahead, base=date.today(), **overrides)

    return build


@pytest.fixture
def client(bundles, stub_forecast) -> TestClient:
    """A client serving both variants, with the forecast answered from memory."""
    stub_forecast()
    return serve(bundles)


@pytest.fixture
def degraded_client(bundles, stub_forecast) -> TestClient:
    """A client whose registry only carries the full model."""
    stub_forecast()
    return serve({"all": bundles["all"]})


@pytest.fixture
def explainable(bundles, stub_forecast):

    stub_forecast()

    rebuilt = {}
    for variant, bundle in bundles.items():
        frame = pd.DataFrame(
            np.random.default_rng(0).normal(size=(40, len(bundle.columns))),
            columns=bundle.columns,
        )
        target = pd.Series([0, 1] * 20)
        rebuilt[variant] = replace(
            bundle,
            model=LogisticRegression(max_iter=200).fit(frame, target),
            params=bundle.params | {"algorithm": "logistic_regression"},
        )
    return serve(rebuilt)


class TestExplanations:

    def test_it_scores_and_explains_in_one_answer(self, explainable, a_flight):
        data = explainable.post("/explanations", json=a_flight()).json()["data"]

        assert 0.0 <= data["delay_probability"] <= 1.0
        assert data["explanations"]

    def test_the_reasons_are_request_columns_not_matrix_columns(self, explainable, a_flight):
        data = explainable.post("/explanations", json=a_flight()).json()["data"]

        reported = [item["column"] for item in data["explanations"]]

        assert reported
        assert not any("_1" in column for column in reported)

    def test_a_contribution_is_signed(self, explainable, a_flight):
        data = explainable.post("/explanations", json=a_flight()).json()["data"]

        assert all(
            isinstance(item["contribution"], float) for item in data["explanations"]
        )

    def test_they_are_ordered_by_how_much_they_moved_the_answer(self, explainable, a_flight):
        data = explainable.post("/explanations", json=a_flight()).json()["data"]

        weights = [abs(item["contribution"]) for item in data["explanations"]]

        assert weights == sorted(weights, reverse=True)

    def test_it_carries_the_same_context_a_prediction_does(self, explainable, a_flight):
        data = explainable.post("/explanations", json=a_flight()).json()["data"]

        assert {"variant", "threshold", "weather", "approximated"} <= set(data)

    def test_a_prediction_can_ask_for_them_too(self, explainable, a_flight):
        data = explainable.post(
            "/predictions?explain=true", json=a_flight()
        ).json()["data"]

        assert data["explanations"]

    def test_a_prediction_leaves_them_out_by_default(self, explainable, a_flight):
        data = explainable.post("/predictions", json=a_flight()).json()["data"]

        assert "explanations" not in data

    def test_a_flight_missing_a_column_the_models_read_is_refused(self, client, a_flight):
        payload = a_flight()
        payload.pop("OriginCongestion")

        assert client.post("/explanations", json=payload).status_code == 422

    def test_with_no_model_loaded_it_answers_503(self, stub_forecast, a_flight):
        stub_forecast()

        assert serve({}).post("/explanations", json=a_flight()).status_code == 503


class TestApproximatedInputsAreReported:

    @pytest.fixture
    def leaning_client(self, bundles, stub_forecast):
        stub_forecast()
        ranked = {"OriginCongestion": 0.5, "DestCongestion": 0.3, "Distance": 0.2}
        return serve({k: replace(v, importance=ranked) for k, v in bundles.items()})

    def test_a_complete_request_reports_nothing(self, leaning_client, body):
        data = leaning_client.post("/predictions", json=body()).json()["data"]

        assert data["approximated"] == []

    def test_a_column_sent_as_null_is_named(self, leaning_client, body):
        data = leaning_client.post(
            "/predictions", json=body(OriginCongestion=None)
        ).json()["data"]

        assert data["approximated"] == ["OriginCongestion"]

    def test_the_flight_is_still_scored(self, leaning_client, body):
        data = leaning_client.post(
            "/predictions", json=body(OriginCongestion=None)
        ).json()["data"]

        assert 0.0 <= data["delay_probability"] <= 1.0

    def test_a_model_with_no_ranking_reports_nothing(self, client, body):
        data = client.post(
            "/predictions", json=body(OriginCongestion=None)
        ).json()["data"]

        assert data["approximated"] == []

    def test_every_row_of_a_batch_is_judged_on_its_own(self, leaning_client, body):
        complete = body()
        partial = body(DestCongestion=None)

        results = leaning_client.post(
            "/batch-predictions", json=[complete, partial]
        ).json()["data"]["results"]

        assert results[0]["approximated"] == []
        assert results[1]["approximated"] == ["DestCongestion"]


class TestPredictions:
    def test_a_valid_flight_is_scored(self, client, a_flight):
        response = client.post("/predictions", json=a_flight())

        assert response.status_code == 200
        data = response.json()["data"]
        assert 0.0 <= data["delay_probability"] <= 1.0
        assert data["is_delayed"] in (0, 1)

    def test_the_response_carries_the_common_envelope(self, client, a_flight):
        response = client.post("/predictions", json=a_flight()).json()

        assert set(response) >= {"message", "method", "status-code", "timestamp", "url", "data"}
        assert response["method"] == "POST"

    def test_the_answer_names_the_model_that_produced_it(self, client, a_flight):
        """A degraded answer must be recognisable as one."""
        data = client.post("/predictions", json=a_flight()).json()["data"]

        assert data["variant"] == "all"
        assert data["weather"] == "ok"

    def test_a_flight_beyond_the_horizon_is_answered_by_the_fallback(self, client, a_flight):
        data = client.post("/predictions", json=a_flight(days_ahead=40)).json()["data"]

        assert data["variant"] == "noweather"
        assert data["weather"] == "beyond_forecast_horizon"

    def test_the_request_is_echoed_back(self, client, a_flight):
        data = client.post("/predictions", json=a_flight()).json()["data"]

        assert data["input"]["Origin"] == "JFK"

    def test_the_released_threshold_is_reported(self, client, bundles, a_flight):
        data = client.post("/predictions", json=a_flight()).json()["data"]

        assert data["threshold"] == bundles["all"].threshold

    def test_the_threshold_can_be_overridden_per_request(self, client, a_flight):
        data = client.post("/predictions?threshold=0.99", json=a_flight()).json()["data"]

        assert data["threshold"] == 0.99
        assert data["is_delayed"] == 0

    def test_an_impossible_hour_is_refused(self, client, a_flight):
        response = client.post("/predictions", json=a_flight(DepTimeDecimal=25.5))

        assert response.status_code == 422

    def test_a_column_the_models_read_cannot_be_left_out(self, client, a_flight, bundles):
        payload = a_flight()
        read_by_a_model = sorted(required_inputs(bundles, CANDIDATE_INPUTS) - {"FlightDate"})[0]
        del payload[read_by_a_model]

        response = client.post("/predictions", json=payload)

        assert response.status_code == 422
        assert read_by_a_model in response.json()["detail"]

    def test_a_column_no_model_reads_can_be_left_out(self, client, a_flight, bundles):
        """Feature selection dropped it, so asking for it would be noise."""
        ignored = set(CANDIDATE_INPUTS) - required_inputs(bundles, CANDIDATE_INPUTS)
        payload = {k: v for k, v in a_flight().items() if k not in ignored}

        assert client.post("/predictions", json=payload).status_code == 200

    def test_a_flight_needing_an_unregistered_model_answers_503(self, degraded_client, a_flight):
        response = degraded_client.post("/predictions", json=a_flight(days_ahead=40))

        assert response.status_code == 503
        assert "noweather" in response.json()["detail"]


class TestBatchPredictions:
    def test_every_flight_comes_back_scored(self, client, a_flight):
        payload = [a_flight(), a_flight(days_ahead=3), a_flight(days_ahead=40)]

        data = client.post("/batch-predictions", json=payload).json()["data"]

        assert data["batch_size"] == 3
        assert [r["index"] for r in data["results"]] == [0, 1, 2]

    def test_the_batch_is_routed_flight_by_flight(self, client, a_flight):
        payload = [a_flight(), a_flight(days_ahead=40)]

        results = client.post("/batch-predictions", json=payload).json()["data"]["results"]

        assert [r["variant"] for r in results] == ["all", "noweather"]

    def test_an_empty_batch_is_refused(self, client):
        assert client.post("/batch-predictions", json=[]).status_code == 422

    def test_an_oversized_batch_is_refused(self, client, a_flight):
        payload = [a_flight()] * (MAX_BATCH_SIZE + 1)

        response = client.post("/batch-predictions", json=payload)

        assert response.status_code == 422
        assert str(MAX_BATCH_SIZE) in response.json()["detail"]


class TestStatus:
    """The root belongs to the interface now, so the service reports on /status."""

    def test_it_reports_the_models_being_served(self, client):
        data = client.get("/status").json()["data"]

        assert sorted(data["variants"]) == ["all", "noweather"]

    def test_the_root_serves_the_interface(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestLookupPredictions:
    """The auto-lookup path: the caller names a flight, the service recovers it."""

    @pytest.fixture
    def named_flight(self) -> dict:
        return {
            "FlightDate": (date.today() + timedelta(days=2)).isoformat(),
            "MarketingCarrier": "AA",
            "ReportingAirline": "MQ",
            "FlightNumber": 3500,
            "Origin": "DFW",
            "Dest": "LBB",
        }

    @pytest.fixture
    def resolved(self, monkeypatch, body):
        """Stand in for the schedule service, which the tests never call."""

        def build(error=None):
            def fake_resolve(lookups):
                if error is not None:
                    raise error
                return [FlightRequest(**body(base=date.today())) for _ in lookups]

            monkeypatch.setattr(
                "predicting_flight_arrival_delays.app.routers.prediction.resolve", fake_resolve
            )

        return build

    def test_the_answer_carries_its_reasons_when_asked(self, explainable, named_flight, resolved):
        """This is the endpoint a person reaches through the interface, and the
        interface asks for the reasons."""
        resolved()

        data = explainable.post(
            "/predictions/lookup?explain=true", json=named_flight
        ).json()["data"]

        assert data["explanations"]
        assert not any("_1" in item["column"] for item in data["explanations"])

    def test_the_reasons_are_left_out_unless_asked_for(self, explainable, named_flight, resolved):
        """A caller who only wants the probability should not pay for a second pass
        through the transformer."""
        resolved()

        data = explainable.post("/predictions/lookup", json=named_flight).json()["data"]

        assert "explanations" not in data

    def test_a_named_flight_is_scored(self, client, named_flight, resolved):
        resolved()

        response = client.post("/predictions/lookup", json=named_flight)

        assert response.status_code == 200
        assert 0.0 <= response.json()["data"]["delay_probability"] <= 1.0

    def test_the_answer_shows_what_was_recovered(self, client, named_flight, resolved):
        """A caller who sent six fields should be able to see the twenty-eight the
        model was actually given."""
        resolved()

        data = client.post("/predictions/lookup", json=named_flight).json()["data"]

        assert data["resolved"]["Origin"] == "JFK"
        assert "OriginCongestion" in data["resolved"]

    def test_an_unknown_flight_answers_404(self, client, named_flight, resolved):
        resolved(error=FlightNotFoundError("No flight AA3500 on that date."))

        response = client.post("/predictions/lookup", json=named_flight)

        assert response.status_code == 404
        assert "AA3500" in response.json()["detail"]

    def test_an_unknown_airport_answers_404(self, client, named_flight, resolved):
        resolved(error=UnknownAirportError("Unknown airport 'ZZZ'."))

        assert client.post("/predictions/lookup", json=named_flight).status_code == 404

    def test_an_unreachable_schedule_service_answers_502(self, client, named_flight, resolved):
        resolved(error=ScheduleUnavailableError("Schedule service unreachable"))

        response = client.post("/predictions/lookup", json=named_flight)

        assert response.status_code == 502

    def test_a_missing_field_is_refused_before_any_call(self, client, named_flight, resolved):
        resolved(error=AssertionError("the schedule service must not be called"))
        del named_flight["Dest"]

        assert client.post("/predictions/lookup", json=named_flight).status_code == 422

    def test_codes_are_uppercased(self, client, named_flight, resolved):
        resolved()
        named_flight |= {"Origin": "dfw", "MarketingCarrier": " aa "}

        assert client.post("/predictions/lookup", json=named_flight).status_code == 200
