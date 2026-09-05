"""Tests for the endpoints describing the served models."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from predicting_flight_arrival_delays.app.inputs import (
    CANDIDATE_INPUTS,
    FORECAST_INPUTS,
    required_inputs,
)
from predicting_flight_arrival_delays.app.main import app
from predicting_flight_arrival_delays.app.utils import apply_bundles
from predicting_flight_arrival_delays.app.routers import model_info

TOKEN = "a-shared-secret"


@pytest.fixture
def client(bundles) -> TestClient:
    """A client serving both variants, as the lifespan would have left it."""
    app.state.bundles = bundles
    app.state.required_inputs = required_inputs(bundles, CANDIDATE_INPUTS)
    return TestClient(app)


@pytest.fixture
def bare_client() -> TestClient:
    """A client whose registry gave it nothing."""
    app.state.bundles = {}
    app.state.required_inputs = set()
    return TestClient(app)


class TestHyperparameters:
    def test_every_served_variant_is_described(self, client):
        data = client.get("/model/hyperparameters").json()["data"]

        assert sorted(data) == ["all", "noweather"]

    def test_the_estimator_settings_are_separated_from_the_run_setup(self, client):
        data = client.get("/model/hyperparameters").json()["data"]["all"]

        assert data["hyperparameters"] == {"learning_rate": "0.05", "num_leaves": "31"}
        assert data["training"]["algorithm"] == "lightgbm"
        assert "hp_learning_rate" not in data["training"]

    def test_the_run_that_produced_the_model_is_named(self, client, bundles):
        data = client.get("/model/hyperparameters").json()["data"]["all"]

        assert data["run_id"] == bundles["all"].run_id

    def test_one_variant_can_be_asked_for(self, client):
        data = client.get("/model/hyperparameters?variant=noweather").json()["data"]

        assert list(data) == ["noweather"]

    def test_a_variant_that_is_not_served_answers_404(self, client):
        response = client.get("/model/hyperparameters?variant=nocarrier")

        assert response.status_code == 404
        assert "nocarrier" in response.json()["detail"]

    def test_with_no_model_loaded_there_is_nothing_to_describe(self, bare_client):
        assert bare_client.get("/model/hyperparameters").status_code == 503


class TestMetrics:
    def test_the_release_metrics_are_reported(self, client):
        data = client.get("/model/metrics").json()["data"]["all"]

        assert data["metrics"]["pr_auc"] == 0.42
        assert data["metrics"]["roc_auc"] == 0.71

    def test_the_operating_threshold_is_reported_alongside_them(self, client, bundles):
        """It was chosen from those same metrics, and decides every label served."""
        data = client.get("/model/metrics").json()["data"]["all"]

        assert data["operating_threshold"] == bundles["all"].threshold

    def test_each_variant_reports_its_own(self, client, bundles):
        data = client.get("/model/metrics").json()["data"]

        assert data["all"]["operating_threshold"] != data["noweather"]["operating_threshold"]
        assert data["noweather"]["operating_threshold"] == bundles["noweather"].threshold

    def test_with_no_model_loaded_there_is_nothing_to_report(self, bare_client):
        assert bare_client.get("/model/metrics").status_code == 503


class TestInputs:
    def test_it_lists_what_a_request_must_carry(self, client, bundles):
        data = client.get("/model/inputs").json()["data"]

        assert set(data["required"]) == required_inputs(bundles, CANDIDATE_INPUTS)

    def test_required_and_ignored_together_are_the_whole_request(self, client):
        data = client.get("/model/inputs").json()["data"]

        assert set(data["required"]) | set(data["ignored"]) == set(CANDIDATE_INPUTS)
        assert not set(data["required"]) & set(data["ignored"])

    def test_it_names_what_the_service_adds_by_itself(self, client):
        data = client.get("/model/inputs").json()["data"]

        assert "LeadDays" in data["supplied_by_the_service"]
        assert "Temperature2mOrigin" in data["supplied_by_the_service"]

    def test_it_answers_even_with_no_model_loaded(self, bare_client):
        """Unlike the others: a caller can still be told what the request looks like."""
        response = bare_client.get("/model/inputs")

        assert response.status_code == 200
        assert response.json()["data"]["variants"] == []

    def test_the_contract_says_it_comes_from_the_served_models(self, client):
        data = client.get("/model/inputs").json()["data"]

        assert data["derived_from_served_models"] is True

    def test_with_no_model_the_contract_says_it_does_not(self, bare_client):
        data = bare_client.get("/model/inputs").json()["data"]

        assert data["derived_from_served_models"] is False

    def test_with_no_model_the_request_is_only_what_the_forecast_needs(self):
        apply_bundles(app, {})
        data = TestClient(app).get("/model/inputs").json()["data"]

        assert set(data["required"]) == FORECAST_INPUTS & set(CANDIDATE_INPUTS)
        assert data["derived_from_served_models"] is False


class TestReload:
    """Putting a newly registered version into service without a restart."""

    @pytest.fixture
    def token(self, monkeypatch):
        monkeypatch.setenv(model_info.RELOAD_TOKEN_VARIABLE, TOKEN)

    @pytest.fixture
    def registry(self, monkeypatch):
        """Stand in for the registry, which the tests never reach."""

        def build(bundles):
            monkeypatch.setattr(model_info, "load_bundles", lambda: bundles)

        return build

    def test_a_new_version_replaces_the_one_in_service(self, client, token, registry, bundles):
        newer = bundles | {"all": replace(bundles["all"], run_id="run-all-v2")}
        registry(newer)

        data = client.post("/model/reload", headers={"X-Reload-Token": TOKEN}).json()["data"]

        assert data["changed"] == {"all": {"was": "run-all", "now": "run-all-v2"}}
        assert app.state.bundles["all"].run_id == "run-all-v2"

    def test_an_unchanged_version_is_reported_as_unchanged(self, client, token, registry, bundles):
        registry(bundles)

        data = client.post("/model/reload", headers={"X-Reload-Token": TOKEN}).json()["data"]

        assert data["changed"] == {}
        assert data["serving"] == ["all", "noweather"]

    def test_the_request_contract_is_recomputed_with_the_models(
        self, client, token, registry, bundles
    ):
        """A new version selects its own features, so the columns a request must carry
        move with it."""
        registry({"all": bundles["all"]})

        data = client.post("/model/reload", headers={"X-Reload-Token": TOKEN}).json()["data"]

        assert set(data["required_inputs"]) == required_inputs(
            {"all": bundles["all"]}, CANDIDATE_INPUTS
        )
        assert set(data["required_inputs"]) == app.state.required_inputs

    def test_a_variant_the_registry_no_longer_carries_is_reported(
        self, client, token, registry, bundles
    ):
        registry({"all": bundles["all"]})

        data = client.post("/model/reload", headers={"X-Reload-Token": TOKEN}).json()["data"]

        assert data["dropped"] == ["noweather"]

    def test_an_empty_registry_does_not_take_the_service_down(
        self, client, token, registry, bundles
    ):
        """A briefly unreachable registry must not turn a failed deployment into an
        outage."""
        registry({})

        response = client.post("/model/reload", headers={"X-Reload-Token": TOKEN})

        assert response.status_code == 502
        assert app.state.bundles == bundles

    def test_a_wrong_token_is_refused(self, client, token, registry, bundles):
        registry(bundles)

        response = client.post("/model/reload", headers={"X-Reload-Token": "guess"})

        assert response.status_code == 401

    def test_a_token_with_an_accent_is_refused_not_crashed(self, client, monkeypatch, registry, bundles):
        monkeypatch.setenv(model_info.RELOAD_TOKEN_VARIABLE, "segreto-àè")
        registry(bundles)

        assert client.post("/model/reload", headers={"X-Reload-Token": "guess"}).status_code == 401

    def test_a_missing_token_is_refused(self, client, token, registry, bundles):
        registry(bundles)

        assert client.post("/model/reload").status_code == 401

    def test_with_no_secret_configured_the_endpoint_is_off_not_open(
        self, client, monkeypatch, registry, bundles
    ):
        """An endpoint that downloads models must not default to being reachable."""
        monkeypatch.delenv(model_info.RELOAD_TOKEN_VARIABLE, raising=False)
        registry(bundles)

        response = client.post("/model/reload", headers={"X-Reload-Token": TOKEN})

        assert response.status_code == 503
        assert model_info.RELOAD_TOKEN_VARIABLE in response.json()["detail"]
