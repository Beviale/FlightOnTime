"""Tests for predicting_flight_arrival_delays.utils."""

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from predicting_flight_arrival_delays import utils
from predicting_flight_arrival_delays.utils import (
    fetch,
    get_dvc_data_hash,
    get_git_dirty,
    safe_relative_path,
    to_pascal_case,
)


class TestToPascalCase:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("flight_date", "FlightDate"),
            ("temperature_2m", "Temperature2m"),
            ("wind_speed_10m", "WindSpeed10m"),
            ("Origin", "Origin"),
            ("ArrDel15", "ArrDel15"),
            ("Flight_Number_Reporting_Airline", "FlightNumberReportingAirline"),
        ],
    )
    def test_converts_names(self, name, expected):
        """Snake_case and already-PascalCase names both come out PascalCase."""
        assert to_pascal_case(name) == expected

    def test_preserves_inner_capitals(self):
        """Only the first letter of each part is touched, the rest is left alone."""
        assert to_pascal_case("crsDepTime") == "CrsDepTime"

    def test_empty_parts_are_dropped(self):
        """Leading/double underscores produce empty parts, which contribute nothing."""
        assert to_pascal_case("__origin__state") == "OriginState"

    def test_empty_string(self):
        assert to_pascal_case("") == ""


class TestSafeRelativePath:
    def test_path_under_cwd_becomes_relative(self, tmp_path, monkeypatch):
        """A path inside the working directory is shortened for readability."""
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "metrics" / "all" / "lightgbm.json"

        assert safe_relative_path(target) == str(Path("metrics") / "all" / "lightgbm.json")

    def test_path_outside_cwd_stays_absolute(self, tmp_path, monkeypatch):
        """A path elsewhere on disk cannot be relativised, so it is returned as-is."""
        working = tmp_path / "repo"
        working.mkdir()
        monkeypatch.chdir(working)
        outside = tmp_path / "elsewhere" / "model.joblib"

        assert safe_relative_path(outside) == str(outside)


class TestFetch:
    class _Response:
        def __init__(self, payload, error=None):
            self.payload = payload
            self.error = error

        def raise_for_status(self):
            if self.error is not None:
                raise self.error

        def json(self):
            return self.payload

    def test_returns_decoded_json(self, monkeypatch):
        """A successful call returns the parsed body and forwards the parameters."""
        seen = {}

        def fake_get(url, params, timeout):
            seen.update(url=url, params=params, timeout=timeout)
            return self._Response({"hourly": {"time": []}})

        monkeypatch.setattr(utils.requests, "get", fake_get)

        assert fetch("https://example.test/v1", {"latitude": 1.0}) == {"hourly": {"time": []}}
        assert seen["url"] == "https://example.test/v1"
        assert seen["params"] == {"latitude": 1.0}
        assert seen["timeout"] == 180

    def test_custom_timeout_is_forwarded(self, monkeypatch):
        seen = {}

        def fake_get(url, params, timeout):
            seen["timeout"] = timeout
            return self._Response({})

        monkeypatch.setattr(utils.requests, "get", fake_get)
        fetch("https://example.test/v1", {}, timeout=5)

        assert seen["timeout"] == 5

    def test_http_error_propagates(self, monkeypatch):
        """Error statuses are raised, never swallowed into an empty payload."""

        def fake_get(url, params, timeout):
            return self._Response(None, error=utils.requests.HTTPError("500"))

        monkeypatch.setattr(utils.requests, "get", fake_get)

        with pytest.raises(utils.requests.HTTPError):
            fetch("https://example.test/v1", {})


class TestGetDvcDataHash:
    @pytest.fixture
    def dvc_lock(self, tmp_path) -> Path:
        lock = {
            "stages": {
                "split": {
                    "cmd": "python -m ...",
                    "outs": [
                        {"path": "data/processed/selection", "md5": "abc123.dir"},
                        {"path": "data/interim/flights.parquet", "md5": "def456"},
                    ],
                }
            }
        }
        path = tmp_path / "dvc.lock"
        path.write_text(yaml.safe_dump(lock))
        return path

    def test_returns_recorded_hash(self, dvc_lock):
        """The md5 recorded for the requested output is returned verbatim."""
        assert get_dvc_data_hash("data/processed/selection", dvc_lock) == "abc123.dir"

    def test_finds_output_of_any_stage(self, dvc_lock):
        assert get_dvc_data_hash("data/interim/flights.parquet", dvc_lock) == "def456"

    def test_unknown_path_returns_not_found(self, dvc_lock):
        """An output DVC does not track is reported, not raised."""
        assert get_dvc_data_hash("data/processed/nope", dvc_lock) == "not_found"

    def test_missing_lock_file_returns_not_found(self, tmp_path):
        """A missing dvc.lock must not break a training run."""
        assert get_dvc_data_hash("data/processed/selection", tmp_path / "absent.lock") == (
            "not_found"
        )

    def test_a_relative_path_object_is_accepted(self, dvc_lock):
        """Both production callers pass a Path, not the string dvc.lock records."""
        assert get_dvc_data_hash(Path("data/processed/selection"), dvc_lock) == "abc123.dir"

    def test_an_absolute_path_is_resolved_against_the_lock_file(self, dvc_lock):
        """PROCESSED_DATA_DIR is absolute; dvc.lock records repo-relative paths."""
        absolute = dvc_lock.parent / "data" / "processed" / "selection"

        assert get_dvc_data_hash(absolute, dvc_lock) == "abc123.dir"

    def test_a_path_outside_the_repository_is_not_found(self, dvc_lock, tmp_path):
        outside = tmp_path.parent / "somewhere-else" / "selection"

        assert get_dvc_data_hash(outside, dvc_lock) == "not_found"


class TestGetGitDirty:
    def test_true_when_there_are_uncommitted_changes(self, monkeypatch):
        monkeypatch.setattr(
            utils.subprocess, "check_output", lambda cmd: b" M predicting/config.py\n"
        )
        assert get_git_dirty() is True

    def test_false_on_a_clean_tree(self, monkeypatch):
        monkeypatch.setattr(utils.subprocess, "check_output", lambda cmd: b"")
        assert get_git_dirty() is False

    def test_whitespace_only_output_is_clean(self, monkeypatch):
        monkeypatch.setattr(utils.subprocess, "check_output", lambda cmd: b"\n  \n")
        assert get_git_dirty() is False

    def test_none_when_git_is_unavailable(self, monkeypatch):
        def boom(cmd):
            raise subprocess.CalledProcessError(128, cmd)

        monkeypatch.setattr(utils.subprocess, "check_output", boom)
        assert get_git_dirty() is None


class _FakeModelInfo:
    registered_model_version = "7"


class _FakeClient:
    aliases: list[tuple] = []

    def set_registered_model_alias(self, name, alias, version):
        _FakeClient.aliases.append((name, alias, version))

    def get_model_version_by_alias(self, name, alias):
        return type("V", (), {"run_id": f"run-for-{alias}"})()

    def search_model_versions(self, filter_string):
        return [
            type("V", (), {"version": "2", "run_id": "run-2"})(),
            type("V", (), {"version": "10", "run_id": "run-10"})(),
        ]


class TestRegisterModelBundle:
    """register_model_bundle logs model, transformer and columns as one unit."""

    @pytest.fixture
    def fake_mlflow(self, monkeypatch):
        logged = {"artifacts": [], "columns": None}

        class FakeSklearn:
            @staticmethod
            def log_model(model, artifact_path, signature, input_example, registered_model_name):
                logged["registered_model_name"] = registered_model_name
                logged["artifact_path"] = artifact_path
                logged["signature"] = signature
                return _FakeModelInfo()

        class FakeMlflow:
            sklearn = FakeSklearn
            MlflowClient = _FakeClient

            @staticmethod
            def log_artifact(local_path, artifact_path):
                name = Path(local_path).name
                logged["artifacts"].append((name, artifact_path))
                if name == "columns.json":
                    logged["columns"] = json.loads(Path(local_path).read_text())

        _FakeClient.aliases = []
        monkeypatch.setattr(utils, "mlflow", FakeMlflow)
        return logged

    @pytest.fixture
    def transformer(self):
        class RecordingTransformer:
            def __init__(self):
                self.saved_to = None

            def save(self, path):
                self.saved_to = path
                path.write_bytes(b"transformer-state")

        return RecordingTransformer()

    def test_logs_transformer_and_columns_beside_the_model(
        self, fake_mlflow, transformer
    ):
        """Everything needed to prepare data lands under the model's artifact path."""
        utils.register_model_bundle(
            model=object(),
            transformer=transformer,
            columns=["Origin", "Dest", "Distance"],
            registered_model_name="flight-delay-all",
        )

        assert fake_mlflow["registered_model_name"] == "flight-delay-all"
        assert set(fake_mlflow["artifacts"]) == {
            ("transformer.joblib", "model"),
            ("columns.json", "model"),
        }
        assert fake_mlflow["columns"] == ["Origin", "Dest", "Distance"]
        assert transformer.saved_to is not None

    def test_no_signature_without_a_sample(self, fake_mlflow, transformer):
        """Skipping the sample skips signature inference rather than failing."""
        utils.register_model_bundle(
            model=object(),
            transformer=transformer,
            columns=["Distance"],
            registered_model_name="flight-delay-noweather",
        )
        assert fake_mlflow["signature"] is None

    def test_alias_is_promoted_when_given(self, fake_mlflow, transformer):
        utils.register_model_bundle(
            model=object(),
            transformer=transformer,
            columns=["Distance"],
            registered_model_name="flight-delay-all",
            alias="champion",
        )
        assert _FakeClient.aliases == [("flight-delay-all", "champion", "7")]

    def test_no_alias_left_unpromoted(self, fake_mlflow, transformer):
        utils.register_model_bundle(
            model=object(),
            transformer=transformer,
            columns=["Distance"],
            registered_model_name="flight-delay-all",
        )
        assert _FakeClient.aliases == []

    def test_custom_artifact_path_is_used_for_all_three(self, fake_mlflow, transformer):
        utils.register_model_bundle(
            model=object(),
            transformer=transformer,
            columns=["Distance"],
            registered_model_name="flight-delay-all",
            artifact_path="bundle",
        )
        assert fake_mlflow["artifact_path"] == "bundle"
        assert {p for _, p in fake_mlflow["artifacts"]} == {"bundle"}


class TestRegisterModelBundleSignature:
    @pytest.fixture
    def logged(self, monkeypatch):
        record = {}

        class FakeSklearn:
            @staticmethod
            def log_model(model, artifact_path, signature, input_example, registered_model_name):
                record["signature"] = signature
                record["input_example"] = input_example
                return _FakeModelInfo()

        class FakeMlflow:
            sklearn = FakeSklearn
            MlflowClient = _FakeClient

            @staticmethod
            def log_artifact(local_path, artifact_path):
                pass

        monkeypatch.setattr(utils, "mlflow", FakeMlflow)
        monkeypatch.setattr(
            utils, "infer_signature", lambda X, y: {"inputs": list(X.columns)}
        )
        return record

    @pytest.fixture
    def transformer(self):
        class RecordingTransformer:
            def save(self, path):
                path.write_bytes(b"state")

        return RecordingTransformer()

    class _Model:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    def test_the_signature_is_inferred_from_the_sample(self, logged, transformer):
        """The registry stores the model's input/output shape for validation."""
        sample = pd.DataFrame({"Distance": [1.0, 2.0], "Origin_ATL": [1, 0]})

        utils.register_model_bundle(
            model=self._Model(),
            transformer=transformer,
            columns=list(sample.columns),
            registered_model_name="flight-delay-all",
            signature_sample=sample,
        )

        assert logged["signature"] == {"inputs": ["Distance", "Origin_ATL"]}

    def test_the_input_example_is_capped_at_five_rows(self, logged, transformer):
        sample = pd.DataFrame({"Distance": np.arange(50, dtype=float)})

        utils.register_model_bundle(
            model=self._Model(),
            transformer=transformer,
            columns=["Distance"],
            registered_model_name="flight-delay-all",
            signature_sample=sample,
        )

        assert len(logged["input_example"]) == 5


class TestLoadModelBundle:
    @pytest.fixture
    def fake_registry(self, monkeypatch, tmp_path):
        transformer_path = tmp_path / "transformer.joblib"
        transformer_path.write_bytes(b"state")
        columns_path = tmp_path / "columns.json"
        columns_path.write_text(json.dumps(["Distance", "Origin_ATL"]))

        requested = []

        class FakeArtifacts:
            @staticmethod
            def download_artifacts(uri):
                requested.append(uri)
                name = uri.rsplit("/", 1)[-1]
                return str(transformer_path if name == "transformer.joblib" else columns_path)

        class FakeSklearn:
            @staticmethod
            def load_model(uri):
                requested.append(uri)
                return f"model@{uri}"

        class FakeMlflow:
            sklearn = FakeSklearn
            artifacts = FakeArtifacts
            MlflowClient = _FakeClient

        monkeypatch.setattr(utils, "mlflow", FakeMlflow)
        monkeypatch.setattr(utils.joblib, "load", lambda path: "restored-transformer")
        return requested

    def test_all_four_pieces_come_back(self, fake_registry):
        model, transformer, columns, run_id = utils.load_model_bundle(
            "flight-delay-all", stage="champion"
        )

        assert model == "model@runs:/run-for-champion/model"
        assert transformer == "restored-transformer"
        assert columns == ["Distance", "Origin_ATL"]
        assert run_id == "run-for-champion"

    def test_everything_is_read_from_the_same_run(self, fake_registry):
        """The bundle is self-contained: one run holds all three artifacts."""
        utils.load_model_bundle("flight-delay-all", stage="champion")

        assert all("runs:/run-for-champion/" in uri for uri in fake_registry)

    def test_the_latest_version_is_used_without_a_stage(self, fake_registry):
        _, _, _, run_id = utils.load_model_bundle("flight-delay-all")

        assert run_id == "run-10"

    def test_a_custom_artifact_path_is_honoured(self, fake_registry):
        utils.load_model_bundle("flight-delay-all", stage="champion", artifact_path="bundle")

        assert any(uri.endswith("/bundle") for uri in fake_registry)

    def test_nothing_registered_is_reported(self, monkeypatch):
        class EmptyClient(_FakeClient):
            def search_model_versions(self, filter_string):
                return []

        monkeypatch.setattr(utils, "mlflow", type("M", (), {"MlflowClient": EmptyClient}))

        with pytest.raises(FileNotFoundError):
            utils.load_model_bundle("flight-delay-all")


class TestGetRunParams:
    def test_the_logged_parameters_come_back(self, monkeypatch):
        class FakeClient:
            def get_run(self, run_id):
                return type(
                    "R", (), {"data": type("D", (), {"params": {"variant": "noweather"}})()}
                )()

        monkeypatch.setattr(utils, "mlflow", type("M", (), {"MlflowClient": FakeClient}))

        assert utils.get_run_params("run-1") == {"variant": "noweather"}


class TestResolveRunId:
    @pytest.fixture
    def fake_mlflow(self, monkeypatch):
        class FakeMlflow:
            MlflowClient = _FakeClient

        monkeypatch.setattr(utils, "mlflow", FakeMlflow)

    def test_alias_is_resolved_through_the_registry(self, fake_mlflow):
        assert utils._resolve_run_id("flight-delay-all", "champion") == "run-for-champion"

    @pytest.mark.parametrize("stage", ["None", ""])
    def test_latest_version_wins_when_no_stage_given(self, fake_mlflow, stage):
        """Versions are compared numerically, so 10 beats 2."""
        assert utils._resolve_run_id("flight-delay-all", stage) == "run-10"

    def test_raises_when_nothing_is_registered(self, monkeypatch):
        class EmptyClient(_FakeClient):
            def search_model_versions(self, filter_string):
                return []

        monkeypatch.setattr(utils, "mlflow", type("M", (), {"MlflowClient": EmptyClient}))

        with pytest.raises(FileNotFoundError, match="No registered versions"):
            utils._resolve_run_id("flight-delay-all", "None")
