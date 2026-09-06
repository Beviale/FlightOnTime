"""Tests for predicting_flight_arrival_delays.modeling.train."""

from pathlib import Path

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
import yaml

from predicting_flight_arrival_delays.config import DATE_COLUMN, SEED
from predicting_flight_arrival_delays.data.transform import Transformer
from predicting_flight_arrival_delays.modeling import train as train_module
from predicting_flight_arrival_delays.modeling.train import (
    BUILDERS,
    HYPERPARAMS,
    _build_model,
    _load_hyperparams,
    train,
    train_with_transformer,
)


@pytest.fixture
def xy():
    """A small, clearly separable numeric problem: fast and deterministic.

    Classes alternate row by row, so any contiguous slice used as a validation
    split still contains both of them.
    """
    rng = np.random.default_rng(0)
    labels = np.tile([0, 1], 60)
    X = pd.DataFrame(
        {
            "signal": np.where(labels == 1, 1.0, -1.0) + rng.normal(0, 0.5, 120),
            "noise": rng.normal(0, 1, 120),
        }
    )
    return X, pd.Series(labels)


@pytest.fixture
def small_transformer(monkeypatch):
    """Make the Transformer keep categories: the real floor is 1000 rows."""

    def build(**kwargs):
        return Transformer(min_category_count=5, **kwargs)

    monkeypatch.setattr(train_module, "Transformer", build)


class TestLoadHyperparams:
    def test_missing_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No hyperparameters file"):
            _load_hyperparams(tmp_path / "absent.yaml")

    def test_seed_is_injected_into_every_config(self, tmp_path):
        """Every run must be reproducible, whatever the YAML says."""
        path = tmp_path / "hp.yaml"
        path.write_text(yaml.safe_dump({"logistic_regression": {"default": {"max_iter": 100}}}))

        loaded = _load_hyperparams(path)

        assert loaded["logistic_regression"]["default"]["random_state"] == SEED

    def test_declared_values_are_kept(self, tmp_path):
        path = tmp_path / "hp.yaml"
        path.write_text(yaml.safe_dump({"lightgbm": {"fast": {"n_estimators": 42}}}))

        assert _load_hyperparams(path)["lightgbm"]["fast"]["n_estimators"] == 42

    def test_the_shipped_file_covers_every_builder(self):
        assert set(HYPERPARAMS) == set(BUILDERS)

    def test_every_shipped_config_is_seeded(self):
        for configs in HYPERPARAMS.values():
            for params in configs.values():
                assert params["random_state"] == SEED


class TestBuildModel:
    @pytest.mark.parametrize(
        "algorithm,expected",
        [
            ("logistic_regression", LogisticRegression),
            ("random_forest", RandomForestClassifier),
            ("lightgbm", LGBMClassifier),
        ],
    )
    def test_builds_the_right_estimator(self, algorithm, expected):
        model, _ = _build_model(algorithm, "default")

        assert isinstance(model, expected)

    def test_lightgbm_reports_its_early_stopping_rounds(self):
        _, rounds = _build_model("lightgbm", "default")

        assert rounds == HYPERPARAMS["lightgbm"]["default"]["early_stopping_rounds"]

    def test_other_algorithms_never_get_early_stopping(self):
        """Only LightGBM knows what to do with it."""
        _, rounds = _build_model(
            "logistic_regression", "default", {"max_iter": 50, "early_stopping_rounds": 10}
        )

        assert rounds is None

    def test_early_stopping_is_not_passed_to_the_constructor(self):
        """LGBMClassifier has no such parameter; leaving it in would raise."""
        model, _ = _build_model("lightgbm", "default")

        assert "early_stopping_rounds" not in model.get_params()

    def test_json_config_overrides_the_named_one(self):
        model, _ = _build_model("logistic_regression", "default", {"max_iter": 7})

        assert model.max_iter == 7

    def test_the_shared_hyperparameter_table_is_not_mutated(self):
        """_build_model pops from a copy, so repeated calls stay identical."""
        _build_model("lightgbm", "default")

        assert "early_stopping_rounds" in HYPERPARAMS["lightgbm"]["default"]

    def test_unknown_algorithm_raises(self):
        with pytest.raises(KeyError):
            _build_model("xgboost", "default")


class TestTrain:
    def test_uncalibrated_returns_the_bare_estimator(self, xy):
        X, y = xy
        estimator = train(X, y, "logistic_regression", "default", calibrate=False)

        assert isinstance(estimator, LogisticRegression)

    def test_the_fitted_model_learns_the_signal(self, xy):
        X, y = xy
        estimator = train(X, y, "logistic_regression", "default", calibrate=False)

        assert estimator.score(X, y) > 0.9

    def test_calibration_on_a_validation_set_freezes_the_base_model(self, xy):
        """Refitting during calibration would waste the fit and leak the split."""
        X, y = xy
        estimator = train(
            X[:80],
            y[:80],
            "logistic_regression",
            "default",
            calibrate=True,
            X_val=X[80:],
            y_val=y[80:],
        )

        assert isinstance(estimator, CalibratedClassifierCV)
        assert isinstance(estimator.calibrated_classifiers_[0].estimator, FrozenEstimator)

    def test_calibration_without_a_validation_set_uses_internal_cv(self, xy):
        X, y = xy
        estimator = train(X, y, "logistic_regression", "default", calibrate=True)

        assert isinstance(estimator, CalibratedClassifierCV)
        assert len(estimator.calibrated_classifiers_) == 5

    def test_calibrated_probabilities_stay_in_range(self, xy):
        X, y = xy
        estimator = train(X, y, "logistic_regression", "default", calibrate=True)
        proba = estimator.predict_proba(X)[:, 1]

        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_json_config_reaches_the_estimator(self, xy):
        X, y = xy
        estimator = train(
            X,
            y,
            "logistic_regression",
            "default",
            calibrate=False,
            json_config={"max_iter": 33, "random_state": SEED},
        )

        assert estimator.max_iter == 33

    def test_training_is_reproducible(self, xy):
        X, y = xy
        first = train(X, y, "random_forest", "shallow", calibrate=False).predict_proba(X)
        second = train(X, y, "random_forest", "shallow", calibrate=False).predict_proba(X)

        np.testing.assert_allclose(first, second)

    def test_lightgbm_trains_with_a_validation_set(self, xy):
        X, y = xy
        estimator = train(
            X[:80],
            y[:80],
            "lightgbm",
            "fast",
            calibrate=False,
            X_val=X[80:],
            y_val=y[80:],
        )

        assert isinstance(estimator, LGBMClassifier)
        assert estimator.predict_proba(X).shape == (120, 2)

    def test_lightgbm_early_stopping_really_fires(self, xy):
        """LightGBM 4.7 takes eval_X/eval_y and deprecates eval_set. It also
        swallows unknown fit kwargs, so a name that stops matching the API
        would silently train all n_estimators instead of raising.
        """
        X, y = xy
        estimator = train(
            X[:80],
            y[:80],
            "lightgbm",
            "fast",
            calibrate=False,
            json_config={
                "n_estimators": 500,
                "learning_rate": 0.1,
                "verbose": -1,
                "random_state": SEED,
                "early_stopping_rounds": 5,
            },
            X_val=X[80:],
            y_val=y[80:],
        )

        # evals_result_ is populated only when a validation set actually reached fit().
        assert estimator.evals_result_
        assert 0 < estimator.best_iteration_ < 500

    def test_lightgbm_without_a_validation_set_runs_to_the_end(self, xy):
        """The contrast case: no validation set means nothing to stop against."""
        X, y = xy
        estimator = train(
            X,
            y,
            "lightgbm",
            "fast",
            calibrate=False,
            json_config={
                "n_estimators": 20,
                "verbose": -1,
                "random_state": SEED,
                "early_stopping_rounds": 5,
            },
        )

        assert not estimator.evals_result_


class TestTrainWithTransformer:
    def test_returns_transformer_estimator_features_and_columns(
        self, flights_df, small_transformer
    ):
        transformer, estimator, X_fit, columns = train_with_transformer(
            flights_df, "noweather", "logistic_regression", "default", calibrate=False
        )

        assert isinstance(transformer, Transformer)
        assert isinstance(estimator, LogisticRegression)
        assert X_fit.shape[0] == len(flights_df)
        assert len(columns) == X_fit.shape[1]

    def test_onehot_features_reach_the_model_as_a_sparse_matrix(
        self, flights_df, small_transformer
    ):
        _, _, X_fit, _ = train_with_transformer(
            flights_df, "noweather", "logistic_regression", "default", calibrate=False
        )

        assert sp.issparse(X_fit)

    def test_native_features_stay_a_dataframe(self, flights_df, small_transformer):
        """LightGBM reads category dtype directly, so nothing is converted."""
        _, _, X_fit, _ = train_with_transformer(
            flights_df, "noweather", "lightgbm", "fast", calibrate=False
        )

        assert isinstance(X_fit, pd.DataFrame)

    def test_the_transformer_matches_the_model_encoding(self, flights_df, small_transformer):
        """Scikit-learn cannot read string categories; LightGBM can."""
        onehot, _, _, _ = train_with_transformer(
            flights_df, "noweather", "logistic_regression", "default", calibrate=False
        )
        native, _, _, _ = train_with_transformer(
            flights_df, "noweather", "lightgbm", "fast", calibrate=False
        )

        assert onehot.encoding == "onehot"
        assert native.encoding == "native"

    def test_the_date_never_reaches_the_model(self, flights_df, small_transformer):
        _, _, _, columns = train_with_transformer(
            flights_df, "noweather", "logistic_regression", "default", calibrate=False
        )

        assert DATE_COLUMN not in columns

    def test_the_columns_describe_the_matrix_the_model_was_fitted_on(
        self, flights_df, small_transformer
    ):
        """The matrix carries only positions, so this list is the only record of
        what each of them means. train.run() writes it to columns.json."""
        _, _, X_fit, columns = train_with_transformer(
            flights_df, "noweather", "logistic_regression", "default", calibrate=False
        )

        assert len(columns) == X_fit.shape[1]
        assert any(c.startswith("ReportingAirline_") for c in columns)
        assert "Origin" not in columns

    def test_validation_frame_is_aligned_to_training(self, flights_df, small_transformer):
        train_df, validation_df = flights_df.iloc[:200], flights_df.iloc[200:]

        _, _, X_fit, columns = train_with_transformer(
            train_df,
            "noweather",
            "logistic_regression",
            "default",
            calibrate=False,
            evaluation_df=validation_df,
        )

        assert X_fit.shape[0] == len(train_df)
        assert len(columns) == X_fit.shape[1]

    def test_unknown_variant_is_rejected(self, flights_df, small_transformer):
        with pytest.raises(ValueError, match="Unknown variant"):
            train_with_transformer(
                flights_df, "nope", "logistic_regression", "default", calibrate=False
            )


class TestRunCommandGuards:
    @pytest.fixture
    def train_path(self, tmp_path, flights_df):
        path = tmp_path / "train.parquet"
        flights_df.to_parquet(path, index=False)
        return path

    def _invoke(self, tmp_path, **overrides):
        from typer.testing import CliRunner

        options = {
            "--variant": "noweather",
            "--model": "logistic_regression",
            "--config": "default",
            "--models-path": str(tmp_path / "models"),
        }
        options.update(overrides)
        args = [item for pair in options.items() for item in pair]
        return CliRunner().invoke(train_module.app, args)

    def test_unknown_variant(self, tmp_path, train_path):
        result = self._invoke(tmp_path, **{"--variant": "nope", "--train-path": str(train_path)})

        assert result.exit_code == 1

    def test_unknown_config(self, tmp_path, train_path):
        result = self._invoke(tmp_path, **{"--config": "turbo", "--train-path": str(train_path)})

        assert result.exit_code == 1

    def test_empty_training_file(self, tmp_path, flights_df):
        empty = tmp_path / "empty.parquet"
        flights_df.head(0).to_parquet(empty, index=False)

        result = self._invoke(tmp_path, **{"--train-path": str(empty)})

        assert result.exit_code == 1

    def test_missing_validation_file(self, tmp_path, train_path):
        result = self._invoke(
            tmp_path,
            **{
                "--train-path": str(train_path),
                "--validation-path": str(tmp_path / "absent.parquet"),
            },
        )

        assert result.exit_code == 1

    def test_empty_validation_file(self, tmp_path, train_path, flights_df):
        empty = tmp_path / "empty_val.parquet"
        flights_df.head(0).to_parquet(empty, index=False)

        result = self._invoke(
            tmp_path, **{"--train-path": str(train_path), "--validation-path": str(empty)}
        )

        assert result.exit_code == 1


class TestRunCommandRegistration:
    """The optional MLflow half of train.run(), independent of local saving."""

    @pytest.fixture
    def tracking(self, monkeypatch):
        recorded = {"params": {}, "bundles": [], "experiments": [], "runs": []}

        class FakeRun:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class FakeMlflow:
            @staticmethod
            def set_experiment(name):
                recorded["experiments"].append(name)

            @staticmethod
            def start_run(run_name=None):
                recorded["runs"].append(run_name)
                return FakeRun()

            @staticmethod
            def log_params(params):
                recorded["params"].update(params)

        monkeypatch.setattr(train_module.dagshub, "init", lambda **kwargs: None)
        monkeypatch.setattr(train_module, "mlflow", FakeMlflow)
        monkeypatch.setattr(
            train_module, "register_model_bundle", lambda **kw: recorded["bundles"].append(kw)
        )
        return recorded

    @pytest.fixture
    def invoke(self, tmp_path, flights_df, small_transformer):
        from typer.testing import CliRunner

        path = tmp_path / "train.parquet"
        flights_df.to_parquet(path, index=False)

        def run(*extra):
            return CliRunner().invoke(
                train_module.app,
                [
                    "--variant",
                    "noweather",
                    "--model",
                    "logistic_regression",
                    "--train-path",
                    str(path),
                    "--models-path",
                    str(tmp_path / "models"),
                    "--no-calibrate",
                    *extra,
                ],
            )

        return run

    def test_nothing_is_tracked_without_an_experiment(self, invoke, tracking):
        """Local saving and MLflow registration are independent switches."""
        result = invoke()

        assert result.exit_code == 0, result.output
        assert tracking["bundles"] == []

    def test_nothing_is_tracked_without_a_model_name(self, invoke, tracking):
        result = invoke("--experiment", "flight-delay-v3")

        assert result.exit_code == 0, result.output
        assert tracking["bundles"] == []

    def test_the_bundle_is_registered_when_both_are_given(self, invoke, tracking):
        result = invoke(
            "--experiment",
            "flight-delay-v3",
            "--registered-model-name",
            "flight-delay-noweather",
        )

        assert result.exit_code == 0, result.output
        assert len(tracking["bundles"]) == 1
        assert tracking["bundles"][0]["registered_model_name"] == "flight-delay-noweather"

    def test_the_run_records_what_produced_the_model(self, invoke, tracking):
        invoke(
            "--experiment",
            "flight-delay-v3",
            "--registered-model-name",
            "flight-delay-noweather",
        )

        assert tracking["params"]["variant"] == "noweather"
        assert tracking["params"]["algorithm"] == "logistic_regression"
        assert tracking["params"]["encoding"] == "onehot"
        assert tracking["params"]["calibrated"] is False

    def test_the_run_is_named_after_the_combination(self, invoke, tracking):
        invoke(
            "--experiment",
            "flight-delay-v3",
            "--registered-model-name",
            "flight-delay-noweather",
        )

        assert tracking["runs"] == ["noweather__logistic_regression__default"]

    def test_the_alias_is_forwarded(self, invoke, tracking):
        invoke(
            "--experiment",
            "flight-delay-v3",
            "--registered-model-name",
            "flight-delay-noweather",
            "--alias",
            "champion",
        )

        assert tracking["bundles"][0]["alias"] == "champion"

    def test_the_registered_columns_come_from_the_fitted_estimator(self, invoke, tracking):
        invoke(
            "--experiment",
            "flight-delay-v3",
            "--registered-model-name",
            "flight-delay-noweather",
        )
        columns = tracking["bundles"][0]["columns"]

        assert any(c.startswith("ReportingAirline_") for c in columns)
        assert "Origin" not in columns


class TestRunCommand:
    def test_missing_training_file_exits_with_an_error_code(self, tmp_path):
        from typer.testing import CliRunner

        result = CliRunner().invoke(
            train_module.app,
            [
                "--variant",
                "noweather",
                "--model",
                "logistic_regression",
                "--train-path",
                str(tmp_path / "absent.parquet"),
                "--models-path",
                str(tmp_path / "models"),
            ],
        )

        assert result.exit_code == 1

    def test_unknown_model_exits_with_an_error_code(self, tmp_path, flights_df):
        from typer.testing import CliRunner

        path = tmp_path / "train.parquet"
        flights_df.to_parquet(path, index=False)

        result = CliRunner().invoke(
            train_module.app,
            [
                "--variant",
                "noweather",
                "--model",
                "xgboost",
                "--train-path",
                str(path),
                "--models-path",
                str(tmp_path / "models"),
            ],
        )

        assert result.exit_code == 1

    def test_saved_artifacts_form_a_complete_bundle(self, tmp_path, flights_df, small_transformer):
        """A saved model is only usable next to its transformer and columns."""
        from typer.testing import CliRunner

        path = tmp_path / "train.parquet"
        flights_df.to_parquet(path, index=False)
        models_path = tmp_path / "models"

        result = CliRunner().invoke(
            train_module.app,
            [
                "--variant",
                "noweather",
                "--model",
                "logistic_regression",
                "--config",
                "default",
                "--train-path",
                str(path),
                "--models-path",
                str(models_path),
                "--no-calibrate",
            ],
        )

        assert result.exit_code == 0, result.output
        save_dir = models_path / "noweather" / "logistic_regression__default"
        assert (save_dir / "model.joblib").exists()
        assert (save_dir / "transformer.joblib").exists()
        assert (save_dir / "columns.json").exists()
        assert Path(save_dir / "columns.json").read_text().startswith("[")
