"""Tests for predicting_flight_arrival_delays.modeling.evaluate."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from predicting_flight_arrival_delays.config import TARGET
from predicting_flight_arrival_delays.data.features import select_features_variant
from predicting_flight_arrival_delays.data.transform import Transformer
from predicting_flight_arrival_delays.modeling import evaluate as evaluate_module
from predicting_flight_arrival_delays.modeling import train as train_module
from predicting_flight_arrival_delays.modeling.evaluate import (
    _evaluate_on_dataframe,
    evaluate,
)
from predicting_flight_arrival_delays.modeling.train import train_with_transformer

runner = CliRunner()


@pytest.fixture
def separable(stub_estimator):
    """Four rows the model orders perfectly."""
    y = pd.Series([0, 0, 1, 1])
    return pd.DataFrame({"x": [1, 2, 3, 4]}), y, stub_estimator([0.1, 0.2, 0.8, 0.9])


class TestEvaluate:
    def test_without_a_threshold_only_ranking_metrics_are_returned(self, separable):
        """Phase 1 compares candidates; it never picks an operating point."""
        X, y, estimator = separable

        assert set(evaluate(X, y, estimator)) == {"roc_auc", "pr_auc", "brier"}

    def test_with_a_threshold_the_decision_metrics_appear(self, separable):
        X, y, estimator = separable
        metrics = evaluate(X, y, estimator, threshold=0.5)

        assert set(metrics) == {
            "roc_auc", "pr_auc", "brier", "recall", "precision",
            "f1.0", "alert_rate", "threshold",
        }

    def test_perfect_ranking_scores_one(self, separable):
        X, y, estimator = separable
        metrics = evaluate(X, y, estimator)

        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["pr_auc"] == pytest.approx(1.0)

    def test_brier_is_the_mean_squared_probability_error(self, separable):
        X, y, estimator = separable
        expected = np.mean((np.array([0.1, 0.2, 0.8, 0.9]) - y.to_numpy()) ** 2)

        assert evaluate(X, y, estimator)["brier"] == pytest.approx(expected)

    def test_threshold_is_echoed_back(self, separable):
        X, y, estimator = separable

        assert evaluate(X, y, estimator, threshold=0.42)["threshold"] == pytest.approx(0.42)

    def test_alert_rate_is_the_share_flagged_as_delayed(self, separable):
        X, y, estimator = separable

        assert evaluate(X, y, estimator, threshold=0.5)["alert_rate"] == pytest.approx(0.5)

    def test_a_lower_threshold_raises_recall(self, separable):
        X, y, estimator = separable
        strict = evaluate(X, y, estimator, threshold=0.85)
        loose = evaluate(X, y, estimator, threshold=0.15)

        assert loose["recall"] >= strict["recall"]
        assert loose["alert_rate"] > strict["alert_rate"]

    def test_the_fbeta_key_names_its_beta(self, separable):
        """select_and_register scores at beta=1.2, so the key must say so."""
        X, y, estimator = separable

        assert "f1.2" in evaluate(X, y, estimator, threshold=0.5, fbeta=1.2)

    def test_a_threshold_nobody_clears_does_not_divide_by_zero(self, separable):
        X, y, estimator = separable
        metrics = evaluate(X, y, estimator, threshold=0.99)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["alert_rate"] == 0.0

    def test_every_metric_is_a_plain_float(self, separable):
        """These go straight into JSON, where numpy scalars are not serialisable."""
        X, y, estimator = separable
        metrics = evaluate(X, y, estimator, threshold=0.5)

        assert all(type(v) is float for v in metrics.values())
        json.dumps(metrics)


class TestEvaluateOnDataframe:

    @pytest.fixture
    def small_transformer(self, monkeypatch):
        def build(**kwargs):
            return Transformer(min_category_count=5, **kwargs)

        monkeypatch.setattr(train_module, "Transformer", build)

    @pytest.fixture
    def bundle(self, flights_df, small_transformer):
        df = select_features_variant(flights_df, "noweather")
        transformer, estimator, X_fit, columns = train_with_transformer(
            df, "noweather", "logistic_regression", "default", calibrate=False
        )
        return df, transformer, estimator, X_fit, columns

    def test_it_returns_the_full_metric_set(self, bundle):
        df, transformer, estimator, _, columns = bundle

        metrics = _evaluate_on_dataframe(df, "noweather", estimator, transformer, columns, 0.5)

        assert {"roc_auc", "pr_auc", "brier", "threshold"} <= set(metrics)

    def test_a_slice_missing_categories_scores_identically(self, bundle):
        """Same rows, same model: preparing them for scoring must reproduce the
        matrix the training path built for them."""
        df, transformer, estimator, X_fit, columns = bundle
        
        rows = (df["OriginAirportID"] == 10397).to_numpy()

        expected = evaluate(X_fit[rows], df.loc[rows, TARGET], estimator, 0.5)
        actual = _evaluate_on_dataframe(
            df[rows], "noweather", estimator, transformer, columns, 0.5
        )

        assert actual["pr_auc"] == pytest.approx(expected["pr_auc"])
        assert actual["brier"] == pytest.approx(expected["brier"])
        assert actual["roc_auc"] == pytest.approx(expected["roc_auc"])

    def test_the_whole_frame_scores_identically_too(self, bundle):
        df, transformer, estimator, X_fit, columns = bundle

        expected = evaluate(X_fit, df[TARGET], estimator, 0.5)
        actual = _evaluate_on_dataframe(df, "noweather", estimator, transformer, columns, 0.5)

        assert actual["pr_auc"] == pytest.approx(expected["pr_auc"])

    def test_without_a_column_list_it_still_scores(self, bundle):
        """Older bundles carry no columns.json; the caller is warned, not stopped."""
        df, transformer, estimator, _, _ = bundle

        metrics = _evaluate_on_dataframe(df, "noweather", estimator, transformer, None, 0.5)

        assert "pr_auc" in metrics


@pytest.fixture
def saved_bundle(tmp_path, flights_df, monkeypatch):
    """A model saved to disk the way train.run() saves it, plus data to score."""

    def build(**kwargs):
        return Transformer(min_category_count=5, **kwargs)

    monkeypatch.setattr(train_module, "Transformer", build)

    df = select_features_variant(flights_df, "noweather")
    transformer, estimator, _, columns = train_with_transformer(
        df, "noweather", "logistic_regression", "default", calibrate=False
    )

    save_dir = tmp_path / "models" / "noweather" / "logistic_regression__default"
    save_dir.mkdir(parents=True)
    joblib.dump(estimator, save_dir / "model.joblib")
    joblib.dump(transformer, save_dir / "transformer.joblib")
    (save_dir / "columns.json").write_text(json.dumps(columns))

    data_path = tmp_path / "test.parquet"
    df.to_parquet(data_path, index=False)

    return save_dir, data_path, estimator, transformer


class TestEvaluateFromLocalPathHappyPath:
    def test_metrics_land_in_a_variant_folder(self, tmp_path, saved_bundle):
        save_dir, data_path, _, _ = saved_bundle
        metrics_dir = tmp_path / "metrics"

        result = runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-local-path",
                "--model-path", str(save_dir),
                "--evaluate-df-path", str(data_path),
                "--output-path", str(metrics_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        written = metrics_dir / "noweather" / "logistic_regression__default.json"
        assert written.exists()

    def test_the_written_metrics_include_the_operating_point(self, tmp_path, saved_bundle):
        save_dir, data_path, _, _ = saved_bundle
        metrics_dir = tmp_path / "metrics"

        runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-local-path",
                "--model-path", str(save_dir),
                "--evaluate-df-path", str(data_path),
                "--output-path", str(metrics_dir),
                "--threshold", "0.3",
            ],
        )
        payload = json.loads(
            (metrics_dir / "noweather" / "logistic_regression__default.json").read_text()
        )

        assert payload["threshold"] == pytest.approx(0.3)
        assert {"roc_auc", "pr_auc", "brier", "recall", "precision"} <= set(payload)

    def test_a_bundle_without_columns_json_still_scores(self, tmp_path, saved_bundle):
        """Older bundles predate columns.json; the caller is warned, not stopped."""
        save_dir, data_path, _, _ = saved_bundle
        (save_dir / "columns.json").unlink()

        result = runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-local-path",
                "--model-path", str(save_dir),
                "--evaluate-df-path", str(data_path),
                "--output-path", str(tmp_path / "metrics"),
            ],
        )

        assert result.exit_code == 0, result.output


class TestEvaluateFromMlflow:
    @pytest.fixture
    def registry(self, monkeypatch, saved_bundle):
        save_dir, _, estimator, transformer = saved_bundle
      
        columns = json.loads((save_dir / "columns.json").read_text())
        requested = {}

        def fake_load_bundle(registered_model_name, stage="None"):
            requested["name"] = registered_model_name
            requested["stage"] = stage
            return estimator, transformer, columns, "run-1"

        monkeypatch.setattr(evaluate_module.dagshub, "init", lambda **kwargs: None)
        monkeypatch.setattr(evaluate_module, "load_model_bundle", fake_load_bundle)
        monkeypatch.setattr(
            evaluate_module, "get_run_params", lambda run_id: {"variant": "noweather"}
        )
        return requested

    def test_the_variant_is_read_back_from_the_run(self, tmp_path, saved_bundle, registry):
        """Nothing in the model name says which feature set it was trained on."""
        _, data_path, _, _ = saved_bundle
        metrics_dir = tmp_path / "metrics"

        result = runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-mlflow",
                "--registered-model-name", "flight-delay-noweather",
                "--evaluate-df-path", str(data_path),
                "--output-path", str(metrics_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert (metrics_dir / "noweather" / "flight-delay-noweather__None.json").exists()

    def test_the_requested_alias_is_forwarded(self, tmp_path, saved_bundle, registry):
        _, data_path, _, _ = saved_bundle

        runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-mlflow",
                "--registered-model-name", "flight-delay-noweather",
                "--evaluate-df-path", str(data_path),
                "--stage", "champion",
                "--output-path", str(tmp_path / "metrics"),
            ],
        )

        assert registry["name"] == "flight-delay-noweather"
        assert registry["stage"] == "champion"

    def test_a_missing_evaluation_file_exits_with_an_error_code(self, tmp_path, registry):
        result = runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-mlflow",
                "--registered-model-name", "flight-delay-noweather",
                "--evaluate-df-path", str(tmp_path / "absent.parquet"),
            ],
        )

        assert result.exit_code == 1

    def test_an_empty_evaluation_file_exits_with_an_error_code(
        self, tmp_path, saved_bundle, registry
    ):
        _, data_path, _, _ = saved_bundle
        empty = tmp_path / "empty.parquet"
        pd.read_parquet(data_path).head(0).to_parquet(empty, index=False)

        result = runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-mlflow",
                "--registered-model-name", "flight-delay-noweather",
                "--evaluate-df-path", str(empty),
            ],
        )

        assert result.exit_code == 1


class TestEvaluateFromLocalPath:
    def test_missing_evaluation_file_exits_with_an_error_code(self, tmp_path):
        result = runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-local-path",
                "--model-path", str(tmp_path / "all" / "lightgbm__default"),
                "--evaluate-df-path", str(tmp_path / "absent.parquet"),
            ],
        )

        assert result.exit_code == 1

    def test_empty_evaluation_file_exits_with_an_error_code(self, tmp_path, flights_df):
        empty = tmp_path / "empty.parquet"
        flights_df.head(0).to_parquet(empty, index=False)

        result = runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-local-path",
                "--model-path", str(tmp_path / "all" / "lightgbm__default"),
                "--evaluate-df-path", str(empty),
            ],
        )

        assert result.exit_code == 1

    def test_missing_model_file_exits_with_an_error_code(self, tmp_path, flights_df):
        data = tmp_path / "test.parquet"
        flights_df.to_parquet(data, index=False)
        model_dir = tmp_path / "all" / "lightgbm__default"
        model_dir.mkdir(parents=True)

        result = runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-local-path",
                "--model-path", str(model_dir),
                "--evaluate-df-path", str(data),
            ],
        )

        assert result.exit_code == 1

    def test_a_model_without_its_transformer_exits_with_an_error_code(
        self, tmp_path, flights_df
    ):
        """Half a bundle is unusable: the model alone cannot prepare its input."""
        data = tmp_path / "test.parquet"
        flights_df.to_parquet(data, index=False)
        model_dir = tmp_path / "all" / "lightgbm__default"
        model_dir.mkdir(parents=True)
        (model_dir / "model.joblib").write_bytes(b"")

        result = runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-local-path",
                "--model-path", str(model_dir),
                "--evaluate-df-path", str(data),
            ],
        )

        assert result.exit_code == 1

    def test_a_folder_not_named_model__config_exits_with_an_error_code(
        self, tmp_path, flights_df
    ):
        """The variant and config are read back out of the directory layout."""
        data = tmp_path / "test.parquet"
        flights_df.to_parquet(data, index=False)
        model_dir = tmp_path / "all" / "lightgbm"
        model_dir.mkdir(parents=True)
        (model_dir / "model.joblib").write_bytes(b"")
        (model_dir / "transformer.joblib").write_bytes(b"")

        result = runner.invoke(
            evaluate_module.app,
            [
                "evaluate-from-local-path",
                "--model-path", str(model_dir),
                "--evaluate-df-path", str(data),
            ],
        )

        assert result.exit_code == 1
