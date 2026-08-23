"""Tests for train_evaluate_save_metrics.train_and_evaluate - selection phase 1.
"""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from predicting_flight_arrival_delays.data.features import select_features_variant
from predicting_flight_arrival_delays.data.transform import Transformer
from predicting_flight_arrival_delays.modeling import (
    train_evaluate_save_metrics as tesm_module,
)
from predicting_flight_arrival_delays.modeling.train_evaluate_save_metrics import (
    train_and_evaluate,
)


runner = CliRunner()


class _FakeMlflow:
    """Records what phase 1 would have logged, without touching a tracking server."""

    def __init__(self):
        self.logged = []

    def log_metrics(self, metrics, step=None):
        self.logged.append((metrics, step))


@pytest.fixture(autouse=True)
def isolated_mlflow(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setattr(tesm_module, "mlflow", fake)
    return fake


@pytest.fixture(autouse=True)
def small_transformer(monkeypatch):
    """The 1000-row category floor would fold every category into OTHER here."""

    def build(**kwargs):
        return Transformer(min_category_count=5, **kwargs)

    monkeypatch.setattr(tesm_module, "Transformer", build)


@pytest.fixture
def make_folds(tmp_path, flights_df):
    """Build fold directories on disk, each with a deliberately unreadable test set.

    Phase 1 must never open test.parquet. Writing garbage there means any
    attempt to read it fails the test loudly instead of passing unnoticed.
    """
    df = select_features_variant(flights_df, "noweather").sort_values("FlightDate")

    def build(slices, name="selection"):
        root = tmp_path / name
        directories = []
        for index, (train_slice, val_slice) in enumerate(slices, start=1):
            fold = root / f"fold_{index}_with_val"
            fold.mkdir(parents=True)
            df.iloc[train_slice].to_parquet(fold / "train.parquet", index=False)
            df.iloc[val_slice].to_parquet(fold / "validation.parquet", index=False)
            (fold / "test.parquet").write_bytes(b"this is not a parquet file")
            directories.append(fold)
        return directories

    return build


@pytest.fixture
def two_folds(make_folds):
    return make_folds([(slice(0, 120), slice(120, 160)), (slice(0, 200), slice(200, 260))])


@pytest.fixture
def identical_folds(make_folds):
    """Two folds carrying exactly the same rows, so the averaging is predictable."""
    return make_folds(
        [(slice(0, 150), slice(150, 200)), (slice(0, 150), slice(150, 200))],
        name="identical",
    )


def _run(folds, resample="none"):
    return train_and_evaluate(folds, "noweather", "logistic_regression", "default", resample)


class TestMetricNaming:
    def test_every_metric_is_marked_as_validation(self, two_folds):
        """A bare 'pr_auc' here would be indistinguishable from a test score."""
        averages = _run(two_folds)

        assert all(k.endswith("_val") or k == "roc_auc_val_std" for k in averages)

    def test_no_unsuffixed_metric_survives(self, two_folds):
        averages = _run(two_folds)

        assert not {"pr_auc", "roc_auc", "brier"} & set(averages)

    def test_the_ranking_metrics_are_all_there(self, two_folds):
        averages = _run(two_folds)

        assert {"roc_auc_val", "pr_auc_val", "brier_val", "roc_auc_val_std"} == set(averages)

    def test_no_threshold_metrics_are_produced(self, two_folds):
        """Phase 1 picks no operating point: that belongs to the winner's run."""
        averages = _run(two_folds)

        assert not any("threshold" in k or "recall" in k for k in averages)


class TestFoldHandling:
    def test_the_test_set_is_never_opened(self, two_folds):
        """test.parquet holds garbage; reading it would raise, not return."""
        assert _run(two_folds)

    def test_every_fold_is_logged_separately(self, two_folds, isolated_mlflow):
        _run(two_folds)

        assert [step for _, step in isolated_mlflow.logged] == [0, 1]

    def test_per_fold_metrics_are_prefixed(self, two_folds, isolated_mlflow):
        _run(two_folds)
        metrics, _ = isolated_mlflow.logged[0]

        assert all(k.startswith("fold_") for k in metrics)
        assert "fold_pr_auc_val" in metrics

    def test_identical_folds_average_to_themselves(self, identical_folds, isolated_mlflow):
        """Same data twice: the mean is the value and the spread is zero."""
        averages = _run(identical_folds)
        first_fold, _ = isolated_mlflow.logged[0]

        assert averages["roc_auc_val"] == pytest.approx(first_fold["fold_roc_auc_val"])
        assert averages["roc_auc_val_std"] == pytest.approx(0.0)

    def test_the_spread_is_reported_across_folds(self, two_folds):
        """The std is what says whether the estimate is stable over time."""
        averages = _run(two_folds)

        assert averages["roc_auc_val_std"] >= 0.0
        assert np.isfinite(averages["roc_auc_val_std"])

    def test_no_folds_at_all_is_an_error(self):
        with pytest.raises(RuntimeError, match="No usable folds"):
            _run([])

    def test_a_single_fold_has_zero_spread(self, make_folds):
        folds = make_folds([(slice(0, 200), slice(200, 260))], name="one")
        averages = _run(folds)

        assert averages["roc_auc_val_std"] == pytest.approx(0.0)


class TestRunCommand:
    """The Typer entry point: guards, MLflow wiring, and the metrics file."""

    @pytest.fixture
    def orchestration(self, tmp_path, monkeypatch):
        recorded = {"params": {}, "metrics": {}, "tags": {}, "runs": []}

        class FakeRun:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class FakeMlflow:
            @staticmethod
            def enable_system_metrics_logging():
                pass

            @staticmethod
            def set_experiment(name):
                recorded["experiment"] = name

            @staticmethod
            def start_run(run_name=None):
                recorded["runs"].append(run_name)
                return FakeRun()

            @staticmethod
            def log_params(params):
                recorded["params"].update(params)

            @staticmethod
            def log_metrics(metrics, step=None):
                recorded["metrics"].update(metrics)

            @staticmethod
            def set_tag(key, value):
                recorded["tags"][key] = value

        monkeypatch.setattr(tesm_module.dagshub, "init", lambda **kwargs: None)
        monkeypatch.setattr(tesm_module, "mlflow", FakeMlflow)
        monkeypatch.setattr(tesm_module, "get_dvc_data_hash", lambda path: "test-hash")
        monkeypatch.setattr(tesm_module, "get_git_dirty", lambda: False)
        monkeypatch.setattr(
            tesm_module,
            "train_and_evaluate",
            lambda *args, **kwargs: {"pr_auc_val": 0.42, "roc_auc_val_std": 0.01},
        )
        monkeypatch.setattr(tesm_module, "METRICS_DIR", tmp_path / "metrics")
        return recorded

    @pytest.fixture
    def variant_root(self, tmp_path, flights_df):
        """A data-path root laid out as split_data writes it: <root>/<variant>/fold_*."""
        df = select_features_variant(flights_df, "noweather").sort_values("FlightDate")
        fold = tmp_path / "root" / "noweather" / "fold_1_with_val"
        fold.mkdir(parents=True)
        df.iloc[:120].to_parquet(fold / "train.parquet", index=False)
        df.iloc[120:].to_parquet(fold / "validation.parquet", index=False)
        return tmp_path / "root"

    def test_the_averaged_metrics_are_written_for_selection_to_read(
        self, variant_root, orchestration, tmp_path
    ):
        result = runner.invoke(
            tesm_module.app,
            [
                "--variant", "noweather",
                "--model", "logistic_regression",
                "--data-path", str(variant_root),
            ],
        )

        assert result.exit_code == 0, result.output
        written = tmp_path / "metrics" / "selection" / "noweather"
        payload = json.loads((written / "logistic_regression__default.json").read_text())
        assert payload["pr_auc_val"] == pytest.approx(0.42)
        assert payload["resample"] == "none"

    def test_the_run_is_marked_as_not_final(self, variant_root, orchestration):
        """These are the 33 selection runs, not the winner's run."""
        runner.invoke(
            tesm_module.app,
            [
                "--variant", "noweather",
                "--model", "logistic_regression",
                "--data-path", str(variant_root),
            ],
        )

        assert orchestration["params"]["final"] is False
        assert orchestration["params"]["calibrated"] is False

    def test_the_data_version_is_recorded(self, variant_root, orchestration):
        runner.invoke(
            tesm_module.app,
            [
                "--variant", "noweather",
                "--model", "logistic_regression",
                "--data-path", str(variant_root),
            ],
        )

        assert orchestration["params"]["dvc_data_hash"] == "test-hash"
        assert orchestration["tags"]["git_dirty"] is False

    def test_the_hyperparameters_are_logged_with_a_prefix(self, variant_root, orchestration):
        runner.invoke(
            tesm_module.app,
            [
                "--variant", "noweather",
                "--model", "logistic_regression",
                "--data-path", str(variant_root),
            ],
        )

        assert "hp_max_iter" in orchestration["params"]

    @pytest.mark.parametrize(
        "override",
        [
            {"--variant": "nope"},
            {"--model": "xgboost"},
            {"--config": "turbo"},
            {"--resample": "bootstrap"},
        ],
    )
    def test_an_unknown_option_value_stops_the_run(self, variant_root, orchestration, override):
        options = {
            "--variant": "noweather",
            "--model": "logistic_regression",
            "--data-path": str(variant_root),
        }
        options.update(override)
        args = [item for pair in options.items() for item in pair]

        result = runner.invoke(tesm_module.app, args)

        assert result.exit_code == 1

    def test_smote_is_refused_for_a_native_encoding_model(self, variant_root, orchestration):
        """SMOTE interpolates between rows, which category dtypes cannot support."""
        result = runner.invoke(
            tesm_module.app,
            [
                "--variant", "noweather",
                "--model", "lightgbm",
                "--data-path", str(variant_root),
                "--resample", "smote",
            ],
        )

        assert result.exit_code == 1

    def test_a_missing_variant_directory_stops_the_run(
        self, variant_root, orchestration, tmp_path
    ):
        result = runner.invoke(
            tesm_module.app,
            [
                "--variant", "all",
                "--model", "logistic_regression",
                "--data-path", str(variant_root),
            ],
        )

        assert result.exit_code == 1

    def test_folds_without_a_validation_split_stop_the_run(
        self, tmp_path, flights_df, orchestration
    ):
        """Phase 1 scores on validation; without it there is nothing to score."""
        df = select_features_variant(flights_df, "noweather")
        fold = tmp_path / "novalroot" / "noweather" / "fold_1_without_val"
        fold.mkdir(parents=True)
        df.to_parquet(fold / "train.parquet", index=False)

        result = runner.invoke(
            tesm_module.app,
            [
                "--variant", "noweather",
                "--model", "logistic_regression",
                "--data-path", str(tmp_path / "novalroot"),
            ],
        )

        assert result.exit_code == 1


class TestResampling:
    def test_the_strategy_reaches_the_fold_preparation(self, two_folds, monkeypatch):
        seen = []
        real_prepare = tesm_module.prepare_fold

        def spy(train_df, encoding, **kwargs):
            seen.append(kwargs.get("resample"))
            return real_prepare(train_df, encoding, **kwargs)

        monkeypatch.setattr(tesm_module, "prepare_fold", spy)
        _run(two_folds, resample="undersample")

        assert seen == ["undersample", "undersample"]

    def test_results_stay_finite_under_resampling(self, two_folds):
        averages = _run(two_folds, resample="undersample")

        assert all(np.isfinite(v) for v in averages.values())
