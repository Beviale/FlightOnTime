"""Tests for predicting_flight_arrival_delays.modeling.select_and_register."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from predicting_flight_arrival_delays.config import PRODUCTION_VARIANTS
from predicting_flight_arrival_delays.modeling import select_and_register as sar_module
from predicting_flight_arrival_delays.modeling.select_and_register import (
    all_fold_dirs,
    choose_threshold,
    load_metrics,
)

runner = CliRunner()

FOLD_FILES = ("train.parquet", "validation.parquet", "test.parquet")


class TestChooseThreshold:
    def test_perfect_separation_cuts_between_the_classes(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])

        assert choose_threshold(y_true, y_prob) == pytest.approx(0.8)

    def test_the_threshold_is_one_of_the_observed_probabilities(self):
        y_true = pd.Series([0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.15, 0.62, 0.33, 0.71, 0.55, 0.24])

        assert choose_threshold(y_true, y_prob) in set(y_prob)

    def test_a_higher_beta_favours_recall(self):
        """beta>1 weights recall over precision, so the cut moves down."""
        y_true = pd.Series([0, 0, 0, 1, 1, 1, 0, 1])
        y_prob = np.array([0.05, 0.2, 0.45, 0.5, 0.6, 0.85, 0.55, 0.35])

        assert choose_threshold(y_true, y_prob, beta=3.0) <= choose_threshold(
            y_true, y_prob, beta=0.5
        )

    def test_returns_a_plain_float(self):
        """It is logged to MLflow, which will not take a numpy scalar."""
        y_true = pd.Series([0, 1])
        y_prob = np.array([0.2, 0.9])

        assert type(choose_threshold(y_true, y_prob)) is float

    def test_all_positive_labels_are_handled(self):
        y_true = pd.Series([1, 1, 1])
        y_prob = np.array([0.3, 0.6, 0.9])

        assert 0.0 <= choose_threshold(y_true, y_prob) <= 1.0


class TestLoadMetrics:
    @pytest.fixture
    def metrics_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sar_module, "METRICS_DIR", tmp_path)
        directory = tmp_path / "selection" / "all"
        directory.mkdir(parents=True)
        return directory

    def test_files_are_keyed_by_algorithm_and_config(self, metrics_dir):
        (metrics_dir / "lightgbm__default.json").write_text(json.dumps({"pr_auc_val": 0.4}))
        (metrics_dir / "random_forest__deep.json").write_text(json.dumps({"pr_auc_val": 0.3}))

        loaded = load_metrics("all")

        assert set(loaded) == {("lightgbm", "default"), ("random_forest", "deep")}

    def test_the_recorded_metrics_come_back_intact(self, metrics_dir):
        payload = {"pr_auc_val": 0.42, "roc_auc_val": 0.71, "resample": "undersample"}
        (metrics_dir / "lightgbm__default.json").write_text(json.dumps(payload))

        assert load_metrics("all")[("lightgbm", "default")] == payload

    def test_an_empty_directory_stops_the_run(self, metrics_dir):
        """Selecting a winner with nothing to compare would be meaningless."""
        with pytest.raises(SystemExit, match="run training first"):
            load_metrics("all")

    def test_a_missing_variant_directory_stops_the_run(self, metrics_dir):
        with pytest.raises(SystemExit):
            load_metrics("noweather")

    def test_the_winner_is_picked_on_validation_pr_auc(self, metrics_dir):
        """Phase 1 decides on validation only - never on the test set."""
        (metrics_dir / "lightgbm__default.json").write_text(
            json.dumps({"pr_auc_val": 0.41, "resample": "none"})
        )
        (metrics_dir / "random_forest__deep.json").write_text(
            json.dumps({"pr_auc_val": 0.55, "resample": "none"})
        )

        candidates = load_metrics("all")
        winner = max(candidates, key=lambda k: candidates[k]["pr_auc_val"])

        assert winner == ("random_forest", "deep")


class TestRunCommand:
    """The top-level command: pick a winner per production variant, register it."""

    @pytest.fixture
    def orchestration(self, monkeypatch):
        registered = []

        monkeypatch.setattr(sar_module.dagshub, "init", lambda **kwargs: None)
        monkeypatch.setattr(
            sar_module, "mlflow", SimpleNamespace(set_experiment=lambda name: None)
        )
        monkeypatch.setattr(
            sar_module,
            "load_metrics",
            lambda variant: {
                ("lightgbm", "default"): {"pr_auc_val": 0.41, "resample": "none"},
                ("random_forest", "deep"): {"pr_auc_val": 0.62, "resample": "undersample"},
                ("logistic_regression", "default"): {"pr_auc_val": 0.33, "resample": "none"},
            },
        )
        monkeypatch.setattr(
            sar_module,
            "register_winner",
            lambda *args, **kwargs: registered.append((args, kwargs)),
        )
        return registered

    def test_one_winner_per_production_variant(self, orchestration):
        result = runner.invoke(sar_module.app, [])

        assert result.exit_code == 0, result.output
        assert [args[0] for args, _ in orchestration] == PRODUCTION_VARIANTS

    def test_the_winner_is_the_best_validation_pr_auc(self, orchestration):
        """Selection reads validation only - the test set is for reporting."""
        runner.invoke(sar_module.app, [])
        args, _ = orchestration[0]

        assert args[1:3] == ("random_forest", "deep")

    def test_the_winners_resampling_strategy_is_carried_over(self, orchestration):
        runner.invoke(sar_module.app, [])
        args, _ = orchestration[0]

        assert args[3] == "undersample"

    def test_the_alias_is_passed_through(self, orchestration):
        runner.invoke(sar_module.app, ["--alias", "challenger"])
        _, kwargs = orchestration[0]

        assert kwargs["alias"] == "challenger"

    def test_calibration_is_on_by_default(self, orchestration):
        runner.invoke(sar_module.app, [])
        args, _ = orchestration[0]

        assert args[4] is True

    def test_local_saving_is_off_by_default(self, orchestration):
        runner.invoke(sar_module.app, [])
        _, kwargs = orchestration[0]

        assert kwargs["models_path"] is None

    def test_a_failure_exits_with_an_error_code(self, monkeypatch):
        monkeypatch.setattr(sar_module.dagshub, "init", lambda **kwargs: None)
        monkeypatch.setattr(
            sar_module, "mlflow", SimpleNamespace(set_experiment=lambda name: None)
        )

        def boom(variant):
            raise RuntimeError("metrics unreadable")

        monkeypatch.setattr(sar_module, "load_metrics", boom)

        result = runner.invoke(sar_module.app, [])

        assert result.exit_code == 1


class TestAllFoldDirs:
    @pytest.fixture
    def processed_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sar_module, "PROCESSED_DATA_DIR", tmp_path)
        return tmp_path / "selection" / "all"

    def _make_fold(self, root, index, files=FOLD_FILES):
        fold = root / f"fold_{index}_with_val"
        fold.mkdir(parents=True)
        for name in files:
            (fold / name).write_bytes(b"")
        return fold

    def test_folds_come_back_in_numeric_order(self, processed_dir):
        """Sorting by name would put fold_10 before fold_2."""
        for index in (2, 10, 1):
            self._make_fold(processed_dir, index)

        assert [d.name for d in all_fold_dirs("all")] == [
            "fold_1_with_val",
            "fold_2_with_val",
            "fold_10_with_val",
        ]

    def test_no_folds_at_all_is_an_error(self, processed_dir):
        processed_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="No folds found"):
            all_fold_dirs("all")

    def test_a_missing_variant_directory_is_an_error(self, processed_dir):
        with pytest.raises(FileNotFoundError, match="No folds found"):
            all_fold_dirs("all")

    def test_an_incomplete_fold_is_reported_by_name(self, processed_dir):
        """Step 1 needs all three splits from every fold, not just the last."""
        self._make_fold(processed_dir, 1)
        self._make_fold(processed_dir, 2, files=("train.parquet", "validation.parquet"))

        with pytest.raises(FileNotFoundError, match="test.parquet"):
            all_fold_dirs("all")

    def test_stray_files_are_not_mistaken_for_folds(self, processed_dir):
        self._make_fold(processed_dir, 1)
        (processed_dir / "fold_notes.txt").write_text("scratch")

        assert len(all_fold_dirs("all")) == 1
