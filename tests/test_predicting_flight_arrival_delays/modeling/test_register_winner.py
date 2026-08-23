"""Tests for select_and_register.register_winner - selection phase 2.
"""

import json
from types import SimpleNamespace

import pytest

from predicting_flight_arrival_delays.data.features import select_features_variant
from predicting_flight_arrival_delays.data.transform import Transformer
from predicting_flight_arrival_delays.modeling import select_and_register as sar_module
from predicting_flight_arrival_delays.modeling import (
    train_evaluate_save_metrics as tesm_module,
)
from predicting_flight_arrival_delays.modeling.predict import prepare_for_inference


class _FakeRun:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeMlflow:
    def __init__(self):
        self.params, self.metrics, self.tags = {}, {}, {}
        self.inputs, self.run_names = [], []

    def start_run(self, run_name=None):
        self.run_names.append(run_name)
        return _FakeRun()

    def log_input(self, dataset):
        self.inputs.append(dataset)

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def log_metric(self, key, value):
        self.metrics[key] = value


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """Stub the outside world and spy on what register_winner does inside it."""
    fake_mlflow = _FakeMlflow()
    registrations, train_calls, prepare_calls = [], [], []

    monkeypatch.setattr(sar_module, "mlflow", fake_mlflow)
    monkeypatch.setattr(sar_module, "from_pandas", lambda *a, **k: {"name": k.get("name")})
    monkeypatch.setattr(
        sar_module, "register_model_bundle", lambda **kw: registrations.append(kw)
    )
    monkeypatch.setattr(sar_module, "get_dvc_data_hash", lambda *a, **k: "test-hash")
    monkeypatch.setattr(sar_module, "get_git_dirty", lambda: False)

    real_train = sar_module.train_model

    def train_spy(X, y, *args, **kwargs):
        train_calls.append(X.shape[0])
        return real_train(X, y, *args, **kwargs)

    monkeypatch.setattr(sar_module, "train_model", train_spy)

    real_prepare = sar_module.prepare_fold

    def prepare_spy(*args, **kwargs):
        prepare_calls.append(kwargs.get("resample", "none"))
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(sar_module, "prepare_fold", prepare_spy)

    def small_transformer(**kwargs):
        return Transformer(min_category_count=5, **kwargs)

    monkeypatch.setattr(sar_module, "Transformer", small_transformer)
    monkeypatch.setattr(tesm_module, "Transformer", small_transformer)
    monkeypatch.setattr(sar_module, "METRICS_DIR", tmp_path / "metrics")

    return SimpleNamespace(
        mlflow=fake_mlflow,
        registrations=registrations,
        train_calls=train_calls,
        prepare_calls=prepare_calls,
        metrics_dir=tmp_path / "metrics",
    )


@pytest.fixture
def scores(monkeypatch):
    """Pin the evaluation metrics so the baseline guard can be driven either way."""

    def set_pr_auc(pr_auc):
        def fake_evaluate(X, y, estimator, threshold=None, fbeta=1.0):
            return {
                "roc_auc": 0.82,
                "pr_auc": pr_auc,
                "brier": 0.11,
                "recall": 0.70,
                "precision": 0.55,
                f"f{fbeta}": 0.62,
                "alert_rate": 0.44,
                "threshold": float(threshold),
            }

        monkeypatch.setattr(
            sar_module, "evaluate", SimpleNamespace(evaluate=fake_evaluate)
        )

    return set_pr_auc


@pytest.fixture
def folds(tmp_path, flights_df, monkeypatch):
    """Two walk-forward folds on disk, with an expanding training window."""
    df = select_features_variant(flights_df, "noweather").sort_values("FlightDate")
    layout = [
        (slice(0, 100), slice(100, 130), slice(130, 160)),
        (slice(0, 160), slice(160, 200), slice(200, 300)),
    ]
    sizes = []
    for index, (train_rows, val_rows, test_rows) in enumerate(layout, start=1):
        fold = tmp_path / "selection" / "noweather" / f"fold_{index}_with_val"
        fold.mkdir(parents=True)
        df.iloc[train_rows].to_parquet(fold / "train.parquet", index=False)
        df.iloc[val_rows].to_parquet(fold / "validation.parquet", index=False)
        df.iloc[test_rows].to_parquet(fold / "test.parquet", index=False)
        sizes.append((len(df.iloc[train_rows]), len(df.iloc[val_rows])))

    monkeypatch.setattr(sar_module, "PROCESSED_DATA_DIR", tmp_path)
    return SimpleNamespace(sizes=sizes, count=len(layout))


def _register(**overrides):
    defaults = {
        "variant": "noweather",
        "algorithm": "logistic_regression",
        "config": "default",
        "resample": "none",
        "calibrate": False,
    }
    return sar_module.register_winner(**{**defaults, **overrides})


def _winner_file(registry):
    return registry.metrics_dir / "winner" / "noweather" / "logistic_regression__default.json"


class TestHonestNumber:
    def test_every_fold_contributes_to_the_reported_score(self, folds, registry, scores):
        """Averaging over all folds is what makes the number statistically stable."""
        scores(0.9)
        _register()

        assert len(registry.prepare_calls) == folds.count

    def test_the_metrics_land_where_the_report_expects_them(self, folds, registry, scores):
        scores(0.9)
        _register()

        assert _winner_file(registry).exists()

    def test_the_file_records_the_resampling_strategy(self, folds, registry, scores):
        scores(0.9)
        _register(resample="none")
        payload = json.loads(_winner_file(registry).read_text())

        assert payload["resample"] == "none"

    def test_the_spread_across_folds_is_reported(self, folds, registry, scores):
        """Without the std, a single lucky fold looks the same as a stable model."""
        scores(0.9)
        _register()
        payload = json.loads(_winner_file(registry).read_text())

        assert "roc_auc_std" in payload

    def test_the_averaged_metrics_are_the_ones_logged(self, folds, registry, scores):
        scores(0.9)
        _register()

        assert registry.mlflow.metrics["pr_auc"] == pytest.approx(0.9)

    def test_the_resampling_strategy_reaches_every_fold(self, folds, registry, scores):
        scores(0.9)
        _register()

        assert registry.prepare_calls == ["none"] * folds.count


class TestBaselineGuard:
    def test_a_model_no_better_than_chance_is_not_registered(self, folds, registry, scores):
        """PR-AUC below the random-guess baseline means the model has no value."""
        scores(0.01)
        _register()

        assert registry.registrations == []

    def test_the_metrics_are_still_written_when_registration_is_skipped(
        self, folds, registry, scores
    ):
        """The failed attempt must stay on the record, not vanish silently."""
        scores(0.01)
        _register()

        assert _winner_file(registry).exists()

    def test_no_model_is_fitted_for_registration_after_a_failed_guard(
        self, folds, registry, scores
    ):
        """It returns before step 2, so only the per-fold fits happened."""
        scores(0.01)
        _register()

        assert len(registry.train_calls) == folds.count

    def test_a_model_that_beats_chance_is_registered(self, folds, registry, scores):
        scores(0.9)
        _register()

        assert len(registry.registrations) == 1


class TestRegisteredModel:
    def test_it_is_fitted_on_train_and_validation_together(self, folds, registry, scores):
        """The last fold has the most data; step 2 uses all of it."""
        scores(0.9)
        _register()
        last_train, last_validation = folds.sizes[-1]

        assert registry.train_calls[-1] == last_train + last_validation

    def test_one_more_fit_happens_than_there_are_folds(self, folds, registry, scores):
        """The registered model is a separate object from the scored ones."""
        scores(0.9)
        _register()

        assert len(registry.train_calls) == folds.count + 1

    def test_it_is_registered_under_the_variant_name(self, folds, registry, scores):
        scores(0.9)
        _register()

        assert registry.registrations[0]["registered_model_name"] == "flight-delay-noweather"

    def test_the_alias_is_forwarded(self, folds, registry, scores):
        scores(0.9)
        _register(alias="champion")

        assert registry.registrations[0]["alias"] == "champion"

    def test_no_alias_leaves_the_version_unpromoted(self, folds, registry, scores):
        scores(0.9)
        _register()

        assert registry.registrations[0]["alias"] is None

    def test_the_registered_columns_describe_the_training_matrix(self, folds, registry, scores):
        """columns.json is what inference reindexes to, so it has to name the
        columns the model was actually fitted on - post one-hot, post selection."""
        scores(0.9)
        _register()
        columns = registry.registrations[0]["columns"]

        assert len(columns) == registry.mlflow.params["n_features"]
        assert any(c.startswith("Origin_") for c in columns)
        assert "Origin" not in columns

    def test_a_fresh_operating_threshold_is_logged(self, folds, registry, scores):
        """Step 2 picks its own cutoff; it never reuses one from the scoring pass."""
        scores(0.9)
        _register()

        assert 0.0 <= registry.mlflow.metrics["operating_threshold"] <= 1.0


class TestRunMetadata:
    def test_the_run_is_marked_final(self, folds, registry, scores):
        """This is what separates the winner's run from the 33 selection runs."""
        scores(0.9)
        _register()

        assert registry.mlflow.params["final"] is True

    def test_the_run_records_variant_algorithm_and_config(self, folds, registry, scores):
        scores(0.9)
        _register()
        params = registry.mlflow.params

        assert params["variant"] == "noweather"
        assert params["algorithm"] == "logistic_regression"
        assert params["config"] == "default"
        assert params["encoding"] == "onehot"

    def test_the_hyperparameters_are_logged_with_a_prefix(self, folds, registry, scores):
        scores(0.9)
        _register()

        assert "hp_max_iter" in registry.mlflow.params

    def test_the_data_lineage_is_attached(self, folds, registry, scores):
        scores(0.9)
        _register()

        assert registry.mlflow.inputs
        assert registry.mlflow.tags["git_dirty"] is False

    def test_the_run_is_named_after_the_variant(self, folds, registry, scores):
        scores(0.9)
        _register()

        assert registry.mlflow.run_names == ["noweather__final__logistic_regression"]


class TestLocalSaving:
    def test_nothing_is_written_locally_by_default(self, folds, registry, scores, tmp_path):
        scores(0.9)
        _register()

        assert not (tmp_path / "models").exists()

    def test_the_full_bundle_is_saved_when_asked(self, folds, registry, scores, tmp_path):
        """A local model is only usable next to its transformer and columns."""
        scores(0.9)
        models_path = tmp_path / "models"
        _register(models_path=models_path)

        save_dir = models_path / "noweather" / "logistic_regression__default"
        assert (save_dir / "model.joblib").exists()
        assert (save_dir / "transformer.joblib").exists()
        assert (save_dir / "columns.json").exists()


class TestTheRegisteredBundleIsUsable:
    """The end-to-end contract: whatever register_winner hands to the registry
    must be enough for predict.py to score a fresh flight with it."""

    def _bundle(self, registry):
        logged = registry.registrations[0]
        return logged["model"], logged["transformer"], tuple(logged["columns"])

    def test_the_registered_model_scores_fresh_flights(
        self, folds, registry, scores, flights_df
    ):
        scores(0.9)
        _register()
        model, transformer, columns = self._bundle(registry)

        fresh = select_features_variant(flights_df, "noweather").head(5)
        X = prepare_for_inference(fresh, "noweather", transformer, columns)
        proba = model.predict_proba(X)[:, 1]

        assert proba.shape == (5,)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_the_one_hot_block_is_reconstructed(self, folds, registry, scores, flights_df):
        """Reindexing to raw names leaves every dummy at zero, so the model scores
        a flight with no airport at all - and when the widths happen to match it
        does so without raising."""
        scores(0.9)
        _register()
        _, transformer, columns = self._bundle(registry)

        fresh = select_features_variant(flights_df, "noweather").head(5)
        X = prepare_for_inference(fresh, "noweather", transformer, columns)

        dummies = [c for c in X.columns if c.startswith("Origin_")]
        assert dummies
        assert (X[dummies].sum(axis=1) == 1).all()

    def test_a_single_flight_is_enough(self, folds, registry, scores, flights_df):
        """Live requests arrive one at a time, carrying one airport out of many."""
        scores(0.9)
        _register()
        model, transformer, columns = self._bundle(registry)

        fresh = select_features_variant(flights_df, "noweather").head(1)
        X = prepare_for_inference(fresh, "noweather", transformer, columns)

        assert model.predict_proba(X).shape == (1, 2)


class TestCalibration:
    def test_calibration_composes_end_to_end(self, folds, registry, scores):
        """Isotonic calibration wraps both the scored folds and the final model."""
        scores(0.9)
        _register(calibrate=True)

        assert len(registry.registrations) == 1
        assert registry.mlflow.params["calibrated"] is True
