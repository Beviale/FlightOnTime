"""Tests for predicting_flight_arrival_delays.modeling.predict."""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from predicting_flight_arrival_delays.data.features import (
    WEATHER_COLUMNS_DESTINATION,
    WEATHER_COLUMNS_ORIGIN,
    build_xy,
    select_features_variant,
)
from predicting_flight_arrival_delays.data.transform import OTHER, Transformer
from predicting_flight_arrival_delays.modeling import predict as predict_module
from predicting_flight_arrival_delays.modeling.predict import (
    align_to_training_columns,
    has_weather,
    predict,
    prepare_for_inference,
)


runner = CliRunner()

WEATHER_FEATURES = WEATHER_COLUMNS_ORIGIN + WEATHER_COLUMNS_DESTINATION


class TestHasWeather:
    def test_the_pipeline_column_names_are_recognised(self, flights_df):
        assert has_weather(flights_df).all()

    def test_all_weather_present_is_true(self):
        df = pd.DataFrame({c: [1.0, 2.0] for c in WEATHER_FEATURES})

        assert has_weather(df).all()

    def test_a_single_gap_disqualifies_the_row(self):
        """The 'all' model was never trained to see a missing weather value."""
        df = pd.DataFrame({c: [1.0, 2.0] for c in WEATHER_FEATURES})
        df.loc[0, WEATHER_FEATURES[0]] = np.nan

        assert list(has_weather(df)) == [False, True]

    def test_a_missing_weather_code_disqualifies_the_row(self, flights_df):
        """The code counts as part of the block, like any other weather column."""
        df = flights_df.copy()
        df.loc[df.index[0], "WeatherCodeOrigin"] = np.nan

        assert not has_weather(df).iloc[0]
        assert has_weather(df).iloc[1:].all()

    def test_weather_at_one_end_only_is_not_enough(self, flights_df):
        """Both origin and destination feed the 'all' model."""
        df = flights_df.drop(columns=WEATHER_COLUMNS_DESTINATION)

        assert not has_weather(df).any()

    def test_no_weather_columns_at_all_is_false_everywhere(self):
        df = pd.DataFrame({"Origin": ["ATL", "DFW"]})

        assert not has_weather(df).any()

    def test_the_result_keeps_the_input_index(self):
        """predict() uses it to route rows, so the index must line up."""
        df = pd.DataFrame({c: [1.0, 2.0] for c in WEATHER_FEATURES}, index=[7, 9])

        assert list(has_weather(df).index) == [7, 9]


class TestAlignToTrainingColumns:
    @pytest.fixture
    def transformer(self):
        t = Transformer(encoding="onehot")
        t.categorical_columns = ["Origin"]
        return t

    def test_missing_columns_are_filled_with_zeros(self, transformer):
        X = pd.DataFrame({"Origin_ATL": [1]})

        aligned = align_to_training_columns(X, ("Origin_ATL", "Origin_DFW"), transformer)

        assert list(aligned.columns) == ["Origin_ATL", "Origin_DFW"]
        assert aligned["Origin_DFW"].iloc[0] == 0

    def test_unknown_columns_are_dropped(self, transformer):
        X = pd.DataFrame({"Origin_ATL": [1], "Origin_XXX": [1]})

        aligned = align_to_training_columns(X, ("Origin_ATL",), transformer)

        assert list(aligned.columns) == ["Origin_ATL"]

    def test_training_order_is_restored(self, transformer):
        X = pd.DataFrame({"b": [1], "a": [2]})

        aligned = align_to_training_columns(X, ("a", "b"), transformer)

        assert list(aligned.columns) == ["a", "b"]

    def test_native_encoding_recasts_categoricals(self):
        t = Transformer(encoding="native")
        t.categorical_columns = ["Origin"]
        t.category_keep = {"Origin": {"ATL", "DFW"}}
        X = pd.DataFrame({"Origin": ["ATL", "DFW"]})

        aligned = align_to_training_columns(X, ("Origin",), t)

        assert isinstance(aligned["Origin"].dtype, pd.CategoricalDtype)

    def test_native_categories_come_from_training_not_the_batch(self):
        """LightGBM reads the category codes, so a live request carrying one
        airport must not renumber the whole category set."""
        t = Transformer(encoding="native")
        t.categorical_columns = ["Origin"]
        t.category_keep = {"Origin": {"ATL", "DFW", "ORD"}}
        X = pd.DataFrame({"Origin": ["DFW"]})

        aligned = align_to_training_columns(X, ("Origin",), t)

        assert list(aligned["Origin"].cat.categories) == ["ATL", "DFW", "ORD", OTHER]

    def test_an_unseen_airport_falls_into_other(self):
        t = Transformer(encoding="native")
        t.categorical_columns = ["Origin"]
        t.category_keep = {"Origin": {"ATL"}}
        X = pd.DataFrame({"Origin": ["ZZZ"]})

        aligned = align_to_training_columns(X, ("Origin",), t)

        assert aligned["Origin"].isna().all()

    def test_row_count_is_unchanged(self, transformer):
        X = pd.DataFrame({"Origin_ATL": [1, 0, 1]})

        assert len(align_to_training_columns(X, ("Origin_ATL", "Origin_DFW"), transformer)) == 3


class TestPrepareForInference:
    @pytest.fixture
    def fitted(self, flights_df):
        """A transformer fitted the way training fits it, on the 'all' variant."""
        df = select_features_variant(flights_df, "all")
        X, y = build_xy(df)

        transformer = Transformer(min_category_count=5, encoding="onehot")
        X_fit = transformer.fit_transform(X, y)
        transformer.select_features(X_fit, y)
        X_fit = transformer.apply_selection(X_fit)

        cat_cols = [c for c in transformer.categorical_columns if c in X_fit.columns]
        from predicting_flight_arrival_delays.data.transform import encode_categoricals

        X_fit = encode_categoricals(X_fit, cat_cols, "onehot")
        return transformer, tuple(X_fit.columns)

    def test_output_matches_the_training_layout(self, flights_df, fitted):
        transformer, columns = fitted

        X = prepare_for_inference(flights_df, "all", transformer, columns)

        assert list(X.columns) == list(columns)

    def test_one_row_in_one_row_out(self, flights_df, fitted):
        transformer, columns = fitted

        X = prepare_for_inference(flights_df.head(3), "all", transformer, columns)

        assert len(X) == 3

    def test_a_single_flight_is_enough(self, flights_df, fitted):
        """Live requests arrive one at a time, not as a batch."""
        transformer, columns = fitted

        X = prepare_for_inference(flights_df.head(1), "all", transformer, columns)

        assert X.shape == (1, len(columns))

    def test_an_unseen_airport_does_not_widen_the_matrix(self, flights_df, fitted):
        transformer, columns = fitted
        df = flights_df.head(5).copy()
        df["Origin"] = "ZZZ"

        X = prepare_for_inference(df, "all", transformer, columns)

        assert list(X.columns) == list(columns)


class _StubBundle:
    """A model that reports the variant it was asked for, via the probability."""

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, X):
        p = np.full(len(X), self.probability)
        return np.column_stack([1 - p, p])


class TestLoadBundle:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        predict_module._load_bundle.cache_clear()
        yield
        predict_module._load_bundle.cache_clear()

    def test_a_variant_is_fetched_from_the_registry_once(self, monkeypatch):
        """Scoring a batch must not re-download the model for every request."""
        calls = []

        def fake_load(name, stage="None"):
            calls.append((name, stage))
            return "model", "transformer", ["col"], "run-1"

        monkeypatch.setattr(predict_module, "load_model_bundle", fake_load)

        predict_module._load_bundle("all")
        predict_module._load_bundle("all")

        assert calls == [("flight-delay-all", "None")]

    def test_each_variant_gets_its_own_entry(self, monkeypatch):
        calls = []

        def fake_load(name, stage="None"):
            calls.append(name)
            return "model", "transformer", ["col"], "run-1"

        monkeypatch.setattr(predict_module, "load_model_bundle", fake_load)

        predict_module._load_bundle("all")
        predict_module._load_bundle("noweather")

        assert calls == ["flight-delay-all", "flight-delay-noweather"]

    def test_the_model_name_is_built_from_the_variant(self, monkeypatch):
        monkeypatch.setattr(
            predict_module,
            "load_model_bundle",
            lambda name, stage="None": (name, "t", ["c"], "r"),
        )

        model, _, _, _ = predict_module._load_bundle("noweather", "champion")

        assert model == "flight-delay-noweather"


class TestRunCommand:
    @pytest.fixture
    def scored(self, monkeypatch):
        """Stub out the registry and the scoring itself: this is about file I/O."""
        monkeypatch.setattr(predict_module.dagshub, "init", lambda **kwargs: None)

        def fake_predict(df, threshold_all, threshold_noweather):
            return pd.DataFrame(
                {
                    "delay_probability": np.full(len(df), 0.7),
                    "is_delayed": np.ones(len(df), dtype=int),
                    "variant": "all",
                },
                index=df.index,
            )

        monkeypatch.setattr(predict_module, "predict", fake_predict)

    @pytest.fixture
    def flights_parquet(self, tmp_path, flights_df):
        path = tmp_path / "flights.parquet"
        flights_df.head(10).to_parquet(path, index=False)
        return path

    def test_predictions_are_written_to_disk(self, tmp_path, flights_parquet, scored):
        output_path = tmp_path / "predictions.parquet"

        result = runner.invoke(
            predict_module.app, [str(flights_parquet), "--output-path", str(output_path)]
        )

        assert result.exit_code == 0, result.output
        assert len(pd.read_parquet(output_path)) == 10

    def test_a_csv_input_is_accepted_too(self, tmp_path, flights_df, scored):
        csv_path = tmp_path / "flights.csv"
        flights_df.head(4).to_csv(csv_path, index=False)
        output_path = tmp_path / "predictions.parquet"

        result = runner.invoke(
            predict_module.app, [str(csv_path), "--output-path", str(output_path)]
        )

        assert result.exit_code == 0, result.output
        assert len(pd.read_parquet(output_path)) == 4

    def test_the_output_directory_is_created(self, tmp_path, flights_parquet, scored):
        output_path = tmp_path / "deep" / "nested" / "predictions.parquet"

        runner.invoke(
            predict_module.app, [str(flights_parquet), "--output-path", str(output_path)]
        )

        assert output_path.exists()

    def test_a_missing_input_file_is_reported(self, tmp_path, scored):
        result = runner.invoke(
            predict_module.app,
            [str(tmp_path / "absent.parquet"), "--output-path", str(tmp_path / "out.parquet")],
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)


class TestPredict:
    @pytest.fixture
    def routed(self, monkeypatch):
        """Both variants stubbed out: 'all' returns 0.9, 'noweather' returns 0.1."""
        calls = []

        def fake_load_bundle(variant, stage=predict_module.DEFAULT_STAGE):
            calls.append(variant)
            probability = 0.9 if variant == "all" else 0.1
            return _StubBundle(probability), Transformer(), ["x"], "run-id"

        def fake_prepare(df, variant, transformer, training_columns):
            return pd.DataFrame({"x": np.zeros(len(df))}, index=df.index)

        monkeypatch.setattr(predict_module, "_load_bundle", fake_load_bundle)
        monkeypatch.setattr(predict_module, "prepare_for_inference", fake_prepare)
        return calls

    @pytest.fixture
    def mixed_df(self):
        """Two flights with weather, one without."""
        df = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in WEATHER_FEATURES})
        df.loc[2, WEATHER_FEATURES[0]] = np.nan
        return df

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="df is empty"):
            predict(pd.DataFrame(), 0.5, 0.5)

    def test_flights_are_routed_by_the_weather_they_carry(self, mixed_df, routed):
        out = predict(mixed_df, 0.5, 0.5)

        assert list(out["variant"]) == ["all", "all", "noweather"]

    def test_only_the_needed_variants_are_loaded(self, mixed_df, routed):
        predict(mixed_df, 0.5, 0.5)

        assert sorted(routed) == ["all", "noweather"]

    def test_a_single_variant_is_loaded_when_that_is_all_it_takes(self, routed):
        df = pd.DataFrame({c: [1.0, 2.0] for c in WEATHER_FEATURES})

        predict(df, 0.5, 0.5)

        assert routed == ["all"]

    def test_the_original_row_order_is_restored(self, mixed_df, routed):
        """Rows are scored in two batches but must come back as they went in."""
        out = predict(mixed_df, 0.5, 0.5)

        assert list(out.index) == list(mixed_df.index)

    def test_each_variant_uses_its_own_threshold(self, mixed_df, routed):
        out = predict(mixed_df, threshold_all=0.95, threshold_noweather=0.05)

        assert list(out["is_delayed"]) == [0, 0, 1]

    def test_probabilities_and_labels_are_both_returned(self, mixed_df, routed):
        out = predict(mixed_df, 0.5, 0.5)

        assert list(out.columns) == ["delay_probability", "is_delayed", "variant"]
        assert out["delay_probability"].between(0, 1).all()
