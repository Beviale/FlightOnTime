"""Tests for predicting_flight_arrival_delays.data.transform."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from predicting_flight_arrival_delays.config import SERVICE_COLUMNS2, TARGET
from predicting_flight_arrival_delays.data.transform import (
    OTHER,
    Transformer,
    align_columns,
    cramers_v,
    encode_categoricals,
    resample_training_data,
    to_sparse_matrix,
)


@pytest.fixture
def small_df() -> pd.DataFrame:
    """A minimal frame: two categoricals, two numerics, one date, twelve rows."""
    return pd.DataFrame(
        {
            "FlightDate": pd.to_datetime(
                ["2025-01-%02d" % (i + 1) for i in range(12)]
            ),
            "Origin": ["ATL"] * 5 + ["DFW"] * 4 + ["ORD"] * 2 + ["RARE"],
            "Airline": ["AA", "DL"] * 6,
            "Distance": [100.0, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
            "Congestion": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )


@pytest.fixture
def small_y() -> pd.Series:
    return pd.Series([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], name=TARGET)


class TestCramersV:
    def test_perfect_determination_is_one(self):
        """When one column fully determines the other, the association is maximal."""
        x = pd.Series(["a", "a", "b", "b", "c", "c"])
        y = pd.Series(["1", "1", "2", "2", "3", "3"])

        assert cramers_v(x, y) == pytest.approx(1.0)

    def test_independent_columns_are_low(self):
        x = pd.Series(list("ababababab" * 4))
        y = pd.Series(list("xxyy" * 10))

        assert cramers_v(x, y) < 0.5

    def test_constant_column_returns_zero(self):
        """A single-category column has no association to measure."""
        x = pd.Series(["a"] * 10)
        y = pd.Series(list("xy" * 5))

        assert cramers_v(x, y) == 0.0

    def test_is_symmetric(self):
        x = pd.Series(["a", "a", "b", "b", "c", "c"])
        y = pd.Series(["1", "2", "2", "3", "3", "1"])

        assert cramers_v(x, y) == pytest.approx(cramers_v(y, x))


class TestEncodeCategoricals:
    def test_native_uses_category_dtype(self, small_df):
        """LightGBM is the only model that reads categories directly."""
        out = encode_categoricals(small_df, ["Origin", "Airline"], "native")

        assert isinstance(out["Origin"].dtype, pd.CategoricalDtype)
        assert isinstance(out["Airline"].dtype, pd.CategoricalDtype)
        assert "Origin" in out.columns

    def test_onehot_replaces_the_original_columns(self, small_df):
        out = encode_categoricals(small_df, ["Origin"], "onehot")

        assert "Origin" not in out.columns
        assert {"Origin_ATL", "Origin_DFW", "Origin_ORD", "Origin_RARE"} <= set(out.columns)

    def test_onehot_columns_are_stored_sparse(self, small_df):
        """Dense dummies on Origin/Dest are what exhausted the RAM in production."""
        out = encode_categoricals(small_df, ["Origin"], "onehot")

        assert isinstance(out["Origin_ATL"].dtype, pd.SparseDtype)

    def test_onehot_downcasts_dense_floats(self, small_df):
        out = encode_categoricals(small_df, ["Origin"], "onehot")

        assert out["Distance"].dtype == np.float32

    def test_absent_columns_are_ignored(self, small_df):
        out = encode_categoricals(small_df, ["Origin", "NotThere"], "onehot")

        assert "Origin_ATL" in out.columns

    def test_input_is_not_mutated(self, small_df):
        before = small_df.copy()
        encode_categoricals(small_df, ["Origin", "Airline"], "native")

        pd.testing.assert_frame_equal(small_df, before)

    def test_unknown_encoding_is_rejected(self, small_df):
        with pytest.raises(ValueError, match="Unknown encoding"):
            encode_categoricals(small_df, ["Origin"], "ordinal")


class TestToSparseMatrix:
    @pytest.fixture
    def encoded(self, small_df):
        return encode_categoricals(
            small_df.drop(columns=["FlightDate"]), ["Origin", "Airline"], "onehot"
        )

    def test_returns_a_csr_matrix(self, encoded):
        matrix = to_sparse_matrix(encoded)

        assert isinstance(matrix, sp.csr_matrix)

    def test_shape_is_preserved(self, encoded):
        matrix = to_sparse_matrix(encoded)

        assert matrix.shape == encoded.shape

    def test_dtype_is_float32(self, encoded):
        assert to_sparse_matrix(encoded).dtype == np.float32

    def test_dense_columns_come_first(self, encoded):
        """Dense-then-sparse ordering is what keeps train and test comparable."""
        matrix = to_sparse_matrix(encoded).toarray()

        np.testing.assert_allclose(
            matrix[:, 0], encoded["Distance"].to_numpy(dtype="float32")
        )

    def test_train_and_test_columns_correspond(self, encoded):
        """Converting aligned frames separately must yield matching layouts."""
        train, test = encoded.iloc[:8], encoded.iloc[8:]
        train, test = align_columns(train, test, "onehot")

        np.testing.assert_allclose(
            to_sparse_matrix(train).toarray()[:, 0],
            train["Distance"].to_numpy(dtype="float32"),
        )
        np.testing.assert_allclose(
            to_sparse_matrix(test).toarray()[:, 0],
            test["Distance"].to_numpy(dtype="float32"),
        )

    def test_dense_only_frame_still_converts(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        assert to_sparse_matrix(df).shape == (2, 2)

    def test_empty_frame_is_rejected(self):
        with pytest.raises(ValueError, match="no columns"):
            to_sparse_matrix(pd.DataFrame(index=[0, 1]))


class TestAlignColumns:
    def test_missing_onehot_columns_are_added(self):
        """A category absent from the test fold must still be a column of zeros."""
        train = pd.DataFrame({"Origin_ATL": [True], "Origin_DFW": [False]})
        test = pd.DataFrame({"Origin_ATL": [True]})

        _, aligned = align_columns(train, test, "onehot")

        assert list(aligned.columns) == ["Origin_ATL", "Origin_DFW"]
        assert not aligned["Origin_DFW"].any()

    def test_extra_test_columns_are_dropped(self):
        """A category seen only at test time is not something the model knows."""
        train = pd.DataFrame({"Origin_ATL": [True]})
        test = pd.DataFrame({"Origin_ATL": [True], "Origin_XXX": [True]})

        _, aligned = align_columns(train, test, "onehot")

        assert list(aligned.columns) == ["Origin_ATL"]

    def test_column_order_follows_train(self):
        train = pd.DataFrame({"b": [1], "a": [2]})
        test = pd.DataFrame({"a": [3], "b": [4]})

        _, aligned = align_columns(train, test, "onehot")

        assert list(aligned.columns) == ["b", "a"]

    def test_train_is_returned_untouched(self):
        train = pd.DataFrame({"Origin_ATL": [True]})
        test = pd.DataFrame({"Origin_ATL": [False]})

        returned_train, _ = align_columns(train, test, "onehot")

        assert returned_train is train

    def test_native_categories_are_taken_from_train(self):
        """The category set is part of the model's input contract."""
        train = pd.DataFrame({"Origin": pd.Categorical(["ATL", "DFW"])})
        test = pd.DataFrame({"Origin": pd.Categorical(["ATL"])})

        _, aligned = align_columns(train, test, "native")

        assert list(aligned["Origin"].cat.categories) == ["ATL", "DFW"]

    def test_native_unseen_category_becomes_missing(self):
        train = pd.DataFrame({"Origin": pd.Categorical(["ATL", "DFW"])})
        test = pd.DataFrame({"Origin": pd.Categorical(["XXX"])})

        _, aligned = align_columns(train, test, "native")

        assert aligned["Origin"].isna().all()

    def test_test_input_is_not_mutated(self):
        train = pd.DataFrame({"a": [1], "b": [2]})
        test = pd.DataFrame({"a": [3]})
        before = test.copy()

        align_columns(train, test, "onehot")

        pd.testing.assert_frame_equal(test, before)


class TestAlignThenSparse:
    """align_columns() feeds to_sparse_matrix(), which orders columns by dtype.

    A fill-in column created as a plain bool lands in the dense block instead of
    among the one-hot dummies, so the two matrices stop meaning the same thing
    column by column - while their DataFrame column names still match, which is
    what makes the fault invisible.
    """

    @pytest.fixture
    def aligned(self):
        train = pd.DataFrame(
            {"Distance": [1.0, 2.0, 3.0], "Origin": ["ATL", "DFW", "ORD"]}
        )
        test = pd.DataFrame({"Distance": [4.0], "Origin": ["ATL"]})

        return align_columns(
            encode_categoricals(train, ["Origin"], "onehot"),
            encode_categoricals(test, ["Origin"], "onehot"),
            "onehot",
        )

    def test_fill_in_columns_keep_the_sparse_dtype(self, aligned):
        _, test = aligned
        dummies = [c for c in test.columns if c.startswith("Origin_")]

        assert all(isinstance(test[c].dtype, pd.SparseDtype) for c in dummies)

    def test_the_same_category_lands_in_the_same_matrix_column(self, aligned):
        """Train row 0 and test row 0 are both ATL flights."""
        train, test = aligned

        train_matrix = to_sparse_matrix(train).toarray()
        test_matrix = to_sparse_matrix(test).toarray()

        np.testing.assert_array_equal(test_matrix[0][1:], train_matrix[0][1:])

    def test_the_matrices_have_the_same_column_layout(self, aligned):
        train, test = aligned

        def layout(df):
            sparse = [c for c in df.columns if isinstance(df[c].dtype, pd.SparseDtype)]
            return [c for c in df.columns if c not in sparse] + sparse

        assert layout(train) == layout(test)


class TestResampleTrainingData:
    @pytest.fixture
    def imbalanced(self):
        X = pd.DataFrame({"a": np.arange(40, dtype=float), "b": np.arange(40, dtype=float)})
        y = pd.Series([0] * 32 + [1] * 8)
        return X, y

    def test_none_returns_the_data_untouched(self, imbalanced):
        X, y = imbalanced
        X_res, y_res = resample_training_data(X, y, "none", "onehot")

        assert X_res is X
        assert y_res is y

    @pytest.mark.parametrize("method", ["undersample", "oversample"])
    def test_classes_end_up_balanced(self, imbalanced, method):
        X, y = imbalanced
        _, y_res = resample_training_data(X, y, method, "onehot")

        counts = y_res.value_counts()
        assert counts[0] == counts[1]

    def test_undersample_shrinks_the_majority(self, imbalanced):
        X, y = imbalanced
        X_res, _ = resample_training_data(X, y, "undersample", "onehot")

        assert len(X_res) == 16

    def test_oversample_grows_the_minority(self, imbalanced):
        X, y = imbalanced
        X_res, _ = resample_training_data(X, y, "oversample", "onehot")

        assert len(X_res) == 64

    def test_works_on_a_sparse_matrix(self, imbalanced):
        """After to_sparse_matrix the training fold is no longer a DataFrame."""
        X, y = imbalanced
        X_res, y_res = resample_training_data(
            sp.csr_matrix(X.to_numpy()), y, "undersample", "onehot"
        )

        assert sp.issparse(X_res)
        assert len(y_res) == 16

    def test_smote_needs_numeric_features(self, imbalanced):
        """SMOTE interpolates between rows, which category dtypes cannot support."""
        X, y = imbalanced
        with pytest.raises(ValueError, match="SMOTE requires numeric features"):
            resample_training_data(X, y, "smote", "native")

    def test_smote_balances_with_onehot(self, imbalanced):
        X, y = imbalanced
        _, y_res = resample_training_data(X, y, "smote", "onehot")

        counts = y_res.value_counts()
        assert counts[0] == counts[1] == 32

    def test_unknown_method_is_rejected(self, imbalanced):
        X, y = imbalanced
        with pytest.raises(ValueError, match="Unknown resampling method"):
            resample_training_data(X, y, "bootstrap", "onehot")


class TestTransformerFit:
    def test_transform_before_fit_is_refused(self, small_df):
        with pytest.raises(RuntimeError, match="must be fitted"):
            Transformer(delay_rate_columns=[]).transform(small_df)

    def test_column_types_are_inferred(self, small_df, small_y):
        t = Transformer(min_category_count=1, delay_rate_columns=[]).fit(small_df, small_y)

        assert set(t.categorical_columns) == {"Origin", "Airline"}
        assert set(t.numeric_columns) == {"Distance", "Congestion"}

    def test_explicit_column_lists_are_respected(self, small_df, small_y):
        t = Transformer(
            min_category_count=1,
            delay_rate_columns=[],
            categorical_columns=["Origin"],
            numeric_columns=["Distance"],
        ).fit(small_df, small_y)

        assert t.categorical_columns == ["Origin"]
        assert t.numeric_columns == ["Distance"]

    def test_service_columns_never_reach_the_model(self, small_df, small_y):
        """transform() drops SERVICE_COLUMNS2 itself; callers must not do it."""
        t = Transformer(min_category_count=1, delay_rate_columns=[]).fit(small_df, small_y)
        out = t.transform(small_df)

        assert not set(out.columns) & set(SERVICE_COLUMNS2)

    def test_rare_categories_are_folded_into_other(self, small_df, small_y):
        t = Transformer(min_category_count=3, delay_rate_columns=[]).fit(small_df, small_y)
        out = t.transform(small_df)

        assert set(out["Origin"]) == {"ATL", "DFW", OTHER}

    def test_unseen_categories_become_other(self, small_df, small_y):
        """A category the model never trained on carries no reliable signal."""
        t = Transformer(min_category_count=1, delay_rate_columns=[]).fit(small_df, small_y)

        fresh = small_df.copy()
        fresh["Origin"] = "BRAND_NEW"
        out = t.transform(fresh)

        assert (out["Origin"] == OTHER).all()

    def test_missing_numeric_values_get_the_training_median(self, small_df, small_y):
        t = Transformer(min_category_count=1, delay_rate_columns=[]).fit(small_df, small_y)

        with_gap = small_df.head(1).copy()
        with_gap["Distance"] = np.nan
        with_median = small_df.head(1).copy()
        with_median["Distance"] = t.impute_values["Distance"]

        assert t.transform(with_gap)["Distance"].iloc[0] == pytest.approx(
            t.transform(with_median)["Distance"].iloc[0]
        )

    def test_numeric_features_are_standardised(self, small_df, small_y):
        t = Transformer(min_category_count=1, delay_rate_columns=[])
        out = t.fit_transform(small_df, small_y)

        assert out["Distance"].mean() == pytest.approx(0.0, abs=1e-9)
        assert out["Distance"].std(ddof=0) == pytest.approx(1.0)

    def test_fit_transform_drops_the_date(self, small_df, small_y):
        out = Transformer(min_category_count=1, delay_rate_columns=[]).fit_transform(
            small_df, small_y
        )

        assert "FlightDate" not in out.columns

    def test_input_frame_is_not_mutated(self, small_df, small_y):
        before = small_df.copy()
        Transformer(min_category_count=1, delay_rate_columns=[]).fit_transform(
            small_df, small_y
        )

        pd.testing.assert_frame_equal(small_df, before)


class TestTransformerDelayRates:
    @pytest.fixture
    def rate_df(self):
        """Six flights out of two airports, in chronological order."""
        return pd.DataFrame(
            {
                "FlightDate": pd.to_datetime(
                    ["2025-01-01", "2025-01-02", "2025-01-03",
                     "2025-01-04", "2025-01-05", "2025-01-06"]
                ),
                "Origin": ["ATL", "ATL", "ATL", "DFW", "DFW", "DFW"],
                "Distance": [100.0, 200, 300, 400, 500, 600],
            }
        )

    @pytest.fixture
    def rate_y(self):
        return pd.Series([1, 1, 0, 0, 0, 1])

    def test_one_rate_column_per_configured_column(self, rate_df, rate_y):
        t = Transformer(
            min_category_count=1, delay_rate_columns=["Origin"], numeric_columns=["Distance"]
        )
        out = t.fit_transform(rate_df, rate_y)

        assert "OriginDelayRate" in out.columns

    def test_absent_columns_produce_no_rate(self, rate_df, rate_y):
        t = Transformer(min_category_count=1, delay_rate_columns=["Origin", "Dest"])
        out = t.fit_transform(rate_df, rate_y)

        assert "DestDelayRate" not in out.columns

    def test_global_rate_is_the_target_mean(self, rate_df, rate_y):
        t = Transformer(min_category_count=1, delay_rate_columns=["Origin"])
        t.fit(rate_df, rate_y)

        assert t.global_delay_rate_ == pytest.approx(0.5)

    def test_training_rates_exclude_the_row_itself(self, rate_df, rate_y):
        """Leave-one-out expanding rates: a row can never see its own label."""
        t = Transformer(
            min_category_count=1,
            delay_rate_columns=["Origin"],
            numeric_columns=["Distance"],
            categorical_columns=["Origin"],
        )
        rates = t._fit_delay_rates_internal(rate_df, rate_y, rate_df["FlightDate"])

        k = t.delay_rate_shrinkage
        # First flight of each airport has no prior history at all.
        assert rates["OriginDelayRate"].iloc[0] == pytest.approx(0.5)
        assert rates["OriginDelayRate"].iloc[3] == pytest.approx(0.5)
        # Second ATL flight has seen exactly one prior ATL flight, which was late.
        assert rates["OriginDelayRate"].iloc[1] == pytest.approx((1 + k * 0.5) / (1 + k))

    def test_scoring_uses_the_frozen_training_stats(self, rate_df, rate_y):
        t = Transformer(min_category_count=1, delay_rate_columns=["Origin"])
        t.fit(rate_df, rate_y)

        k = t.delay_rate_shrinkage
        rates = t._apply_delay_rates_internal(rate_df)

        # ATL: 2 delays out of 3 flights, over the whole training period.
        assert rates["OriginDelayRate"].iloc[0] == pytest.approx((2 + k * 0.5) / (3 + k))

    def test_unseen_category_falls_back_to_the_global_rate(self, rate_df, rate_y):
        t = Transformer(min_category_count=1, delay_rate_columns=["Origin"])
        t.fit(rate_df, rate_y)

        fresh = rate_df.head(1).copy()
        fresh["Origin"] = "XXX"
        rates = t._apply_delay_rates_internal(fresh)

        assert rates["OriginDelayRate"].iloc[0] == pytest.approx(0.5)

    def test_rates_are_order_independent(self, rate_df, rate_y):
        """The expanding window follows the dates, not the row order."""
        t = Transformer(min_category_count=1, delay_rate_columns=["Origin"])
        shuffled = rate_df.iloc[::-1]
        rates = t._fit_delay_rates_internal(
            shuffled, rate_y.iloc[::-1], shuffled["FlightDate"]
        )

        assert rates["OriginDelayRate"].loc[0] == pytest.approx(0.5)


class TestTransformerSelectFeatures:
    """min_mutual_info=-1.0 disables the information filter, so each test below
    isolates exactly one of the four selection criteria."""

    def test_constant_columns_are_dropped(self, small_df, small_y):
        df = small_df.copy()
        df["AlwaysOne"] = 1

        t = Transformer(min_category_count=1, delay_rate_columns=[], min_mutual_info=-1.0)
        prepared = t.fit_transform(df, small_y)
        t.select_features(prepared, small_y)

        assert "AlwaysOne" in t.dropped_features

    def test_near_duplicate_numeric_features_are_dropped(self, small_df, small_y):
        """Of a correlated pair the later column goes, the first one stays."""
        df = small_df.copy()
        df["DistanceCopy"] = df["Distance"] * 2

        t = Transformer(min_category_count=1, delay_rate_columns=[], min_mutual_info=-1.0)
        prepared = t.fit_transform(df, small_y)
        t.select_features(prepared, small_y)

        assert "DistanceCopy" in t.dropped_features
        assert "Distance" not in t.dropped_features

    def test_redundant_categorical_features_are_dropped(self, small_df, small_y):
        """One column fully determining another means one of them is noise."""
        df = small_df.copy()
        df["OriginAlias"] = df["Origin"].map({"ATL": "a", "DFW": "d", "ORD": "o", "RARE": "r"})

        t = Transformer(min_category_count=1, delay_rate_columns=[], min_mutual_info=-1.0)
        prepared = t.fit_transform(df, small_y)
        t.select_features(prepared, small_y)

        assert "OriginAlias" in t.dropped_features
        assert "Origin" not in t.dropped_features

    def test_uninformative_features_are_dropped(self, small_df, small_y):
        """An impossible mutual-information floor drops everything."""
        t = Transformer(min_category_count=1, delay_rate_columns=[], min_mutual_info=10.0)
        prepared = t.fit_transform(small_df, small_y)
        t.select_features(prepared, small_y)

        assert set(t.dropped_features) == set(prepared.columns)

    def test_mutual_information_is_estimated_on_a_sample(self, small_df, small_y):
        """Mutual information is too expensive on millions of rows, so above
        mi_sample_size it is estimated on a random subset instead."""
        t = Transformer(
            min_category_count=1, delay_rate_columns=[], mi_sample_size=5, min_mutual_info=-1.0
        )
        prepared = t.fit_transform(small_df, small_y)
        t.select_features(prepared, small_y)

        # The information filter is disabled here, so only the correlated pair goes:
        # the sampled estimate ran and contributed nothing, which is the point.
        assert t.dropped_features == ["Congestion"]

    def test_the_sample_is_reproducible(self, small_df, small_y):
        """Sampling must not make the selected feature set vary run to run."""
        selections = []
        for _ in range(2):
            t = Transformer(
                min_category_count=1, delay_rate_columns=[], mi_sample_size=5,
                min_mutual_info=0.05,
            )
            prepared = t.fit_transform(small_df, small_y)
            t.select_features(prepared, small_y)
            selections.append(t.dropped_features)

        assert selections[0] == selections[1]

    def test_dropped_features_are_deduplicated_and_sorted(self, small_df, small_y):
        t = Transformer(min_category_count=1, delay_rate_columns=[], min_mutual_info=10.0)
        prepared = t.fit_transform(small_df, small_y)
        t.select_features(prepared, small_y)

        assert t.dropped_features == sorted(set(t.dropped_features))

    def test_apply_selection_removes_exactly_those_columns(self, small_df, small_y):
        df = small_df.copy()
        df["AlwaysOne"] = 1

        t = Transformer(min_category_count=1, delay_rate_columns=[], min_mutual_info=-1.0)
        prepared = t.fit_transform(df, small_y)
        t.select_features(prepared, small_y)
        reduced = t.apply_selection(prepared)

        assert set(reduced.columns) == set(prepared.columns) - set(t.dropped_features)

    def test_apply_selection_tolerates_absent_columns(self, small_df, small_y):
        """Evaluation frames may already lack a column selection wanted dropped."""
        t = Transformer(min_category_count=1, delay_rate_columns=[])
        prepared = t.fit_transform(small_df, small_y)
        t.dropped_features = ["NotThere"]

        pd.testing.assert_frame_equal(t.apply_selection(prepared), prepared)

    def test_selection_is_a_no_op_before_it_runs(self, small_df, small_y):
        t = Transformer(min_category_count=1, delay_rate_columns=[])
        prepared = t.fit_transform(small_df, small_y)

        assert t.dropped_features == []
        pd.testing.assert_frame_equal(t.apply_selection(prepared), prepared)


class TestTransformerPersistence:
    def test_round_trip_preserves_behaviour(self, tmp_path, small_df, small_y):
        t = Transformer(min_category_count=3, delay_rate_columns=["Origin"])
        expected = t.fit_transform(small_df, small_y)

        path = tmp_path / "artifacts" / "transformer.joblib"
        t.save(path)
        restored = Transformer.load(path)

        pd.testing.assert_frame_equal(restored.transform(small_df), expected)

    def test_save_creates_parent_directories(self, tmp_path, small_df, small_y):
        t = Transformer(min_category_count=1, delay_rate_columns=[]).fit(small_df, small_y)
        path = tmp_path / "deep" / "nested" / "transformer.joblib"

        t.save(path)

        assert path.exists()

    def test_fitted_state_survives(self, tmp_path, small_df, small_y):
        t = Transformer(min_category_count=3, delay_rate_columns=["Origin"])
        t.fit(small_df, small_y)
        t.select_features(t.transform(small_df), small_y)

        path = tmp_path / "transformer.joblib"
        t.save(path)
        restored = Transformer.load(path)

        assert restored.category_keep == t.category_keep
        assert restored.impute_values == t.impute_values
        assert restored.dropped_features == t.dropped_features
        assert restored.global_delay_rate_ == t.global_delay_rate_
