"""Tests for predicting_flight_arrival_delays.modeling.train_evaluate_save_metrics."""

import pandas as pd
import pytest
import scipy.sparse as sp

from predicting_flight_arrival_delays.config import TARGET, DATE_COLUMN
from predicting_flight_arrival_delays.data.features import select_features_variant
from predicting_flight_arrival_delays.data.transform import Transformer
from predicting_flight_arrival_delays.modeling import (
    train_evaluate_save_metrics as tesm_module,
)
from predicting_flight_arrival_delays.modeling.train_evaluate_save_metrics import prepare_fold


@pytest.fixture(autouse=True)
def small_transformer(monkeypatch):
    """The 1000-row category floor would fold every category into OTHER here."""

    def build(**kwargs):
        return Transformer(min_category_count=5, **kwargs)

    monkeypatch.setattr(tesm_module, "Transformer", build)


@pytest.fixture
def fold(flights_df):
    """One walk-forward fold: train, validation and test, split by date."""
    df = select_features_variant(flights_df, "noweather").sort_values(DATE_COLUMN)
    return df.iloc[:180], df.iloc[180:240], df.iloc[240:]


class TestPrepareFold:
    def test_onehot_yields_sparse_matrices(self, fold):
        """Dense one-hot on Origin/Dest is what exhausted the RAM in production."""
        train_df, validation_df, test_df = fold

        X_fit, _, X_val, _, X_test, _ = prepare_fold(
            train_df, "onehot", test_df=test_df, validation_df=validation_df
        )

        assert sp.issparse(X_fit)
        assert sp.issparse(X_val)
        assert sp.issparse(X_test)

    def test_native_keeps_dataframes(self, fold):
        """LightGBM reads category dtype directly, so nothing is converted."""
        train_df, validation_df, test_df = fold

        X_fit, _, X_val, _, X_test, _ = prepare_fold(
            train_df, "native", test_df=test_df, validation_df=validation_df
        )

        assert isinstance(X_fit, pd.DataFrame)
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)

    def test_every_split_ends_up_with_the_same_width(self, fold):
        """Alignment happens before the sparse conversion, while names still exist."""
        train_df, validation_df, test_df = fold

        X_fit, _, X_val, _, X_test, _ = prepare_fold(
            train_df, "onehot", test_df=test_df, validation_df=validation_df
        )

        assert X_fit.shape[1] == X_val.shape[1] == X_test.shape[1]

    def test_row_counts_are_preserved_without_resampling(self, fold):
        train_df, validation_df, test_df = fold

        X_fit, y_fit, X_val, y_val, X_test, y_test = prepare_fold(
            train_df, "onehot", test_df=test_df, validation_df=validation_df
        )

        assert X_fit.shape[0] == len(train_df) == len(y_fit)
        assert X_val.shape[0] == len(validation_df) == len(y_val)
        assert X_test.shape[0] == len(test_df) == len(y_test)

    def test_validation_is_optional(self, fold):
        train_df, _, test_df = fold

        _, _, X_val, y_val, X_test, _ = prepare_fold(train_df, "onehot", test_df=test_df)

        assert X_val is None
        assert y_val is None
        assert X_test is not None

    def test_test_is_optional(self, fold):
        """Phase 1 never touches the test set - it does not even read the file."""
        train_df, validation_df, _ = fold

        _, _, X_val, _, X_test, y_test = prepare_fold(
            train_df, "onehot", validation_df=validation_df
        )

        assert X_test is None
        assert y_test is None
        assert X_val is not None

    def test_resampling_balances_only_the_training_fold(self, fold):
        train_df, validation_df, test_df = fold

        _, y_fit, _, y_val, _, y_test = prepare_fold(
            train_df, "onehot", test_df=test_df,
            validation_df=validation_df, resample="undersample",
        )

        counts = y_fit.value_counts()
        assert counts[0] == counts[1]
        assert len(y_val) == len(validation_df)
        assert len(y_test) == len(test_df)

    def test_resampling_works_after_the_sparse_conversion(self, fold):
        train_df, _, _ = fold

        X_fit, y_fit, _, _, _, _ = prepare_fold(
            train_df, "onehot", resample="oversample"
        )

        assert sp.issparse(X_fit)
        assert X_fit.shape[0] == len(y_fit)

    def test_smote_is_refused_for_native_encoding(self, fold):
        train_df, _, _ = fold

        with pytest.raises(ValueError, match="SMOTE requires numeric features"):
            prepare_fold(train_df, "native", resample="smote")

    def test_the_target_is_split_off_not_encoded(self, fold):
        train_df, _, _ = fold

        _, y_fit, _, _, _, _ = prepare_fold(train_df, "onehot")

        assert y_fit.name == TARGET
        assert set(y_fit.unique()) <= {0, 1}
