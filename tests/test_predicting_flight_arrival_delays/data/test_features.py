"""Tests for predicting_flight_arrival_delays.data.features."""

import numpy as np
import pandas as pd
import pytest

from predicting_flight_arrival_delays.config import SERVICE_COLUMNS, TARGET
from predicting_flight_arrival_delays.data.features import (
    CARRIER_COLUMNS,
    NOWEATHER_DROP,
    VARIANTS,
    build_xy,
    describe_variants,
    get_feature_columns,
    select_features_variant,
)


class TestGetFeatureColumns:
    def test_unknown_variant_is_rejected(self, flights_df):
        """A typo in the variant name must fail loudly, not silently train on 'all'."""
        with pytest.raises(ValueError, match="Unknown variant"):
            get_feature_columns(flights_df, "noweathr")

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_target_and_service_columns_are_always_excluded(self, flights_df, variant):
        """No variant may leak the target or a service column into the model."""
        cols = get_feature_columns(flights_df, variant)

        assert TARGET not in cols
        assert not set(cols) & set(SERVICE_COLUMNS)

    def test_all_variant_keeps_everything_else(self, flights_df):
        expected = [
            c for c in flights_df.columns if c not in set(SERVICE_COLUMNS) | {TARGET}
        ]
        assert get_feature_columns(flights_df, "all") == expected

    def test_noweather_drops_weather_and_lead_days(self, flights_df):
        cols = get_feature_columns(flights_df, "noweather")

        assert not set(cols) & set(NOWEATHER_DROP)
        assert "LeadDays" not in cols

    def test_nocarrier_drops_carrier_identity_only(self, flights_df):
        cols = get_feature_columns(flights_df, "nocarrier")
        expected = [c for c in get_feature_columns(flights_df, "all") if c not in CARRIER_COLUMNS]
        
        assert cols == expected

    def test_original_column_order_is_preserved(self, flights_df):
        """Column order feeds straight into the one-hot layout, so it must be stable."""
        cols = get_feature_columns(flights_df, "noweather")
        positions = [flights_df.columns.get_loc(c) for c in cols]

        assert positions == sorted(positions)

    def test_absent_columns_are_simply_ignored(self, flights_df):
        """A variant may name columns a given dataframe does not carry."""
        df = flights_df.drop(columns=["OriginCarrier", "DestCarrier"])
        cols = get_feature_columns(df, "nocarrier")

        assert "ReportingAirline" not in cols


class TestSelectFeaturesVariant:
    def test_returns_features_plus_target(self, flights_df):
        out = select_features_variant(flights_df, "noweather")

        assert list(out.columns) == get_feature_columns(flights_df, "noweather") + [TARGET]

    def test_rows_with_missing_weather_are_dropped(self, flights_df):
        """'all' is only served to flights whose weather actually matched."""
        df = flights_df.copy()
        df.loc[df.index[:10], "Temperature2mOrigin"] = np.nan

        out = select_features_variant(df, "all", drop_missing_weather=True)

        assert len(out) == len(df) - 10

    def test_missing_weather_kept_when_disabled(self, flights_df):
        df = flights_df.copy()
        df.loc[df.index[:10], "Temperature2mOrigin"] = np.nan

        out = select_features_variant(df, "all", drop_missing_weather=False)

        assert len(out) == len(df)

    def test_a_missing_weather_code_drops_the_row_too(self, flights_df):
        """The code is part of the weather block: an incomplete block means the
        flight belongs to 'noweather'."""
        df = flights_df.copy()
        df.loc[df.index[:7], "WeatherCodeOrigin"] = np.nan

        out = select_features_variant(df, "all", drop_missing_weather=True)

        assert len(out) == len(df) - 7

    def test_noweather_ignores_missing_weather(self, flights_df):
        """Rows without weather are exactly what the 'noweather' model is for."""
        df = flights_df.copy()
        df.loc[df.index[:10], "Temperature2mOrigin"] = np.nan

        out = select_features_variant(df, "noweather", drop_missing_weather=True)

        assert len(out) == len(df)

    def test_input_dataframe_is_not_mutated(self, flights_df):
        before = flights_df.copy()
        select_features_variant(flights_df, "noweather")

        pd.testing.assert_frame_equal(flights_df, before)


class TestBuildXy:
    def test_variant_none_keeps_every_column_but_the_target(self, flights_df):
        X, y = build_xy(flights_df)

        assert TARGET not in X.columns
        assert list(X.columns) == [c for c in flights_df.columns if c != TARGET]
        assert len(y) == len(flights_df)

    def test_variant_applies_the_column_filter(self, flights_df):
        X, _ = build_xy(flights_df, "noweather")

        assert list(X.columns) == get_feature_columns(flights_df, "noweather")

    def test_x_and_y_stay_aligned(self, flights_df):
        """Weather-driven row dropping must not desynchronise features and target."""
        df = flights_df.copy()
        df.loc[df.index[:10], "PrecipitationDest"] = np.nan

        X, y = build_xy(df, "all")

        assert len(X) == len(y)
        assert X.index.equals(y.index)

    def test_target_values_are_untouched(self, flights_df):
        _, y = build_xy(flights_df)

        assert y.name == TARGET
        assert set(y.unique()) <= {0, 1}

    def test_unknown_variant_propagates(self, flights_df):
        with pytest.raises(ValueError, match="Unknown variant"):
            build_xy(flights_df, "nope")


class TestDescribeVariants:
    def test_runs_over_every_variant(self, flights_df):
        """Purely a logging helper: it must not raise or return anything."""
        assert describe_variants(flights_df) is None
