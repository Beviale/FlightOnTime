"""Tests for predicting_flight_arrival_delays.data.split_data."""

import pandas as pd
import pytest
from typer.testing import CliRunner

from predicting_flight_arrival_delays.config import DATE_COLUMN, TARGET
from predicting_flight_arrival_delays.data.split_data import Fold, app, make_folds

runner = CliRunner()


@pytest.fixture
def dated_df():
    """One flight a day, 1 January to 10 April 2025."""
    dates = pd.date_range("2025-01-01", "2025-04-10", freq="D")
    return pd.DataFrame({DATE_COLUMN: dates, "value": range(len(dates))})


class TestFold:
    def test_str_shows_both_windows(self):
        fold = Fold(
            index=2,
            train_start=pd.Timestamp("2025-01-01"),
            test_start=pd.Timestamp("2025-03-01"),
            test_end=pd.Timestamp("2025-04-01"),
        )

        assert str(fold) == "fold 2: train 2025-01-01..2025-03-01 | test 2025-03-01..2025-04-01"


class TestMakeFolds:
    def test_one_fold_per_cut_point(self, dated_df):
        """Each cut point opens a test window; the last one runs to the end."""
        folds = make_folds(dated_df, "2025-02-01,2025-03-01")

        assert len(folds) == 2

    def test_folds_are_numbered_from_one(self, dated_df):
        folds = make_folds(dated_df, "2025-02-01,2025-03-01")

        assert [f.index for f in folds] == [1, 2]

    def test_training_window_expands(self, dated_df):
        """Every fold trains from the very first flight; only the end moves."""
        folds = make_folds(dated_df, "2025-02-01,2025-03-01")

        assert {f.train_start for f in folds} == {pd.Timestamp("2025-01-01")}
        assert [f.test_start for f in folds] == [
            pd.Timestamp("2025-02-01"),
            pd.Timestamp("2025-03-01"),
        ]

    def test_test_windows_are_contiguous(self, dated_df):
        folds = make_folds(dated_df, "2025-02-01,2025-03-01")

        for previous, following in zip(folds, folds[1:]):
            assert previous.test_end == following.test_start

    def test_last_window_ends_one_day_past_the_data(self, dated_df):
        """The end boundary is exclusive, so the final day must still be included."""
        folds = make_folds(dated_df, "2025-02-01")

        assert folds[-1].test_end == pd.Timestamp("2025-04-11")

    def test_cut_points_are_sorted(self, dated_df):
        folds = make_folds(dated_df, "2025-03-01,2025-02-01")

        assert [f.test_start for f in folds] == [
            pd.Timestamp("2025-02-01"),
            pd.Timestamp("2025-03-01"),
        ]

    def test_whitespace_around_cut_points_is_tolerated(self, dated_df):
        assert len(make_folds(dated_df, " 2025-02-01 , 2025-03-01 ")) == 2

    @pytest.mark.parametrize("cut", ["2024-12-01", "2025-06-01"])
    def test_cut_point_outside_the_data_is_rejected(self, dated_df, cut):
        """A cut outside the range would silently produce an empty split."""
        with pytest.raises(ValueError, match="outside data range"):
            make_folds(dated_df, cut)

    def test_error_names_the_offending_cut_points(self, dated_df):
        with pytest.raises(ValueError, match="2025-06-01"):
            make_folds(dated_df, "2025-02-01,2025-06-01")


class TestSplitFoldsCommand:
    @pytest.fixture
    def preprocessed(self, tmp_path, flights_df):
        path = tmp_path / "flights_preprocessed.parquet"
        flights_df.to_parquet(path, index=False)
        return path

    @pytest.fixture
    def output_dir(self, tmp_path, preprocessed):
        out = tmp_path / "selection"
        result = runner.invoke(
            app,
            [
                "--input-path", str(preprocessed),
                "--output-dir", str(out),
                "--variants", "all",
                "--cut-points", "2025-02-01,2025-02-20,2025-03-10",
            ],
        )
        assert result.exit_code == 0, result.output
        return out

    def test_one_directory_per_fold(self, output_dir):
        fold_dirs = sorted((output_dir / "all").glob("fold_*"))

        assert [d.name for d in fold_dirs] == [
            "fold_1_with_val",
            "fold_2_with_val",
            "fold_3_with_val",
        ]

    def test_each_fold_has_all_three_splits(self, output_dir):
        for fold_dir in (output_dir / "all").glob("fold_*"):
            assert (fold_dir / "train.parquet").exists()
            assert (fold_dir / "validation.parquet").exists()
            assert (fold_dir / "test.parquet").exists()

    def test_splits_are_chronological(self, output_dir):
        """Validation must sit strictly after train, and test after validation."""
        fold_dir = output_dir / "all" / "fold_1_with_val"
        train = pd.read_parquet(fold_dir / "train.parquet")
        validation = pd.read_parquet(fold_dir / "validation.parquet")
        test = pd.read_parquet(fold_dir / "test.parquet")

        assert train[DATE_COLUMN].max() < validation[DATE_COLUMN].min()
        assert validation[DATE_COLUMN].max() < test[DATE_COLUMN].min()

    def test_splits_do_not_overlap(self, output_dir):
        fold_dir = output_dir / "all" / "fold_2_with_val"
        train = pd.read_parquet(fold_dir / "train.parquet")
        test = pd.read_parquet(fold_dir / "test.parquet")

        assert train[DATE_COLUMN].max() < test[DATE_COLUMN].min()

    def test_training_window_grows_across_folds(self, output_dir):
        sizes = [
            len(pd.read_parquet(output_dir / "all" / f"fold_{i}_with_val" / "train.parquet"))
            for i in (1, 2, 3)
        ]

        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_variant_columns_are_applied(self, output_dir):
        """The folds carry the variant's feature set, not the raw dataset."""
        train = pd.read_parquet(output_dir / "all" / "fold_1_with_val" / "train.parquet")

        assert TARGET in train.columns
        assert "TailNumber" not in train.columns

    def test_val_frac_zero_writes_only_train_and_test(self, tmp_path, preprocessed):
        out = tmp_path / "no_val"
        result = runner.invoke(
            app,
            [
                "--input-path", str(preprocessed),
                "--output-dir", str(out),
                "--variants", "noweather",
                "--cut-points", "2025-03-01",
                "--val-frac", "0",
            ],
        )

        assert result.exit_code == 0, result.output
        fold_dir = out / "noweather" / "fold_1_without_val"
        assert (fold_dir / "train.parquet").exists()
        assert not (fold_dir / "validation.parquet").exists()

    def test_several_variants_in_one_run(self, tmp_path, preprocessed):
        out = tmp_path / "multi"
        result = runner.invoke(
            app,
            [
                "--input-path", str(preprocessed),
                "--output-dir", str(out),
                "--variants", "all",
                "--variants", "noweather",
                "--cut-points", "2025-03-01",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (out / "all").is_dir()
        assert (out / "noweather").is_dir()

    def test_missing_input_exits_with_an_error_code(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "--input-path", str(tmp_path / "absent.parquet"),
                "--output-dir", str(tmp_path / "out"),
                "--cut-points", "2025-03-01",
            ],
        )

        assert result.exit_code == 1

    def test_invalid_cut_point_exits_with_an_error_code(self, tmp_path, preprocessed):
        result = runner.invoke(
            app,
            [
                "--input-path", str(preprocessed),
                "--output-dir", str(tmp_path / "out"),
                "--variants", "all",
                "--cut-points", "2030-01-01",
            ],
        )

        assert result.exit_code == 1
