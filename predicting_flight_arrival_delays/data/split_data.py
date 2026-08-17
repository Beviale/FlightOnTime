"""Walk-forward splitting by flight date.

The evaluation is repeated over several expanding-window folds: the training
period grows, the test period slides forward. Metrics are averaged across folds,
which also shows how stable performance is over time.

    Fold 1: train [start .. t1)   test [t1 .. t2)
    Fold 2: train [start .. t2)   test [t2 .. t3)
    Fold 3: train [start .. t3)   test [t3 .. end)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd
from loguru import logger
import typer
from predicting_flight_arrival_delays.config import (
    DATE_COLUMN,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    PRODUCTION_VARIANTS,
)
from predicting_flight_arrival_delays.data.features import select_features_variant, VARIANTS
from predicting_flight_arrival_delays.utils import safe_relative_path

app = typer.Typer()


@dataclass
class Fold:
    """Represents a single walk-forward cross-validation fold boundary.

    Attributes:
        index (int): Sequential index of the fold (1-based).
        train_start (pd.Timestamp): Start timestamp for the training period.
        test_start (pd.Timestamp): Start timestamp for the test period (end of training).
        test_end (pd.Timestamp): End timestamp for the test period.
    """

    index: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __str__(self) -> str:
        return (
            f"fold {self.index}: train {self.train_start.date()}..{self.test_start.date()} "
            f"| test {self.test_start.date()}..{self.test_end.date()}"
        )


def make_folds(df: pd.DataFrame, cut_points: str) -> list[Fold]:
    """Calculate expanding-window fold boundaries based on specified cut points.

    Args:
        df (pd.DataFrame): Dataframe containing flight records with a valid date column.
        cut_points (str): Comma-separated date strings marking the start of each test split.

    Returns:
        list[Fold]: A list of Fold data structures containing start and end timestamps.

    Raises:
        ValueError: If any cut point falls outside the minimum and maximum date range of the dataset.
    """
    dates = pd.to_datetime(df[DATE_COLUMN]).astype("datetime64[ns]")
    start, end = dates.min(), dates.max() + pd.DateOffset(days=1)

    cuts = [pd.Timestamp(c.strip()) for c in cut_points.split(",")]
    invalid = [c for c in cuts if not (start < c < end)]
    if invalid:
        raise ValueError(
            f"Cut points outside data range {start.date()}..{end.date()}: "
            f"{[str(c.date()) for c in invalid]}"
        )

    boundaries = sorted(cuts) + [end]
    folds = [
        Fold(
            index=i + 1,
            train_start=start,
            test_start=boundaries[i],
            test_end=boundaries[i + 1],
        )
        for i in range(len(boundaries) - 1)
    ]

    for f in folds:
        logger.info(str(f))
    return folds


@app.command()
def split_folds(
    input_path: Path = typer.Option(
        INTERIM_DATA_DIR / "flights_preprocessed.parquet",
        help="Path to preprocessed Parquet dataset",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "selection",
        help="Root directory where processed fold subdirectories will be created",
    ),
    variants: List[str] = typer.Option(
        VARIANTS,
        help="List of feature variants to generate splits for",
    ),
    cut_points: str = typer.Option(
        "2025-07-01,2025-10-01,2026-01-01",
        help="Comma-separated dates where each walk-forward test window starts",
    ),
    val_frac: float = typer.Option(
        0.2,
        help="Fraction of training data to reserve for validation split (0.0 disables validation)",
    ),
) -> None:
    """Generates walk-forward folds and writes train/validation/test Parquet splits to disk.

    Args:
        input_path (Path): File path to input preprocessed Parquet dataset.
        output_dir (Path): Destination base directory for processed splits.
        variants (List[str]): Feature set variants to generate splits for.
        cut_points (str): Comma-separated string of split dates.
        val_frac (float): Fraction of training window to carve out for validation.
    """
    try:
        complete_df = pd.read_parquet(input_path)
        complete_df[DATE_COLUMN] = pd.to_datetime(complete_df[DATE_COLUMN]).astype("datetime64[ns]")
        complete_df = complete_df.sort_values(DATE_COLUMN)

        for variant in variants:
            logger.info(f"Processing variant '{variant}'...")
            df = select_features_variant(complete_df.copy(), variant)
            folds = make_folds(df, cut_points=cut_points)

            for fold in folds:
                fold_suffix = "with_val" if val_frac > 0 else "without_val"
                fold_dir = output_dir / variant / f"fold_{fold.index}_{fold_suffix}"
                fold_dir.mkdir(parents=True, exist_ok=True)

                train_df = df[
                    (df[DATE_COLUMN] >= fold.train_start)
                    & (df[DATE_COLUMN] < fold.test_start)
                ]

                test_df = df[
                    (df[DATE_COLUMN] >= fold.test_start)
                    & (df[DATE_COLUMN] < fold.test_end)
                ]

                validation_df = None
                if val_frac > 0:
                    cutoff = train_df[DATE_COLUMN].quantile(1 - val_frac)
                    validation_df = train_df[train_df[DATE_COLUMN] > cutoff]
                    train_df = train_df[train_df[DATE_COLUMN] <= cutoff].copy()

                train_path = fold_dir / "train.parquet"
                test_path = fold_dir / "test.parquet"

                train_df.to_parquet(train_path, index=False)
                test_df.to_parquet(test_path, index=False)

                if validation_df is not None:
                    validation_path = fold_dir / "validation.parquet"
                    validation_df.to_parquet(validation_path, index=False)
                    logger.info(
                        f"Fold {fold.index} written: train={len(train_df)} rows -> {safe_relative_path(train_path)} | "
                        f"val={len(validation_df)} rows -> {safe_relative_path(validation_path)} | "
                        f"test={len(test_df)} rows -> {safe_relative_path(test_path)}"
                    )
                else:
                    logger.info(
                        f"Fold {fold.index} written: train={len(train_df)} rows -> {safe_relative_path(train_path)} | "
                        f"test={len(test_df)} rows -> {safe_relative_path(test_path)}"
                    )

        logger.success(f"All the folds have been created and saved at {output_dir}")
    except Exception as e:
        logger.exception(f"An error occurred while creating the folds: {e}")
        raise typer.Exit(code=1) from None



@app.command()
def split_final_folds(
    input_path: Path = typer.Option(
        INTERIM_DATA_DIR / "flights_preprocessed.parquet",
        help="Path to preprocessed Parquet dataset",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "final",
        help="Root directory where processed fold subdirectories will be created",
    ),
    variants: List[str] = typer.Option(
        PRODUCTION_VARIANTS,
        help="List of feature variants to generate splits for",
    ),
    val_frac: float = typer.Option(
        0.2,
        help="Fraction of data to reserve for validation split (0.0 disables validation)",
    ),
) -> None:
    """Generates a fold for each variant and writes train and validation Parquet splits to disk.

    Args:
        input_path (Path): File path to input preprocessed Parquet dataset.
        output_dir (Path): Destination base directory for processed splits.
        variants (List[str]): Feature set variants to generate splits for.
        val_frac (float): Fraction of data to reserve for validation split.
    """
    try:
        complete_df = pd.read_parquet(input_path)
        complete_df[DATE_COLUMN] = pd.to_datetime(complete_df[DATE_COLUMN]).astype("datetime64[ns]")
        complete_df = complete_df.sort_values(DATE_COLUMN)

    
        for variant in variants:
            logger.info(f"Processing variant '{variant}'...")
            df = select_features_variant(complete_df.copy(), variant)

            cutoff = df[DATE_COLUMN].quantile(1 - val_frac)
            train_df = df[df[DATE_COLUMN] <= cutoff]
            validation_df = df[df[DATE_COLUMN] > cutoff]
    
            variant_dir = output_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
        
            train_path = variant_dir / "train.parquet"
            validation_path = variant_dir / "validation.parquet"
    
            train_df.to_parquet(train_path, index=False)
            validation_df.to_parquet(validation_path, index=False)
    
            logger.info(
                f"{variant}: train={len(train_df)} rows -> {safe_relative_path(train_path)} | "
                f"val={len(validation_df)} rows -> {safe_relative_path(validation_path)}"
            )
    
        logger.success(f"Final train/validation splits written to {output_dir}")
    except Exception as e:
        logger.exception(f"An error occurred while creating the folds: {e}")
        raise typer.Exit(code=1) from None

if __name__ == "__main__":
    app()