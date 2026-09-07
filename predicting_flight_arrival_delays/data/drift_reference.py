
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from predicting_flight_arrival_delays.app.enrichment.builder import FEATURE_FRAME_COLUMNS
from predicting_flight_arrival_delays.config import (
    DRIFT_REFERENCE_CSV,
    DRIFT_REFERENCE_ROWS,
    INTERIM_DATA_DIR,
    SEED,
)

app = typer.Typer()


@app.command()
def build(
    source: Path = INTERIM_DATA_DIR / "flights_preprocessed.parquet",
    destination: Path = DRIFT_REFERENCE_CSV,
    rows: int = DRIFT_REFERENCE_ROWS,
) -> None:
    """Draw the reference sample and write it out.

    Args:
        source: The preprocessed flights the models were trained on.
        destination: Where the sample goes.
        rows: How many flights to keep.

    Raises:
        typer.Exit: If the training frame has not been pulled.
    """
    if not source.exists():
        logger.error(f"{source} is missing - run `dvc pull` first.")
        raise typer.Exit(code=1)

    frame = pd.read_parquet(source)
    comparable = [column for column in frame.columns if column in set(FEATURE_FRAME_COLUMNS)]
    sample = frame[comparable].sample(n=min(rows, len(frame)), random_state=SEED)

    destination.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(destination, index=False)

    size_mb = destination.stat().st_size / 1024**2
    logger.success(
        f"Drift reference: {len(sample)} of {len(frame)} flights, "
        f"{len(comparable)} of {frame.shape[1]} columns, {size_mb:.1f} MB"
    )


if __name__ == "__main__":
    app()
