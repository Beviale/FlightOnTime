
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
import pandas as pd

from predicting_flight_arrival_delays.config import PRODUCTION_FEATURES_CSV

OBSERVED_AT = "ObservedAt"


def record_features(
    frame: pd.DataFrame,
    variants: list[str] | None = None,
    path: Path = PRODUCTION_FEATURES_CSV,
) -> int:
    """Append one row per scored flight to the production log.

    Args:
        frame: The enriched feature frame, exactly as the models received it.
        variants: The model that answered each row, if known, kept alongside so the
            comparison can be run per variant.
        path: Where to append. The header is written only when the file is new.

    Returns:
        How many rows were written, zero if the write failed.
    """
    if frame.empty:
        return 0

    try:
        rows = frame.copy()
        rows[OBSERVED_AT] = datetime.now(UTC).isoformat(timespec="seconds")
        if variants is not None:
            rows["Variant"] = variants

        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path, mode="a", index=False, header=not path.exists())
    except Exception as error:
        logger.warning(f"Could not record {len(frame)} scored flights for drift: {error}")
        return 0

    return len(frame)
