# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "deepchecks==0.19.1",
#   "setuptools<81",
#   "scikit-learn<1.6",
#   "pandas<2.3",
#   "numpy<2",
#   "pyarrow",
# ]
# ///
"""Compare what the service scored against what it was trained on."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


NOT_FEATURES = {
    "ObservedAt",
    "Variant",
    "FlightDate",
    "LeadDays",
    "Month",
    "DaysToNearestHoliday",
}


def build_reference(source: Path, destination: Path, rows: int) -> pd.DataFrame:
    """Draw a reference sample from the training frame, once.

    Args:
        source: The preprocessed parquet the models were trained on.
        destination: Where to cache the sample.
        rows: How many rows to keep.

    Returns:
        The reference frame.

    Raises:
        SystemExit: If neither the cache nor the training frame exists.
    """
    if destination.exists():
        return pd.read_csv(destination, low_memory=False)

    if not source.exists():
        raise SystemExit(
            f"No reference at {destination} and no training frame at {source}. "
            "Run 'dvc pull data/interim/flights_preprocessed.parquet' first."
        )

    frame = pd.read_parquet(source)
    sample = frame.sample(n=min(rows, len(frame)), random_state=42)

    destination.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(destination, index=False)
    print(f"Reference built from {len(frame)} training rows, kept {len(sample)}.")
    return sample


def recent(production: pd.DataFrame, days: int, column: str = "ObservedAt") -> pd.DataFrame:
    """Keep only the rows scored within the window.

    Args:
        production: Everything the service has logged.
        days: How far back to look. Zero or less means the whole file.
        column: The timestamp the collector writes on every row.

    Returns:
        The rows inside the window, or all of them if there is no way to tell.
    """
    if days <= 0:
        return production

    if column not in production.columns:
        print(f"No {column} column: comparing the whole file.")
        return production

    stamped = pd.to_datetime(production[column], errors="coerce", utc=True)
    cutoff = datetime.now(UTC) - timedelta(days=days)
    return production[stamped >= cutoff]


def shared_features(reference: pd.DataFrame, production: pd.DataFrame) -> list[str]:
    """The columns both sides carry and that describe the flight itself."""
    return sorted((set(reference.columns) & set(production.columns)) - NOT_FEATURES)


def drift_scores(reference: pd.DataFrame, production: pd.DataFrame, columns: list[str]) -> dict:
    """Run the check and reduce it to one number per column.

    Args:
        reference: The training sample.
        production: What the service has scored.
        columns: The columns to compare.

    Returns:
        Column name to drift score.
    """
    reference, production = reference[columns], production[columns]

    categorical = [c for c in columns if not pd.api.types.is_numeric_dtype(reference[c])]

    result = FeatureDrift().run(
        train_dataset=Dataset(reference, label=None, cat_features=categorical),
        test_dataset=Dataset(production, label=None, cat_features=categorical),
    )
    return {name: float(info["Drift score"]) for name, info in result.value.items()}


def main() -> int:
    """Write one drift report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production", type=Path, default=ROOT / "data/interim/production_features.csv"
    )
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "data/interim/drift_reference.csv"
    )
    parser.add_argument(
        "--training", type=Path, default=ROOT / "data/interim/flights_preprocessed.parquet"
    )
    parser.add_argument("--reports", type=Path, default=ROOT / "reports/drift")
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="How many days back to compare. 0 compares the whole file.",
    )
    parser.add_argument("--reference-rows", type=int, default=50_000)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    if not args.production.exists():
        print(f"Nothing scored yet: {args.production} does not exist.")
        return 0

    logged = pd.read_csv(args.production, low_memory=False)
    production = recent(logged, args.days)

    if len(production) < args.min_rows:
        window = "in total" if args.days <= 0 else f"in the last {args.days} days"
        print(
            f"Only {len(production)} rows {window} of {len(logged)} logged, "
            f"{args.min_rows} needed. Nothing to compare."
        )
        return 0

    reference = build_reference(args.training, args.reference, args.reference_rows)
    columns = shared_features(reference, production)
    if not columns:
        raise SystemExit("The two frames share no comparable column.")

    scores = drift_scores(reference, production, columns)
    above = sorted(
        (c for c, s in scores.items() if s >= args.threshold), key=scores.get, reverse=True
    )

    report = {
        "check": "FeatureDrift",
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "window_days": args.days,
        "rows": {
            "reference": len(reference),
            "production": len(production),
            "logged": len(logged),
        },
        "threshold": args.threshold,
        "scores": dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)),
        "drifted": above,
    }

    args.reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    path = args.reports / f"drift_{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    window = "ever" if args.days <= 0 else f"in the last {args.days} days"
    print(f"Compared {len(columns)} columns over {len(production)} flights scored {window}.")
    if above:
        print(f"Above {args.threshold}: " + ", ".join(f"{c} ({scores[c]:.3f})" for c in above))
    else:
        print(f"Nothing above {args.threshold}. Highest: {max(scores, key=scores.get)}")
    print(f"Report: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
