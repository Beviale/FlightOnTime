"""Build airports.csv from BTS's own T_MASTER_CORD table."""

import zipfile
from pathlib import Path
import pandas as pd
import typer
from loguru import logger
from playwright.sync_api import sync_playwright
from timezonefinder import TimezoneFinder

from predicting_flight_arrival_delays.config import (
    EXTERNAL_DATA_DIR,
    EXTERNAL_RAW_DATA_DIR,
    MASTER_CORD_PAGE,
    T_MASTER_CORD_FILE_NAME,
    REQUIRED_AIRPORTS_COLUMNS
)
from predicting_flight_arrival_delays.utils import safe_relative_path, to_pascal_case

app = typer.Typer()

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download() -> None:
    """Download and extract the T_MASTER_CORD table via Playwright.

    Automates a headless browser session to navigate to the BTS TranStats page,
    triggers the download button, extracts the resulting ZIP file into
    EXTERNAL_RAW_DATA_DIR, and cleans up the temporary archive.
    """
    EXTERNAL_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = EXTERNAL_RAW_DATA_DIR / f"{T_MASTER_CORD_FILE_NAME}.zip"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        logger.info(f"Opening {MASTER_CORD_PAGE}")
        page.goto(MASTER_CORD_PAGE, timeout=60_000)

        with page.expect_download(timeout=60_000) as download_info:
            page.click("#btnDownload")
        download_obj = download_info.value

        download_obj.save_as(zip_path)
        browser.close()

    logger.success(f"Saved {safe_relative_path(zip_path)}")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(EXTERNAL_RAW_DATA_DIR)
    zip_path.unlink()
    logger.success(f"Extracted to {safe_relative_path(EXTERNAL_RAW_DATA_DIR)}")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _build(output_path: Path) -> None:
    """Build the airport reference table used by the rest of the pipeline.

    Reads the raw T_MASTER_CORD CSV, filters for active US airports, converts
    coordinates to numeric values, calculates timezones via coordinates,
    restricts columns to REQUIRED_AIRPORTS_COLUMNS + TIMEZONE, and renames columns to PascalCase.

    Args:
        output_path (Path): File path where the processed CSV table will be saved.
    """
    csv_path = EXTERNAL_RAW_DATA_DIR / f"{T_MASTER_CORD_FILE_NAME}.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found -- run `download` first.")

    cord = pd.read_csv(csv_path)

    missing_cols = [c for c in REQUIRED_AIRPORTS_COLUMNS if c not in cord.columns]
    if missing_cols:
        raise SystemExit(f"Missing columns in {csv_path.name}: {missing_cols}")

    before = len(cord)
    cord = cord[cord["AIRPORT_COUNTRY_CODE_ISO"] == "US"].copy()
    logger.info(f"Filtered to US: {before} -> {len(cord)} airports")

    before = len(cord)
    cord = cord[cord["AIRPORT_IS_LATEST"] == 1].copy()
    logger.info(f"Filtered to current records: {before} -> {len(cord)} airports")

    cord["LATITUDE"] = pd.to_numeric(cord["LATITUDE"], errors="coerce")
    cord["LONGITUDE"] = pd.to_numeric(cord["LONGITUDE"], errors="coerce")
    cord = cord.dropna(subset=["LATITUDE", "LONGITUDE"])
    cord = cord.drop_duplicates(subset=["AIRPORT_ID"])

    tf = TimezoneFinder()
    cord["TIMEZONE"] = cord.apply(
        lambda r: tf.timezone_at(lat=r["LATITUDE"], lng=r["LONGITUDE"]), axis=1
    )

    n_missing_tz = cord["TIMEZONE"].isna().sum()
    if n_missing_tz:
        logger.warning(f"{n_missing_tz} airports got no timezone match from coordinates")

    result = cord[["AIRPORT_ID", "AIRPORT", "LATITUDE", "LONGITUDE", "TIMEZONE"]].rename(
        columns={
            "AIRPORT": "Iata",
            **{c: to_pascal_case(c) for c in ["AIRPORT_ID", "LATITUDE", "LONGITUDE", "TIMEZONE"]},
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.success(f"Saved {len(result)} airports to {safe_relative_path(output_path)}")


# ---------------------------------------------------------------------------
# CLI Command
# ---------------------------------------------------------------------------

@app.command()
def run(
    force_download: bool = typer.Option(
        True, help="Re-download even if the file exists."
    ),
    output_path: Path = typer.Option(
        EXTERNAL_DATA_DIR / "airports.csv", help="Output CSV file path for processed airport data."
    ),
) -> None:
    """Fetch the raw airport table if needed, then build the reference CSV.

    Args:
        force_download (bool): Force re-downloading the raw source table 
            even if the corresponding CSV already exists on disk.
        output_path (Path): Path where the final processed CSV file will be saved.
    """
    try:
        csv_path = EXTERNAL_RAW_DATA_DIR / f"{T_MASTER_CORD_FILE_NAME}.csv"
        if force_download or not csv_path.exists():
            _download()
        _build(output_path)
    except Exception as e:
        logger.exception(f"An error occurred while creating the airport reference table: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()