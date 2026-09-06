"""Download the BTS On-Time Performance dataset."""

import shutil
import zipfile

from loguru import logger
import requests
from tqdm import tqdm
import typer

from predicting_flight_arrival_delays.config import (
    BTS_BASE_URL,
    RAW_DATA_DIR,
    YEARS_DATA,
)

app = typer.Typer()


@app.command()
def download_bts_data(
    years: list[int] = typer.Option(list(YEARS_DATA), "--year", help="Year(s) to download."),
    force: bool = typer.Option(
        False, "--force", help="Re-download months already present, overwriting them."
    ),
):
    """Download and extract BTS On-Time Performance data for the given years.

    Fetches one ZIP file per month from the BTS TranStats archive, extracts it
    into a per-month folder under RAW_DATA_DIR, and removes the ZIP afterwards.
    Months already present are skipped unless --force is given. Months not yet
    published on BTS (HTTP 404) are logged and skipped.

    Args:
        years: Year(s) to download. Defaults to YEARS_DATA from config.
        force: Re-download and overwrite months already on disk.
    """
    try:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}

        months = [(y, m) for y in years for m in range(1, 13)]

        if force:
            months_to_download = months
            logger.warning(f"--force given: re-downloading all {len(months)} months")
        else:
            months_to_download = []
            for year, month in months:
                extract_dir = RAW_DATA_DIR / f"{year}_{month:02d}"
                if extract_dir.exists() and any(extract_dir.iterdir()):
                    logger.info(f"Skipping {year}-{month:02d} (already present)")
                    continue
                months_to_download.append((year, month))

        failed_months: list[str] = []

        # Second pass: download
        for year, month in tqdm(months_to_download, desc="Downloading BTS data"):
            extract_dir = RAW_DATA_DIR / f"{year}_{month:02d}"
            url = BTS_BASE_URL.format(year=year, month=month)
            zip_path = RAW_DATA_DIR / f"bts_{year}_{month}.zip"

            try:
                logger.info(f"Downloading {year}-{month:02d}")
                response = requests.get(url, headers=headers, timeout=120)

                # A month not yet published on BTS returns 404: log and move on
                if response.status_code == 404:
                    logger.warning(f"{year}-{month:02d} not available on BTS (404), skipping")
                    continue
                response.raise_for_status()

                if extract_dir.exists():
                    shutil.rmtree(extract_dir)

                zip_path.write_bytes(response.content)

                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(extract_dir)

                for html_file in extract_dir.glob("*.html"):
                    html_file.unlink()

            except requests.RequestException as e:
                logger.error(f"HTTP error for {year}-{month:02d}: {e}")
                failed_months.append(f"{year}-{month:02d}")
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
            except zipfile.BadZipFile:
                logger.error(f"Corrupted ZIP downloaded for {year}-{month:02d}")
                failed_months.append(f"{year}-{month:02d}")
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
            except Exception as e:
                logger.error(f"Error processing {year}-{month:02d}: {e}")
                failed_months.append(f"{year}-{month:02d}")
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
            finally:
                if zip_path.exists():
                    zip_path.unlink()

        if failed_months:
            logger.error(
                f"Download completed with {len(failed_months)} error(s). "
                f"Failed months: {', '.join(failed_months)}"
            )
            raise typer.Exit(code=1)

        logger.success(f"Dataset downloaded successfully to {RAW_DATA_DIR}.")
    except Exception as e:
        logger.exception(f"An error occurred while setting up the download process: {e}")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
