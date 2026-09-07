"""Running the drift comparison on its own, once a day."""

from pathlib import Path
import shutil
import subprocess
import sys

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from predicting_flight_arrival_delays import config
from predicting_flight_arrival_delays.config import (
    DRIFT_JOB_HOUR,
    DRIFT_JOB_MINUTE,
    DRIFT_JOB_TIMEOUT_SECONDS,
    DRIFT_REFERENCE_ROWS,
    DRIFT_SCORE_THRESHOLD,
    DRIFT_WINDOW_DAYS,
    MIN_DRIFT_ROWS,
    PROJ_ROOT,
)

DRIFT_SCRIPT = PROJ_ROOT / "scripts" / "drift_report.py"
JOB_ID = "drift_report"

scheduler = AsyncIOScheduler()


def _runner() -> Path | None:
    """The uv executable, which builds the script's pinned environment.

    Returns:
        Its path, or None if uv is not on this machine.
    """
    found = shutil.which("uv")
    return Path(found) if found else None


def drift_job() -> None:
    if not DRIFT_SCRIPT.exists():
        logger.warning(f"No drift script at {DRIFT_SCRIPT}; skipping.")
        return

    uv = _runner()
    if uv is None:
        logger.warning("uv is not installed here, so the drift script cannot be run.")
        return

    
    settings = [
        "--days",
        str(DRIFT_WINDOW_DAYS),
        "--min-rows",
        str(MIN_DRIFT_ROWS),
        "--threshold",
        str(DRIFT_SCORE_THRESHOLD),
        "--reference-rows",
        str(DRIFT_REFERENCE_ROWS),
    ]

    logger.info(f"Running the drift comparison over the last {DRIFT_WINDOW_DAYS} days")
    try:
        finished = subprocess.run(
            [str(uv), "run", str(DRIFT_SCRIPT), *settings],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            timeout=DRIFT_JOB_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"The drift comparison ran past {DRIFT_JOB_TIMEOUT_SECONDS}s and was killed.")
        return
    except Exception as error:
        logger.error(f"Could not run the drift comparison: {error}")
        return

    output = (finished.stdout or finished.stderr or "").strip()
    if finished.returncode == 0:
        logger.success(f"Drift comparison finished. {output.splitlines()[-1] if output else ''}")
    else:
        logger.error(f"Drift comparison failed ({finished.returncode}): {output[-2000:]}")


def start_scheduler() -> bool:
    """Schedule the daily comparison, if it is wanted and not already running.

    Returns:
        Whether the scheduler is running when this returns.
    """
    if not config.DRIFT_SCHEDULE_ENABLED:
        logger.info("Drift schedule is off; no comparison will run on its own.")
        return False

    if scheduler.running:
        return True

    scheduler.add_job(
        drift_job,
        trigger=CronTrigger(hour=DRIFT_JOB_HOUR, minute=DRIFT_JOB_MINUTE),
        id=JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    try:
        scheduler.start()
    except Exception as error:
        logger.warning(f"Could not start the drift scheduler: {error}")
        return False

    logger.info(f"Drift comparison scheduled daily at {DRIFT_JOB_HOUR:02d}:{DRIFT_JOB_MINUTE:02d}")
    return True


def shutdown_scheduler() -> None:
    """Stop the scheduler without waiting for a comparison in flight.
    """
    if not scheduler.running:
        return

    try:
        scheduler.shutdown(wait=False)
    except RuntimeError as error:
        logger.debug(f"Drift scheduler was already past stopping: {error}")
    else:
        logger.info("Drift scheduler stopped")


if __name__ == "__main__":
    drift_job()
    sys.exit(0)
