"""Turn the pytest HTML report into a PDF."""

from pathlib import Path

from loguru import logger
from playwright.sync_api import sync_playwright
import typer

from predicting_flight_arrival_delays.config import PYTEST_REPORTS_DIR

app = typer.Typer()


@app.command()
def convert(
    source: Path = PYTEST_REPORTS_DIR / "report.html",
    destination: Path = PYTEST_REPORTS_DIR / "report.pdf",
    title: str = "FlightOnTime - test report",
    keep_html: bool = typer.Option(False, help="Leave the HTML in place as well"),
) -> None:
    """Render the report and write it as a PDF.

    Args:
        source: The HTML pytest-html wrote.
        destination: Where the PDF goes.
        title: What to put at the top, in place of the source filename.
        keep_html: Whether to leave the HTML behind.

    Raises:
        typer.Exit: If there is no report to convert.
    """
    if not source.exists():
        logger.error(f"No report at {source} - run pytest first.")
        raise typer.Exit(code=1)

    destination.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(source.resolve().as_uri(), wait_until="networkidle")

        page.evaluate(
            """(title) => {
                document.querySelectorAll('.collapsed')
                    .forEach(e => e.classList.remove('collapsed'));
                document.querySelectorAll('details').forEach(e => e.open = true);
                document.title = title;
                const heading = document.querySelector('h1');
                if (heading) heading.textContent = title;
            }""",
            title,
        )

        page.pdf(
            path=str(destination),
            format="A4",
            print_background=True,
            margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"},
        )
        browser.close()

    size_kb = destination.stat().st_size / 1024
    logger.success(f"{destination}  ({size_kb:.0f} KB)")

    if not keep_html:
        source.unlink()
        logger.info(f"Removed {source.name}; the PDF is the report now.")


if __name__ == "__main__":
    app()
