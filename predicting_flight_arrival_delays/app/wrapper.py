"""What the interface calls.

Gradio binds a Python function to every button and panel, and those functions live
here. Each one takes what the widgets hold, sends a request to the service's own
API, and turns the answer into the markdown or the table the page displays.
"""
from typing import Any

import httpx
from loguru import logger
import pandas as pd

from predicting_flight_arrival_delays.config import API_URL

TIMEOUT_SECONDS = 60.0

VARIANT_LABEL = {
    "all": "full model, weather included",
    "noweather": "fallback model, no weather",
}
WEATHER_LABEL = {
    "ok": "forecast retrieved",
    "beyond_forecast_horizon": "flight too far ahead for a forecast to exist",
    "unavailable": "weather service unreachable",
    "unknown_airport": "airport with no known coordinates",
}


async def call(method: str, path: str, **kwargs) -> dict[str, Any]:
    """Send one request to the service's own API.

    Args:
        method: HTTP method.
        path: Path on the API, starting with a slash.
        **kwargs: Passed through to httpx.

    Returns:
        The decoded body on success, or {"error": "..."} describing what went wrong.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        try:
            response = await client.request(method, f"{API_URL}{path}", **kwargs)
        except httpx.HTTPError as e:
            logger.error(f"{method} {path} failed: {e}")
            return {"error": f"Service unreachable: {e}"}

    if response.is_success:
        return response.json()

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    logger.warning(f"{method} {path} answered {response.status_code}: {detail}")
    return {"error": f"{response.status_code} - {detail}"}


def render_prediction(data: dict[str, Any]) -> str:
    """Turn one scored flight into something a person can read.

    Args:
        data: The data block of a prediction response.

    Returns:
        Markdown stating the risk, and what the answer is based on.
    """
    probability = data["delay_probability"]
    delayed = data["is_delayed"]
    verdict = "🔴 At risk of delay" if delayed else "🟢 Expected on time"

    variant = VARIANT_LABEL.get(data["variant"], data["variant"])
    weather = WEATHER_LABEL.get(data["weather"], data["weather"])

    lines = [
        f"# {verdict}",
        f"## Probability of delay: {probability:.1%}",
        "",
        f"Threshold above which a flight is called late: **{data['threshold']:.1%}**.",
        f"Answered by the **{variant}** — {weather}.",
    ]
    if data["variant"] == "noweather":
        lines.append(
            "\n> Without a forecast this estimate is less informed than the one the "
            "main model would have given."
        )

    explanations = data.get("explanations") or []
    if explanations:
        lines.append("\n### What pushed this answer")
        lines += [
            f"- **{item['column']}** pushed "
            + ("towards a delay" if item["contribution"] > 0 else "towards arriving on time")
            + f" ({item['contribution']:+.3f})"
            for item in explanations
        ]

    approximated = data.get("approximated") or []
    if approximated:
        lines.append(
            "\n> ⚠️ **This answer may be less accurate than it looks.** The model "
            "leans heavily on the following, and the request left them out, so a "
            "training average stood in for each: "
            + ", ".join(f"`{column}`" for column in approximated)
            + ". Sending them should give a sharper answer."
        )
    return "\n".join(lines)


async def predict_lookup(
    flight_date, marketing_carrier, operating_carrier, number, origin, dest
) -> str:
    """Score a flight the user only named.

    Args:
        flight_date: Departure date, as the date picker gives it.
        marketing_carrier: The code the flight is sold under.
        operating_carrier: The code of the airline flying it.
        number: Flight number.
        origin: Departure airport, IATA.
        dest: Arrival airport, IATA.

    Returns:
        Markdown with the answer, or with what went wrong.
    """
    if not all([flight_date, marketing_carrier, operating_carrier, number, origin, dest]):
        return "Fill in every field."

    payload = {
        "FlightDate": str(flight_date)[:10],
        "MarketingCarrier": marketing_carrier,
        "ReportingAirline": operating_carrier,
        "FlightNumber": int(number),
        "Origin": origin,
        "Dest": dest,
    }

    body = await call("POST", "/predictions/lookup?explain=true", json=payload)
    if "error" in body:
        return f"### Could not answer\n\n{body['error']}"

    data = body["data"]
    resolved = data["resolved"]
    return render_prediction(data) + (
        f"\n\n---\n**Flight found:** {resolved['Origin']} → {resolved['Dest']}, "
        f"scheduled departure at {resolved['DepTimeDecimal']:.2f} local, "
        f"{resolved['Distance']:.0f} miles."
    )


async def predict_batch(file) -> pd.DataFrame:
    """Score a CSV of flights described in full.

    Args:
        file: The uploaded file, as Gradio hands it over.

    Returns:
        One row per flight with the answer, or a single row describing the failure.
    """
    if file is None:
        return pd.DataFrame({"error": ["Upload a CSV first."]})

    try:
        flights = pd.read_csv(file)
    except Exception as e:
        logger.error(f"Could not read the uploaded CSV: {e}")
        return pd.DataFrame({"error": [f"Unreadable CSV: {e}"]})

    payload = flights.where(pd.notna(flights), None).to_dict(orient="records")
    body = await call("POST", "/batch-predictions", json=payload)
    if "error" in body:
        return pd.DataFrame({"error": [body["error"]]})

    return pd.DataFrame(
        [
            {
                "flight": row["index"],
                "verdict": "🔴 At risk" if row["is_delayed"] else "🟢 On time",
                "probability": f"{row['delay_probability']:.1%}",
                "model": row["variant"],
                "weather": WEATHER_LABEL.get(row["weather"], row["weather"]),
            }
            for row in body["data"]["results"]
        ]
    )


def _as_markdown_table(rows: dict[str, Any]) -> str:
    """Render a flat mapping as a two-column table."""
    lines = ["| | |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in sorted(rows.items())]
    return "\n".join(lines)


async def get_metrics() -> str:
    """Report how each served model scored when it was released."""
    body = await call("GET", "/model/metrics")
    if "error" in body:
        return f"### Metrics unavailable\n\n{body['error']}"

    blocks = []
    for variant, info in sorted(body["data"].items()):
        numbers = {key: f"{value:.4f}" for key, value in info["metrics"].items()}
        blocks.append(
            f"## {variant} — {VARIANT_LABEL.get(variant, '')}\n\n"
            f"Operating threshold: **{info['operating_threshold']:.3f}**\n\n"
            + _as_markdown_table(numbers)
            + f"\n\nRun: `{info['run_id']}`"
        )
    return "\n\n---\n\n".join(blocks)


async def get_hyperparameters() -> str:
    """Report how each served model was configured and trained."""
    body = await call("GET", "/model/hyperparameters")
    if "error" in body:
        return f"### Hyperparameters unavailable\n\n{body['error']}"

    blocks = []
    for variant, info in sorted(body["data"].items()):
        blocks.append(
            f"## {variant}\n\n### The algorithm's own hyperparameters\n\n"
            + _as_markdown_table(info["hyperparameters"])
            + "\n\n### How the run was configured\n\n"
            + _as_markdown_table(info["training"])
        )
    return "\n\n---\n\n".join(blocks)


async def get_inputs() -> str:
    """Report which columns a manual request has to carry."""
    body = await call("GET", "/model/inputs")
    if "error" in body:
        return f"### Not available\n\n{body['error']}"

    data = body["data"]
    if not data.get("derived_from_served_models", True):
        return (
            "## No model is in service"
        )

    return (
        "## Columns a manual request must carry\n\n"
        f"{', '.join(f'`{c}`' for c in data['required'])}\n\n"
        "## Columns the served models do not read\n\n"
        "Feature selection dropped them, so sending them changes nothing.\n\n"
        f"{', '.join(f'`{c}`' for c in data['ignored']) or '_none_'}\n\n"
        "## Added by the service\n\n"
        f"{', '.join(f'`{c}`' for c in data['supplied_by_the_service'])}"
    )
