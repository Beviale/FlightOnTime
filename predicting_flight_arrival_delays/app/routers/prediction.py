"""Prediction endpoints, manual-entry path.

The caller supplies every feature but the weather; the service adds the forecast, and
whether it could is what decides which model answers. Both endpoints report the variant
that served the request and why the weather was or was not available, so a degraded
answer is never mistaken for a full one.
"""

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger

from predicting_flight_arrival_delays.app.enrichment.aerodatabox import (
    FlightNotFoundError,
    ScheduleUnavailableError,
)
from predicting_flight_arrival_delays.app.enrichment.builder import build_feature_frame
from predicting_flight_arrival_delays.app.enrichment.identity import (
    AmbiguousAirportError,
    UnknownAirportError,
)
from predicting_flight_arrival_delays.app.enrichment.lookup import resolve
from predicting_flight_arrival_delays.app.inference import ModelUnavailableError, score
from predicting_flight_arrival_delays.app.schema import FlightLookupRequest, FlightRequest
from predicting_flight_arrival_delays.app.utils import (
    construct_response,
    get_bundles,
    get_required_inputs,
)
from predicting_flight_arrival_delays.config import MAX_BATCH_SIZE

router = APIRouter(tags=["Prediction"])

Threshold = Query(
    default=None,
    gt=0,
    lt=1,
    description=(
        "One cutoff for whichever variant answers, replacing the threshold it was "
        "released with."
    ),
)


def check_inputs(flights: list[FlightRequest], required: set[str]) -> None:
    """Refuse a flight that leaves out a column one of the served models reads.

    Which columns those are is decided by the models themselves, so this cannot live
    in the request schema: it changes when a new version is registered.

    Args:
        flights: The flights being scored.
        required: The columns the loaded models need, from app.inputs.

    Raises:
        HTTPException: 422, naming the missing columns and the flight they belong to.
    """
    for index, flight in enumerate(flights):
        missing = sorted(required - flight.supplied())
        if missing:
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail=(
                    f"Flight {index} leaves out {len(missing)} columns the served models "
                    f"read: {', '.join(missing)}. GET /model/inputs lists them all."
                ),
            )


def run_scoring(
    request: Request, flights: list[FlightRequest], threshold: float | None
) -> list[dict[str, Any]]:
    """Enrich, score, and pair each result back with the flight that asked for it.

    Args:
        request: The incoming request, carrying the loaded bundles.
        flights: The flights to score.
        threshold: Optional override of the released operating threshold.

    Returns:
        One result per flight, in the order they were sent.

    Raises:
        HTTPException: 422 if a flight leaves out a column a served model reads,
            503 if the model a flight needs is not loaded.
    """
    check_inputs(flights, get_required_inputs(request))

    frame, weather_status = build_feature_frame(flights)

    try:
        scored = score(frame, get_bundles(request), threshold)
    except ModelUnavailableError as e:
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail=str(e)) from e

    return [
        {
            "input": flight.model_dump(mode="json"),
            "delay_probability": float(row.delay_probability),
            "is_delayed": int(row.is_delayed),
            "variant": row.variant,
            "threshold": float(row.threshold),
            "weather": status,
        }
        for flight, row, status in zip(
            flights, scored.itertuples(), weather_status, strict=True
        )
    ]


@router.post("/predictions")
@construct_response
def predict(request: Request, payload: FlightRequest, threshold: float | None = Threshold):
    """Score one scheduled flight."""
    result = run_scoring(request, [payload], threshold)[0]

    logger.success(
        f"{payload.ReportingAirline}{payload.FlightNumberReportingAirline} "
        f"{payload.Origin}-{payload.Dest}: {result['delay_probability']:.3f} "
        f"({result['variant']}, weather {result['weather']})"
    )
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": result,
    }


@router.post("/batch-predictions")
@construct_response
def predict_batch(
    request: Request, payload: list[FlightRequest], threshold: float | None = Threshold
):
    """Score several scheduled flights in one request.

    Raises:
        HTTPException: 422 if the batch is empty or larger than MAX_BATCH_SIZE.
    """
    if not payload:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail="The batch carries no flights."
        )
    if len(payload) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=f"Batch of {len(payload)} exceeds the limit of {MAX_BATCH_SIZE} flights.",
        )

    results = run_scoring(request, payload, threshold)
    for index, result in enumerate(results):
        result["index"] = index

    logger.success(f"Scored a batch of {len(results)} flights")
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"results": results, "batch_size": len(results)},
    }


@router.post("/predictions/lookup")
@construct_response
def predict_lookup(
    request: Request, payload: FlightLookupRequest, threshold: float | None = Threshold
):
    """Score a flight the caller only named.

    Raises:
        HTTPException: 404 if the flight or an airport is unknown, 502 if the schedule
            service cannot be reached, 503 if the model is not loaded.
    """
    try:
        flights = resolve([payload])
    except (FlightNotFoundError, UnknownAirportError, AmbiguousAirportError) as e:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e)) from e
    except ScheduleUnavailableError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_GATEWAY, detail=str(e)) from e

    result = run_scoring(request, flights, threshold)[0]
    result["resolved"] = flights[0].model_dump(mode="json")

    logger.success(
        f"{payload.MarketingCarrier}{payload.FlightNumber} {payload.Origin}-{payload.Dest} "
        f"(operated by {payload.ReportingAirline}): {result['delay_probability']:.3f} "
        f"({result['variant']}, weather {result['weather']})"
    )
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": result,
    }
