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
import pandas as pd

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
from predicting_flight_arrival_delays.app.inference import (
    ModelUnavailableError,
    prepared_matrix,
    score,
)
from predicting_flight_arrival_delays.app.inputs import approximated_inputs, complete_frame
from predicting_flight_arrival_delays.app.schema import FlightLookupRequest, FlightRequest
from predicting_flight_arrival_delays.app.utils import (
    construct_response,
    get_bundles,
    get_required_inputs,
)
from predicting_flight_arrival_delays.config import (
    EXPLANATION_COLUMN_COUNT,
    IMPORTANT_COLUMN_SHARE,
    MAX_BATCH_SIZE,
)
from predicting_flight_arrival_delays.modeling.explainability import (
    request_column_contributions,
    waterfall_terms,
)

router = APIRouter(tags=["Prediction"])

Explain = Query(
    default=False,
    description=(
        "Also report which columns pushed the answer where it went. Off by default."
    ),
)

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


def _score(
    request: Request, flights: list[FlightRequest], threshold: float | None
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Validate, enrich and score.

    Args:
        request: The incoming request, carrying the loaded bundles.
        flights: The flights to score.
        threshold: Optional override of the released operating threshold.

    Returns:
        The feature frame.

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

    return frame, weather_status, scored


def run_scoring(
    request: Request,
    flights: list[FlightRequest],
    threshold: float | None,
    explain: bool = False,
) -> list[dict[str, Any]]:
    """Enrich, score, and pair each result back with the flight that asked for it.

    Args:
        request: The incoming request, carrying the loaded bundles.
        flights: The flights to score.
        threshold: Optional override of the released operating threshold.
        explain: Whether each result should also say what pushed it.

    Returns:
        One result per flight, in the order they were sent.

    Raises:
        HTTPException: 422 if a flight leaves out a column a served model reads,
            503 if the model a flight needs is not loaded.
    """
    frame, weather_status, scored = _score(request, flights, threshold)
    bundles = get_bundles(request)

    results = []
    for position, (flight, row, status) in enumerate(
        zip(flights, scored.itertuples(), weather_status, strict=True)
    ):
        result = {
            "input": flight.model_dump(mode="json"),
            "delay_probability": float(row.delay_probability),
            "is_delayed": int(row.is_delayed),
            "variant": row.variant,
            "threshold": float(row.threshold),
            "weather": status,
            "approximated": approximated_inputs(
                flight, bundles[row.variant], IMPORTANT_COLUMN_SHARE
            ),
        }
        if explain:
            one = frame.iloc[[position]]
            result["explanations"] = _explain(one, bundles[row.variant])
            result["waterfall"] = _waterfall(
                one, bundles[row.variant], float(row.delay_probability)
            )
        results.append(result)

    return results


@router.post("/predictions")
@construct_response
def predict(
    request: Request,
    payload: FlightRequest,
    threshold: float | None = Threshold,
    explain: bool = Explain,
):
    """Score one scheduled flight."""
    result = run_scoring(request, [payload], threshold, explain)[0]

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
    request: Request,
    payload: FlightLookupRequest,
    threshold: float | None = Threshold,
    explain: bool = Explain,
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

    result = run_scoring(request, flights, threshold, explain)[0]
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


def _waterfall(frame: pd.DataFrame, bundle, probability: float) -> dict[str, Any] | None:
    """The same explanation, with what it takes to close on the answer given.

    Args:
        frame: The feature frame for that one flight.
        bundle: The model that answered.
        probability: The probability the caller was given.

    Returns:
        Base value, leading contributions, the summed rest and the calibration step,
        or None if the estimator cannot be read.
    """
    matrix = prepared_matrix(complete_frame(frame, bundle.transformer), bundle)
    return waterfall_terms(
        bundle.model,
        matrix,
        bundle.transformer,
        bundle.params.get("algorithm", ""),
        probability,
        EXPLANATION_COLUMN_COUNT,
    )


def _explain(frame: pd.DataFrame, bundle) -> list[dict[str, Any]]:
    """What pushed one flight's answer where it went, in the caller's vocabulary.

    Args:
        frame: The feature frame for that one flight.
        bundle: The model that answered.

    Returns:
        The leading contributions, or an empty list if the estimator cannot be read.
    """
    matrix = prepared_matrix(complete_frame(frame, bundle.transformer), bundle)
    contributions = request_column_contributions(
        bundle.model,
        matrix,
        bundle.transformer,
        bundle.params.get("algorithm", ""),
        EXPLANATION_COLUMN_COUNT,
    )
    if not contributions:
        logger.warning(f"No explanation available for the {bundle.variant} model.")
    return contributions


@router.post("/explanations")
@construct_response
def explain(request: Request, payload: FlightRequest, threshold: float | None = Threshold):
    """Score one flight and say what pushed the answer where it went.

    A contribution is positive when it pushed towards a delay and negative when it
    pushed towards an on-time arrival.
    """
    frame, weather_status, scored = _score(request, [payload], threshold)
    row = next(scored.itertuples())
    bundle = get_bundles(request)[row.variant]

    logger.success(
        f"Explained {payload.Origin}-{payload.Dest}: "
        f"{row.delay_probability:.3f} ({row.variant})"
    )
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "input": payload.model_dump(mode="json"),
            "delay_probability": float(row.delay_probability),
            "is_delayed": int(row.is_delayed),
            "variant": row.variant,
            "threshold": float(row.threshold),
            "weather": weather_status[0],
            "approximated": approximated_inputs(
                payload, bundle, IMPORTANT_COLUMN_SHARE
            ),
            "explanations": _explain(frame, bundle),
            "waterfall": _waterfall(frame, bundle, float(row.delay_probability)),
        },
    }
