"""What the served models are, what a request has to tell them, and swapping them."""

from http import HTTPStatus
import os
import secrets
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request
from loguru import logger

from predicting_flight_arrival_delays.app.inputs import CANDIDATE_INPUTS
from predicting_flight_arrival_delays.app.utils import (
    Bundle,
    apply_bundles,
    construct_response,
    get_bundles,
    get_required_inputs,
    load_bundles,
)
from predicting_flight_arrival_delays.data.features import (
    WEATHER_COLUMNS_DESTINATION,
    WEATHER_COLUMNS_ORIGIN,
)

router = APIRouter(tags=["Model"])


HYPERPARAMETER_PREFIX = "hp_"

Variant = Query(default=None, description="Narrow the answer to one served variant.")

# To reload the registered final model variants
RELOAD_TOKEN_VARIABLE = "MODEL_RELOAD_TOKEN"


def check_reload_token(supplied: str | None) -> None:
    """Refuse a reload that does not carry the configured secret.

    Args:
        supplied: The token the caller sent, if any.

    Raises:
        HTTPException: 503 if no secret is configured, so the endpoint is off rather
            than open; 401 if the token does not match.
    """
    expected = os.environ.get(RELOAD_TOKEN_VARIABLE, "").strip()
    if not expected:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=f"Reloading is disabled: {RELOAD_TOKEN_VARIABLE} is not set.",
        )
    if not supplied or not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED, detail="Invalid or missing reload token."
        )


def served(request: Request, variant: str | None) -> dict[str, Bundle]:
    """The bundles an answer should cover.

    Args:
        request: The incoming request.
        variant: One variant to narrow to, or None for every served one.

    Returns:
        The bundles to report on, keyed by variant.

    Raises:
        HTTPException: 503 if no model is loaded, 404 if the variant asked for is not
            among those that are.
    """
    bundles = get_bundles(request)
    if not bundles:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="No model is loaded, so there is nothing to describe.",
        )
    if variant is None:
        return bundles
    if variant not in bundles:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"'{variant}' is not being served. Loaded: {', '.join(sorted(bundles))}.",
        )
    return {variant: bundles[variant]}


@router.get("/model/hyperparameters")
@construct_response
def hyperparameters(request: Request, variant: str | None = Variant):
    """Report how each served model was configured and trained."""
    data: dict[str, Any] = {}
    for name, bundle in served(request, variant).items():
        tuned = {
            key.removeprefix(HYPERPARAMETER_PREFIX): value
            for key, value in bundle.params.items()
            if key.startswith(HYPERPARAMETER_PREFIX)
        }
        data[name] = {
            "run_id": bundle.run_id,
            "hyperparameters": tuned,
            "training": {
                key: value
                for key, value in bundle.params.items()
                if not key.startswith(HYPERPARAMETER_PREFIX)
            },
        }

    return {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": data}


@router.get("/model/metrics")
@construct_response
def metrics(request: Request, variant: str | None = Variant):
    """Report how each served model scored when it was released."""
    data: dict[str, Any] = {
        name: {
            "run_id": bundle.run_id,
            "operating_threshold": bundle.threshold,
            "metrics": bundle.metrics,
        }
        for name, bundle in served(request, variant).items()
    }

    return {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": data}


@router.get("/model/inputs")
@construct_response
def inputs(request: Request):
    """List the columns a request has to carry, and those it need not.

    The served models decide this between them: a column both of them dropped during
    feature selection is not worth asking for.
    """
    bundles = get_bundles(request)
    required = get_required_inputs(request)
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "derived_from_served_models": bool(bundles),
            "required": sorted(required),
            "ignored": sorted(set(CANDIDATE_INPUTS) - required),
            "supplied_by_the_service": [
                "LeadDays",
                *WEATHER_COLUMNS_ORIGIN,
                *WEATHER_COLUMNS_DESTINATION,
            ],
            "variants": sorted(bundles),
        },
    }


@router.post("/model/reload")
@construct_response
def reload(request: Request, x_reload_token: str | None = Header(default=None)):
    """Put the versions currently in the registry into service.

    Raises:
        HTTPException: 401 or 503 if the token is missing or invalid, 502 if the
            registry yielded nothing while models were already being served.
    """
    check_reload_token(x_reload_token)

    previous = {variant: bundle.run_id for variant, bundle in get_bundles(request).items()}
    fresh = load_bundles()

    if not fresh and previous:
        logger.error("Reload found no model; keeping the ones already in service")
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail="The registry yielded no model; the ones already in service were kept.",
        )

    apply_bundles(request.app, fresh)

    changed = {
        variant: {"was": previous.get(variant), "now": bundle.run_id}
        for variant, bundle in fresh.items()
        if previous.get(variant) != bundle.run_id
    }
    logger.success(f"Reloaded; {len(changed)} variant(s) changed version")

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "serving": sorted(fresh),
            "changed": changed,
            "dropped": sorted(set(previous) - set(fresh)),
            "required_inputs": sorted(request.app.state.required_inputs),
        },
    }
