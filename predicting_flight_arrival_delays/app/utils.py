"""Shared plumbing for the serving layer: response shape and model bundles."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from typing import Any

import dagshub
from fastapi import FastAPI, Request
from loguru import logger

from predicting_flight_arrival_delays.app.inputs import CANDIDATE_INPUTS, required_inputs
from predicting_flight_arrival_delays.config import (
    DAGSHUB_REPO_NAME,
    DAGSHUB_REPO_OWNER,
    DEFAULT_THRESHOLD,
    SERVED_VARIANTS,
    WINNER_MODEL_STAGE,
)
from predicting_flight_arrival_delays.utils import (
    get_run_metrics,
    get_run_params,
    load_bundle_importance,
    load_model_bundle,
)

THRESHOLD_METRIC = "operating_threshold"


@dataclass(frozen=True)
class Bundle:
    """One served model, with everything needed to score a flight with it.
    """

    variant: str
    model: Any
    transformer: Any
    columns: list[str]
    run_id: str
    threshold: float
    params: dict[str, str]
    metrics: dict[str, float]
    importance: dict[str, float] = field(default_factory=dict)


def registered_name(variant: str) -> str:
    """The registry name a variant is published under.

    Args:
        variant: Production variant, e.g. "all".

    Returns:
        The registered model name.
    """
    return f"flight-delay-{variant}"


def load_bundle(variant: str, stage: str = WINNER_MODEL_STAGE) -> Bundle:
    """Fetch one variant from the model registry, threshold included.

    Args:
        variant: Production variant to load.
        stage: Registry stage or alias to resolve; "None" means the latest version.

    Returns:
        The loaded bundle.

    Raises:
        FileNotFoundError: If the variant has never been registered.
    """
    name = registered_name(variant)
    logger.info(f"Loading {name} ({stage}) from the registry")

    model, transformer, columns, run_id = load_model_bundle(name, stage=stage)
    params, metrics = get_run_params(run_id), get_run_metrics(run_id)
    importance = load_bundle_importance(run_id)

    threshold = metrics.get(THRESHOLD_METRIC)
    if threshold is None:
        logger.warning(
            f"{name} run {run_id} logged no {THRESHOLD_METRIC}; "
            f"falling back to {DEFAULT_THRESHOLD}"
        )
        threshold = DEFAULT_THRESHOLD

    logger.success(f"{name} loaded from run {run_id}, threshold {threshold:.3f}")
    return Bundle(
        variant=variant,
        model=model,
        transformer=transformer,
        columns=columns,
        run_id=run_id,
        threshold=float(threshold),
        params=params,
        metrics=metrics,
        importance=importance,
    )


def load_bundles() -> dict[str, Bundle]:
    """Load every served variant from the registry.

    Returns:
        The variants that loaded, keyed by variant.
    """
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

    bundles = {}
    for variant in SERVED_VARIANTS:
        try:
            bundles[variant] = load_bundle(variant)
        except Exception as e:
            logger.error(f"Could not load the '{variant}' model: {e}")

    if not bundles:
        logger.error("No model loaded - every prediction will answer 503.")
    return bundles


def apply_bundles(app: FastAPI, bundles: dict[str, Bundle]) -> None:
    """Put a set of models into service, and the request contract that goes with them.

    Args:
        app: The application to update.
        bundles: The models to serve.
    """
    app.state.bundles = bundles
    app.state.required_inputs = required_inputs(bundles, CANDIDATE_INPUTS)

    logger.info(f"Serving variants: {sorted(bundles) or 'none'}")
    if bundles:
        ignored = sorted(set(CANDIDATE_INPUTS) - app.state.required_inputs)
        logger.info(
            f"Requests must carry {len(app.state.required_inputs)} of "
            f"{len(CANDIDATE_INPUTS)} columns; feature selection makes the rest "
            f"unnecessary: {ignored or 'none'}"
        )
    else:
        logger.warning(
            "With no model loaded, only the columns the weather forecast needs are asked for. "
        )


def get_bundles(request: Request) -> dict[str, Bundle]:
    """Read the loaded bundles off the application state.

    Args:
        request: The incoming request.

    Returns:
        The bundles that loaded at startup, keyed by variant. Empty if none did.
    """
    return getattr(request.app.state, "bundles", {})


def get_required_inputs(request: Request) -> set[str]:
    """Read the columns the loaded models need off the application state.

    Args:
        request: The incoming request.

    Returns:
        The set computed at startup, or an empty set if no model is loaded.
    """
    return getattr(request.app.state, "required_inputs", set())


def construct_response(f):
    """Wrap an endpoint's result in the API's common envelope.
    """

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        result = f(request, *args, **kwargs)
        response = {
            "message": result["message"],
            "method": request.method,
            "status-code": result["status-code"],
            "timestamp": datetime.now(UTC).isoformat(),
            "url": request.url._url,
        }
        if "data" in result:
            response["data"] = result["data"]
        return response

    return wrap
