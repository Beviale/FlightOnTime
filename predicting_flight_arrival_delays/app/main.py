"""FlightOnTime serving application.

Models are fetched from the MLflow registry once at startup and kept in application
state. Both production variants are loaded: 'all' answers whenever the weather resolved,
'noweather' covers the flights it did not.
"""

import asyncio
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request
import gradio as gr
from loguru import logger

from predicting_flight_arrival_delays.app import ui
from predicting_flight_arrival_delays.app.drift_scheduler import (
    shutdown_scheduler,
    start_scheduler,
)
from predicting_flight_arrival_delays.app.monitoring import instrumentator
from predicting_flight_arrival_delays.app.routers import model_info, prediction
from predicting_flight_arrival_delays.app.utils import (
    apply_bundles,
    construct_response,
    load_bundles,
)


async def _load_in_background(app: FastAPI) -> None:
    await asyncio.to_thread(load_bundles, lambda ready: apply_bundles(app, ready))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hold the served models, and what they expect to be asked for."""
    apply_bundles(app, {})
    loading = asyncio.create_task(_load_in_background(app))
    start_scheduler()

    try:
        yield
    finally:
        loading.cancel()
        shutdown_scheduler()
        app.state.bundles = {}
        app.state.required_inputs = set()
        logger.info("Released the served models on shutdown")


app = FastAPI(
    title="FlightOnTime - Flight Arrival Delay Prediction",
    description=("Calibrated delay risk for scheduled U.S. flights, before departure."),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(prediction.router)
app.include_router(model_info.router)


@app.get("/status", tags=["General"])
@construct_response
def status(request: Request):
    """Report what the service is and which models are answering."""
    bundles = getattr(request.app.state, "bundles", {})
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "message": "Welcome to FlightOnTime!",
            "variants": {
                variant: {"run_id": bundle.run_id, "threshold": bundle.threshold}
                for variant, bundle in bundles.items()
            },
            "docs": "/docs",
            "interface": "/",
        },
    }


instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

app = gr.mount_gradio_app(app, ui.build(), path="/")
