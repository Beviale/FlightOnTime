"""Prometheus metrics for the serving application.
"""

import os

from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "flightontime")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "api")

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/gradio_api.*", "/theme.css", "/assets/.*"],
    inprogress_name=f"{NAMESPACE}_inprogress",
    inprogress_labels=True,
)

for metric in (
    metrics.latency(
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
    ),
    metrics.requests(
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
    ),
    metrics.request_size(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM),
    metrics.response_size(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM),
):
    instrumentator.add(metric)


requests_total = Counter(
    f"{NAMESPACE}_{SUBSYSTEM}_scoring_requests_total",
    "Scoring requests received, by endpoint and shape.",
    ["endpoint", "kind"],
)

flights_scored_total = Counter(
    f"{NAMESPACE}_{SUBSYSTEM}_flights_scored_total",
    "Flights scored, by the model that answered and how its weather resolved.",
    ["variant", "weather"],
)

outcomes_total = Counter(
    f"{NAMESPACE}_{SUBSYSTEM}_outcomes_total",
    "Flights called delayed or on time, at the threshold in force.",
    ["outcome"],
)

approximated_total = Counter(
    f"{NAMESPACE}_{SUBSYSTEM}_approximated_flights_total",
    "Flights scored while missing a column the model leans on.",
)

errors_total = Counter(
    f"{NAMESPACE}_{SUBSYSTEM}_errors_total",
    "Requests refused before an answer could be produced.",
    ["endpoint", "reason"],
)

delay_probability = Histogram(
    f"{NAMESPACE}_{SUBSYSTEM}_delay_probability",
    "Predicted delay risk. Its shape is the drift signal available before the labels are.",
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0],
)

batch_size = Histogram(
    f"{NAMESPACE}_{SUBSYSTEM}_batch_size",
    "Flights per batch request.",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200],
)

scoring_seconds = Histogram(
    f"{NAMESPACE}_{SUBSYSTEM}_scoring_seconds",
    "Time to enrich and score, without the HTTP overhead around it.",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)


def record_scoring(endpoint: str, kind: str, results: list[dict], elapsed: float) -> None:
    """Record one scoring request and every flight it answered.

    Args:
        endpoint: The route, as it should read on a dashboard.
        kind: 'single', 'batch' or 'lookup'.
        results: The per-flight results the endpoint is about to return.
        elapsed: Seconds spent enriching and scoring.
    """
    requests_total.labels(endpoint=endpoint, kind=kind).inc()
    scoring_seconds.labels(endpoint=endpoint).observe(elapsed)

    if kind == "batch":
        batch_size.observe(len(results))

    for result in results:
        flights_scored_total.labels(variant=result["variant"], weather=result["weather"]).inc()
        outcomes_total.labels(outcome="delayed" if result["is_delayed"] else "on_time").inc()
        delay_probability.observe(result["delay_probability"])

        if result.get("approximated"):
            approximated_total.inc()


def record_error(endpoint: str, reason: str) -> None:
    """Record a request that never reached an answer.

    Args:
        endpoint: The route, as it should read on a dashboard.
        reason: Short slug for the cause, e.g. 'model_unavailable'.
    """
    errors_total.labels(endpoint=endpoint, reason=reason).inc()
