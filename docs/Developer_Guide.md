# Developer Guide

How to set the project up, run the pipeline, and extend it without breaking the parts
that depend on what you changed.

For what the system *is*, see the [README](../README.md). For the data and the models, see
the [dataset card](Dataset_Card.md) and the [model card](Model_Card.md).

## Contents

1. [Prerequisites](#prerequisites)
2. [Environment setup](#1-environment-setup)
3. [Fetching data and models](#2-fetching-data-and-models)
4. [Running the pipeline](#3-running-the-pipeline)
5. [Adding a model configuration](#4-adding-a-model-configuration)
6. [Adding a feature variant](#5-adding-a-feature-variant)
7. [Adding a feature](#6-adding-a-feature)
8. [Working on the service](#7-working-on-the-service)
9. [Tests and quality gates](#8-tests-and-quality-gates)
10. [Monitoring locally](#9-monitoring-locally)
11. [Deploying](#10-deploying)
12. [Quick reference](#quick-reference)

## Prerequisites

- **Python 3.12** — pinned in `pyproject.toml` as `~=3.12.0`
- **uv** — [install](https://docs.astral.sh/uv/getting-started/installation/)
- **DVC** — comes with the `pipeline` dependency group
- **Docker** — only for the container and the monitoring stack
- Accounts: **DagsHub** (data remote and MLflow registry), **RapidAPI** (AeroDataBox, for
  the auto-lookup path only)

## 1. Environment setup

```bash
git clone https://github.com/Beviale/FlightOnTime.git
cd FlightOnTime
uv sync
```

`uv sync` installs the main dependencies plus three groups, declared in `pyproject.toml`:

| Group | Holds | Reaches the Docker image? |
|-------|-------|---------------------------|
| *(main)* | fastapi, gradio, lightgbm, shap, skops, apscheduler | **yes** |
| `dev` | pytest, ruff, great-expectations, pynblint | no |
| `pipeline` | dvc, mlflow, playwright, pyarrow | no |
| `notebooks` | jupyterlab, seaborn | no |

The Dockerfile runs `uv sync --no-default-groups`, so **anything the running service needs
must be a main dependency**. This has bitten twice: `skops` lives under `mlflow` in the
`pipeline` group, and its absence made every prediction answer 503 in production while
working perfectly in development.

### Environment variables

```bash
# .env in the project root
DAGSHUB_USER_TOKEN=<token>       # MLflow registry; also the DVC remote password
RAPIDAPI_KEY=<key>               # AeroDataBox, auto-lookup only
MODEL_RELOAD_TOKEN=<any secret>  # optional, guards POST /model/reload
DRIFT_SCHEDULE_ENABLED=0         # optional, turns the nightly drift job off
```

### DVC credentials

The remote is an **HTTP** remote, not S3, so it takes basic auth rather than access keys:

```bash
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <your_dagshub_username>
uv run dvc remote modify origin --local password <your_dagshub_token>
```

These land in `.dvc/config.local`, which is gitignored — which is why CI has to recreate
them on every run.

## 2. Fetching data and models

```bash
uv run dvc pull                    # everything: several gigabytes
```

Usually you want less than that:

| You are working on | Pull |
|--------------------|------|
| The API | `data/external/airports.csv data/external/airport_identity.csv` |
| Data validation | `data/raw data/interim/flights_preprocessed.parquet` |
| Behavioural tests | `data/processed/selection` |
| Drift | `data/interim/drift_reference.csv` |

Models come from the MLflow registry rather than DVC, and are fetched at startup — nothing
to pull.

## 3. Running the pipeline

```bash
uv run dvc repro
```

> ⚠️ **`dvc repro` deletes a stage's outputs before re-running it**, and it pulls in
> upstream stages it considers stale. `download_weather` holds around 2,800 files from a
> rate-limited public API; re-fetching them is hours of work and cannot be undone by the
> run itself.
>
> Use `--single-item` when you mean one stage and nothing before it:
>
> ```bash
> uv run dvc repro --single-item build_drift_reference
> ```
>
> And when a stage is stale only because a file it does not really depend on changed, tell
> DVC the outputs are still good instead of rebuilding them:
>
> ```bash
> uv run dvc commit -f
> ```

### The stages

| # | Stage | Produces |
|---|-------|----------|
| 1 | `download_bts` | `data/raw/` — monthly extracts |
| 2 | `build_airports` | `data/external/airports.csv` |
| 3 | `prepare_flights` | `data/interim/flights_features.parquet` |
| 4 | `build_airport_identity` | `data/external/airport_identity.csv` |
| 5 | `download_weather` | `data/external/weather/` |
| 6 | `join_weather` | `data/interim/flights_preprocessed.parquet` |
| 7 | `build_drift_reference` | `data/interim/drift_reference.csv` |
| 8 | `split_data_cv` | `data/processed/selection/` — folds per variant |
| 9 | `train_evaluate_save_metrics` | `metrics/selection/` — 24 instances via `foreach` |
| 10 | `select_and_register` | `metrics/winner/`, and the registered models |

### Useful subsets

```bash
uv run dvc repro split_data_cv                       # up to the folds
uv run dvc repro train_evaluate_save_metrics         # all 24 candidates
uv run dvc repro select_and_register                 # pick and register
```

To run one candidate without DVC:

```bash
uv run predicting_flight_arrival_delays/modeling/train_evaluate_save_metrics.py \
  --variant all --model lightgbm --config deep
```

## 4. Adding a model configuration

### Step 1 — the hyperparameters

`predicting_flight_arrival_delays/modeling/hyperparams.yaml`, under the algorithm:

```yaml
lightgbm:
  very_deep:
    n_estimators: 800
    learning_rate: 0.02
    num_leaves: 255
    verbose: -1
    n_jobs: -1
    early_stopping_rounds: 50
```

Nothing else is needed for an existing algorithm — the config name is read from here.

### Step 2 — the pipeline

`dvc.yaml`, in the `foreach` list of `train_evaluate_save_metrics`:

```yaml
      - { variant: all, model: lightgbm, config: very_deep}
```

### Step 3 — run it

```bash
uv run dvc repro train_evaluate_save_metrics
```

`select_and_register` picks the best configuration per variant on validation PR-AUC, so a
new candidate competes automatically. It will not register anything that fails to beat a
random-guess PR-AUC baseline.

### Adding a whole algorithm

Three more edits:

1. **`config.py`** — declare how it encodes categoricals:

   ```python
   ENCODING = {
       "logistic_regression": "onehot",
       "random_forest": "onehot",
       "lightgbm": "native",
       "catboost": "native",     # new
   }
   ```

2. **`modeling/train.py`** — construct it in `build_estimator`.

3. **`hyperparams.yaml`** — at least one configuration.

The encoding choice matters more than it looks. `onehot` builds a sparse matrix and
computes `feature_means` for the linear explanation path; `native` keeps pandas
`category` dtypes and uses SHAP `TreeExplainer`. Getting it wrong means the model is
scored on one feature set and registered with another — a bug this project has already
had.

## 5. Adding a feature variant

A variant is a feature set defined by what it **removes**.

### Step 1 — declare it

`predicting_flight_arrival_delays/data/features.py`:

```python
NOROTATION_DROP = ["AircraftDailyLegs", "LegPosition", "ScheduledTurnaround"]

VARIANTS: dict[str, list[str]] = {
    "all": [],
    "noweather": NOWEATHER_DROP,
    "nocarrier": CARRIER_COLUMNS,
    "norotation": NOROTATION_DROP,   # new
}
```

### Step 2 — decide whether it is served

`config.py`:

```python
PRODUCTION_VARIANTS = ["all", "noweather"]   # add it here only if it should be served
```

An unserved variant is an ablation: it is trained and measured, and the comparison tells
you what those columns are worth. `nocarrier` exists for exactly that reason.

### Step 3 — the pipeline

Add the variant to `split_data_cv`'s outputs and to the `foreach` list, one entry per
configuration you want to try.

### Step 4 — run it

```bash
uv run dvc repro split_data_cv
uv run dvc repro train_evaluate_save_metrics
```

**If you serve it,** `app/inference.py` must know when to route a flight to it. The
existing rule is weather: a flight whose forecast resolved goes to `all`, otherwise to
`noweather`.

## 6. Adding a feature

Most features are derived in `data/preprocess.py`. Adding one touches more than that file,
and the order matters.

1. **`preprocess.py`** — compute it. If it comes from a raw BTS column, add that column to
   `KEEP_COLUMNS` in `config.py` first.
2. **`config.py`** — if the model must never see it, add it to `SERVICE_COLUMNS`. Join
   keys and identifiers belong there.
3. **`data/transform.py`** — if it is categorical and high-cardinality, consider
   `RATE_ONLY_COLUMNS` so it is replaced by its historical delay rate rather than one-hot
   encoded. If it is cyclical, add it to `CYCLICAL_COLUMNS` with its period.
4. **`app/schema.py`** — if a caller must supply it, add the field with its validation.
5. **`app/enrichment/builder.py`** — if the service derives it, add it to
   `FEATURE_FRAME_COLUMNS` and compute it there too.
6. **`tests/test_flight_data/`** — add an expectation for it.

**The trap.** Steps 1 and 5 are two separate implementations of the same feature: one for
training, one for serving. If they disagree, the model sees one thing in training and
another in production, and nothing fails loudly. The behavioural tests exist partly to
catch this.

Feature selection runs automatically and may drop what you added — correlation above 0.98,
Cramér's V association above 0.98, or mutual information below 1e-5. The log says which,
and why.

## 7. Working on the service

```bash
uv run uvicorn predicting_flight_arrival_delays.app.main:app --port 7860 --reload
```

- `http://localhost:7860` — Gradio
- `http://localhost:7860/docs` — OpenAPI
- `http://localhost:7860/status` — which variants are answering

### Structure

```
app/
├── main.py          FastAPI app, lifespan, Gradio mount
├── schema.py        request and response models
├── inputs.py        which columns each model needs
├── inference.py     route each flight to a variant, apply thresholds
├── utils.py         load bundles from the registry
├── monitoring.py    Prometheus metrics
├── drift.py         log what was scored
├── routers/         prediction.py, model_info.py
└── enrichment/      everything the caller does not send
```

### Two things about startup

**Models load in the background.** The lifespan opens the port first and fetches the models
after. On a small container one artifact takes over four minutes; blocking on that means
uvicorn never binds, the host's health check restarts the container, and the replacement
finds port 7860 still held — a restart loop that never serves a request. Until the models
land, predictions answer 503, which is what they already do when nothing is registered.

**A newly registered model does not appear on its own.** `load_bundles` runs once. To pick
up a new version without restarting:

```bash
curl -X POST -H "X-Reload-Token: $MODEL_RELOAD_TOKEN" http://localhost:7860/model/reload
```

### Loading models from the registry

Use `models:/<name>@<alias>`, not `runs:/<run_id>/model`. MLflow 3 records the logged
model's location in the run as a **local temp directory of the machine that trained it**,
so resolving through the run fails everywhere else. `load_model_bundle` already does this;
the transformer and column list are plain run artifacts and are fetched the other way.

## 8. Tests and quality gates

```bash
uv run pytest                                          # everything
uv run pytest tests/test_predicting_flight_arrival_delays -q   # unit, no data needed
uv run pytest tests/test_flight_data -q                # needs data/raw and the parquet
uv run pytest tests/test_behavioral_model -q           # needs the registered champions
```

An HTML report is written to `reports/pytest/report.html` on every run.

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

Ruff covers `predicting_flight_arrival_delays/`, `tests/` and `scripts/`, set by `include`
in `pyproject.toml`. The CI workflow applies the fixes and **pushes them back to your
branch**, so `git pull` before continuing after a CI run.

### The data suites

Run as scripts, they validate every row and write a JSON report; run under pytest, they
sample 200,000 rows and only assert.

```bash
uv run python tests/test_flight_data/raw_test.py        # → reports/great_expectations/
uv run python tests/test_flight_data/processed_test.py
```

### CI

| Workflow | Trigger | Does |
|----------|---------|------|
| `ruff.yml` | pull request | lint, autofix, push the fixes back, then fail on what is left |
| `tests.yml` | pull request | `unit` job (no credentials) and `data` job (DVC + registry) |
| `pynblint.yml` | PR touching `notebooks/**` | notebook linting |
| `deploy.yml` | push to `main` | build the `deploy` branch, push to the Space, wait for `RUNNING` |

Required secrets: `DAGSHUB_TOKEN`, `HF_TOKEN`. Required variables: `DAGSHUB_USER`,
`HF_SPACE`.

## 9. Monitoring locally

```bash
docker compose up --build
```

| | URL |
|---|---|
| Service | http://localhost:7860 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:4444 |
| Locust | http://localhost:8089 |

### Load testing

Locust generates complete flights from twelve real airport ids. By default they depart
10–40 days out, past the five-day forecast horizon, so the run stays off Open-Meteo — fifty
simulated users would exhaust a free tier in under two minutes and you would be measuring
their rate limiter. To exercise the weather path deliberately, with few users:

```bash
LOCUST_LEAD_DAYS_MIN=1 LOCUST_LEAD_DAYS_MAX=4 uvx locust -f locust/locustfile.py \
  --host http://127.0.0.1:7860
```

### Adding a metric

Define it in `app/monitoring.py`, increment it in `record_scoring` or at the call site.
Nothing else: Prometheus scrapes `/metrics` and stores whatever it finds, and
`prometheus.yml` does not need to know the metric exists.

### Drift

```bash
uv run scripts/drift_report.py              # last 7 days
uv run scripts/drift_report.py --days 30
uv run scripts/drift_report.py --days 0     # the whole log
```

The script declares its own environment in a PEP 723 header, and `uv run` builds it. It has
to: Deepchecks pins scikit-learn below 1.6 and numpy below 2, while this project trains on
1.9 and 2.5. Do not try to add Deepchecks to `pyproject.toml`.

The scheduler in `app/drift_scheduler.py` shells out to the same script nightly, passing the
settings from `config.py`.

## 10. Deploying

A merge to `main` runs `deploy.yml`, which builds a `deploy` branch holding only what the
service needs, pushes it to the Space, and polls until Hugging Face reports `RUNNING`.

**When adding a file the service needs at runtime**, two places have to know:

1. `.dockerignore` — an exception, if it lives under an excluded directory
2. `deploy.yml` — the `ITEMS` list, or it never reaches the deploy branch

`data/` is excluded wholesale except three files: the two airport tables and the drift
reference. Adding a fourth means editing both.

**System libraries** go in the Dockerfile's `apt-get` line. `libgomp1` is there because
LightGBM's OpenMP runtime is not a Python package and nothing pulls it in.

## Quick reference

| Task | Command |
|------|---------|
| Install | `uv sync` |
| Fetch the airport tables | `uv run dvc pull data/external/airports.csv data/external/airport_identity.csv` |
| Run the API | `uv run uvicorn predicting_flight_arrival_delays.app.main:app --port 7860` |
| Run the full stack | `docker compose up --build` |
| Unit tests | `uv run pytest tests/test_predicting_flight_arrival_delays -q` |
| Lint and format | `uv run ruff check --fix . && uv run ruff format .` |
| One pipeline stage | `uv run dvc repro --single-item <stage>` |
| Accept stale stages | `uv run dvc commit -f` |
| Upload data | `uv run dvc push` |
| Drift report | `uv run scripts/drift_report.py` |
| Reload the served model | `curl -X POST -H "X-Reload-Token: $TOKEN" .../model/reload` |
