# Model Card — FlightOnTime

## Overview

FlightOnTime answers one question: **how likely is this scheduled flight to arrive at
least fifteen minutes late?** It answers before departure, from the timetable and a
weather forecast, and it returns a calibrated probability rather than a verdict.

Two models are registered and served together.

| | `flight-delay-all` | `flight-delay-noweather` |
|---|---|---|
| **Algorithm** | LightGBM (`deep`) | LightGBM (`default`) |
| **Features** | 38 | 26 |
| **Calibration** | isotonic | isotonic |
| **Operating threshold** | 0.2403 | 0.2339 |
| **Registry alias** | `Champion` | `Champion` |
| **Version** | 1 | 1 |

The second is not a lesser model kept in reserve; it is the answer to a specific failure.
A flight more than five days out has no forecast, and Open-Meteo can time out. Rather than
refuse, the service routes that flight to a variant that was never trained on weather at
all. **Every response names the variant that produced it**, so a degraded answer is
distinguishable from a complete one.

## Intended use

**What it is for.** Telling a traveller, an airline analyst or a scheduling tool how much
delay risk a specific flight carries, early enough to act — to leave a longer connection,
to rebook, to staff a gate differently.

**What it is not for.** Compensation decisions, contractual claims, or anything where a
probability would be read as a fact. The model's discrimination is modest by construction
(see *Limitations*), and it says nothing about *why* a particular flight will be late.

**Not a safety system.** It has no bearing on flight safety and is not used in any
operational decision about whether a flight departs.

## How it works

### Inputs

A request describes one scheduled flight. Depending on which model serves it, between 20
and 27 columns are read; `GET /model/inputs` reports exactly which are required and which
the service can fill in.

The caller never sends the weather. The service resolves the airports, fetches the
forecast for the scheduled departure and arrival hours, counts how many other flights
share those airports in those hours, and assembles the feature frame itself.

### Output

```json
{
  "delay_probability": 0.34,
  "is_delayed": 1,
  "variant": "all",
  "threshold": 0.2403,
  "weather": "resolved",
  "approximated": []
}
```

`delay_probability` is the calibrated probability. `is_delayed` applies the operating
threshold. `variant` and `weather` say which model answered and why. `approximated` names
any column the caller omitted that this model leans on.

A model can be scored without every column it was trained on. What the request leaves
out, `complete_frame` fills in before the matrix reaches the estimator: a missing numeric
column takes the **median the transformer learned while fitting**, and a missing
categorical one takes `OTHER`, the same bucket that swallowed every rare level during
training.

So the service names them. `approximated` lists the columns that were absent **and** that
this particular model leans on, ordered by how much it leans:

    "approximated": ["OriginCarrier", "ScheduledTurnaround"]

Two conditions have to hold before a column appears there. It has to be missing — either
not sent at all, or sent as `null`, which counts the same. And it has to carry at least
`IMPORTANT_COLUMN_SHARE` of the model's total importance, five per cent by default:
below that the absence is not worth a caller's attention, and listing every omitted
column would make the field noise.

With `?explain=true`, the answer also carries per-column contributions and a waterfall
that closes on the probability given.

### Training

- **Walk-forward cross-validation**, three folds, train and test contiguous in time.
- **Calibration**: `CalibratedClassifierCV` with isotonic regression over a
  `FrozenEstimator` — the base model is not refit, only its probabilities are mapped.
- **Threshold**: chosen by maximising the F-beta score on the same held-out window the
  calibration is fitted on.
- **Measurement and deployment are two different fits.** Every number reported here comes
  from models fitted on a fold's train, calibrated and thresholded on its validation, and
  scored on its test — none of which is kept. The model that gets *registered* is then
  refit on the last fold's train and validation together, the widest and most recent
  window available, and takes its calibration and its threshold from that fold's test,
  which the refit has never seen.
- **No resampling.** Oversampling, undersampling and SMOTE were all evaluated; none
  improved ROC-AUC, and all three damaged calibration, which is the property this service
  actually sells.

### Hyperparameters

`flight-delay-all` selected `deep`; `flight-delay-noweather` selected `default`.

| | `deep` | `default` |
|---|---|---|
| `n_estimators` | 500 | 300 |
| `learning_rate` | 0.03 | 0.05 |
| `num_leaves` | 127 | 63 |
| `early_stopping_rounds` | 50 | 50 |

LightGBM uses native categorical handling rather than one-hot; the linear and forest
candidates use one-hot.

## Performance

Measured on the held-out test folds — the future relative to every fold's training window.

| Variant | ROC-AUC | PR-AUC | Brier | Recall | Precision | F1.2 | Alert rate |
|---------|---------|--------|-------|--------|-----------|------|-----------|
| **all** | 0.7074 | 0.4433 | 0.1555 | 0.610 | 0.358 | 0.469 | 38.4% |
| **noweather** | 0.6910 | 0.4241 | 0.1599 | 0.651 | 0.332 | 0.459 | 45.0% |

### How to read these

**The Brier score.** 0.1555 sounds good until you know the scale. The base rate across the
test folds is 22.26%, so a model that always answered "22.26%" — looking at nothing —
scores 0.1730. The improvement is:

```
Brier Skill Score = 1 − 0.1555 / 0.1730 = 0.10
```

Ten per cent better than knowing nothing. That is a real gain, and it is small. It is
also close to the ceiling: the Brier decomposes into uncertainty minus resolution plus
reliability, and calibration has already driven the reliability term near zero. Lowering
it further needs a **better model** (i.e. it is a current limitation).

**ROC-AUC 0.707.** Out of a hundred pairs of one delayed and one on-time flight, the model
orders about seventy-one correctly. Enough to rank flights usefully; not enough to call one.

### What each feature group is worth

Read off the difference between variants:

| Removed | ROC-AUC cost |
|---------|-------------|
| Weather | 1.5 points |
| Carrier delay rates | 0.4 points |

And the algorithm matters: LightGBM beats logistic regression by roughly 4 points, which
is the non-linearity in the problem. A delay depends on the *combination* of airport, hour
and weather, not on their sum.

### What the model leans on

For `flight-delay-noweather`, by importance folded back onto the columns a caller sends:

```
Month                  0.094
ScheduledTurnaround    0.066
DayOfWeek              0.063
ReportingAirline       0.035
AircraftDailyLegs      0.024
...
DestCongestion         0.010
OriginCongestion       0.002
```

## Explainability

Two paths, both additive, so contributions sum to the answer rather than approximately to
it.

**Tree models** use SHAP `TreeExplainer`. **Logistic regression** uses the closed form
`coefᵢ · (xᵢ − E[xᵢ])`, verified to agree with `shap.LinearExplainer` — which is why
`feature_means.json` is logged for one-hot models and absent for LightGBM, where
TreeExplainer derives its own base value from the trees.

Contributions are folded from encoded columns back onto the columns a caller actually
sent, so an explanation speaks in `OriginCarrier` rather than in `OriginCarrier_11298AA`.
The waterfall identity closes to 1e-9.

## Limitations

**Performance not so good** The strongest predictors of a delay are
unknowable before departure: whether the inbound aircraft is already late, what the
weather actually did, whether crew scheduling breaks. The model sees the plan and a
forecast.

**Performance declines with the forecast horizon.** A flight five days out has a vaguer
forecast than one tomorrow; beyond five days it has none, and the fallback model answers.

**Trained on 2025–2026 only.**

**Domestic US only.** No international flights, no other jurisdictions.

**It learns airline reputation.** Carrier delay rates are among its strongest features, so
a prediction embeds a judgement about specific airlines drawn from the public record. The
`nocarrier` variant measures the cost of removing it: 0.6 ROC-AUC points.

**Fold-to-fold variance is real.** ROC-AUC declines across folds, and the cause is the
validation period rather than the model: the holiday feature's sign inverts in fold 3, and
day-level variance triples between the first fold and the third.

## Monitoring

The service exposes Prometheus metrics at `/metrics`. Two of them concern this model
directly:

- **`flights_scored_total{variant,weather}`** — the share answered by the fallback. When
  the forecast path breaks, the service returns 200 quickly using `noweather`; from the
  HTTP layer that looks like perfect health, and this label is the only thing that shows
  otherwise.
- **`delay_probability`** — the distribution of what the model predicts. The true label
  arrives weeks after the flight lands, so the shape of the predictions is the earliest
  drift signal available.

A nightly Deepchecks job compares a rolling seven-day window of scored flights against a
fixed sample of the training distribution. Reports land in `reports/drift/`.

## Reproducing

```bash
uv run dvc repro
```

Registration happens in `select_and_register`, which picks the best configuration per
variant on validation, refuses to register anything that fails to beat a random-guess
PR-AUC baseline, and promotes the winner under the `Champion` alias.

Every registered version carries its transformer and its column order in the same MLflow
artifact path, so a version is self-contained: everything needed to prepare data for it is
recoverable from the registry alone.

Runs: [DagsHub MLflow](https://dagshub.com/Beviale/FlightOnTime.mlflow).

## See also

- [Dataset_Card.md](Dataset_Card.md) — the data, its defects, and how it is split
- [FlightOnTime_ML_Canvas.md](FlightOnTime_ML_Canvas.md) — the problem framing
- [Risk_Classification.md](Risk_Classification.md) — assessment under the AI Act
