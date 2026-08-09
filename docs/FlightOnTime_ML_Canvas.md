# FLIGHTONTIME - MACHINE LEARNING CANVAS

**Designed by:** Alessandro Bevilacqua
**Date:** 09/08/2026
**Iteration:** 3

---

## 1. Prediction Task

FlightOnTime performs a **binary classification** task on scheduled U.S. commercial flights, predicting whether a flight will arrive **15 minutes or more late** - the official on-time threshold used by the U.S. Bureau of Transportation Statistics (BTS).

There are two possible outcomes: **delayed** or **on time**. The output is a probability between 0 and 1, plus the binary label based on a chosen threshold.

The prediction is made **before departure**, using only information known at that point. Fields known only after departure (actual departure time, actual delay, delay cause) are never used as input. Predicting cancellations, diversions, or the exact number of delay minutes is out of scope.

The outcome becomes known when the aircraft lands, but it enters the training data only once BTS publishes it, about a month later. Feedback on production performance is therefore always delayed by roughly that much.

---

## 2. Decisions

The system supports **passengers**, **airport staff**, and **airline planners**. It is **advisory**: it informs a human decision and never acts on its own.

What a prediction suggests depends on who is asking. A **passenger** facing a high risk may add extra time before a connection, choose another flight, or arrange a backup plan. **Airport staff** may anticipate pressure on gates and adjust ground handling accordingly. **Airline planners** benefit most from the explanation behind the prediction: knowing whether a route is fragile because of its departure time, its airport, or its exposure to weather points to different corrective actions, such as reshaping the schedule or adding slack to a rotation. A low risk, for all three, means continuing as planned.

Two ways to use the system: **manual entry** (the user types in the flight details, and the system adds weather) and **auto-lookup** (the user gives only the carrier, the flight number and the departure date, and the system fetches everything else automatically).

A default decision threshold is chosen during development (see §6), but it is exposed as a configuration parameter rather than hard-coded, so that an integrating application can tune it properly based on its users' needs.

Predictions are also flagged as low-confidence when the input is unusual - i.e. when some key features take values that are rare or extreme compared to the training data.

---

## 3. Value Proposition

FlightOnTime gives a calibrated delay probability before departure. Passengers planning a connection or a pickup can judge how much extra time to leave instead of guessing, while airport staff and airline planners can anticipate pressure on specific gates, routes, and time slots ahead of time.

Available as a REST API, meant to be integrated into flight booking applications and internal airline or airport management systems.

---

## 4. Data Collection

Training data is drawn from a rolling window, split into expanding-window folds by date rather than at random: flights close together in time share weather and airport congestion, which breaks the independence a random split assumes.

Flight data comes from **BTS**, published on a regular monthly schedule with about a one-month delay. Labels require no manual annotation, since BTS publishes them directly.

Each flight is also matched with weather forecasts at origin and destination, retrieved from the **Open-Meteo** API, and with holiday flags built using the Python `holidays` library, including the distance in days to the nearest federal holiday. Weather is always a forecast, never the observed outcome. A forecast lead time (`forecast_lead_days`) is assigned to every training record, sampled at random between 0 and N days. At inference, the lead time is set by the real gap between the request and the flight date.

Airport coordinates and timezones, needed to match weather to each training flight record and to convert local scheduled times to UTC, come from BTS's own Master Coordinate table (T_MASTER_CORD).

Whenever new months of BTS flight data are ingested for training, T_MASTER_CORD is refreshed alongside them: an airport that opens after the reference table was last built would have no coordinates or timezone, and its flights would fall into the `noweather` path by default - not because a forecast was genuinely unavailable, but because the pipeline never had a location to look one up for. At inference, the auto-lookup path does not rely on this stored reference table at
all: the flight-schedule lookup service used to resolve carrier, flight number and date into a route also returns the origin and destination coordinates and their timezones directly, always current.

Incoming data is monitored periodically to detect meaningful shifts that would justify retraining the model.

---

## 5. Data Sources

| Source | Provides |
|---|---|
| BTS On-Time Performance | Flight records, features and training labels |
| Open-Meteo (Historical Forecast and Previous Runs) | Archived weather forecasts at different lead times, for training |
| Open-Meteo Forecast API | Weather forecast at inference time |
| Python `holidays` library | Federal holiday flags and distance to the nearest holiday |
| BTS Master Coordinate (T_MASTER_CORD) | Airport coordinates and derived timezone, for training |
| Flight-schedule lookup service (AeroDataBox) | Route, schedule, distance, coordinates and timezone, when only carrier, flight number and departure date are given at inference time |

---

## 6. Impact Simulation

**Evaluation protocol.** Candidate models are compared using expanding-window folds: each fold trains on everything up to a cut-off date and tests on the following period, with the training window growing at each step. The split follows time rather than random sampling, since flights close together in time share similar weather and airport congestion, which breaks the independence a random split would assume. Metrics are averaged across folds, and their spread is reported alongside the average, since a single split would reflect whichever period happened to fall on the test side rather
than the model's general behaviour.

**Choosing what to optimize for.** Accuracy is not used: with delays in the minority, a model that always predicts "on time" would score well while being useless. Model selection is driven by ROC-AUC and PR-AUC, which do not depend on where the decision threshold is set. Predicted probabilities are also checked for calibration, because the system reports a risk level rather than only a plain delayed/on-time answer, and that number is only useful if it can be trusted as stated. The decision threshold is chosen only afterwards, on a held-out slice of each training window never seen by the model, by maximising F-beta on delayed flights: missing a real delay costs more than a false positive.

**Fairness.** Group-fairness metrics do not apply here: the system does not process personal data and does not score individual people. Two other risks are checked instead: whether carrier identity is unfairly blamed for delays actually caused by congested routes or airports, and whether predictions based on limited data are presented with more confidence than they deserve.

**Target performance.** No published study matches this setup closely enough to set an absolute target, so no fixed value is committed to at this stage. The model must at least outperform a majority-class predictor.

---

## 7. Making Predictions

| | Manual entry | Auto-lookup |
|---|---|---|
| Latency target (median) | Under 250 ms | Under 700 ms |
| External calls | Weather service | Flight-schedule + weather service |
| If the weather is unavailable | Falls back to the `noweather` model (see §8) | Falls back to the `noweather` model (see §8) |
| If a step fails | Returns a clear error | Returns a clear error |

These latency targets are provisional and they refer to a single flight. They will be verified under load and adjusted once the response times of the external services are measured in practice.

Batch scoring of multiple flights in one request is also supported. Compute target is a CPU-only container; no GPU is needed for training or inference. A small Gradio interface is included as a way to try the API directly.

---

## 8. Building Models

Three model variants are trained and tracked. **all** uses the full feature set and is the main model. **noweather** drops the weather features and is used whenever no forecast is available - either because the flight is further ahead than the forecast horizon, or because the weather service cannot be reached. Both are production models, serving different situations rather than competing for the same role. **nocarrier** removes carrier identity and exists only as an analysis variant, to check whether the carrier carries real signal or mostly reflects the routes and airports it serves.

Three algorithms are compared for each variant: Logistic Regression, Random Forest, and LightGBM, each with hyperparameter selection. Candidates are evaluated with expanding-window folds, so selection reflects performance across several periods rather than one. Model selection is driven by ROC-AUC and PR-AUC, which do not depend on a decision threshold.

For each production variant, the winning configuration is trained once more on a held-out slice of data to fix the decision threshold, then retrained from scratch on the full dataset with that threshold kept fixed - the model benefits from every available record, while the threshold is still chosen only on data it never trained on. The final model is calibrated so that the probabilities it reports can be read properly. Only these final models are released - that is, registered in the model registry and served by the API.

Model explainability is provided through SHAP, which breaks down each individual prediction into the contribution of every feature. This lets users see which factors - route, schedule, weather, or calendar effects - pushed a specific delay prediction up or down.

Retraining is not on a fixed cadence: monitoring periodically checks for a drop in performance or a meaningful shift in incoming data, and retraining is triggered only when one of these checks finds one.

---

## 9. Features

All features are available before departure, drawn from flight schedule data, calendar information, and weather forecasts at origin and destination. No field known only after departure is used, and no feature is built from past outcomes without explicit leakage controls.

Feature selection uses only criteria that do not depend on a specific algorithm — removing constant, redundant, and uninformative columns — so the comparison between algorithms is not confounded by different input sets. Missing values are imputed and categorical features are then encoded to match what each algorithm can use natively, rather than forcing all three onto the same representation.

Airports and carriers with little historical support are grouped into an `OTHER` category.

---

## 10. Monitoring

Service health (latency, error rate, resource usage, uptime) is tracked separately for each input path (manual entry and auto-lookup) and for each model variant serving the request (`all` or `noweather`). External services are monitored on their own.

Incoming data is compared against the training distribution to detect drift. Model performance metrics are recomputed on production data once labels become available and compared against the values recorded when the model was released - with calibration checked separately. A drop in either performance or data stability triggers retraining. A new model replaces the one currently in use only if it improves on ROC-AUC, PR-AUC, and calibration.

Whether a user actually avoided a missed connection cannot be measured directly. Usage volume gives a partial indication, but this remains a real limit: the system can show that its predictions are accurate and well calibrated, not that anyone was better off.

---

> Based on the Machine Learning Canvas v1.1 by Louis Dorard, Ph.D. — CC BY-SA 4.0 — [ownml.co](https://ownml.co)